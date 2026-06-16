#!/usr/bin/env python3
"""
Anthropic Voice Agent Bridge
=============================

Bridges phone calls (SIP/RTP) to the Anthropic Messages API, using
pluggable local/cloud STT and TTS providers in place of a realtime voice
WebSocket (Anthropic has no native voice API).

Drop-in replacement for `xai_voice_bridge.XAIVoiceBridge`: exposes the same
interface that `openclaw_voice.py`'s `_run_call` drives (`connect`,
`disconnect`, `send_audio`, `receive_audio`, `is_speaking`, `running`,
`_listen_task`, `on_tool_call`, `outbound_queue`).

AUDIO PIPELINE (turn-based, VAD-segmented):
============================================
  Phone (SIP/RTP, u-law @ 8kHz)
       |
       |  VAD-segmented utterance buffer
       v
  STTProvider.transcribe()  -> text
       v
  Anthropic Messages API (+ tool use loop)  -> text
       v
  TTSProvider.synthesize()  -> u-law @ 8kHz
       v
  outbound_queue (160-byte RTP frames) -> Phone
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from audio import VoiceActivityDetector
from providers.audio_utils import (
    RTP_FRAME_BYTES,
    ulaw_to_pcm16,
    resample_pcm16,
    pcm_any_rate_to_ulaw_8k,
    split_into_rtp_frames,
)
from providers.stt_base import STTProvider
from providers.tts_base import TTSProvider

logger = logging.getLogger(__name__)

# Emoji / symbol ranges Kokoro (and most TTS) can't pronounce.
_STRIP_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # misc symbols, pictographs, emoticons, transport
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001FA00-\U0001FA9F"  # chess, medical, etc.
    "‍"                 # zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Sentence splitter: only break on ! and ? — these are unambiguous sentence
# boundaries in conversational text.  We intentionally leave "." alone to
# avoid false splits on abbreviations ("Dr.", "Mr.", "e.g.", URLs, etc.)
# which are common in Claude responses.  Splitting on just !? is strictly
# better than no splitting: single-sentence replies are unchanged, and any
# response with ! or ? gets streamed sentence-by-sentence.
_SENT_SPLIT_RE = re.compile(r'(?<=[!?])\s+')


def _split_sentences(text: str) -> list:
    """
    Split `text` into streaming-TTS chunks on ! and ? boundaries.

    Falls back to the whole text as a single chunk if there are no splits.
    """
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts if parts else [text.strip()]


def _clean_for_tts(text: str) -> str:
    """Strip emoji/symbols and normalise whitespace before sending to TTS."""
    return _STRIP_RE.sub('', text).strip()


# VAD-based turn segmentation tuning
SILENCE_FRAMES_THRESHOLD = 25   # 25 * 20ms = 500ms of silence ends an utterance
MIN_SPEECH_FRAMES = 6           # 6 * 20ms = 120ms minimum speech to count as an utterance
MAX_BUFFER_FRAMES = 1500        # 30s safety cap


class AnthropicVoiceBridge:
    """
    Bridge between SIP/RTP audio and the Anthropic Messages API.

    Unlike `XAIVoiceBridge` (a realtime bidirectional WebSocket), this is a
    turn-based pipeline: it buffers caller audio using VAD until it detects
    end-of-utterance, transcribes it, sends it to Claude (handling any tool
    calls), then synthesizes and queues the reply.
    """

    def __init__(
        self,
        api_key: str,
        stt_provider: STTProvider,
        tts_provider: TTSProvider,
        model: str = "claude-sonnet-4-6",
        instructions: Optional[str] = None,
        tools: Optional[list] = None,
        max_tokens: int = 1024,
        vad_aggressiveness: int = 2,
        greeting: Optional[str] = "Hello, this is OpenClaw. How can I help you?",
    ):
        self.api_key = api_key
        self.stt = stt_provider
        self.tts = tts_provider
        self.model = model
        self.instructions = instructions or "You are a helpful voice assistant."
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.greeting = greeting

        self._client: Optional[AsyncAnthropic] = None
        self._vad = VoiceActivityDetector(aggressiveness=vad_aggressiveness)

        self.running = False
        self.is_speaking = False
        self.session_id: Optional[str] = None
        self.outbound_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._listen_task: Optional[asyncio.Task] = None

        # Conversation state (Anthropic Messages format)
        self.messages: List[Dict[str, Any]] = []

        # Turn-detection state
        self._buffer: List[bytes] = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False
        self._turn_processing = False

        # Callbacks (mirrors XAIVoiceBridge)
        self.on_transcript: Optional[Callable[[str, str], None]] = None
        self.on_tool_call: Optional[Callable[[str, str, dict], Any]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialize the Anthropic client and queue an opening greeting."""
        try:
            self._client = AsyncAnthropic(api_key=self.api_key)
            self.session_id = str(uuid.uuid4())
            self.running = True

            # Keep an "always-running" task so callers can `await bridge._listen_task`
            # and `.cancel()` it on hangup, matching XAIVoiceBridge's interface.
            self._listen_task = asyncio.create_task(self._idle_loop())

            if self.greeting:
                asyncio.create_task(self._speak(self.greeting))
                self.messages.append({"role": "assistant", "content": self.greeting})

            logger.info(f"Anthropic voice bridge ready (session={self.session_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic bridge: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False

    async def _idle_loop(self):
        """Placeholder background task - keeps `_listen_task` awaitable/cancelable."""
        try:
            while self.running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def disconnect(self):
        """Tear down the bridge."""
        self.running = False
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        try:
            await self.stt.close()
            await self.tts.close()
        except Exception as e:
            logger.warning(f"Provider close error: {e}")
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Anthropic voice bridge disconnected")

    async def update_session(self, instructions: Optional[str] = None, tools: Optional[list] = None):
        """Update instructions/tools mid-call (e.g. after PIN authentication)."""
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = tools

    # ------------------------------------------------------------------
    # Audio in (SIP -> bridge)
    # ------------------------------------------------------------------

    async def send_audio(self, ulaw_audio: bytes):
        """
        Feed one RTP frame (160 bytes u-law @ 8kHz) from the phone into the
        VAD-based turn buffer. When end-of-utterance is detected, kicks off
        STT -> Anthropic -> TTS processing in the background.
        """
        if not self.running or self._turn_processing:
            return

        try:
            is_speech = self._vad.is_speech(ulaw_audio)
        except Exception:
            is_speech = False

        if is_speech:
            self._buffer.append(ulaw_audio)
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            self._in_speech = True
        elif self._in_speech:
            self._buffer.append(ulaw_audio)
            self._silence_frame_count += 1
            if (self._silence_frame_count >= SILENCE_FRAMES_THRESHOLD
                    and self._speech_frame_count >= MIN_SPEECH_FRAMES):
                utterance = self._buffer
                self._reset_buffer()
                self._turn_processing = True
                asyncio.create_task(self._process_utterance(utterance))
        else:
            # Leading silence before any speech - don't let it grow unbounded.
            if len(self._buffer) > 5:
                self._buffer.clear()

        if len(self._buffer) > MAX_BUFFER_FRAMES:
            # Safety valve: force-process a runaway utterance (e.g. VAD never
            # sees silence because of a noisy line).
            utterance = self._buffer
            self._reset_buffer()
            self._turn_processing = True
            asyncio.create_task(self._process_utterance(utterance))

    def _reset_buffer(self):
        self._buffer = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False

    # ------------------------------------------------------------------
    # Audio out (bridge -> SIP)
    # ------------------------------------------------------------------

    async def receive_audio(self) -> Optional[bytes]:
        """Pop one 160-byte u-law RTP frame for playback, or None if idle."""
        try:
            frame = self.outbound_queue.get_nowait()
            return frame
        except asyncio.QueueEmpty:
            if self.outbound_queue.empty() and not self._turn_processing:
                self.is_speaking = False
            return None

    # ------------------------------------------------------------------
    # Turn pipeline: STT -> Anthropic -> TTS
    # ------------------------------------------------------------------

    async def _process_utterance(self, frames: List[bytes]):
        try:
            ulaw_bytes = b"".join(frames)
            pcm8k = ulaw_to_pcm16(ulaw_bytes)
            target_rate = getattr(self.stt, "input_sample_rate", 16000)
            pcm_target, _ = resample_pcm16(pcm8k, 8000, target_rate)

            text = await self.stt.transcribe(pcm_target)
            text = (text or "").strip()
            if not text:
                logger.debug("Empty transcript, ignoring utterance")
                return

            logger.info(f"User: {text}")
            if self.on_transcript:
                self.on_transcript("user", text)

            reply = await self._call_anthropic(text)
            reply = (reply or "").strip()
            if not reply:
                return

            logger.info(f"Assistant: {reply}")
            if self.on_transcript:
                self.on_transcript("assistant", reply)

            await self._speak(reply)
        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self._turn_processing = False

    async def _speak(self, text: str):
        """
        Synthesize `text` sentence-by-sentence and stream to the outbound
        queue.

        Splitting into sentences lets playback start on the first sentence
        while subsequent sentences are still being synthesized, cutting
        perceived latency significantly for multi-sentence replies.
        """
        text = _clean_for_tts(text)
        if not text:
            return

        self.is_speaking = True
        sentences = _split_sentences(text)

        for sentence in sentences:
            if not self.running:
                break
            sentence = _clean_for_tts(sentence)
            if not sentence:
                continue
            try:
                ulaw_audio = await self.tts.synthesize(sentence)
                if ulaw_audio:
                    for frame in split_into_rtp_frames(ulaw_audio):
                        await self.outbound_queue.put(frame)
            except Exception as e:
                logger.error(f"TTS error ({sentence!r:.40}): {e}")
                # Continue with remaining sentences rather than going silent.

    async def _call_anthropic(self, user_text: str) -> str:
        """Send `user_text` to Claude, resolving any tool_use turns, and
        return the final assistant text."""
        if not self._client:
            return ""

        self.messages.append({"role": "user", "content": user_text})
        anthropic_tools = _convert_tools(self.tools)

        # Bound the tool-use loop so a misbehaving tool can't hang the call.
        for _ in range(5):
            try:
                kwargs = dict(
                    model=self.model,
                    system=self.instructions,
                    messages=self.messages,
                    max_tokens=self.max_tokens,
                )
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                response = await self._client.messages.create(**kwargs)
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                if self.on_error:
                    self.on_error(str(e))
                return "Sorry, I'm having trouble reaching the assistant right now."

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                )

            # Resolve tool_use blocks
            tool_results = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                logger.info(f"Tool call: {block.name}({block.input})")
                try:
                    if self.on_tool_call:
                        result = await self.on_tool_call(block.id, block.name, block.input)
                    else:
                        result = f"No handler for tool '{block.name}'"
                except Exception as e:
                    result = f"Error: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

            self.messages.append({"role": "user", "content": tool_results})

        return "Sorry, that's taking longer than expected. Let's try something else."


def _convert_tools(openai_style_tools: list) -> list:
    """Convert OpenAI/xAI-style function tool defs to Anthropic tool defs.

    Input:  {"type": "function", "name": ..., "description": ..., "parameters": {...}}
    Output: {"name": ..., "description": ..., "input_schema": {...}}
    """
    converted = []
    for tool in openai_style_tools or []:
        converted.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted
