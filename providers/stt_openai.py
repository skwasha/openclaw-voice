#!/usr/bin/env python3
"""
OpenAI Whisper API STT provider.

Wraps the audio in a WAV container and sends it to OpenAI's transcription
endpoint (whisper-1 / gpt-4o-transcribe).

Install:
    pip install openai

Env:
    OPENAI_API_KEY
"""

import io
import logging
import os
import wave

from .stt_base import STTProvider

logger = logging.getLogger(__name__)


class OpenAISTTProvider(STTProvider):
    input_sample_rate = 16000

    def __init__(
        self,
        api_key: str = "",
        model: str = "whisper-1",
        language: str = "en",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.language = language
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    def _to_wav(self, pcm16_audio: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.input_sample_rate)
            wf.writeframes(pcm16_audio)
        buf.seek(0)
        buf.name = "audio.wav"  # OpenAI SDK inspects .name for content-type
        return buf

    async def transcribe(self, pcm16_audio: bytes) -> str:
        if not pcm16_audio:
            return ""

        client = self._ensure_client()
        wav_file = self._to_wav(pcm16_audio)

        try:
            resp = await client.audio.transcriptions.create(
                model=self.model,
                file=wav_file,
                language=self.language,
            )
        except Exception as e:
            logger.error(f"OpenAI STT error: {e}")
            return ""

        text = (resp.text or "").strip()
        logger.debug(f"OpenAI STT transcript: {text!r}")
        return text

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None
