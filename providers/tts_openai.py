#!/usr/bin/env python3
"""
OpenAI TTS provider.

Uses the OpenAI Audio Speech API (`tts-1` / `tts-1-hd` / `gpt-4o-mini-tts`).
Requests raw 24kHz 16-bit PCM and resamples down to u-law @ 8kHz for RTP.

Install:
    pip install openai

Env:
    OPENAI_API_KEY
"""

import logging
import os

from .tts_base import TTSProvider
from .audio_utils import pcm_any_rate_to_ulaw_8k

logger = logging.getLogger(__name__)

OPENAI_PCM_SAMPLE_RATE = 24000


class OpenAITTSProvider(TTSProvider):
    def __init__(
        self,
        api_key: str = "",
        model: str = "tts-1",
        voice: str = "nova",
        instructions: str = "",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        client = self._ensure_client()

        kwargs = dict(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",  # raw 24kHz 16-bit signed PCM, little-endian
        )
        if self.instructions:
            kwargs["instructions"] = self.instructions

        try:
            response = await client.audio.speech.create(**kwargs)
            pcm24k = response.read() if hasattr(response, "read") else response.content
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return b""

        return pcm_any_rate_to_ulaw_8k(pcm24k, OPENAI_PCM_SAMPLE_RATE)

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None
