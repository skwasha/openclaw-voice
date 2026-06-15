#!/usr/bin/env python3
"""
ElevenLabs TTS provider.

Calls the ElevenLabs text-to-speech REST API directly via aiohttp and
requests raw PCM output, then resamples down to u-law @ 8kHz for RTP.

Env:
    ELEVENLABS_API_KEY
"""

import logging
import os

import aiohttp

from .tts_base import TTSProvider
from .audio_utils import pcm_any_rate_to_ulaw_8k

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class ElevenLabsTTSProvider(TTSProvider):
    def __init__(
        self,
        api_key: str = "",
        voice_id: str = "",
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_24000",
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.sample_rate = int(output_format.split("_")[-1])
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        if not self.api_key or not self.voice_id:
            logger.error("ElevenLabs TTS not configured (missing api_key or voice_id)")
            return b""

        session = await self._ensure_session()
        url = API_URL.format(voice_id=self.voice_id)
        params = {"output_format": self.output_format}
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
        }

        try:
            async with session.post(url, params=params, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"ElevenLabs TTS HTTP {resp.status}: {body[:200]}")
                    return b""
                pcm = await resp.read()
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return b""

        return pcm_any_rate_to_ulaw_8k(pcm, self.sample_rate)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
