#!/usr/bin/env python3
"""
Kokoro TTS provider (local, MIT licensed).

Runs the Kokoro-82M model locally via the `kokoro` pip package. No API key,
no per-call cost - good fit for a permanent voice daemon.

Install:
    pip install kokoro soundfile

Kokoro generates audio at 24kHz; we resample down to u-law @ 8kHz for RTP.
"""

import asyncio
import logging

import numpy as np

from .tts_base import TTSProvider
from .audio_utils import pcm_any_rate_to_ulaw_8k

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


class KokoroTTSProvider(TTSProvider):
    """Local Kokoro TTS. Lazily loads the model on first use."""

    def __init__(
        self,
        model: str = "kokoro-v1.0",
        voice: str = "af_heart",
        lang_code: str = "a",
        device: str = "cpu",
        speed: float = 1.0,
    ):
        self.model_name = model
        self.voice = voice
        self.lang_code = lang_code
        self.device = device
        self.speed = speed
        self._pipeline = None
        self._lock = asyncio.Lock()

    def _load_pipeline(self):
        from kokoro import KPipeline

        logger.info(f"Loading Kokoro pipeline (lang_code={self.lang_code}, device={self.device})...")
        pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        logger.info("Kokoro pipeline loaded")
        return pipeline

    async def _ensure_pipeline(self):
        if self._pipeline is None:
            async with self._lock:
                if self._pipeline is None:
                    loop = asyncio.get_event_loop()
                    self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
        return self._pipeline

    def _generate(self, text: str) -> np.ndarray:
        pipeline = self._pipeline
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=self.voice, speed=self.speed):
            # `audio` may be a torch.Tensor or numpy array depending on version
            arr = audio
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            chunks.append(np.asarray(arr, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            return b""

        await self._ensure_pipeline()

        loop = asyncio.get_event_loop()
        samples = await loop.run_in_executor(None, self._generate, text)
        if samples.size == 0:
            return b""

        from .audio_utils import float32_to_pcm16
        pcm24k = float32_to_pcm16(samples)
        return pcm_any_rate_to_ulaw_8k(pcm24k, KOKORO_SAMPLE_RATE)

    async def close(self):
        self._pipeline = None
