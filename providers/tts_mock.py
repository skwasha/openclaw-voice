#!/usr/bin/env python3
"""Mock TTS provider - generates silence/tone instead of real speech."""

import logging

import numpy as np

from .tts_base import TTSProvider
from .audio_utils import float32_to_pcm16, pcm_any_rate_to_ulaw_8k

logger = logging.getLogger(__name__)


class MockTTSProvider(TTSProvider):
    """Generates a short tone (or silence) sized roughly to the text length.

    Useful for exercising the call pipeline without a real TTS engine or
    model download.
    """

    def __init__(self, tone_hz: float = 0.0, sample_rate: int = 8000):
        # tone_hz = 0 -> silence. Set e.g. 440 for an audible test tone.
        self.tone_hz = tone_hz
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        # ~150 words/min reading speed -> rough duration estimate, clamped.
        words = max(1, len(text.split()))
        duration_s = min(8.0, max(0.5, words / 2.5))
        n_samples = int(self.sample_rate * duration_s)

        if self.tone_hz > 0:
            t = np.arange(n_samples) / self.sample_rate
            samples = 0.2 * np.sin(2 * np.pi * self.tone_hz * t).astype(np.float32)
        else:
            samples = np.zeros(n_samples, dtype=np.float32)

        pcm = float32_to_pcm16(samples)
        logger.debug(f"MockTTS: synthesized {duration_s:.2f}s for {words} words")
        return pcm_any_rate_to_ulaw_8k(pcm, self.sample_rate)
