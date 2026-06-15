#!/usr/bin/env python3
"""Abstract base class for Speech-to-Text providers."""

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """
    Converts a buffered utterance (linear PCM16 audio) into text.

    Implementations receive 16-bit linear PCM audio at `input_sample_rate`
    (defaults to 16kHz, the rate `anthropic_voice_bridge.py` resamples to
    before calling `transcribe`).
    """

    #: Sample rate (Hz) the bridge should resample audio to before calling
    #: `transcribe`. Override in subclasses if a provider wants a different rate.
    input_sample_rate: int = 16000

    @abstractmethod
    async def transcribe(self, pcm16_audio: bytes) -> str:
        """
        Transcribe a single utterance.

        Args:
            pcm16_audio: 16-bit signed linear PCM, mono, at `input_sample_rate`.

        Returns:
            The transcribed text (empty string if nothing recognizable).
        """
        raise NotImplementedError

    async def close(self):
        """Optional cleanup hook (close HTTP sessions, unload models, etc.)."""
        pass
