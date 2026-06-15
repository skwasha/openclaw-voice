#!/usr/bin/env python3
"""Abstract base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """
    Converts text into G.711 u-law audio @ 8kHz, ready to be split into
    160-byte RTP frames by `audio_utils.split_into_rtp_frames`.

    Implementations are responsible for synthesizing audio at whatever
    native sample rate they produce and resampling/encoding down to
    u-law @ 8kHz before returning (see `audio_utils.pcm_any_rate_to_ulaw_8k`).
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech for `text`.

        Returns:
            G.711 u-law encoded audio bytes @ 8kHz (any length; the caller
            will split it into 20ms/160-byte RTP frames).
        """
        raise NotImplementedError

    async def close(self):
        """Optional cleanup hook (close HTTP sessions, unload models, etc.)."""
        pass
