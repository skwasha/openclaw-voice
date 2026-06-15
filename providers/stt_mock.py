#!/usr/bin/env python3
"""Mock STT provider - for testing the call pipeline without a real model."""

import logging

from .stt_base import STTProvider

logger = logging.getLogger(__name__)


class MockSTTProvider(STTProvider):
    """Returns a fixed canned transcript regardless of audio content.

    Useful for exercising the SIP/RTP -> Anthropic -> TTS pipeline end to
    end without installing faster-whisper or hitting an API.
    """

    def __init__(self, fixed_text: str = "Hello, this is a test."):
        self.fixed_text = fixed_text

    async def transcribe(self, pcm16_audio: bytes) -> str:
        logger.debug(f"MockSTT: received {len(pcm16_audio)} bytes, returning fixed text")
        return self.fixed_text
