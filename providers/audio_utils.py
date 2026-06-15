#!/usr/bin/env python3
"""
Audio conversion helpers shared by STT/TTS providers.

The SIP/RTP side always speaks G.711 u-law @ 8kHz, 20ms (160-byte) frames.
STT/TTS providers generally want 16-bit linear PCM at 8kHz, 16kHz, 22.05kHz
or 24kHz. These helpers convert between those representations using the
stdlib `audioop` module (already a dependency via audio.py) plus numpy.
"""

import audioop
import numpy as np

RTP_FRAME_BYTES = 160   # 160 bytes u-law = 20ms @ 8kHz
RTP_FRAME_MS = 20
SIP_SAMPLE_RATE = 8000


def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """G.711 u-law -> 16-bit linear PCM (same sample rate, 8kHz)."""
    return audioop.ulaw2lin(ulaw_bytes, 2)


def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """16-bit linear PCM -> G.711 u-law (same sample rate)."""
    return audioop.lin2ulaw(pcm_bytes, 2)


def resample_pcm16(pcm_bytes: bytes, in_rate: int, out_rate: int, state=None):
    """Resample 16-bit mono linear PCM. Returns (data, new_state)."""
    if in_rate == out_rate:
        return pcm_bytes, state
    return audioop.ratecv(pcm_bytes, 2, 1, in_rate, out_rate, state)


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """16-bit linear PCM bytes -> float32 numpy array in [-1, 1]."""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 numpy array in [-1, 1] -> 16-bit linear PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def ulaw_8k_to_pcm16_16k(ulaw_bytes: bytes) -> bytes:
    """Convenience: RTP u-law @ 8kHz -> linear PCM @ 16kHz (for Whisper)."""
    pcm8k = ulaw_to_pcm16(ulaw_bytes)
    pcm16k, _ = resample_pcm16(pcm8k, 8000, 16000)
    return pcm16k


def pcm_any_rate_to_ulaw_8k(pcm_bytes: bytes, in_rate: int) -> bytes:
    """Convenience: linear PCM at any rate -> RTP u-law @ 8kHz."""
    pcm8k, _ = resample_pcm16(pcm_bytes, in_rate, 8000)
    return pcm16_to_ulaw(pcm8k)


def split_into_rtp_frames(ulaw_bytes: bytes) -> list[bytes]:
    """Split u-law audio into 160-byte (20ms) RTP frames, dropping the remainder."""
    frames = []
    for i in range(0, len(ulaw_bytes) - RTP_FRAME_BYTES + 1, RTP_FRAME_BYTES):
        frames.append(ulaw_bytes[i:i + RTP_FRAME_BYTES])
    return frames
