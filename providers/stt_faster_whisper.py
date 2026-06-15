#!/usr/bin/env python3
"""
faster-whisper STT provider (local, CTranslate2-based Whisper).

Runs entirely on Noguchi's CPU (or GPU if available) - no per-call API cost.

Install:
    pip install faster-whisper
"""

import asyncio
import logging

from .stt_base import STTProvider
from .audio_utils import pcm16_to_float32

logger = logging.getLogger(__name__)


class FasterWhisperSTTProvider(STTProvider):
    input_sample_rate = 16000

    def __init__(
        self,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        cpu_threads: int = 1,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self._model = None
        self._lock = asyncio.Lock()

    def _load_model(self):
        from faster_whisper import WhisperModel

        logger.info(f"Loading faster-whisper model '{self.model_name}' "
                    f"(device={self.device}, compute_type={self.compute_type}, "
                    f"cpu_threads={self.cpu_threads})...")
        # cpu_threads=1: ctranslate2 spawns its own OpenMP thread pool, which
        # can otherwise contend/segfault with torch's MKL OpenMP runtime
        # loaded by the TTS provider in the same process (see
        # KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS notes in openclaw_voice.py).
        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        logger.info("faster-whisper model loaded")
        return model

    async def _ensure_model(self):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, self._load_model)
        return self._model

    def _run_transcribe(self, audio) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # caller already does VAD-based turn segmentation
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, pcm16_audio: bytes) -> str:
        if not pcm16_audio:
            return ""

        await self._ensure_model()

        audio = pcm16_to_float32(pcm16_audio)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, self._run_transcribe, audio)
        except Exception as e:
            logger.error(f"faster-whisper transcription error: {e}")
            return ""

        logger.debug(f"faster-whisper transcript: {text!r}")
        return text

    async def close(self):
        self._model = None
