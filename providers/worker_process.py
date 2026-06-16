#!/usr/bin/env python3
"""
Process-isolated provider wrappers.

Runs each heavy provider (Kokoro TTS, faster-whisper STT) in its own
subprocess so that torch's OpenMP runtime and ctranslate2's OpenMP runtime
never share the same process.  This eliminates the duplicate-OpenMP segfault
that occurs with cpu_threads > 1 on macOS Intel (where torch 2.2.2 is the
latest available wheel and ships Intel MKL's libiomp5).

Architecture
------------
Main process  <--Pipe-->  Worker process (one per provider)
  ProcessProvider.call()     _worker_target(): load model, serve requests

Each worker:
  - Gets its own clean OpenMP runtime (no duplicate-library conflict)
  - Has OMP/MKL/KMP_NUM_THREADS constraints removed (set by main process to
    work around the same conflict) so it can use multiple CPU threads freely
  - Loads the underlying model lazily on the first request (no cold-start on
    spawn; startup just forks the interpreter, which is fast)

Usage (via __init__.py factories - no need to use this module directly):
  ProcessSTTAdapter('providers.stt_faster_whisper', 'FasterWhisperSTTProvider', **kwargs)
  ProcessTTSAdapter('providers.tts_kokoro', 'KokoroTTSProvider', **kwargs)
"""

import asyncio
import importlib
import logging
import multiprocessing
import os
import sys
from typing import Any

from .stt_base import STTProvider
from .tts_base import TTSProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------

def _worker_target(
    pipe: Any,
    inherited_sys_path: list,
    provider_module: str,
    provider_class_name: str,
    init_kwargs: dict,
) -> None:
    """
    Subprocess entry point.  Imports and instantiates the provider, then
    serves ('call', method, args, kwargs) messages over `pipe` until a
    ('shutdown',) message arrives.

    Sends back ('ok', result) or ('error', message_str).
    """
    # Restore parent's sys.path so relative intra-package imports work.
    sys.path[:] = inherited_sys_path

    # Remove the single-thread constraints set in the main process to work
    # around the duplicate-OpenMP conflict.  This worker has only ONE library
    # (either torch OR ctranslate2) so there is no conflict here, and we
    # want to use multiple threads for real throughput.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_NUM_THREADS"):
        os.environ.pop(var, None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    module = importlib.import_module(provider_module)
    cls = getattr(module, provider_class_name)
    provider = cls(**init_kwargs)

    try:
        while True:
            try:
                msg = pipe.recv()
            except EOFError:
                break  # main process closed the connection

            if msg[0] == "shutdown":
                break

            _, method, args, kwargs = msg
            try:
                coro = getattr(provider, method)(*args, **kwargs)
                result = loop.run_until_complete(coro)
                pipe.send(("ok", result))
            except Exception as exc:
                pipe.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            loop.run_until_complete(provider.close())
        except Exception:
            pass
        loop.close()
        pipe.close()


# ---------------------------------------------------------------------------
# Main-process side
# ---------------------------------------------------------------------------

class ProcessProvider:
    """
    Spawns a persistent worker subprocess for a provider and proxies calls
    over a multiprocessing Pipe.

    Thread-safety: one in-flight request at a time (asyncio.Lock).  STT and
    TTS are always called sequentially within a turn so this is never a
    bottleneck.
    """

    def __init__(
        self,
        provider_module: str,
        provider_class_name: str,
        **init_kwargs: Any,
    ):
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._conn = parent_conn
        self._lock = asyncio.Lock()
        self._class_name = provider_class_name

        self._process = ctx.Process(
            target=_worker_target,
            args=(
                child_conn,
                sys.path[:],
                provider_module,
                provider_class_name,
                init_kwargs,
            ),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # parent only needs its end
        logger.info(
            f"{provider_class_name} worker started (pid={self._process.pid})"
        )

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            loop = asyncio.get_event_loop()
            self._conn.send(("call", method, args, kwargs))
            try:
                response = await loop.run_in_executor(None, self._conn.recv)
            except EOFError:
                raise RuntimeError(
                    f"{self._class_name} worker process died unexpectedly"
                )
            if response[0] == "error":
                raise RuntimeError(f"{self._class_name} worker error: {response[1]}")
            return response[1]

    def shutdown(self) -> None:
        try:
            self._conn.send(("shutdown",))
        except Exception:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning(f"{self._class_name} worker did not exit cleanly; terminating")
            self._process.terminate()
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provider adapters (implement STTProvider / TTSProvider interfaces)
# ---------------------------------------------------------------------------

class ProcessSTTAdapter(STTProvider):
    """
    Runs an STTProvider in a subprocess.

    `input_sample_rate` is exposed as a class attribute so
    AnthropicVoiceBridge can read it without hitting the subprocess.
    Hardcoded to 16000 Hz (faster-whisper's required input rate).
    """

    input_sample_rate: int = 16000

    def __init__(
        self,
        provider_module: str,
        provider_class_name: str,
        **init_kwargs: Any,
    ):
        self._proc = ProcessProvider(provider_module, provider_class_name, **init_kwargs)

    async def transcribe(self, pcm16_audio: bytes) -> str:
        return await self._proc.call("transcribe", pcm16_audio)

    async def close(self) -> None:
        self._proc.shutdown()


class ProcessTTSAdapter(TTSProvider):
    """Runs a TTSProvider in a subprocess."""

    def __init__(
        self,
        provider_module: str,
        provider_class_name: str,
        **init_kwargs: Any,
    ):
        self._proc = ProcessProvider(provider_module, provider_class_name, **init_kwargs)

    async def synthesize(self, text: str) -> bytes:
        return await self._proc.call("synthesize", text)

    async def close(self) -> None:
        self._proc.shutdown()
