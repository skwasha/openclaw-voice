"""
Pluggable STT/TTS provider layer for OpenClaw Voice.

This package adds a provider-agnostic audio pipeline (STT -> Anthropic ->
TTS) on top of the existing SIP/RTP core, plus factory functions that build
the configured providers from `config.yaml`.
"""

from .stt_base import STTProvider
from .tts_base import TTSProvider

from .stt_mock import MockSTTProvider
from .tts_mock import MockTTSProvider

import logging

logger = logging.getLogger(__name__)


def create_tts_provider(config: dict) -> TTSProvider:
    """Build the configured TTS provider.

    `config` is the top-level config dict; the active provider is selected
    via `config["tts"]["provider"]` (kokoro | openai | elevenlabs | mock).
    """
    tts_cfg = config.get("tts", {})
    provider = (tts_cfg.get("provider") or "mock").lower()

    if provider == "kokoro":
        from .worker_process import ProcessTTSAdapter
        return ProcessTTSAdapter(
            "providers.tts_kokoro", "KokoroTTSProvider",
            **tts_cfg.get("kokoro", {}),
        )
    elif provider == "openai":
        from .tts_openai import OpenAITTSProvider
        return OpenAITTSProvider(**tts_cfg.get("openai", {}))
    elif provider == "elevenlabs":
        from .tts_elevenlabs import ElevenLabsTTSProvider
        return ElevenLabsTTSProvider(**tts_cfg.get("elevenlabs", {}))
    elif provider == "mock":
        return MockTTSProvider(**tts_cfg.get("mock", {}))
    else:
        logger.warning(f"Unknown TTS provider '{provider}', falling back to mock")
        return MockTTSProvider()


def create_stt_provider(config: dict) -> STTProvider:
    """Build the configured STT provider.

    `config` is the top-level config dict; the active provider is selected
    via `config["stt"]["provider"]` (faster_whisper | openai | mock).
    """
    stt_cfg = config.get("stt", {})
    provider = (stt_cfg.get("provider") or "mock").lower()

    if provider in ("faster_whisper", "faster-whisper", "whisper_local"):
        from .worker_process import ProcessSTTAdapter
        return ProcessSTTAdapter(
            "providers.stt_faster_whisper", "FasterWhisperSTTProvider",
            **stt_cfg.get("faster_whisper", {}),
        )
    elif provider == "openai":
        from .stt_openai import OpenAISTTProvider
        return OpenAISTTProvider(**stt_cfg.get("openai", {}))
    elif provider == "mock":
        return MockSTTProvider(**stt_cfg.get("mock", {}))
    else:
        logger.warning(f"Unknown STT provider '{provider}', falling back to mock")
        return MockSTTProvider()


def create_bridge(config: dict, instructions: str, tools: list):
    """Build the voice bridge selected by `config["engine"]` (anthropic | xai).

    Returns an object implementing the bridge interface consumed by
    `openclaw_voice.py`'s `_run_call` (connect/disconnect/send_audio/
    receive_audio/is_speaking/running/_listen_task/on_tool_call/outbound_queue).
    """
    engine = (config.get("engine") or "xai").lower()

    if engine == "anthropic":
        from anthropic_voice_bridge import AnthropicVoiceBridge

        anth_cfg = config.get("anthropic", {})
        stt = create_stt_provider(config)
        tts = create_tts_provider(config)
        kwargs = dict(
            api_key=anth_cfg.get("api_key", ""),
            stt_provider=stt,
            tts_provider=tts,
            model=anth_cfg.get("model", "claude-sonnet-4-6"),
            instructions=instructions,
            tools=tools,
            max_tokens=anth_cfg.get("max_tokens", 1024),
        )
        if "greeting" in anth_cfg:
            kwargs["greeting"] = anth_cfg["greeting"]
        return AnthropicVoiceBridge(**kwargs)

    elif engine == "xai":
        from xai_voice_bridge import XAIVoiceBridge

        xai_cfg = config.get("xai", {})
        return XAIVoiceBridge(
            api_key=xai_cfg.get("api_key", ""),
            voice=xai_cfg.get("voice", "Rex"),
            instructions=instructions,
            tools=tools,
        )

    else:
        raise ValueError(f"Unknown voice engine '{engine}' (expected 'anthropic' or 'xai')")


__all__ = [
    "STTProvider",
    "TTSProvider",
    "MockSTTProvider",
    "MockTTSProvider",
    "create_tts_provider",
    "create_stt_provider",
    "create_bridge",
]
