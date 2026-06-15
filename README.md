# OpenClaw Voice

An [OpenClaw](https://github.com/hakangit/openclaw) plugin that bridges SIP telephone calls (e.g. a VoIP.ms DID/sub-account) to an AI voice agent. Registers as a SIP extension, answers inbound calls (with optional PIN auth), and connects callers to the agent in real time.

Two voice engines are supported, selected via `engine:` in `config.yaml`:

- **`anthropic`** (default) — Claude via the Messages API, with pluggable
  local/cloud STT and TTS providers (Kokoro + faster-whisper by default, no
  per-call API cost).
- **`xai`** — the original realtime xAI Grok Voice WebSocket bridge.

## Architecture

```
                                   ┌─ engine: anthropic ─────────────────┐
                                   │  STT -> Claude (Messages API) -> TTS │
Phone Call ←→ SIP/RTP ←→ openclaw_voice.py ←→                            │
                                   │  engine: xai                        │
                                   └─ xAI Grok Voice WebSocket  ──────────┘
                              ↓
                         HTTP API (:8079)
                              ↓
                    OpenClaw Gateway ←→ Tools
```

### Components

| File | Purpose |
|------|---------|
| `openclaw_voice.py` | Main daemon — SIP registration, call routing, HTTP API |
| `sip_client.py` | Minimal SIP/RTP client (no external SIP library) |
| `anthropic_voice_bridge.py` | Claude Messages API bridge: VAD-segmented STT -> Claude -> TTS |
| `xai_voice_bridge.py` | WebSocket client for xAI Grok Voice API (alternative engine) |
| `providers/` | Pluggable STT/TTS provider interfaces + implementations + bridge factory |
| `voice_bridge.py` | Standalone xAI audio bridge example/entry point (legacy) |
| `audio.py` | Voice Activity Detection (WebRTC VAD) and echo suppression |
| `ov_cli.py` | CLI for making calls and checking status via the HTTP API |

### Pluggable STT/TTS providers (`providers/`)

| Provider type | Options (`config.yaml`) | Notes |
|---|---|---|
| TTS | `kokoro` (default), `openai`, `elevenlabs`, `mock` | Kokoro runs locally on CPU, MIT licensed, no per-call cost |
| STT | `faster_whisper` (default), `openai`, `mock` | faster-whisper runs locally on CPU, no per-call cost |

Each provider implements a small abstract interface (`providers/tts_base.py`,
`providers/stt_base.py`) — `synthesize(text) -> u-law bytes` and
`transcribe(pcm16_audio) -> text`. Swapping providers is a one-line config
change; `providers/__init__.py` has the factory functions and
`create_bridge()` which builds the configured engine.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and SIP credentials
   # Edit config.yaml to set your SIP extension, voice, and instructions
   ```

3. **Run**:
   ```bash
   python openclaw_voice.py
   ```

## Configuration

### `.env`

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (engine: anthropic) |
| `OPENAI_API_KEY` | Optional - if using OpenAI for TTS/STT |
| `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` | Optional - if using ElevenLabs TTS |
| `XAI_API_KEY` | xAI API key (engine: xai only) |
| `SIP_PASSWORD` | VoIP.ms sub-account SIP password |
| `OPENCLAW_GATEWAY_TOKEN` | OpenClaw gateway auth token |
| `OPENCLAW_AUTH_PIN` | PIN for inbound call authentication |

### `config.yaml`

Covers the voice `engine` (anthropic/xai), TTS/STT provider selection, SIP
server (VoIP.ms sub-account), audio format, daemon settings, and auth.
Environment variables are substituted using `${VAR}` syntax.

## CLI Usage

```bash
# Make an outbound call
python ov_cli.py call 5551234567

# List active calls
python ov_cli.py calls

# Hang up a call
python ov_cli.py hangup <call-id>

# Health check
python ov_cli.py status
```

## HTTP API

The daemon exposes a local HTTP API on the configured port (default 8079):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/call` | Initiate outbound call |
| `GET` | `/calls` | List active calls |
| `DELETE` | `/call/{id}` | Hang up a call |
| `GET` | `/health` | Health check |
| `POST` | `/call/{id}/conference` | Bridge call into conference |

## OpenClaw Integration

When connected to an OpenClaw gateway, the voice agent can execute tools (queries, commands, API calls) through the gateway's tool-calling interface. Configure the gateway connection in `config.yaml`:

```yaml
openclaw:
  gateway_port: 18789
  gateway_token: ${OPENCLAW_GATEWAY_TOKEN}
  session_key: "openclaw-voice"
```

## Deployment (systemd)

```bash
sudo cp openclaw-voice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openclaw-voice
```

The service file expects the project at `/opt/openclaw_voice` with a virtualenv at `/opt/openclaw_voice/venv`. Adjust paths as needed.

## Auth

Inbound calls can require a DTMF PIN before connecting to the voice agent. Configure in `config.yaml`:

```yaml
auth:
  pin: ${OPENCLAW_AUTH_PIN}
  whitelist: ["+15551234567"]  # numbers that skip PIN
```

## Design

- **No SIP library dependency**: `sip_client.py` implements SIP/RTP directly on UDP sockets.
- **Concurrent calls**: Multiple simultaneous calls, each with its own RTP port and bridge instance.
- **`engine: xai`** - xAI Grok natively supports G.711 u-law at 8kHz, matching SIP exactly. No audio resampling, sub-700ms latency, realtime bidirectional WebSocket.
- **`engine: anthropic`** - turn-based pipeline: WebRTC VAD segments caller
  audio into utterances, which are transcribed (STT), sent to Claude
  (resolving any tool calls), and the reply is synthesized (TTS) and queued
  back as RTP frames. All resampling between 8kHz SIP audio and each
  provider's native rate happens in `providers/audio_utils.py`.

## License

MIT
