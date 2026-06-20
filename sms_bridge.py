"""
VoIP.ms SMS bridge.

Outbound: a plain GET to VoIP.ms's REST API with method=sendSMS.
Inbound:  VoIP.ms sends a GET request to a callback URL you configure in the
          portal (Manage DIDs -> your DID -> SMS/MMS URL Callback) with query
          params `to`, `from`, `message`, `files`, `id`, `date`, and an
          optional `api_key` you choose yourself (VoIP.ms just echoes whatever
          you put in the callback URL template — it's not an API key VoIP.ms
          issues, it's how *we* verify the request came from our own callback
          config and not a random GET to the same port).

          If "URL Callback Retry" is enabled in the portal, VoIP.ms expects a
          literal "ok" response body; otherwise it resends the same message
          every 30 minutes. openclaw_voice.py's webhook handler returns "ok"
          immediately and processes the message in the background so a slow
          LLM reply never risks a duplicate-send retry.

Inbound routing/handling (caller tiers, OpenClaw forwarding, daily-log
writes) lives in openclaw_voice.py since it needs the same caller-identity
table and memory store as voice calls. This module only knows how to talk to
the VoIP.ms REST API.
"""

import logging

import aiohttp

logger = logging.getLogger(__name__)

VOIPMS_API_URL = "https://voip.ms/api/v1/rest.php"


async def send_sms(config: dict, dst: str, message: str) -> dict:
    """
    Send an SMS via the VoIP.ms REST API (method=sendSMS).

    Returns the parsed JSON response, e.g. {"status": "success"} or
    {"status": "error", ...}. On a transport-level failure (network error,
    bad config) returns {"status": "error", "error": "<description>"}.
    """
    sms_cfg = config.get('sms', {})
    api_username = sms_cfg.get('api_username', '')
    api_password = sms_cfg.get('api_password', '')
    did = sms_cfg.get('did', '')

    for label, val in (("api_username", api_username), ("api_password", api_password), ("did", did)):
        if not val or str(val).startswith('${'):
            return {"status": "error", "error": f"SMS not configured: missing {label}"}

    params = {
        "api_username": api_username,
        "api_password": api_password,
        "method": "sendSMS",
        "did": str(did),
        "dst": str(dst),
        "message": message,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                VOIPMS_API_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                # VoIP.ms returns JSON but doesn't always set the content-type
                # header correctly, so don't let aiohttp gate on it.
                data = await resp.json(content_type=None)
                if data.get("status") == "success":
                    logger.info(f"SMS sent to {dst} ({len(message)} chars)")
                else:
                    logger.warning(f"VoIP.ms sendSMS failed for {dst}: {data}")
                return data
    except Exception as e:
        logger.error(f"sendSMS error: {e}")
        return {"status": "error", "error": str(e)}
