"""
Katie's shared memory integration for OpenClaw Voice.

At call start, reads three sources from AGENT_MEMORY_DIR
(default: /Users/sascha/clawd):

    MEMORY.md          — long-term curated memory (most important)
    USER.md            — stable background on the user
    memory/YYYY-MM-DD  — today's and yesterday's daily logs

All three are injected into the system prompt so the voice agent
has the same context as the text-chat Katie.

At call end, a lightweight Claude Haiku call summarises the conversation
and appends a dated block to today's daily log:

    ## Voice Call — HH:MM TZ
    - bullet points...

This is the exact file text-chat Katie reads at session start, so the
next text session will see what was discussed on the call.

If AGENT_MEMORY_DIR is not configured, falls back to per-caller markdown
files under memory_dir/  (simple caller-profile mode, no daily log).
"""

import asyncio
import logging
import re
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path(__file__).parent / "user_memory"
_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Katie's shared memory (reads from /Users/sascha/clawd)
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Reads Katie's canonical memory files and writes call summaries back
    so every channel (voice, text, WhatsApp, etc.) shares the same context.
    """

    def __init__(self, agent_dir: Path):
        self.agent_dir = Path(agent_dir)
        self._memory_md: str = ""
        self._user_md: str = ""
        self._daily: dict[str, str] = {}   # iso-date -> content
        self._active_note_lines: list[str] = []  # mid-call notes pending flush
        self.load()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self):
        """Load all relevant memory files from disk."""
        self._memory_md = self._read(self.agent_dir / "MEMORY.md")
        self._user_md   = self._read(self.agent_dir / "USER.md")

        daily_dir = self.agent_dir / "memory"
        today     = date.today()
        yesterday = today - timedelta(days=1)
        for d in (today, yesterday):
            content = self._read(daily_dir / f"{d.isoformat()}.md")
            if content:
                self._daily[d.isoformat()] = content

        loaded = sum([
            bool(self._memory_md),
            bool(self._user_md),
            len(self._daily),
        ])
        logger.info(
            f"AgentMemory loaded from {self.agent_dir}: "
            f"MEMORY.md={'yes' if self._memory_md else 'no'}, "
            f"USER.md={'yes' if self._user_md else 'no'}, "
            f"daily logs={len(self._daily)}"
        )

    def exists(self) -> bool:
        return bool(self._memory_md or self._user_md or self._daily)

    # ------------------------------------------------------------------
    # Inject into system prompt
    # ------------------------------------------------------------------

    def to_system_block(self) -> str:
        """Return a formatted block to append to the system prompt."""
        parts = []

        if self._user_md:
            parts.append(f"USER BACKGROUND:\n{self._user_md}")

        if self._memory_md:
            parts.append(f"LONG-TERM MEMORY:\n{self._memory_md}")

        today_iso = date.today().isoformat()
        yesterday_iso = (date.today() - timedelta(days=1)).isoformat()
        for iso, label in ((today_iso, "TODAY"), (yesterday_iso, "YESTERDAY")):
            if iso in self._daily:
                parts.append(f"DAILY LOG ({label}, {iso}):\n{self._daily[iso]}")

        if not parts:
            return ""

        return (
            "\n\n---\n"
            "KATIE'S MEMORY (shared context — use this to personalize your responses):\n\n"
            + "\n\n".join(parts)
            + "\n---"
        )

    # ------------------------------------------------------------------
    # Mid-call: immediate note via update_memory tool
    # ------------------------------------------------------------------

    def append_note(self, note: str):
        """
        Record an explicit mid-call note.  Held in memory until
        write_call_summary() flushes everything to the daily log.
        If the call ends abnormally, flush_notes() writes them standalone.
        """
        self._active_note_lines.append(note.strip())
        logger.info(f"Mid-call note queued: {note!r:.80}")

    def flush_notes(self):
        """Write any queued notes to today's daily log immediately (no AI summary)."""
        if not self._active_note_lines:
            return
        timestamp = datetime.now().strftime("%H:%M %Z").strip()
        lines = "\n".join(f"- {n}" for n in self._active_note_lines)
        block = f"\n\n## Voice Call Notes — {timestamp}\n{lines}\n"
        self._append_to_daily(block)
        self._active_note_lines.clear()

    # ------------------------------------------------------------------
    # Post-call: AI summary → daily log
    # ------------------------------------------------------------------

    async def write_call_summary(
        self,
        messages: list,
        api_key: str,
        caller_number: str = "",
        assistant_name: str = "Katie",
        model: str = _EXTRACTION_MODEL,
    ):
        """
        Generate a brief bullet-point summary of the call with Claude Haiku
        and append it to today's daily log as a Voice Call section.

        Runs in the background after hangup — does not block anything.
        """
        try:
            # Build transcript (text only)
            lines = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    lines.append(f"{role.upper()}: {content.strip()}")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            lines.append(f"{role.upper()}: {block['text'].strip()}")
                        elif hasattr(block, "type") and block.type == "text":
                            lines.append(f"{role.upper()}: {block.text.strip()}")

            if not lines:
                # Nothing to summarise — just flush any queued notes
                self.flush_notes()
                return

            convo = "\n".join(lines)
            extra_notes = (
                "\nExplicitly noted mid-call:\n" + "\n".join(f"- {n}" for n in self._active_note_lines)
                if self._active_note_lines else ""
            )
            caller_ctx = f" from {caller_number}" if caller_number else ""

            prompt = f"""You are writing a brief summary of a voice call{caller_ctx} with {assistant_name} \
to append to a daily activity log.

CALL TRANSCRIPT:
{convo}{extra_notes}

Write 3–7 concise bullet points covering:
- What was discussed or decided
- Any tasks started, completed, or mentioned
- Anything worth remembering for next time (preferences, facts, context)

Rules:
- Each bullet starts with "- "
- Be specific and concrete, not generic
- Skip pleasantries and small talk
- Do NOT include a header line — just the bullets

Return ONLY the bullet points, nothing else."""

            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                bullets = response.content[0].text.strip()
            finally:
                await client.close()

            timestamp = datetime.now().strftime("%H:%M %Z").strip()
            block = f"\n\n## Voice Call — {timestamp}\n{bullets}\n"
            self._active_note_lines.clear()
            self._append_to_daily(block)
            logger.info(f"Voice call summary written to daily log ({len(bullets)} chars)")

        except Exception as e:
            logger.error(f"Failed to write call summary: {e}")
            # Best-effort: at least flush any explicit notes
            self.flush_notes()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_to_daily(self, block: str):
        """Append `block` to today's daily log, creating the file if needed."""
        today_iso = date.today().isoformat()
        daily_dir = self.agent_dir / "memory"
        daily_dir.mkdir(parents=True, exist_ok=True)
        log_path = daily_dir / f"{today_iso}.md"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)
        logger.info(f"Appended to {log_path}")

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return ""


# ---------------------------------------------------------------------------
# Fallback: per-caller profile files (no AGENT_MEMORY_DIR)
# ---------------------------------------------------------------------------

def _norm_number(number: str) -> str:
    digits = re.sub(r'\D', '', number)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    return digits


class CallerMemory:
    """
    Simple per-caller markdown profiles under memory_dir/{number}.md.
    Used when AGENT_MEMORY_DIR is not configured.
    Updated via Claude Haiku after each call.
    """

    def __init__(self, number: str, memory_dir: Path = DEFAULT_MEMORY_DIR):
        self.number = number
        self._norm = _norm_number(number)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.memory_dir / f"{self._norm}.md"
        self.content: str = ""
        self._load()

    def exists(self) -> bool:
        return bool(self.content.strip())

    def to_system_block(self) -> str:
        if not self.content.strip():
            return ""
        return (
            "\n\n---\n"
            "ABOUT THIS CALLER:\n"
            f"{self.content.strip()}\n"
            "---"
        )

    def append_note(self, note: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        addition = f"\n- [{timestamp}] {note.strip()}"
        self.content = (self.content.rstrip() + addition) if self.content else f"# Caller {self.number}\n{addition}"
        self._save()

    def flush_notes(self):
        pass  # notes are written immediately in append_note

    async def write_call_summary(
        self,
        messages: list,
        api_key: str,
        caller_number: str = "",
        assistant_name: str = "Katie",
        model: str = _EXTRACTION_MODEL,
    ):
        """Rewrite caller profile with updated facts extracted from the call."""
        try:
            lines = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    lines.append(f"{role.upper()}: {content.strip()}")

            if not lines:
                return

            convo = "\n".join(lines)
            existing = f"\nEXISTING PROFILE:\n{self.content}\n" if self.content else "\n(New caller — no existing profile.)\n"

            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": f"""Update this caller profile for {assistant_name}.
{existing}
CALL TRANSCRIPT ({datetime.now().strftime('%Y-%m-%d')}):
{convo}

Rewrite the profile as clean Markdown. Include who they are, what they're working on, preferences, \
and any specific facts worth remembering. Preserve existing facts; update stale ones; add new ones.
Return ONLY the updated Markdown, nothing else."""}],
                )
                self.content = response.content[0].text.strip()
                self._save()
            finally:
                await client.close()

        except Exception as e:
            logger.error(f"CallerMemory update failed for {self._norm}: {e}")

    def _load(self):
        self.content = self._path.read_text(encoding="utf-8").strip() if self._path.exists() else ""

    def _save(self):
        self._path.write_text(self.content.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_memory(number: str, config: dict):
    """
    Return the appropriate memory object for this caller.

    If memory.agent_dir is configured (points at /Users/sascha/clawd or similar),
    use AgentMemory — reads shared MEMORY.md / USER.md / daily logs.
    Otherwise fall back to per-caller CallerMemory files.
    """
    agent_dir = config.get('memory', {}).get('agent_dir', '')
    if agent_dir and not str(agent_dir).startswith('${'):
        return AgentMemory(Path(agent_dir))

    fallback_dir = Path(config.get('memory', {}).get('dir', 'user_memory'))
    return CallerMemory(number, memory_dir=fallback_dir)
