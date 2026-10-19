"""Small shared helpers (no project imports, safe to use from any module)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    """Single timestamp format for the whole system: ISO-8601 UTC with 'Z'."""
    return datetime.now(timezone.utc).strftime(TS_FORMAT)


def utc_param(value: str | None, name: str = "timestamp") -> str | None:
    """Normalise a user-supplied ISO-8601 timestamp to the stored UTC 'Z' form.

    Accepts '2026-10-14T00:00:00Z', '2026-10-14T09:00:00+09:00', '2026-10-14' (midnight UTC).
    Naive values are taken as UTC. Anything else raises ValueError, so a typo can never
    silently select the wrong time window.
    """
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} must be ISO-8601, e.g. 2026-10-14T00:00:00Z or 2026-10-14T09:00:00+09:00") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(TS_FORMAT)


def safe_name(value: str | None, fallback: str = "unknown", max_len: int = 64) -> str:
    """File-system-safe token derived from a serial / filename: no separators, no traversal."""
    s = _NAME_RE.sub("_", Path(value or "").name.strip()) if value else ""
    s = s.strip("._")[:max_len]
    return s or fallback
