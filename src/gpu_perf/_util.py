"""Tiny shared helpers (kept dependency-free so they run anywhere)."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. '2026-09-13T18:22:05.123456+00:00'."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    """Human-sortable run id: '<prefix>-YYYYmmdd-HHMMSS'."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def hostname() -> str:
    """Best-effort node hostname (used for run provenance)."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()
