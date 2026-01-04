from __future__ import annotations

import json
import os
import time
from typing import Any


def _env_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return _env_truthy(os.getenv("CAAL_AGENT_NDJSON_LOG"))


def enabled() -> bool:
    """Public helper to check if NDJSON logging is enabled."""
    return _enabled()


def _log_path() -> str:
    return os.getenv("CAAL_AGENT_LOG_PATH", "/app/logs/agent.log")


def log_event(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """
    Best-effort NDJSON logger for the agent (disabled unless CAAL_AGENT_NDJSON_LOG is truthy).

    Important: do not log secrets/PII; keep payloads small (previews not full prompts).
    """
    if not _enabled():
        return

    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "sessionId": session_id or os.getenv("CAAL_AGENT_LOG_SESSION_ID", "agent"),
            "runId": run_id or os.getenv("CAAL_AGENT_LOG_RUN_ID", "run1"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"

        # Multi-process safe-ish append (LiveKit spawns multiple processes)
        # Use flock if available; otherwise just append.
        try:
            import fcntl  # type: ignore

            with open(path, "a", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
                f.write(line)
                f.flush()
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        except Exception:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
    except Exception:
        # Best effort only: never break agent due to logging.
        return


