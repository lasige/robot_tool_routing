"""Compact terminal logging shared by routing and evaluation clients."""

_TERMINAL_MAX = 120


def log(area: str, message: str, **details):
    """
    Write a terminal log line, truncating long values for readability.
    """
    prefix = f"[{area}] {message}"

    # ── terminal (truncated) ─────────────────────────────────
    terminal_parts = [prefix]
    for key, value in details.items():
        s = str(value)
        if len(s) > _TERMINAL_MAX:
            s = s[:_TERMINAL_MAX] + "…"
        terminal_parts.append(f"{key}={s!r}")
    print(" | ".join(terminal_parts), "\n")
