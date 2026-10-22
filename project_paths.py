"""Resolve project data paths independently of the checkout directory name."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def resolve_from_project(path: str) -> str:
    """Resolve project-relative paths, including legacy Single-Robot/ paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    if candidate.parts and candidate.parts[0] == "Single-Robot":
        candidate = Path(*candidate.parts[1:])
    return str(PROJECT_DIR / candidate)
