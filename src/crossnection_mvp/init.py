"""Crossnection package root.

Exposes package version and convenience import for the crew builder so that
external callers can simply do:

    from crossnection import get_crew

which returns an instance of `CrossnectionMvpCrew` ready to run.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("crossnection-mvp")
except PackageNotFoundError:  # pragma: no cover – not installed
    __version__ = "0.1.0"

# Convenience re‑export -----------------------------------------------------#

def get_crew():  # noqa: D401 – simple function
    """Return a lazily‑constructed instance of :class:`CrossnectionMvpCrew`."""

    # Local import to avoid heavy deps at import‑time
    from crossnection.crew import CrossnectionMvpCrew  # type: ignore

    return CrossnectionMvpCrew()
