"""crossnection/agents package

Exposes shorthand import helpers for the three custom agents while ensuring
that `class_path` references in agents.yaml resolve correctly.

Example:
    from crossnection.agents import DataAgent, StatsAgent, ExplainAgent
"""
from __future__ import annotations

from .data_agent import DataAgent
from .stats_agent import StatsAgent
from .explain_agent import ExplainAgent

__all__ = [
    "DataAgent",
    "StatsAgent",
    "ExplainAgent",
]
