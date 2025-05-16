"""crossnection_mvp/agents/stats_agent.py

Fully‑functional **StatsAgent** subclass used by Crossnection MVP.
It orchestrates the statistical analysis on the unified dataset produced by
DataAgent, delegating heavy calculations to `CrossStatEngineTool`.

Why a subclass?
---------------
* Enables standalone use in notebooks/unit tests without starting the whole
  CrewAI flow.
* Centralises defaults so YAML and code do not drift.
* Provides helper `run_stats_pipeline()` that bundles the three StatsAgent tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import logging
import pandas as pd

import crewai as cr
from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool

logger = logging.getLogger(__name__)


class StatsAgent(cr.BaseAgent):
    """Custom StatsAgent with a convenience statistical pipeline wrapper."""

    # Defaults mirrored in agents.yaml
    _ROLE = "Statistical Insight Generator"
    _GOAL = (
        "Compute correlations, rank driver impact on KPI, and detect outliers "
        "in the unified dataset."
    )
    _TOOLS = [CrossStatEngineTool.name]

    # ------------------------------------------------------------------
    # CrewAI hooks
    # ------------------------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:  # noqa: D401 – simple wrapper
        defaults = {
            "role": self._ROLE,
            "goal": self._GOAL,
            "tools": self._TOOLS,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    # ------------------------------------------------------------------
    # Convenience API (bypass Crew execution)
    # ------------------------------------------------------------------

    def run_stats_pipeline(
        self,
        unified_dataset: str | Path | pd.DataFrame,
        *,
        kpi: str,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """End‑to‑end execution of the three StatsAgent tasks.

        Parameters
        ----------
        unified_dataset
            Path, CSV string, or DataFrame returned by DataAgent.
        kpi
            Target KPI column name.
        top_k
            How many drivers to keep in the impact ranking.

        Returns
        -------
        correlation_matrix, impact_ranking, outlier_report
            *correlation_matrix* – list of dicts with r & p‑value per driver
            *impact_ranking*    – top‑k drivers sorted by composite score
            *outlier_report*    – dict with details of flagged rows/drivers
        """
        # Load dataset ---------------------------------------------------
        if isinstance(unified_dataset, pd.DataFrame):
            df = unified_dataset
        elif isinstance(unified_dataset, Path):
            df = pd.read_csv(unified_dataset)
        else:  # assume CSV string/bytes
            df = pd.read_csv(pd.compat.StringIO(unified_dataset))

        if kpi not in df.columns:
            raise KeyError(f"KPI column '{kpi}' not found in dataset")

        tool = CrossStatEngineTool()

        # ------------------------------------------------ correlation ----
        logger.info("[StatsAgent] Computing correlation matrix…")
        corr_json = tool.run(df_csv=df.to_csv(index=False), kpi=kpi, mode="correlation")
        correlation_matrix: List[Dict[str, Any]] = json.loads(corr_json)

        # ------------------------------------------------ ranking ------
        logger.info("[StatsAgent] Building impact ranking (top %s)…", top_k)
        rank_json = tool.run(
            df_csv=df.to_csv(index=False), kpi=kpi, mode="ranking", top_k=top_k
        )
        impact_ranking: List[Dict[str, Any]] = json.loads(rank_json)["ranking"]

        # ------------------------------------------------ outliers ------
        logger.info("[StatsAgent] Detecting outliers…")
        outlier_json = tool.run(df_csv=df.to_csv(index=False), kpi=kpi, mode="outliers")
        outlier_report: Dict[str, Any] = json.loads(outlier_json)

        logger.info("[StatsAgent] Pipeline completed (drivers=%s)", len(impact_ranking))
        return correlation_matrix, impact_ranking, outlier_report

    # ------------------------------------------------------------------
    # Friendly repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401 – simple representation
        return f"<StatsAgent role='{self._ROLE}'>"
