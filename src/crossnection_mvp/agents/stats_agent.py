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
from crossnection_mvp.utils.context_decorators import with_context_io
from crossnection_mvp.utils.context_store import ContextStore
from crossnection_mvp.utils.error_handling import with_robust_error_handling

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

    @with_robust_error_handling(
        log_level="ERROR",
        stage_name="stats_pipeline",
        custom_fallback={
            "correlation_matrix": [],
            "impact_ranking": [],
            "outlier_report": {"outliers": []}
        }
    )
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

        # Salviamo il dataset nel Context Store
        store = ContextStore.get_instance()
        store.save_dataframe("unified_dataset", df)

        tool = CrossStatEngineTool()

        # ------------------------------------------------ correlation ----
        logger.info("[StatsAgent] Computing correlation matrix…")
        corr_json = tool.run(df_csv=df.to_csv(index=False), kpi=kpi, mode="correlation")
        correlation_matrix: List[Dict[str, Any]] = json.loads(corr_json)
        store.save_json("correlation_matrix", correlation_matrix)

        # ------------------------------------------------ ranking ------
        logger.info("[StatsAgent] Building impact ranking (top %s)…", top_k)
        rank_json = tool.run(
            df_csv=df.to_csv(index=False), kpi=kpi, mode="ranking", top_k=top_k
        )
        impact_ranking_data = json.loads(rank_json)
        impact_ranking: List[Dict[str, Any]] = impact_ranking_data["ranking"]
        store.save_json("impact_ranking", impact_ranking_data)

        # ------------------------------------------------ outliers ------
        logger.info("[StatsAgent] Detecting outliers…")
        outlier_json = tool.run(df_csv=df.to_csv(index=False), kpi=kpi, mode="outliers")
        outlier_report: Dict[str, Any] = json.loads(outlier_json)
        store.save_json("outlier_report", outlier_report)

        logger.info("[StatsAgent] Pipeline completed (drivers=%s)", len(impact_ranking))
        return correlation_matrix, impact_ranking, outlier_report
    
    @with_context_io(
        input_keys={"df_csv": "unified_dataset"}, 
        output_key="correlation_matrix", 
        output_type="json"
    )
    @with_robust_error_handling(
        stage_name="compute_correlations",
        store_error_key="correlation_matrix"
    )
    def compute_correlations(self, **kwargs):
        """
        Computes correlation between each driver and KPI.
        Enhanced error handling and fallback logic.
        """
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "compute_correlations"
            
        # Estrai parametri dall'input
        df_csv = kwargs.get("df_csv", "")
        kpi = kwargs.get("kpi", "value_speed")
        mode = kwargs.get("mode", "correlation")
        
        logger.info(f"StatsAgent.compute_correlations called with kpi={kpi}, mode={mode}, df_csv_len={len(df_csv) if df_csv else 0}")
        
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        result = tool.run(df_csv=df_csv, kpi=kpi, mode=mode)
        return json.loads(result) if isinstance(result, str) else result

    @with_context_io(
        input_keys={
            "df_csv": "unified_dataset", 
            "correlation_matrix": "correlation_matrix"
        }, 
        output_key="impact_ranking", 
        output_type="json"
    )
    @with_robust_error_handling(
        stage_name="rank_impact",
        store_error_key="impact_ranking"
    )
    def rank_impact(self, **kwargs):
        """
        Ranks drivers by impact based on correlation data.
        Enhanced with error recovery.
        """
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "rank_impact"
            
        # Estrai parametri dall'input
        df_csv = kwargs.get("df_csv", "")
        kpi = kwargs.get("kpi", "value_speed")
        correlation_matrix = kwargs.get("correlation_matrix", [])
        
        logger.info(f"StatsAgent.rank_impact called with kpi={kpi}, matrix_len={len(correlation_matrix)}")
        
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        result = tool.run(df_csv=df_csv, kpi=kpi, mode="ranking")
        return json.loads(result) if isinstance(result, str) else result

    @with_context_io(
        input_keys={"df_csv": "unified_dataset"}, 
        output_key="outlier_report", 
        output_type="json"
    )
    @with_robust_error_handling(
        stage_name="detect_outliers",
        store_error_key="outlier_report"
    )
    def detect_outliers(self, **kwargs):
        """
        Identifies outliers in the driver datasets.
        Enhanced with error handling.
        """
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "detect_outliers"
            
        # Estrai parametri dall'input
        df_csv = kwargs.get("df_csv", "")
        kpi = kwargs.get("kpi", "value_speed")
        
        logger.info(f"StatsAgent.detect_outliers called with kpi={kpi}, df_csv_len={len(df_csv) if df_csv else 0}")
        
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        result = tool.run(df_csv=df_csv, kpi=kpi, mode="outliers")
        return json.loads(result) if isinstance(result, str) else result

    # ------------------------------------------------------------------
    # Friendly repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401 – simple representation
        return f"<StatsAgent role='{self._ROLE}'>"