"""crossnection_mvp/agents/data_agent.py

Fully‑functional **DataAgent** subclass used by Crossnection MVP.
The agent is lightweight: heavy ETL lives in `CrossDataProfilerTool`, but this
class is handy for:

* Manual invocation in notebooks / unit‑tests without spinning an entire Crew.
* Centralising defaults (role, goal, tools) so YAML and code stay in sync.
* Providing a higher‑level helper `run_data_pipeline()` that combines the three
  DataAgent tasks in one call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import logging
import pandas as pd

import crewai as cr
from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool

logger = logging.getLogger(__name__)


class DataAgent(cr.BaseAgent):
    """Custom DataAgent with a convenience ETL pipeline wrapper."""

    # Defaults mirrored in agents.yaml
    _ROLE = "Data Integration & Validation"
    _GOAL = (
        "Ingest CSV driver datasets, validate & profile them, discover a join‑key, "
        "and emit a single cleaned table ready for statistics."
    )
    _TOOLS = [CrossDataProfilerTool.name]

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

    def run_data_pipeline(self, csv_dir: str | Path, *, kpi: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """End‑to‑end execution of the three DataAgent tasks.

        Parameters
        ----------
        csv_dir
            Directory containing one CSV per driver.
        kpi
            Column name of the KPI (just stored in the report; not used here).

        Returns
        -------
        unified_df, data_report
            *unified_df* is a pandas DataFrame (cleaned & merged)  
            *data_report* is the profiling summary as a Python dict.
        """
        csv_dir = Path(csv_dir)
        if not csv_dir.is_dir():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        tool = CrossDataProfilerTool()

        logger.info("[DataAgent] Profiling & cleaning driver CSVs in %s", csv_dir)
        artefacts = tool.run(csv_folder=str(csv_dir), kpi=kpi, mode="full_pipeline")

        unified_csv: str = artefacts["unified_dataset_csv"]
        data_report_json: str = artefacts["data_report_json"]

        unified_df = pd.read_csv(pd.compat.StringIO(unified_csv))
        data_report: Dict[str, Any] = pd.json.loads(data_report_json)  # type: ignore[attr-defined]

        logger.info("[DataAgent] Pipeline completed (rows=%s, cols=%s)", *unified_df.shape)
        return unified_df, data_report

    # ------------------------------------------------------------------
    # Friendly repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401 – simple representation
        return f"<DataAgent role='{self._ROLE}'>"
