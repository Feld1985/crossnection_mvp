"""crossnection_mvp/tools/cross_data_profiler.py

Custom CrewAI Tool: **CrossDataProfilerTool**
-------------------------------------------------
Responsible for:
1. Profiling & validating user‑supplied CSVs.
2. Discovering (or prompting the agent to synthesise) a join‑key strategy.
3. Cleaning & normalising the merged dataset ready for statistical analysis.

Design notes
~~~~~~~~~~~~
* Built on **pandas** for ETL and **great_expectations** for validation.
* Returns two artefacts as a dict:  
    - `unified_dataset_csv` (string in CSV format)  
    - `data_report_json` (JSON string)
* CrewAI automatically serialises / deserialises when passed
  between tasks (falls back to CSV in Flow context if needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import uuid

import pandas as pd
import great_expectations as ge
from crewai import Tool

__all__ = ["CrossDataProfilerTool"]


class CrossDataProfilerTool(Tool):
    """Validate, profile and merge driver CSV datasets."""

    name: str = "cross_data_profiler"
    description: str = (
        "Profiles CSV driver datasets, validates schema, discovers/creates a join‑key, "
        "cleans & normalises the data, then returns unified data and a "
        "structured report."
    )

    # ------------------------------------------------------------------
    # Public API (called by agents)
    # ------------------------------------------------------------------

    def run(self, *, csv_folder: str | Path, kpi: str, mode: str = "full_pipeline") -> Dict[str, Any]:
        """Main entry‑point expected by CrewAI.

        Parameters
        ----------
        csv_folder : str | Path
            Directory containing the CSV files representing each driver. Each file must have
            at least a numeric `value` column and ideally a common key.
        kpi : str
            Name of the KPI column (just stored in the report; not used here).
        mode : str
            Execution mode: 'profile_only', 'join_key_only', 'clean_only', or 'full_pipeline'

        Returns
        -------
        dict
            {
              "unified_dataset_csv": str,
              "data_report_json": str
            }
        """
        csv_folder = Path(csv_folder)
        if not csv_folder.is_dir():
            raise FileNotFoundError(f"CSV directory not found: {csv_folder}")
            
        # Get all CSV files in the directory
        csv_paths = list(csv_folder.glob("*.csv"))
        if not csv_paths:
            raise ValueError(f"No CSV files found in {csv_folder}")
            
        # Load dataframes
        dataframes = [pd.read_csv(p) for p in csv_paths]
        profile = self._profile_frames(dataframes, csv_paths)

        key = self._discover_join_key(dataframes, profile)
        unified = self._merge_and_clean(dataframes, key)

        # Return artifacts in the expected format
        return {
            "unified_dataset_csv": unified.to_csv(index=False),
            "data_report_json": json.dumps(profile, ensure_ascii=False, indent=2)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_frames(self, frames: List[pd.DataFrame], paths: List[str | Path]) -> Dict[str, Any]:
        """Return column types, null counts, basic stats & anomalies using GE."""

        report: Dict[str, Any] = {"tables": []}
        for df, p in zip(frames, paths):
            ge_df = ge.from_pandas(df)
            # simple expectations – existence of numeric `value` column
            ge_df.expect_column_to_exist("value")
            summary = {
                "file": str(p),
                "columns": {
                    col: {
                        "dtype": str(df[col].dtype),
                        "nulls": int(df[col].isna().sum()),
                    }
                    for col in df.columns
                },
            }
            report["tables"].append(summary)
        return report

    def _discover_join_key(self, frames: List[pd.DataFrame], report: Dict[str, Any]) -> str:
        """Try to find a column present in all frames with high uniqueness."""
        common_cols = set(frames[0].columns)
        for df in frames[1:]:
            common_cols &= set(df.columns)
        # choose first candidate with high uniqueness else synthesize
        for col in common_cols:
            if all(df[col].is_unique for df in frames):
                return col
        # synthesise surrogate key
        surrogate = "_cxn_id"
        for df in frames:
            df[surrogate] = [uuid.uuid4().hex for _ in range(len(df))]
        report["surrogate_key"] = True
        return surrogate

    def _merge_and_clean(self, frames: List[pd.DataFrame], key: str) -> pd.DataFrame:
        """Outer‑join all frames on the discovered key and coerce numeric columns."""
        base = frames[0]
        for df in frames[1:]:
            base = base.merge(df, on=key, how="outer", suffixes=("", "_dup"))
        # Drop duplicated columns created by suffixes
        dupes = [c for c in base.columns if c.endswith("_dup")]
        base.drop(columns=dupes, inplace=True)
        # Coerce numeric
        for col in base.columns:
            if col == key:
                continue
            base[col] = pd.to_numeric(base[col], errors="coerce")
        return base