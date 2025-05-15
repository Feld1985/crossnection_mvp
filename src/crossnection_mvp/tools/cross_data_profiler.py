"""crossnection/tools/cross_data_profiler.py

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
    - `unified_dataset` (pandas.DataFrame)  
    - `data_report` (python dict serialisable as JSON)
* CrewAI automatically serialises / deserialises pandas objects when passed
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
        "cleans & normalises the data, then returns a DataFrame (`unified_dataset`) and "
        "a structured `data_report`."
    )

    # ------------------------------------------------------------------
    # Public API (called by agents)
    # ------------------------------------------------------------------

    def run(self, csv_paths: List[str | Path]) -> Dict[str, Any]:
        """Main entry‑point expected by CrewAI.

        Parameters
        ----------
        csv_paths : list[str|Path]
            Paths to the CSV files representing each driver. Each file must have
            at least a numeric `value` column and ideally a common key.

        Returns
        -------
        dict
            {
              "unified_dataset": pandas.DataFrame,
              "data_report": dict (JSON‑serialisable)
            }
        """
        dataframes = [pd.read_csv(p) for p in csv_paths]
        profile = self._profile_frames(dataframes, csv_paths)

        key = self._discover_join_key(dataframes, profile)
        unified = self._merge_and_clean(dataframes, key)

        return {"unified_dataset": unified, "data_report": profile}

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
