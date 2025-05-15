"""crossnection/tools/cross_stat_engine.py

Custom CrewAI Tool: CrossStatEngineTool
======================================

Numerical “engine” invoked by *StatsAgent* to transform the unified dataset
in tre artefatti principali:

1. **correlation_matrix**  – coefficiente r (Pearson o Spearman) e p-value per
   ogni driver rispetto al KPI.
2. **impact_ranking**      – classifica dei driver in base a uno *score*
   composito: |r| normalizzato × -log10 p-value  (più alto = maggiore impatto).
3. **outlier_report**      – individua gli outlier su ciascun driver usando
   doppio criterio **Z-score** (> |3|) e **IQR**.

Tutti e tre i metodi sono richiamabili singolarmente tramite `run(mode=…)`
oppure direttamente come funzioni di libreria.

Dipendenze: pandas · numpy · scipy · statsmodels
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.weightstats import ztest

# ---------------------------------------------------------------------------#
# Helper utilities
# ---------------------------------------------------------------------------#


def _choose_corr(x: pd.Series, y: pd.Series) -> str:
    """Pearson se entrambe le serie sono Gauss-like, Spearman altrimenti."""
    if x.skew() < 1 and y.skew() < 1:
        return "pearson"
    return "spearman"


def _zscore_outliers(series: pd.Series, z_thresh: float = 3.0) -> List[int]:
    """Return row-indices considered outliers by Z-score."""
    z = np.abs(sp_stats.zscore(series, nan_policy="omit"))
    return series.index[z > z_thresh].tolist()


def _iqr_outliers(series: pd.Series, iqr_mult: float = 1.5) -> List[int]:
    """Row-indices flagged by Tukey’s IQR rule."""
    q1, q3 = np.nanpercentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    return series.index[(series < lower) | (series > upper)].tolist()


# ---------------------------------------------------------------------------#
# Correlation & ranking
# ---------------------------------------------------------------------------#


def correlation_matrix(df: pd.DataFrame, *, kpi: str) -> pd.DataFrame:
    """Compute r & p for ogni colonna numerica vs kpi."""
    rows = []
    y = df[kpi]
    for col in df.select_dtypes(include="number").columns:
        if col == kpi:
            continue
        x = df[col]
        method = _choose_corr(x, y)
        if method == "pearson":
            r, p = sp_stats.pearsonr(x, y, nan_policy="omit")
        else:
            r, p = sp_stats.spearmanr(x, y, nan_policy="omit")
        rows.append({"driver_name": col, "method": method, "r": r, "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value")


def impact_ranking(corr_df: pd.DataFrame, top_k: int | None = None) -> List[Dict[str, Any]]:
    """Blend effect size & significance in a single score and sort."""
    # Score: |r| normalizzato 0-1  ×  -log10(p)
    r_abs = corr_df["r"].abs()
    r_norm = (r_abs - r_abs.min()) / (r_abs.max() - r_abs.min() + 1e-9)
    score = r_norm * -np.log10(corr_df["p_value"].clip(lower=1e-12))
    ranked = corr_df.assign(score=score).sort_values("score", ascending=False)
    if top_k:
        ranked = ranked.head(top_k)
    return ranked.to_dict(orient="records")


# ---------------------------------------------------------------------------#
# Outlier detection
# ---------------------------------------------------------------------------#


def outlier_report(df: pd.DataFrame, *, kpi: str) -> Dict[str, Any]:
    """Return list of outlier points per driver (index, driver, method)."""
    outliers = []
    for col in df.select_dtypes(include="number").columns:
        if col == kpi:
            continue
        z_idx = _zscore_outliers(df[col])
        iqr_idx = _iqr_outliers(df[col])
        combined = sorted(set(z_idx) | set(iqr_idx))
        for idx in combined:
            outliers.append({"row": int(idx), "driver": col})
    return {"kpi": kpi, "outliers": outliers}


# ---------------------------------------------------------------------------#
# CrewAI Tool wrapper
# ---------------------------------------------------------------------------#


class CrossStatEngineTool:
    """
    Tool registrabile in CrewAI.

    Usage via task:
        tool_name: cross_stat_engine
        args:
            mode: correlation
            kpi: "First Pass Yield"
            df_csv: "{{ unified_dataset }}"
    """

    name = "cross_stat_engine"
    description = "Statistical engine: correlations, impact ranking, outlier detection."

    def run(
        self,
        *,
        df_csv: str | bytes,
        kpi: str,
        mode: str = "correlation",
        top_k: int | None = 10,
    ) -> str:
        """
        Parameters
        ----------
        df_csv
            Unified dataset as CSV string/bytes (passed from DataAgent).
        kpi
            Column name of the target KPI.
        mode
            'correlation', 'ranking', or 'outliers'.
        top_k
            How many drivers to keep in ranking (ignored otherwise).

        Returns
        -------
        JSON string result – CrewAI converts to python dict automatically.
        """
        df = pd.read_csv(pd.compat.StringIO(df_csv) if isinstance(df_csv, str) else df_csv)

        if mode == "correlation":
            corr = correlation_matrix(df, kpi=kpi)
            return corr.to_json(orient="records")

        if mode == "ranking":
            corr_df = correlation_matrix(df, kpi=kpi)
            ranked = impact_ranking(corr_df, top_k=top_k)
            return json.dumps(
                {"kpi_name": kpi, "ranking": ranked},
                ensure_ascii=False,
                indent=2,
            )

        if mode == "outliers":
            report = outlier_report(df, kpi=kpi)
            return json.dumps(report, ensure_ascii=False, indent=2)

        raise ValueError("mode must be 'correlation', 'ranking', or 'outliers'")
