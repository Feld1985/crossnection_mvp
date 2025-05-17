"""crossnection_mvp/tools/cross_stat_engine.py

Custom CrewAI Tool: CrossStatEngineTool
======================================

Numerical "engine" invoked by *StatsAgent* to transform the unified dataset
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
from typing import Any, Dict, List, Union
from pydantic import BaseModel

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.weightstats import ztest
from crewai.tools import BaseTool
from pathlib import Path

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
    """Row-indices flagged by Tukey's IQR rule."""
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
    
    # Aggiungi categoria di forza
    for i, row in ranked.iterrows():
        if abs(row["r"]) > 0.7:
            ranked.at[i, "strength"] = "Strong"
        elif abs(row["r"]) > 0.3:
            ranked.at[i, "strength"] = "Moderate"
        else:
            ranked.at[i, "strength"] = "Weak"
            
        sign = "positive" if row["r"] > 0 else "negative"
        significant = "statistical significance" if row["p_value"] < 0.05 else "moderate confidence"
        ranked.at[i, "explanation"] = f"{ranked.at[i, 'strength']} {sign} relationship with {significant}"
    
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

class CrossStatEngineToolSchema(BaseModel):
    input: Union[str, Dict[str, Any]]
    
class CrossStatEngineTool(BaseTool):
    """
    Tool registrabile in CrewAI.

    Usage via task:
        tool_name: cross_stat_engine
        args:
            mode: correlation
            kpi: "First Pass Yield"
            df_csv: "{{ unified_dataset }}"
    """

    name: str = "cross_stat_engine"
    description: str = "Statistical engine: correlations, impact ranking, outlier detection."
    args_schema = CrossStatEngineToolSchema

    def _run(self, input: Union[str, Dict[str, Any]]) -> str:
        """
        Main entry point required by BaseTool, handling both dict and string inputs.
        """
        from pathlib import Path
        import pandas as pd
        
        print(f"DEBUG: CrossStatEngineTool received raw input: {input}")
        
        # Parse input if it's a string
        if isinstance(input, str):
            try:
                input_data = json.loads(input)
                print(f"DEBUG: Parsed JSON string to dict: {input_data}")
            except json.JSONDecodeError:
                # Se l'input è una stringa CSV diretta
                if input.startswith("join_key,timestamp,value"):
                    print(f"DEBUG: Detected direct CSV content")
                    input_data = {"df_csv": input, "kpi": "First Pass Yield", "mode": "correlation"}
                else:
                    # Fallback per stringhe semplici
                    print(f"DEBUG: Input is a simple string, using as KPI: {input}")
                    input_data = {"df_csv": "", "kpi": input, "mode": "correlation"}
        elif isinstance(input, dict):
            # Struttura CrewAI con {"input": {...}}
            if "input" in input and isinstance(input["input"], dict):
                input_data = input["input"]
                print(f"DEBUG: Extracted input data from CrewAI format: {input_data}")
            else:
                input_data = input
                print(f"DEBUG: Using input dict directly: {input_data}")
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")
        
        # Verifica di avere i parametri necessari
        df_csv = input_data.get("df_csv", "")
        
        # Se non c'è df_csv ma c'è qualcos'altro, prova a usare quello
        if not df_csv and "unified_dataset" in input_data:
            df_csv = input_data["unified_dataset"]
            print(f"DEBUG: Using unified_dataset as df_csv")
        
        # Se ancora non c'è df_csv, usa il dataset integrato
        if not df_csv:
            # Se siamo qui, probabilmente l'agente sta passando i parametri errati
            # Carica il dataset unificato da un file conosciuto
            csv_path = Path("examples/driver_csvs/unified_dataset.csv")
            if csv_path.exists():
                print(f"DEBUG: Loading default dataset from {csv_path}")
                df_csv = csv_path.read_text()
            else:
                # Se non esiste, crea un dataset integrato dai file originali
                from pathlib import Path
                import pandas as pd
                
                print(f"DEBUG: Creating unified dataset from source files")
                # Carica i file CSV originali
                csv_files = list(Path("examples/driver_csvs").glob("*.csv"))
                if not csv_files:
                    raise ValueError("No CSV files found in examples/driver_csvs")
                
                dataframes = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    # Rinomina 'value' per distinguere le colonne
                    df = df.rename(columns={"value": f"value_{file.stem}"})
                    dataframes.append(df)
                
                # Unisci i dataframe
                base = dataframes[0]
                for df in dataframes[1:]:
                    base = base.merge(df, on="join_key", how="outer", suffixes=("", "_dup"))
                
                # Rimuovi colonne duplicate
                dupes = [c for c in base.columns if c.endswith("_dup")]
                base.drop(columns=dupes, inplace=True)
                
                df_csv = base.to_csv(index=False)
        
        kpi = input_data.get("kpi", "First Pass Yield")
        mode = input_data.get("mode", "correlation")
        top_k = input_data.get("top_k", 10)
        
        print(f"DEBUG: Using parameters: df_csv={df_csv[:50]}..., kpi={kpi}, mode={mode}")
        
        # Call the original implementation
        return self.run(df_csv=df_csv, kpi=kpi, mode=mode, top_k=top_k)

    def run(
        self,
        *,
        df_csv: str | bytes,
        kpi: str,
        mode: str = "correlation",
        top_k: int | None = 10,
    ) -> str:
        """
        Original implementation with full type annotations.
        
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
        from io import StringIO
        df = pd.read_csv(StringIO(df_csv)) if isinstance(df_csv, str) else pd.read_csv(df_csv)

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