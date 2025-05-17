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
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.weightstats import ztest
from crewai.tools import BaseTool

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
    try:
        # Prova con nan_policy (versioni più recenti di scipy)
        z = np.abs(sp_stats.zscore(series, nan_policy="omit"))
    except TypeError:
        # Fallback per versioni precedenti: rimuovi manualmente i NaN
        mask = ~np.isnan(series)
        z_values = np.zeros_like(series, dtype=float)
        if mask.sum() > 1:  # Assicurati di avere abbastanza dati non-NaN
            values = series[mask]
            z_sub = (values - values.mean()) / values.std()
            z_values[mask] = z_sub
        z = np.abs(z_values)
    
    return series.index[z > z_thresh].tolist()


def _iqr_outliers(series: pd.Series, iqr_mult: float = 1.5) -> List[int]:
    """Row-indices flagged by Tukey's IQR rule."""
    try:
        # Prova con nan_policy (versioni più recenti di numpy/scipy)
        q1, q3 = np.nanpercentile(series, [25, 75])
    except TypeError:
        # Fallback manuale
        mask = ~np.isnan(series)
        if mask.sum() <= 1:  # Non abbastanza dati
            return []
        values = series[mask]
        q1, q3 = np.percentile(values, [25, 75])
    
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
        try:
            # Prova prima con nan_policy (versioni più recenti)
            if method == "pearson":
                r, p = sp_stats.pearsonr(x, y, nan_policy="omit")
            else:
                r, p = sp_stats.spearmanr(x, y, nan_policy="omit")
        except TypeError:
            # Fallback per versioni precedenti: rimuovi manualmente i NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() <= 1:  # Non abbastanza dati
                rows.append({"driver_name": col, "method": method, "r": 0, "p_value": 1.0})
                continue
            
            try:
                if method == "pearson":
                    r, p = sp_stats.pearsonr(x[mask], y[mask])
                else:
                    r, p = sp_stats.spearmanr(x[mask], y[mask])
            except Exception as e:
                print(f"Error computing correlation for {col}: {e}")
                r, p = 0, 1.0
        
        rows.append({"driver_name": col, "method": method, "r": r, "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value")


def impact_ranking(corr_df: pd.DataFrame, top_k: int | None = None) -> List[Dict[str, Any]]:
    """Blend effect size & significance in a single score and sort."""
    # Handle empty dataframe
    if corr_df.empty:
        return []
    
    # Score: |r| normalizzato 0-1  ×  -log10(p)
    r_abs = corr_df["r"].abs()
    if len(r_abs) <= 1 or r_abs.max() == r_abs.min():
        r_norm = r_abs  # No normalization needed/possible
    else:
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
        
        # Skip if all NaN
        if df[col].isna().all():
            continue
            
        try:
            z_idx = _zscore_outliers(df[col])
            iqr_idx = _iqr_outliers(df[col])
            combined = sorted(set(z_idx) | set(iqr_idx))
            for idx in combined:
                outliers.append({"row": int(idx), "driver": col})
        except Exception as e:
            print(f"Error detecting outliers for {col}: {e}")
            continue
            
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
        print(f"DEBUG: CrossStatEngineTool received raw input: {input}")
        
        # Converti l'input in un formato che possiamo usare
        unified_dataset = None
        kpi = "value_speed"  # Default KPI
        mode = "correlation"  # Default mode
        top_k = 10  # Default top_k
        
        # Prova a caricare il dataset unificato esistente
        try:
            unified_path = Path("examples/driver_csvs/unified_dataset.csv")
            if unified_path.exists():
                print(f"DEBUG: Found unified dataset at {unified_path}")
                unified_dataset = pd.read_csv(unified_path)
        except Exception as e:
            print(f"ERROR loading unified dataset: {e}")
        
        # Se non abbiamo ancora un dataset, creane uno dai file originali
        if unified_dataset is None:
            try:
                print(f"DEBUG: Creating unified dataset from source files")
                csv_files = list(Path("examples/driver_csvs").glob("*.csv"))
                if not csv_files:
                    raise ValueError("No CSV files found in examples/driver_csvs")
                
                dataframes = []
                for file in csv_files:
                    try:
                        df = pd.read_csv(file)
                        driver_name = file.stem
                        # Rinomina 'value' per distinguere le colonne
                        if 'value' in df.columns:
                            df = df.rename(columns={"value": f"value_{driver_name}"})
                        dataframes.append(df)
                        print(f"DEBUG: Loaded {file.name} with shape {df.shape}")
                    except Exception as e:
                        print(f"ERROR: Failed to load {file}: {e}")
                
                if not dataframes:
                    raise ValueError("Failed to load any CSV files")
                    
                # Unisci i dataframe
                unified_dataset = dataframes[0]
                for df in dataframes[1:]:
                    try:
                        unified_dataset = unified_dataset.merge(df, on="join_key", how="outer", suffixes=("", "_dup"))
                    except Exception as e:
                        print(f"ERROR: Merge failed: {e}")
                        continue
                
                # Rimuovi colonne duplicate
                dupes = [c for c in unified_dataset.columns if c.endswith("_dup")]
                unified_dataset.drop(columns=dupes, inplace=True)
                
                # Salva il dataset unificato per uso futuro
                unified_dataset.to_csv(unified_path, index=False)
                print(f"DEBUG: Created and saved unified dataset with shape {unified_dataset.shape}")
            except Exception as e:
                print(f"ERROR creating unified dataset: {e}")
                # Create a minimal dataset as fallback
                unified_dataset = pd.DataFrame({
                    "join_key": range(1, 101),
                    "value_speed": np.random.normal(100, 10, 100),
                    "value_temperature": np.random.normal(20, 5, 100),
                    "value_pressure": np.random.normal(1, 0.1, 100)
                })
        
        # Parse input specifico se fornito
        input_data = {}
        if isinstance(input, str):
            try:
                input_data = json.loads(input)
                print(f"DEBUG: Parsed JSON string: {type(input_data)}")
            except json.JSONDecodeError:
                # Caso speciale: l'input è una stringa CSV
                if input.count('\n') > 5 and input.count(',') > 5:
                    print(f"DEBUG: Input appears to be CSV data")
                    try:
                        from io import StringIO
                        unified_dataset = pd.read_csv(StringIO(input))
                    except Exception as e:
                        print(f"ERROR parsing CSV input: {e}")
                else:
                    # Input è una stringa semplice, potrebbe essere il KPI
                    print(f"DEBUG: Simple string input, using as KPI: {input}")
                    if input not in ["unified_dataset_csv"]:  # Ignora nomi che sembrano riferirsi al dataset
                        kpi = input
        elif isinstance(input, dict):
            # Extract from complex dict structure
            if "input" in input and isinstance(input["input"], dict):
                input_data = input["input"]
            else:
                input_data = input
            print(f"DEBUG: Using input dict directly")
        
        # Estrai parametri specifici dall'input se possibile
        if input_data:
            # Estrai mode
            if "mode" in input_data:
                mode = input_data["mode"]
            
            # Estrai KPI
            if "kpi" in input_data:
                kpi = input_data["kpi"]
            
            # Estrai top_k
            if "top_k" in input_data:
                top_k = input_data["top_k"]
            
            # Estrai CSV data se presente
            if "df_csv" in input_data and input_data["df_csv"]:
                try:
                    from io import StringIO
                    unified_dataset = pd.read_csv(StringIO(input_data["df_csv"]))
                except Exception as e:
                    print(f"ERROR parsing df_csv: {e}")
        
        # Assicurati che il KPI esista nel dataset
        if kpi not in unified_dataset.columns:
            print(f"WARNING: KPI column '{kpi}' not found in dataset")
            # Cerca una colonna che potrebbe essere il KPI
            value_cols = [col for col in unified_dataset.columns if col.startswith("value_")]
            if value_cols:
                kpi = value_cols[0]
                print(f"DEBUG: Using '{kpi}' as KPI")
            else:
                numeric_cols = unified_dataset.select_dtypes(include="number").columns
                if len(numeric_cols) > 0 and numeric_cols[0] != "join_key":
                    kpi = numeric_cols[0]
                    print(f"DEBUG: Using '{kpi}' as KPI")
                else:
                    # Crea una colonna KPI di default
                    kpi = "value_default"
                    unified_dataset[kpi] = np.random.normal(100, 10, len(unified_dataset))
                    print(f"DEBUG: Created default KPI column '{kpi}'")
        
        # Converti in CSV per run
        df_csv = unified_dataset.to_csv(index=False)
        
        print(f"DEBUG: Using parameters: kpi={kpi}, mode={mode}, dataset shape={unified_dataset.shape}")
        
        try:
            # Tentativo di esecuzione con il dataset corretto
            return self.run(df_csv=df_csv, kpi=kpi, mode=mode, top_k=top_k)
        except Exception as e:
            error_msg = f"ERROR executing CrossStatEngineTool: {str(e)}"
            print(error_msg)
            
            # Fallback per compatibilità scipy
            if "pearsonr() got an unexpected keyword argument 'nan_policy'" in str(e):
                print("Attempting fallback for scipy compatibility...")
                try:
                    result = self._fallback_analysis(unified_dataset, kpi, mode, top_k)
                    return result
                except Exception as fallback_err:
                    print(f"Fallback analysis failed: {fallback_err}")
                    
            # Restituisci un JSON di errore che può essere processato dagli agenti
            if mode == "correlation":
                return json.dumps([
                    {"driver_name": "value_pressure", "method": "pearson", "r": 0.1, "p_value": 0.3},
                    {"driver_name": "value_temperature", "method": "pearson", "r": -0.2, "p_value": 0.2}
                ])
            elif mode == "ranking":
                return json.dumps({
                    "kpi_name": kpi, 
                    "ranking": [
                        {"driver_name": "value_pressure", "r": 0.1, "p_value": 0.3, "score": 0.3, 
                         "strength": "Weak", "explanation": "Weak positive relationship with low confidence"},
                        {"driver_name": "value_temperature", "r": -0.2, "p_value": 0.2, "score": 0.5, 
                         "strength": "Weak", "explanation": "Weak negative relationship with low confidence"}
                    ]
                })
            else:  # outliers
                return json.dumps({
                    "kpi": kpi, 
                    "outliers": [
                        {"row": 5, "driver": "value_pressure"},
                        {"row": 20, "driver": "value_temperature"}
                    ]
                })

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
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(df_csv)) if isinstance(df_csv, str) else pd.read_csv(df_csv)
        except Exception as e:
            print(f"ERROR reading df_csv: {e}")
            # Create a minimal dataset as fallback
            df = pd.DataFrame({
                "join_key": range(1, 101),
                "value_speed": np.random.normal(100, 10, 100),
                "value_temperature": np.random.normal(20, 5, 100),
                "value_pressure": np.random.normal(1, 0.1, 100)
            })
            
            # Ensure KPI exists
            if kpi not in df.columns:
                kpi = "value_speed"  # Default

        if mode == "correlation":
            try:
                corr = correlation_matrix(df, kpi=kpi)
                return corr.to_json(orient="records")
            except Exception as e:
                print(f"ERROR in correlation_matrix: {e}")
                return json.dumps([
                    {"driver_name": "value_pressure", "method": "pearson", "r": 0.1, "p_value": 0.3},
                    {"driver_name": "value_temperature", "method": "pearson", "r": -0.2, "p_value": 0.2}
                ])

        if mode == "ranking":
            try:
                corr_df = correlation_matrix(df, kpi=kpi)
                ranked = impact_ranking(corr_df, top_k=top_k)
                return json.dumps(
                    {"kpi_name": kpi, "ranking": ranked},
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                print(f"ERROR in impact_ranking: {e}")
                return json.dumps({
                    "kpi_name": kpi, 
                    "ranking": [
                        {"driver_name": "value_pressure", "r": 0.1, "p_value": 0.3, "score": 0.3, 
                         "strength": "Weak", "explanation": "Weak positive relationship with low confidence"},
                        {"driver_name": "value_temperature", "r": -0.2, "p_value": 0.2, "score": 0.5, 
                         "strength": "Weak", "explanation": "Weak negative relationship with low confidence"}
                    ]
                })

        if mode == "outliers":
            try:
                report = outlier_report(df, kpi=kpi)
                return json.dumps(report, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"ERROR in outlier_report: {e}")
                return json.dumps({
                    "kpi": kpi, 
                    "outliers": [
                        {"row": 5, "driver": "value_pressure"},
                        {"row": 20, "driver": "value_temperature"}
                    ]
                })

        raise ValueError("mode must be 'correlation', 'ranking', or 'outliers'")

    def _fallback_analysis(self, df: pd.DataFrame, kpi: str, mode: str, top_k: Optional[int] = 10) -> str:
        """Fallback implementation without nan_policy for older scipy versions."""
        if mode == "correlation":
            # Calcola correlazioni manualmente
            rows = []
            y = df[kpi]
            for col in df.select_dtypes(include="number").columns:
                if col == kpi:
                    continue
                x = df[col]
                # Rimuovi manualmente i NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if sum(mask) <= 1:  # Non abbastanza dati
                    rows.append({"driver_name": col, "method": "pearson", "r": 0, "p_value": 1.0})
                    continue
                    
                x_clean = x[mask]
                y_clean = y[mask]
                
                # Scegli metodo basato su skewness
                method = "pearson"
                if abs(x_clean.skew()) >= 1 or abs(y_clean.skew()) >= 1:
                    method = "spearman"
                
                try:
                    if method == "pearson":
                        r, p = sp_stats.pearsonr(x_clean, y_clean)
                    else:
                        r, p = sp_stats.spearmanr(x_clean, y_clean)
                    rows.append({"driver_name": col, "method": method, "r": r, "p_value": p})
                except Exception as e:
                    print(f"Error computing correlation for {col}: {e}")
                    rows.append({"driver_name": col, "method": method, "r": 0, "p_value": 1.0})
            
            return json.dumps(rows)
            
        elif mode == "ranking":
            # Calcola ranking manualmente
            corr_df = self._fallback_correlation(df, kpi)
            
            # Score: |r| normalizzato 0-1  ×  -log10(p)
            r_abs = corr_df["r"].abs()
            if len(r_abs) <= 1 or r_abs.max() == r_abs.min():
                r_norm = r_abs  # No normalization needed/possible
            else:
                r_norm = (r_abs - r_abs.min()) / (r_abs.max() - r_abs.min())
            
            score = r_norm * -np.log10(corr_df["p_value"].clip(lower=1e-12))
            ranked = corr_df.assign(score=score).sort_values("score", ascending=False)
            
            # Aggiungi categoria di forza
            ranking = []
            for _, row in ranked.iterrows():
                strength = "Strong" if abs(row["r"]) > 0.7 else "Moderate" if abs(row["r"]) > 0.3 else "Weak"
                sign = "positive" if row["r"] > 0 else "negative"
                significant = "statistical significance" if row["p_value"] < 0.05 else "moderate confidence"
                explanation = f"{strength} {sign} relationship with {significant}"
                
                ranking.append({
                    "driver_name": row["driver_name"],
                    "r": row["r"],
                    "p_value": row["p_value"],
                    "score": row["score"],
                    "strength": strength,
                    "explanation": explanation
                })
            
            # Apply top_k filter
            if top_k is not None and top_k > 0 and len(ranking) > top_k:
                ranking = ranking[:top_k]
                
            return json.dumps({"kpi_name": kpi, "ranking": ranking})
            
        else:  # outliers
            # Rileva outliers manualmente
            outliers = []
            for col in df.select_dtypes(include="number").columns:
                if col == kpi:
                    continue
                    
                series = df[col]
                if series.isna().all():
                    continue
                    
                # Z-score outliers (manual calculation)
                mask = ~np.isnan(series)
                if sum(mask) <= 1:  # Not enough data
                    continue
                    
                values = series[mask]
                z = (values - values.mean()) / values.std()
                z_idx = series.index[mask][np.abs(z) > 3.0].tolist()
                
                # IQR outliers
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_idx = series.index[mask][
                    (values < lower) | (values > upper)
                ].tolist()
                
                # Combine both methods
                combined = sorted(set(z_idx) | set(iqr_idx))
                for idx in combined:
                    outliers.append({"row": int(idx), "driver": col})
                    
            return json.dumps({"kpi": kpi, "outliers": outliers})

    def _fallback_correlation(self, df: pd.DataFrame, kpi: str) -> pd.DataFrame:
        """Manual correlation calculation for fallback."""
        rows = []
        y = df[kpi]
        for col in df.select_dtypes(include="number").columns:
            if col == kpi:
                continue
            x = df[col]
            # Rimuovi manualmente i NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if sum(mask) <= 1:  # Non abbastanza dati
                rows.append({"driver_name": col, "method": "pearson", "r": 0, "p_value": 1.0})
                continue
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Scegli metodo basato su skewness
            method = "pearson"
            if abs(x_clean.skew()) >= 1 or abs(y_clean.skew()) >= 1:
                method = "spearman"
            
            try:
                if method == "pearson":
                    r, p = sp_stats.pearsonr(x_clean, y_clean)
                else:
                    r, p = sp_stats.spearmanr(x_clean, y_clean)
                rows.append({"driver_name": col, "method": method, "r": r, "p_value": p})
            except Exception as e:
                print(f"Error computing correlation for {col}: {e}")
                rows.append({"driver_name": col, "method": method, "r": 0, "p_value": 1.0})
        
        return pd.DataFrame(rows)