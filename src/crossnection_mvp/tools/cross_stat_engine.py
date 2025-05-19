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
import logging
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.weightstats import ztest
from crewai.tools import BaseTool
from crossnection_mvp.utils.metadata_loader import enrich_driver_names
from crossnection_mvp.utils.context_store import ContextStore

# Configura logger
logger = logging.getLogger(__name__)

# Costanti per messaggi di errore
ERROR_STATE_KEY = "error_state"
ERROR_MESSAGE_KEY = "error_message"
USER_MESSAGE_KEY = "user_message"

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
    """Compute r & p per ogni colonna numerica vs kpi."""
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
                logger.error(f"Error computing correlation for {col}: {e}")
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
    
    # Ottieni le descrizioni arricchite dei driver
    driver_names = ranked["driver_name"].tolist()
    enriched_names = enrich_driver_names(driver_names)
    
    # Aggiungi descrizioni ai risultati
    ranked_dict = ranked.to_dict(orient="records")
    for item in ranked_dict:
        driver = item["driver_name"]
        if driver in enriched_names:
            item["driver_description"] = enriched_names[driver]
    
    return ranked_dict


# ---------------------------------------------------------------------------#
# Outlier detection
# ---------------------------------------------------------------------------#


# Trova la funzione outlier_report e modifica per garantire formato standard
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
            logger.error(f"Error detecting outliers for {col}: {e}")
            continue
    
    # Assicurati di avere una struttura standard
    return {
        "kpi": kpi, 
        "outliers": outliers,
        "success": True,
        "summary": f"Found {len(outliers)} outliers across {len(set(o['driver'] for o in outliers))} drivers"
    }


# ---------------------------------------------------------------------------#
# CrewAI Tool wrapper
# ---------------------------------------------------------------------------#

class CrossStatEngineToolSchema(BaseModel):
    input: Optional[Union[str, Dict[str, Any]]] = None
    
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

    def _run(self, input: Union[str, Dict[str, Any]] = None, **kwargs) -> str:
        """
        Main entry point required by BaseTool
                    
        Adattato per accettare parametri sia direttamente che nel formato {"input": ...}
        """
        # Gestisci il caso in cui l'input arriva come kwargs invece che sotto la chiave 'input'
        if input is None and kwargs:
            # Adatta la struttura per renderla compatibile
            input = kwargs
        
        logger.info(f"CrossStatEngineTool received input: {str(input)[:100]}...")
        
        # Converti l'input in un formato che possiamo usare
        unified_dataset = None
        kpi = "value_speed"  # Default KPI
        mode = "correlation"  # Default mode
        top_k = 10  # Default top_k
        
        # Prova a caricare il dataset unificato dal Context Store
        try:
            store = ContextStore.get_instance()
            unified_dataset = store.load_dataframe("unified_dataset")
            logger.info("Successfully loaded unified dataset from Context Store")
        except Exception as e:
            logger.warning(f"Failed to load unified dataset from Context Store: {e}")
            unified_dataset = None
        
        # Se non abbiamo ancora un dataset dal Context Store, prova il file legacy
        if unified_dataset is None:
            try:
                unified_path = Path("examples/driver_csvs/unified_dataset.csv")
                if unified_path.exists():
                    logger.info(f"Found unified dataset at {unified_path}")
                    unified_dataset = pd.read_csv(unified_path)
                    # Salva nel Context Store per uso futuro
                    store = ContextStore.get_instance()
                    store.save_dataframe("unified_dataset", unified_dataset)
            except Exception as e:
                logger.error(f"Error loading unified dataset from file: {e}")
        
        # Se non abbiamo ancora un dataset, creane uno dai file originali
        if unified_dataset is None:
            try:
                logger.info("Creating unified dataset from source files")
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
                        logger.info(f"Loaded {file.name} with shape {df.shape}")
                    except Exception as e:
                        logger.error(f"Failed to load {file}: {e}")
                
                if not dataframes:
                    raise ValueError("Failed to load any CSV files")
                    
                # Unisci i dataframe
                unified_dataset = dataframes[0]
                for df in dataframes[1:]:
                    try:
                        unified_dataset = unified_dataset.merge(df, on="join_key", how="outer", suffixes=("", "_dup"))
                    except Exception as e:
                        logger.error(f"Merge failed: {e}")
                        continue
                
                # Rimuovi colonne duplicate
                dupes = [c for c in unified_dataset.columns if c.endswith("_dup")]
                unified_dataset.drop(columns=dupes, inplace=True)
                
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_dataframe("unified_dataset", unified_dataset)
                
                # Salva anche il file legacy per compatibilità
                unified_path = Path("examples/driver_csvs/unified_dataset.csv")
                unified_dataset.to_csv(unified_path, index=False)
                
                logger.info(f"Created and saved unified dataset with shape {unified_dataset.shape}")
            except Exception as e:
                logger.error(f"Error creating unified dataset: {e}")
                # Invece di creare dati casuali, indica chiaramente l'errore
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Non è stato possibile caricare o creare un dataset valido. Verifica che i file CSV siano presenti e nel formato corretto.",
                    "columns": []
                }
                return json.dumps(error_result)
        
        # Parse input specifico se fornito
        input_data = {}
        if isinstance(input, str):
            try:
                input_data = json.loads(input)
                logger.info(f"Parsed JSON string input")
            except json.JSONDecodeError:
                # Caso speciale: l'input è una stringa CSV
                if input.count('\n') > 5 and input.count(',') > 5:
                    logger.info("Input appears to be CSV data")
                    try:
                        from io import StringIO
                        unified_dataset = pd.read_csv(StringIO(input))
                        # Salva nel Context Store
                        store = ContextStore.get_instance()
                        store.save_dataframe("unified_dataset", unified_dataset)
                    except Exception as e:
                        logger.error(f"Error parsing CSV input: {e}")
                else:
                    # Input è una stringa semplice, potrebbe essere il KPI
                    logger.info(f"Simple string input, using as KPI: {input}")
                    if input not in ["unified_dataset_csv"]:  # Ignora nomi che sembrano riferirsi al dataset
                        kpi = input
        elif isinstance(input, dict):
            # Extract from complex dict structure
            if "input" in input and isinstance(input["input"], dict):
                input_data = input["input"]
            else:
                input_data = input
            logger.info("Using input dict directly")
        
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
            
            # Se c'è un riferimento al Context Store, usalo
            if "ref" in input_data and isinstance(input_data["ref"], str):
                try:
                    store = ContextStore.get_instance()
                    ref_path = input_data["ref"]
                    if ref_path.endswith(".csv"):
                        unified_dataset = store.load_dataframe(ref_path.split(".")[0])
                    logger.info(f"Loaded dataset from Context Store reference: {ref_path}")
                except Exception as e:
                    logger.error(f"Error loading from Context Store reference: {e}")
            
            # Estrai CSV data se presente
            if "df_csv" in input_data and input_data["df_csv"]:
                try:
                    from io import StringIO
                    df_csv = input_data["df_csv"]
                    if isinstance(df_csv, dict) and "path" in df_csv:
                        # È un riferimento al Context Store
                        store = ContextStore.get_instance()
                        try:
                            path_parts = df_csv["path"].split("/")
                            artifact_name = path_parts[-1].split(".")[0]
                            unified_dataset = store.load_dataframe(artifact_name)
                        except Exception as e:
                            logger.error(f"Error loading from Context Store path: {e}")
                    else:
                        # È un contenuto CSV diretto
                        unified_dataset = pd.read_csv(StringIO(input_data["df_csv"]))
                        # Salva nel Context Store
                        store = ContextStore.get_instance()
                        store.save_dataframe("unified_dataset", unified_dataset)
                except Exception as e:
                    logger.error(f"Error parsing df_csv: {e}")
                    
        # Gestione per casi speciali di input non standard
        if isinstance(input_data, dict) and 'description' in input_data and 'type' in input_data:
        # Rileva tentativi di rilevamento outlier con formato non standard
            if input_data.get('type') in ['outlier detection', 'outliers']:
                logger.info("Detected non-standard outlier detection request, converting to standard format")
                mode = "outliers"
                # Usa kpi di default se non specificato
                if 'kpi' not in input_data:
                    kpi = "value_speed"  # o recupera dal context store
        
        # Assicurati che il KPI esista nel dataset
        kpi_ok = True
        if unified_dataset is not None and kpi not in unified_dataset.columns:
            logger.warning(f"KPI column '{kpi}' not found in dataset")
            kpi_ok = False
            # Cerca una colonna che potrebbe essere il KPI
            value_cols = [col for col in unified_dataset.columns if col.startswith("value_")]
            if value_cols:
                kpi = value_cols[0]
                logger.info(f"Using '{kpi}' as KPI")
                kpi_ok = True
            else:
                numeric_cols = unified_dataset.select_dtypes(include="number").columns
                if len(numeric_cols) > 0 and numeric_cols[0] != "join_key":
                    kpi = numeric_cols[0]
                    logger.info(f"Using '{kpi}' as KPI")
                    kpi_ok = True
                else:
                    # Invece di creare una colonna random, segnala l'errore
                    error_result = {
                        ERROR_STATE_KEY: True,
                        ERROR_MESSAGE_KEY: f"KPI '{kpi}' non trovato e nessuna colonna numerica valida disponibile",
                        USER_MESSAGE_KEY: f"Non è stato possibile trovare il KPI specificato '{kpi}'. Assicurati che il file CSV contenga una colonna con questo nome o una colonna che inizi con 'value_'.",
                        "columns": list(unified_dataset.columns) if unified_dataset is not None else []
                    }
                    store = ContextStore.get_instance()
                    if mode == "correlation":
                        store.save_json("correlation_matrix", error_result)
                    elif mode == "ranking":
                        store.save_json("impact_ranking", error_result)
                    else:  # outliers
                        store.save_json("outlier_report", error_result)
                    return json.dumps(error_result)
        
        # Converti in CSV per run
        if unified_dataset is not None:
            df_csv = unified_dataset.to_csv(index=False)
            logger.info(f"Using parameters: kpi={kpi}, mode={mode}, dataset shape={unified_dataset.shape}")
        else:
            # Errore: non abbiamo un dataset
            error_result = {
                ERROR_STATE_KEY: True,
                ERROR_MESSAGE_KEY: "Nessun dataset valido disponibile",
                USER_MESSAGE_KEY: "Non è stato possibile caricare o creare un dataset valido. Verifica che i file CSV siano presenti e nel formato corretto."
            }
            store = ContextStore.get_instance()
            if mode == "correlation":
                store.save_json("correlation_matrix", error_result)
            elif mode == "ranking":
                store.save_json("impact_ranking", error_result)
            else:  # outliers
                store.save_json("outlier_report", error_result)
            return json.dumps(error_result)
        
        try:
            # Tentativo di esecuzione con il dataset corretto
            result = self.run(df_csv=df_csv, kpi=kpi, mode=mode, top_k=top_k)
            
            # Verifica se il risultato contiene già un errore
            if isinstance(result, str):
                try:
                    result_obj = json.loads(result)
                    if isinstance(result_obj, dict) and result_obj.get(ERROR_STATE_KEY, False):
                        # Già contiene un errore, passa avanti
                        pass
                    else:
                        # Aggiungi flag di successo
                        if isinstance(result_obj, dict):
                            result_obj["success"] = True
                            result = json.dumps(result_obj)
                except:
                    # Non è JSON o parsing fallito, lascia invariato
                    pass
            
            # Salva anche il risultato nel Context Store
            store = ContextStore.get_instance()
            if mode == "correlation":
                corr_data = json.loads(result) if isinstance(result, str) else result
                store.save_json("correlation_matrix", corr_data)
            elif mode == "ranking":
                ranking_data = json.loads(result) if isinstance(result, str) else result
                store.save_json("impact_ranking", ranking_data)
            elif mode == "outliers":
                outlier_data = json.loads(result) if isinstance(result, str) else result
                store.save_json("outlier_report", outlier_data)
                
            return result
        except Exception as e:
            error_msg = f"ERROR executing CrossStatEngineTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Fallback per compatibilità scipy
            if "pearsonr() got an unexpected keyword argument 'nan_policy'" in str(e):
                logger.info("Attempting fallback for scipy compatibility...")
                try:
                    result = self._fallback_analysis(unified_dataset, kpi, mode, top_k)
                    return result
                except Exception as fallback_err:
                    logger.error(f"Fallback analysis failed: {fallback_err}", exc_info=True)
                    
            # Restituisci un JSON di errore che può essere processato dagli agenti
            error_result = {
                ERROR_STATE_KEY: True,
                ERROR_MESSAGE_KEY: str(e),
                USER_MESSAGE_KEY: self._user_friendly_error_message(e, mode)
            }
            
            if mode == "correlation":
                # Aggiungi una minima struttura di fallback, ma con flag di errore
                error_result["drivers"] = []
                store = ContextStore.get_instance()
                store.save_json("correlation_matrix", error_result)
                return json.dumps(error_result)
            elif mode == "ranking":
                # Aggiungi una minima struttura di fallback, ma con flag di errore
                error_result["kpi_name"] = kpi
                error_result["ranking"] = []
                store = ContextStore.get_instance()
                store.save_json("impact_ranking", error_result)
                return json.dumps(error_result)
            else:  # outliers
                # Aggiungi una minima struttura di fallback, ma con flag di errore
                error_result["kpi"] = kpi
                error_result["outliers"] = []
                store = ContextStore.get_instance()
                store.save_json("outlier_report", error_result)
                return json.dumps(error_result)

    def _user_friendly_error_message(self, error: Exception, mode: str) -> str:
        """Genera un messaggio di errore comprensibile per l'utente in base al tipo di errore."""
        error_str = str(error).lower()
        
        if "pearsonr() got an unexpected keyword argument" in error_str:
            return "C'è un problema di compatibilità con la libreria statistica. Ci scusiamo per l'inconveniente. Il team è stato notificato per risolvere il problema."
        
        if "cannot open file" in error_str or "no such file" in error_str:
            return "Non è stato possibile trovare i file di dati necessari. Assicurati di aver caricato correttamente i file CSV e riprova."
        
        if "invalid literal for" in error_str or "could not convert" in error_str:
            return "Alcuni dati nei file CSV non sono nel formato previsto. Verifica che i tuoi file contengano dati numerici validi per le analisi statistiche."
        
        if "merge" in error_str or "join" in error_str:
            return "Si è verificato un problema durante l'unione dei dataset. Verifica che tutti i file CSV contengano la stessa chiave di join 'join_key'."
        
        # Messaggi specifici per modalità
        mode_messages = {
            "correlation": "Non è stato possibile calcolare le correlazioni tra i driver e il KPI. ",
            "ranking": "Non è stato possibile creare la classifica di impatto dei driver. ",
            "outliers": "Non è stato possibile rilevare gli outlier nel dataset. "
        }
        
        # Messaggio di base + suggerimento generico
        base_message = mode_messages.get(mode, "Si è verificato un errore durante l'analisi. ")
        return base_message + "Verifica i tuoi dati e riprova. Se il problema persiste, contatta il supporto tecnico."

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
            
            # Salva il DataFrame nel Context Store per riferimento futuro
            store = ContextStore.get_instance()
            store.save_dataframe("unified_dataset", df)
        except Exception as e:
            logger.error(f"Error reading df_csv: {e}", exc_info=True)
            # Prova a caricare dal Context Store
            try:
                store = ContextStore.get_instance()
                df = store.load_dataframe("unified_dataset")
                logger.info("Loaded unified dataset from Context Store as fallback")
            except Exception as store_err:
                logger.error(f"Error loading from Context Store: {store_err}", exc_info=True)
                # Invece di creare dati casuali, restituisci un messaggio di errore
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Non è stato possibile leggere il dataset. Verifica che il formato CSV sia valido."
                }
                return json.dumps(error_result)
            
            # Ensure KPI exists
            if kpi not in df.columns:
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: f"KPI column '{kpi}' not found in dataset",
                    USER_MESSAGE_KEY: f"Il KPI '{kpi}' non è presente nel dataset. Colonne disponibili: {', '.join(df.columns)}",
                    "available_columns": list(df.columns)
                }
                return json.dumps(error_result)

        if mode == "correlation":
            try:
                corr = correlation_matrix(df, kpi=kpi)
                result = corr.to_json(orient="records")
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("correlation_matrix", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in correlation_matrix: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante il calcolo delle correlazioni. Assicurati che i driver contengano dati numerici validi.",
                    "drivers": []
                }
                # Salva il fallback nel Context Store
                store = ContextStore.get_instance()
                store.save_json("correlation_matrix", error_result)
                return json.dumps(error_result)

        if mode == "ranking":
            try:
                corr_df = correlation_matrix(df, kpi=kpi)
                ranked = impact_ranking(corr_df, top_k=top_k)
                result = json.dumps(
                    {"kpi_name": kpi, "ranking": ranked, "success": True},
                    ensure_ascii=False,
                    indent=2,
                )
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("impact_ranking", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in impact_ranking: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante la creazione del ranking dei driver. Verifica che ci siano sufficienti dati per l'analisi statistica.",
                    "kpi_name": kpi, 
                    "ranking": []
                }
                # Salva il fallback nel Context Store
                store = ContextStore.get_instance()
                store.save_json("impact_ranking", error_result)
                return json.dumps(error_result)

        if mode == "outliers":
            try:
                report = outlier_report(df, kpi=kpi)
                # Assicurati che la struttura sia sempre corretta
                if "outliers" not in report:
                    report["outliers"] = []
                if "kpi" not in report:
                    report["kpi"] = kpi
                report["success"] = True
                report["summary"] = f"Found {len(report['outliers'])} outliers across {len(set(o.get('driver', '') for o in report['outliers']))} drivers"
                if "success" not in report:
                    report["success"] = True
                    
                result = json.dumps(report, ensure_ascii=False, indent=2)
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("outlier_report", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in outlier_report: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante il rilevamento degli outlier. Verifica che il dataset contenga sufficienti dati validi.",
                    "kpi": kpi, 
                    "outliers": []
                }
                # Salva il fallback nel Context Store
                store = ContextStore.get_instance()
                store.save_json("outlier_report", error_result)
                return json.dumps(error_result)

        error_result = {
            ERROR_STATE_KEY: True,
            ERROR_MESSAGE_KEY: f"Invalid mode: {mode}",
            USER_MESSAGE_KEY: f"Modalità '{mode}' non valida. Le modalità disponibili sono: 'correlation', 'ranking', 'outliers'."
        }
        return json.dumps(error_result)

    def _fallback_analysis(self, df: pd.DataFrame, kpi: str, mode: str, top_k: Optional[int] = 10) -> str:
        """Fallback implementation without nan_policy for older scipy versions."""
        if mode == "correlation":
            try:
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
                        logger.error(f"Error computing correlation for {col}: {e}")
                        rows.append({"driver_name": col, "method": method, "r": 0, "p_value": 1.0})
                
                result = json.dumps(rows)
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("correlation_matrix", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in fallback correlation analysis: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante l'analisi di correlazione alternativa. Verifica che i dati siano nel formato corretto.",
                    "drivers": []
                }
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("correlation_matrix", error_result)
                return json.dumps(error_result)
            
        elif mode == "ranking":
            try:
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
                    
                result = json.dumps({"kpi_name": kpi, "ranking": ranking, "success": True})
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("impact_ranking", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in fallback ranking analysis: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante la creazione del ranking di impatto alternativo. Verifica che i dati siano nel formato corretto.",
                    "kpi_name": kpi,
                    "ranking": []
                }
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("impact_ranking", error_result)
                return json.dumps(error_result)
            
        else:  # outliers
            try:
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
                    
                result = json.dumps({"kpi": kpi, "outliers": outliers, "success": True})
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("outlier_report", json.loads(result))
                return result
            except Exception as e:
                logger.error(f"Error in fallback outlier analysis: {e}", exc_info=True)
                error_result = {
                    ERROR_STATE_KEY: True,
                    ERROR_MESSAGE_KEY: str(e),
                    USER_MESSAGE_KEY: "Si è verificato un errore durante il rilevamento di outlier alternativo. Verifica che i dati siano nel formato corretto.",
                    "kpi": kpi,
                    "outliers": []
                }
                # Salva nel Context Store
                store = ContextStore.get_instance()
                store.save_json("outlier_report", error_result)
                return json.dumps(error_result)

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
                logger.error(f"Error computing correlation for {col}: {e}")
                rows.append({"driver_name": col, "method": method, "r": 0, "p_value": 1.0})
        
        return pd.DataFrame(rows)