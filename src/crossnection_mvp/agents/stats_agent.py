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


class StatsAgent(cr.Agent):
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
        # Aggiungi debug per verificare il dataframe
        print(f"\n----------------------------------------------------------------")
        print(f"DEBUG STATS_AGENT: Starting stats pipeline")
        print(f"DEBUG STATS_AGENT: KPI={kpi}, top_k={top_k}")
        print(f"DEBUG STATS_AGENT: unified_dataset type={type(unified_dataset)}")
        print(f"----------------------------------------------------------------\n")
        
        # Load dataset ---------------------------------------------------
        if isinstance(unified_dataset, pd.DataFrame):
            df = unified_dataset
            print(f"DEBUG STATS_AGENT: Using provided DataFrame directly")
        elif isinstance(unified_dataset, Path):
            df = pd.read_csv(unified_dataset)
            print(f"DEBUG STATS_AGENT: Loaded DataFrame from path: {unified_dataset}")
        else:  # assume CSV string/bytes
            # Verifica se è un percorso file
            if isinstance(unified_dataset, str) and ('\\' in unified_dataset or '/' in unified_dataset):
                file_path = Path(unified_dataset)
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    print(f"DEBUG STATS_AGENT: Loaded DataFrame from file path: {file_path}")
                else:
                    # Prova a caricare dal Context Store se sembra un percorso al Context Store
                    store = ContextStore.get_instance()
                    try:
                        # Estrai nome base del file
                        artifact_name = "unified_dataset"
                        df = store.load_dataframe(artifact_name)
                        print(f"DEBUG STATS_AGENT: Loaded DataFrame from Context Store: {artifact_name}")
                    except Exception as e:
                        print(f"DEBUG STATS_AGENT: Failed to load from Context Store: {e}")
                        # Fallback alla lettura come contenuto CSV
                        df = pd.read_csv(pd.compat.StringIO(unified_dataset))
                        print(f"DEBUG STATS_AGENT: Loaded DataFrame from CSV string content")
            else:
                df = pd.read_csv(pd.compat.StringIO(unified_dataset))
                print(f"DEBUG STATS_AGENT: Loaded DataFrame from CSV string content")

        # Aggiungi debug per verificare il dataframe
        print(f"\n----------------------------------------------------------------")
        print(f"DEBUG STATS_AGENT: Dataset loaded, shape={df.shape}")
        print(f"DEBUG STATS_AGENT: Columns={df.columns.tolist()}")
        print(f"DEBUG STATS_AGENT: Looking for KPI='{kpi}'")
        print(f"DEBUG STATS_AGENT: KPI in columns? {kpi in df.columns}")
        print(f"----------------------------------------------------------------\n")
        
        # Normalizza i nomi delle colonne
        try:
            # Converti tutti i nomi di colonna a stringhe e rimuovi spazi
            df.columns = [str(col).strip() for col in df.columns]
            print(f"DEBUG STATS_AGENT: Normalized column names: {df.columns.tolist()}")
            print(f"DEBUG STATS_AGENT: KPI in normalized columns? {kpi in df.columns}")
        except Exception as e:
            print(f"ERROR normalizing columns: {e}")

        if kpi not in df.columns:
            raise KeyError(f"KPI column '{kpi}' not found in dataset")

        # Salviamo il dataset nel Context Store
        store = ContextStore.get_instance()
        store.save_dataframe("unified_dataset", df)

        tool = CrossStatEngineTool()

        # ------------------------------------------------ correlation ----
        logger.info("[StatsAgent] Computing correlation matrix…")
        df_csv = df.to_csv(index=False)
        corr_json = tool.run(df_csv=df_csv, kpi=kpi, mode="correlation")
        correlation_matrix: List[Dict[str, Any]] = json.loads(corr_json)
        store.save_json("correlation_matrix", correlation_matrix)

        # ------------------------------------------------ ranking ------
        logger.info("[StatsAgent] Building impact ranking (top %s)…", top_k)
        rank_json = tool.run(
            df_csv=df_csv, kpi=kpi, mode="ranking", top_k=top_k
        )
        impact_ranking_data = json.loads(rank_json)
        impact_ranking: List[Dict[str, Any]] = impact_ranking_data["ranking"]
        store.save_json("impact_ranking", impact_ranking_data)

        # ------------------------------------------------ outliers ------
        logger.info("[StatsAgent] Detecting outliers…")
        outlier_json = tool.run(df_csv=df_csv, kpi=kpi, mode="outliers")
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
        
        # Verifica se df_csv è un path invece che contenuto
        if isinstance(df_csv, str) and ('\\' in df_csv or '/' in df_csv):
            logger.info(f"df_csv appears to be a path: {df_csv}")
            
            try:
                # Prova a caricare il dataframe direttamente
                path = Path(df_csv)
                if path.exists():
                    logger.info(f"Loading DataFrame from file: {path}")
                    df = pd.read_csv(path)
                    # Converti di nuovo in CSV per il tool
                    df_csv = df.to_csv(index=False)
                else:
                    # Prova a caricare dal Context Store
                    store = ContextStore.get_instance()
                    logger.info("Loading DataFrame from Context Store")
                    df = store.load_dataframe("unified_dataset")
                    df_csv = df.to_csv(index=False)
            except Exception as e:
                logger.error(f"Error loading dataframe: {e}")
                # Continua con df_csv originale
        
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        # Assicurati di passare i parametri corretti
        result = tool.run(df_csv=df_csv, kpi=kpi, mode="correlation")
        
        # Normalizza il risultato
        if isinstance(result, str):
            try:
                result_obj = json.loads(result)
            except json.JSONDecodeError:
                # Fallback se non è JSON valido
                result_obj = []
        else:
            result_obj = result
        
        # Assicurati che il risultato abbia il formato corretto
        if isinstance(result_obj, list):
            normalized_results = []
            for item in result_obj:
                if isinstance(item, dict):
                    # Assicurati che tutti i campi necessari siano presenti
                    driver_name = item.get("driver_name", item.get("driver", ""))
                    r_value = item.get("r", item.get("correlation", 0))
                    p_value = item.get("p_value", 1.0)
                    
                    # Converti i valori a numeri se sono stringhe
                    if isinstance(r_value, str):
                        try:
                            r_value = float(r_value)
                        except ValueError:
                            r_value = 0
                    
                    if isinstance(p_value, str):
                        try:
                            p_value = float(p_value)
                        except ValueError:
                            p_value = 1.0
                    
                    normalized_item = {
                        "driver_name": driver_name,
                        "r": r_value,
                        "p_value": p_value,
                        "method": item.get("method", "pearson")
                    }
                    normalized_results.append(normalized_item)
            
            # Aggiorna result_obj con i risultati normalizzati
            result_obj = normalized_results
        
        # Salva nel Context Store
        store = ContextStore.get_instance()
        store.save_json("correlation_matrix", result_obj)
        
        return result_obj

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
        
        # Verifica se df_csv è un path invece che contenuto
        if isinstance(df_csv, str) and ('\\' in df_csv or '/' in df_csv):
            logger.info(f"df_csv appears to be a path: {df_csv}")
            
            try:
                # Prova a caricare il dataframe direttamente
                path = Path(df_csv)
                if path.exists():
                    logger.info(f"Loading DataFrame from file: {path}")
                    df = pd.read_csv(path)
                    # Converti di nuovo in CSV per il tool
                    df_csv = df.to_csv(index=False)
                else:
                    # Prova a caricare dal Context Store
                    store = ContextStore.get_instance()
                    logger.info("Loading DataFrame from Context Store")
                    df = store.load_dataframe("unified_dataset")
                    df_csv = df.to_csv(index=False)
            except Exception as e:
                logger.error(f"Error loading dataframe: {e}")
                # Continua con df_csv originale
        
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        result = tool.run(df_csv=df_csv, kpi=kpi, mode="ranking")
        
        # Normalizza il risultato
        if isinstance(result, str):
            try:
                result_obj = json.loads(result)
            except:
                # Fallback se non è JSON valido
                result_obj = {"kpi_name": kpi, "ranking": []}
        else:
            result_obj = result
        
        # Assicurati che il risultato abbia il formato corretto
        if isinstance(result_obj, dict):
            if "ranking" not in result_obj:
                result_obj["ranking"] = []
            if "kpi_name" not in result_obj:
                result_obj["kpi_name"] = kpi
        else:
            result_obj = {"kpi_name": kpi, "ranking": []}
        
        # Normalizza i valori numerici
        ranking = result_obj.get("ranking", [])
        normalized_ranking = []
        
        for item in ranking:
            if isinstance(item, dict):
                # Normalizza i valori numerici
                r_value = item.get("r", 0)
                p_value = item.get("p_value", 1.0)
                score = item.get("score", 0)
                
                # Converti a numeri se sono stringhe
                if isinstance(r_value, str):
                    try:
                        r_value = float(r_value)
                    except ValueError:
                        r_value = 0
                
                if isinstance(p_value, str):
                    try:
                        p_value = float(p_value)
                    except ValueError:
                        p_value = 1.0
                        
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0
                
                normalized_item = {
                    **item,
                    "r": r_value,
                    "p_value": p_value,
                    "score": score
                }
                normalized_ranking.append(normalized_item)
        
        result_obj["ranking"] = normalized_ranking
        
        # Salva nel Context Store
        store = ContextStore.get_instance()
        store.save_json("impact_ranking", result_obj)
        
        return result_obj

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
        """Identifica outlier nei dataset dei driver."""
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "detect_outliers"
            
        # Estrai parametri dall'input
        df_csv = kwargs.get("df_csv", "")
        kpi = kwargs.get("kpi", "value_speed")
        
        logger.info(f"StatsAgent.detect_outliers called with kpi={kpi}, df_csv_len={len(df_csv) if df_csv else 0}")
        
        # Verifica se df_csv è un path invece che contenuto
        if isinstance(df_csv, str) and ('\\' in df_csv or '/' in df_csv):
            logger.info(f"df_csv appears to be a path: {df_csv}")
            
            try:
                # Prova a caricare il dataframe direttamente
                path = Path(df_csv)
                if path.exists():
                    logger.info(f"Loading DataFrame from file: {path}")
                    df = pd.read_csv(path)
                    # Converti di nuovo in CSV per il tool
                    df_csv = df.to_csv(index=False)
                else:
                    # Prova a caricare dal Context Store
                    store = ContextStore.get_instance()
                    logger.info("Loading DataFrame from Context Store")
                    df = store.load_dataframe("unified_dataset")
                    df_csv = df.to_csv(index=False)
            except Exception as e:
                logger.error(f"Error loading dataframe: {e}")
                # Continua con df_csv originale
        
        # IMPORTANTE: Usa esplicitamente mode="outliers"
        # Ottieni lo strumento e eseguilo
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        tool = CrossStatEngineTool()
        
        # Esecuzione esplicita con mode="outliers"
        result = tool.run(df_csv=df_csv, kpi=kpi, mode="outliers")
        
        # Normalizza il risultato
        if isinstance(result, str):
            try:
                result_obj = json.loads(result)
            except:
                # Fallback se non è JSON valido
                result_obj = {"kpi": kpi, "outliers": [], "success": False}
        else:
            result_obj = result
        
        # Assicurati che la struttura sia sempre corretta
        if "outliers" not in result_obj:
            result_obj["outliers"] = []
        if "kpi" not in result_obj:
            result_obj["kpi"] = kpi
        if "success" not in result_obj:
            result_obj["success"] = True
        
        # Salva nel Context Store per sicurezza
        store = ContextStore.get_instance()
        store.save_json("outlier_report", result_obj)
        
        return result_obj

    # ------------------------------------------------------------------
    # Friendly repr for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401 – simple representation
        return f"<StatsAgent role='{self._ROLE}'>"