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
import json

import crewai as cr
from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool
from crossnection_mvp.utils.context_decorators import with_context_io
from crossnection_mvp.utils.error_handling import with_robust_error_handling

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

    @with_robust_error_handling(
        stage_name="data_pipeline",
        log_level="ERROR"
    )
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
    
    @with_context_io(
        output_key="data_report", 
        output_type="json"
    )
    @with_robust_error_handling(
        stage_name="profile_validate",
        store_error_key="data_report"
    )
    def profile_validate_dataset(self, **kwargs):
        """Profila e valida i file CSV dalla directory di input."""
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "profile_validate_dataset"
            
        # Estrazione e validazione parametri
        csv_folder = kwargs.get("csv_folder", kwargs.get("_original_csv_folder", "examples/driver_csvs"))
        kpi = kwargs.get("kpi", "Default KPI")
        mode = kwargs.get("mode", "full_pipeline")
        
        logger.info(f"DataAgent.profile_validate_dataset called with csv_folder={csv_folder}, kpi={kpi}")
        
        # Verifica esistenza directory
        csv_folder_path = Path(csv_folder)
        if not csv_folder_path.is_dir() and Path("examples/driver_csvs").is_dir():
            csv_folder = "examples/driver_csvs"
            logger.warning(f"Directory {csv_folder_path} not found, using examples/driver_csvs instead")
        
        # Ottieni il tool e eseguilo
        from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool
        tool = CrossDataProfilerTool()
        
        result = tool.run(csv_folder=csv_folder, kpi=kpi, mode=mode)
        return result
    
    @with_context_io(
        input_keys={"data_report": "data_report"}, 
        output_key="join_key_strategy", 
        output_type="json"
    )
    @with_robust_error_handling(
        stage_name="join_key_strategy",
        store_error_key="join_key_strategy"
    )
    def join_key_strategy(self, **kwargs):
        """Analizza il data_report per scoprire o generare una join-key unica."""
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "join_key_strategy"
            
        # Estrai data_report dall'input
        data_report = kwargs.get("data_report", {})
        data_report_ref = kwargs.get("data_report_ref", "")
        
        # Gestione avanzata dei riferimenti
        if not data_report and isinstance(data_report_ref, str) and len(data_report_ref) > 0:
            try:
                # Se è un riferimento al Context Store o un percorso file
                store = ContextStore.get_instance()
                logger.info(f"Attempting to load data_report from reference: {data_report_ref}")
                
                # Estrai il nome base senza estensione
                path = Path(data_report_ref)
                base_name = path.stem
                if '.' in base_name:  # Gestisce nomi del tipo 'data_report.v1'
                    base_name = base_name.split('.')[0]
                    
                # Carica dal Context Store
                data_report = store.load_json(base_name)
                logger.info(f"Successfully loaded data_report from Context Store using key: {base_name}")
            except Exception as e:
                logger.warning(f"Failed to load data_report from reference: {e}")
                # Fallback al default
                data_report = {}
        
        # Verifica se data_report contiene un errore
        if isinstance(data_report, dict) and data_report.get("error_state", False):
            logger.warning("Data report contains error state, passing through")
            return {
                "error_state": True,
                "error_message": data_report.get("error_message", "Unknown error"),
                "user_message": "Non è stato possibile determinare una strategia di join-key a causa di errori nei dati",
                "strategy": "error",
                "key_name": "none"
            }
                
        # Analizza le colonne dai report delle tabelle
        tables = data_report.get("tables", [])
        common_columns = set()
        
        # Trova colonne comuni in tutte le tabelle
        if tables:
            first_table_columns = set(tables[0].get("columns", {}).keys())
            common_columns = first_table_columns.copy()
            
            for table in tables[1:]:
                table_columns = set(table.get("columns", {}).keys())
                common_columns &= table_columns
                
        # Cerca candidati join-key (priorità: join_key, id, key)
        join_key_candidates = ["join_key", "id", "key"]
        selected_key = None
        
        for candidate in join_key_candidates:
            if candidate in common_columns:
                selected_key = candidate
                break
                
        # Se non troviamo una chiave comune, suggerisci di generarne una
        if not selected_key:
            logger.warning("No common join key found, suggesting synthetic key")
            return {
                "strategy": "generate",
                "key_name": "_cxn_id",
                "reason": "No common key columns found across tables"
            }
            
        # Se troviamo una chiave, verifica che sia un buon candidato (no null, tipo consistente)
        valid_key = True
        key_type = None
        
        for table in tables:
            if selected_key in table.get("columns", {}):
                key_info = table["columns"][selected_key]
                
                # Verifica null values
                if key_info.get("nulls", 0) > 0:
                    valid_key = False
                    logger.warning(f"Key '{selected_key}' has null values in some tables")
                    
                # Verifica consistenza del tipo
                current_type = key_info.get("dtype", "")
                if key_type is None:
                    key_type = current_type
                elif key_type != current_type:
                    valid_key = False
                    logger.warning(f"Key '{selected_key}' has inconsistent types: {key_type} vs {current_type}")
        
        # Restituisci la strategia appropriata
        if valid_key:
            return {
                "strategy": "use_existing",
                "key_name": selected_key,
                "key_type": key_type
            }
        else:
            return {
                "strategy": "fuzzy_match",
                "key_name": selected_key,
                "reason": "Key has null values or inconsistent types",
                "fallback": "_cxn_id"
            }
    
    @with_context_io(
        input_keys={
            "join_key_strategy": "join_key_strategy",
            "data_report": "data_report"
        },
        output_key="unified_dataset",
        output_type="dataframe"
    )
    @with_robust_error_handling(
        stage_name="clean_normalize",
        store_error_key="unified_dataset_error"
    )
    def clean_normalize_dataset(self, **kwargs):
        """Applica regole di pulizia e unisce tutti i dataset dei driver."""
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "clean_normalize_dataset"
            
        # Estrai parametri dall'input
        join_key_strategy = kwargs.get("join_key_strategy", {})
        data_report = kwargs.get("data_report", {})
        
        # Verifica se join_key_strategy contiene un errore
        if isinstance(join_key_strategy, dict) and join_key_strategy.get("error_state", False):
            logger.warning("Join key strategy contains error state, cannot proceed")
            raise ValueError(join_key_strategy.get("user_message", "Errore nella strategia join-key"))
            
        # Verifica se data_report contiene un errore
        if isinstance(data_report, dict) and data_report.get("error_state", False):
            logger.warning("Data report contains error state, cannot proceed")
            raise ValueError(data_report.get("user_message", "Errore nel report dei dati"))
            
        # Ottieni parametri dalla strategia
        strategy_type = join_key_strategy.get("strategy", "use_existing")
        key_name = join_key_strategy.get("key_name", "join_key")
        
        # Carica i CSV dal report
        csv_files = []
        tables = data_report.get("tables", [])
        
        for table in tables:
            file_path = table.get("file", "")
            if file_path:
                csv_files.append(Path(file_path))
                
        if not csv_files:
            raise ValueError("No CSV files found in data report")
            
        # Carica i dataframe
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Salva il nome del file come attributo
                df._file_path = str(csv_file)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load CSV file {csv_file}: {e}")
                raise ValueError(f"Errore nel caricamento del file {csv_file.name}")
                
        if not dataframes:
            raise ValueError("Nessun dataframe caricato con successo")
            
        # Applica strategia join-key
        if strategy_type == "use_existing":
            # Verifica che la chiave esista in tutti i dataframe
            for i, df in enumerate(dataframes):
                if key_name not in df.columns:
                    raise ValueError(f"Join key '{key_name}' non trovata nel dataframe {i+1}")
        elif strategy_type == "generate":
            # Genera una chiave sintetica per ogni dataframe
            logger.info(f"Generating synthetic key '{key_name}'")
            for df in dataframes:
                df[key_name] = range(1, len(df) + 1)
        elif strategy_type == "fuzzy_match":
            # Implementazione semplificata: usa la chiave esistente ma puliscila
            logger.info(f"Using fuzzy match on key '{key_name}'")
            fallback = join_key_strategy.get("fallback", "_cxn_id")
            
            for df in dataframes:
                if key_name in df.columns:
                    # Pulisci la chiave esistente (converti a stringa, rimuovi spazi)
                    df[key_name] = df[key_name].astype(str).str.strip()
                else:
                    # Usa fallback se la chiave non esiste
                    logger.warning(f"Key '{key_name}' not found, using fallback '{fallback}'")
                    df[fallback] = range(1, len(df) + 1)
                    key_name = fallback
        
        # Unisci i dataframe
        base_df = dataframes[0]
        for df in dataframes[1:]:
            try:
                base_df = base_df.merge(df, on=key_name, how="outer", suffixes=("", "_dup"))
            except Exception as e:
                logger.error(f"Error merging dataframes: {e}")
                raise ValueError(f"Errore nella fusione dei dataframe: {e}")
                
        # Rimuovi colonne duplicate
        dupes = [c for c in base_df.columns if c.endswith("_dup")]
        if dupes:
            logger.warning(f"Removing {len(dupes)} duplicate columns")
            base_df.drop(columns=dupes, inplace=True)
            
        # Normalizza i tipi di dato
        for col in base_df.columns:
            if col == key_name:
                continue
                
            try:
                if "timestamp" in col.lower() or "date" in col.lower():
                    # Converti a datetime se possibile
                    base_df[col] = pd.to_datetime(base_df[col], errors="ignore")
                else:
                    # Prova a convertire a numerico
                    base_df[col] = pd.to_numeric(base_df[col], errors="ignore")
            except Exception as e:
                logger.warning(f"Failed to normalize column {col}: {e}")
                
        return base_df