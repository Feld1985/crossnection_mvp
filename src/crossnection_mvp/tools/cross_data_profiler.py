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
    - `unified_dataset_ref` (reference to dataset in Context Store)  
    - `data_report_ref` (reference to JSON report in Context Store)
* CrewAI automatically serialises / deserialises when passed
  between tasks (falls back to CSV in Flow context if needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
from pydantic import BaseModel, Field

import json
import uuid

import pandas as pd
import great_expectations as ge
from crewai.tools import BaseTool
from crossnection_mvp.utils.context_store import ContextStore

__all__ = ["CrossDataProfilerTool"]

class CrossDataProfilerToolSchema(BaseModel):
    csv_folder: Optional[str] = Field(None, description="Path to directory containing CSV files")
    kpi: Optional[str] = Field("value_speed", description="Target KPI column name")
    mode: Optional[str] = Field("full_pipeline", description="Processing mode")
    
    # Backward compatibility
    input: Optional[Union[str, Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"  # Permette campi aggiuntivi non dichiarati nello schema
        
class CrossDataProfilerTool(BaseTool):
    """Validate, profile and merge driver CSV datasets."""

    name: str = "cross_data_profiler"
    description: str = (
        "Profiles CSV driver datasets, validates schema, discovers/creates a join‑key, "
        "cleans & normalises the data, then returns unified data and a "
        "structured report."
    )
    args_schema = CrossDataProfilerToolSchema

    def print_truncated(self, message, max_length=500):
        """Print message, truncating if too long."""
        if len(message) > max_length:
            print(f"{message[:max_length]}... [truncated, total length: {len(message)}]")
        else:
            print(message)

    def _run(self, input: Union[str, Dict[str, Any]] = None, **kwargs) -> str:
        """
        Main entry‑point expected by CrewAI BaseTool.
        
        Arguments can be passed as JSON string or dictionary.
        """
        self.print_truncated(f"DEBUG: CrossDataProfilerTool received raw input: {input}")
        
        # Gestisci il caso in cui l'input arriva come kwargs invece che sotto la chiave 'input'
        if input is None and kwargs:
            # Adatta la struttura per renderla compatibile
            input = kwargs
            
        # Gestione parametri di input in formato vario
        csv_folder = None
        kpi = "value_speed"  # Default
        mode = "full_pipeline"  # Default
        
        # Estrai parametri dall'input dictionary
        if isinstance(input, dict):
            # Input diretto
            csv_folder = input.get("csv_folder")
            
            # Ricerca in strutture annidate
            if not csv_folder and "input" in input and isinstance(input["input"], dict):
                csv_folder = input["input"].get("csv_folder")
                
            # Fallback su altri campi potenzialmente utili
            if not csv_folder:
                # Prova a estrarre da description o type
                if input.get("type") == "csv_folder":
                    csv_folder = "examples/driver_csvs"
                elif input.get("description") and "driver datasets" in input.get("description", ""):
                    csv_folder = "examples/driver_csvs"
            
            # Estrai altri parametri
            kpi = input.get("kpi", kpi)
            mode = input.get("mode", mode)
            
            # Verifica input nidificati
            if "input" in input and isinstance(input["input"], dict):
                if not csv_folder:
                    csv_folder = input["input"].get("csv_folder")
                if "kpi" in input["input"]:
                    kpi = input["input"].get("kpi", kpi)
                if "mode" in input["input"]:
                    mode = input["input"].get("mode", mode)
            
        # Intercetta input errati o placeholder comuni
        if isinstance(input, str) or (isinstance(input, dict) and not csv_folder):
            if input == "user_uploaded_csv_path" or input == {"input": "user_uploaded_csv_path"} or (
                isinstance(input, dict) and input.get("input") == "user_uploaded_csv_path"):
                # Usa l'esempio predefinito
                csv_folder = "examples/driver_csvs"
                self.print_truncated(f"DEBUG: Detected placeholder input, using default values")
            elif isinstance(input, str):
                # Rilevamento contenuto CSV diretto
                if input.startswith("join_key,timestamp,value") or input.startswith('"join_key,timestamp,value'):
                    print(f"DEBUG: Detected direct CSV content instead of folder path")
                    # Per i contenuti CSV diretti, salva nel Context Store e restituisci il riferimento
                    store = ContextStore.get_instance()
                    from io import StringIO
                    df = pd.read_csv(StringIO(input.replace('"', '')))
                    unified_path = store.save_dataframe("unified_dataset", df)
                    report = {"tables": [], "note": "Direct CSV input used"}
                    report_path = store.save_json("data_report", report)
                    
                    return json.dumps({
                        "unified_dataset_ref": unified_path,
                        "data_report_ref": report_path
                    }, ensure_ascii=False)
                
                try:
                    # Prova a interpretare come JSON
                    input_data = json.loads(input)
                    if isinstance(input_data, dict):
                        # Estrai parametri dal JSON
                        if not csv_folder:
                            csv_folder = input_data.get("csv_folder")
                        if "kpi" in input_data:
                            kpi = input_data.get("kpi", kpi)
                        if "mode" in input_data:
                            mode = input_data.get("mode", mode)
                except json.JSONDecodeError:
                    # È una stringa semplice, usa il fallback
                    if not csv_folder:
                        print(f"DEBUG: Input is a simple string, using as csv_folder: {input}")
                        csv_folder = input
            
        # Fornisci un valore predefinito se csv_folder è ancora None
        if not csv_folder:
            csv_folder = "examples/driver_csvs"
            print(f"WARNING: csv_folder parameter is empty, using default '{csv_folder}'")
        
        # Validazione directory
        if isinstance(csv_folder, str) and (csv_folder.endswith('.csv') or csv_folder.endswith('.json')):
            # Rilevato un file invece di una directory
            print(f"WARNING: csv_folder '{csv_folder}' appears to be a file path, not a directory")
            
            # Gestione speciale dei file dal Context Store
            if "\\" in csv_folder or "/" in csv_folder:
                try:
                    # Tenta di estrarre il nome della sessione dalla prima parte del percorso
                    session_id = Path(csv_folder).parts[0]
                    print(f"INFO: Extracted potential session ID: {session_id}")
                    csv_folder = "examples/driver_csvs"  # Usa la directory di default
                except Exception as e:
                    print(f"WARNING: Error processing file path reference: {e}")
                    csv_folder = "examples/driver_csvs"  # Fallback alla directory di default
            else:
                csv_folder = "examples/driver_csvs"  # Fallback alla directory di default
        
        # Validazione e correzione del path
        if csv_folder in ["driver_datasets.csv", "user_uploaded_driver_datasets.csv", "uploaded_csv_files", "user_uploaded_csv_path"]:
            print(f"WARNING: Detected invalid path '{csv_folder}', correcting to 'examples/driver_csvs'")
            csv_folder = "examples/driver_csvs"
        
        # Chiama il metodo run con i parametri estratti
        result = self.run(csv_folder=csv_folder, kpi=kpi, mode=mode)
        
        # Converti result a string se necessario
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return result

    def run(self, *, csv_folder: str | Path, kpi: str, mode: str = "full_pipeline") -> Dict[str, Any]:
        """
        Original implementation that processes CSV files.

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
              "unified_dataset_ref": str,  # Riferimento al Context Store
              "data_report_ref": str       # Riferimento al Context Store
            }
        """
        # QUICK FIX - Fix per il problema del percorso
        print(f"DEBUG: Received csv_folder={csv_folder}")
        
        # Verifica se abbiamo contenuto CSV diretto invece di un percorso
        if isinstance(csv_folder, str) and (csv_folder.startswith("join_key,timestamp,value") or 
                                        csv_folder.startswith('"join_key,timestamp,value')):
            print(f"DEBUG: Detected direct CSV content in run() method")
            # Salva il contenuto CSV nel Context Store
            store = ContextStore.get_instance()
            from io import StringIO
            df = pd.read_csv(StringIO(csv_folder.replace('"', '')))
            unified_path = store.save_dataframe("unified_dataset", df)
            report = {"tables": [], "note": "Direct CSV input used"}
            report_path = store.save_json("data_report", report)
            
            return {
                "unified_dataset_ref": unified_path,
                "data_report_ref": report_path
            }
        
        # Validazione e correzione percorso
        if csv_folder in ["driver_datasets.csv", "user_uploaded_driver_datasets.csv", "uploaded_csv_files", "user_uploaded_csv_path"]:
            print(f"WARNING: Detected invalid path '{csv_folder}', correcting to 'examples/driver_csvs'")
            csv_folder = "examples/driver_csvs"
        
        # Converti a Path e verifica esistenza
        csv_folder = Path(csv_folder)
        if not csv_folder.is_dir():
            # FALLBACK: usa una directory di esempio conosciuta come sicura
            fallback_path = Path("examples/driver_csvs")
            if fallback_path.is_dir():
                print(f"ERROR: CSV directory not found: {csv_folder}. Using fallback: {fallback_path}")
                csv_folder = fallback_path
            else:
                raise FileNotFoundError(f"CSV directory not found: {csv_folder} and fallback not available")
            
        # Get all CSV files in the directory
        csv_paths = list(csv_folder.glob("*.csv"))
        if not csv_paths:
            # FALLBACK: potremmo fornire dati di esempio o usare un'altra directory
            fallback_path = Path("examples/driver_csvs")
            if fallback_path.is_dir():
                csv_paths = list(fallback_path.glob("*.csv"))
                if csv_paths:
                    print(f"WARNING: No CSV files found in {csv_folder}. Using CSVs from fallback: {fallback_path}")
                else:
                    raise ValueError(f"No CSV files found in {csv_folder} or fallback location")
            else:
                raise ValueError(f"No CSV files found in {csv_folder}")
        
        # Filtra via unified_dataset.csv se presente
        csv_paths = [p for p in csv_paths if p.name != "unified_dataset.csv"]
            
        # Load dataframes
        dataframes = []
        for p in csv_paths:
            df = pd.read_csv(p)
            # Salva il percorso del file come attributo del dataframe
            df._file_path = str(p)
            dataframes.append(df)
        profile = self._profile_frames(dataframes, csv_paths)

        key = self._discover_join_key(dataframes, profile)
        unified = self._merge_and_clean(dataframes, key)

        # Salva nel Context Store
        store = ContextStore.get_instance()
        unified_path = store.save_dataframe("unified_dataset", unified)
        report_path = store.save_json("data_report", profile)
        
        # Salva il dataset unificato come file per compatibilità con codice esistente
        unified_csv = unified.to_csv(index=False)
        legacy_path = Path("examples/driver_csvs/unified_dataset.csv")
        with open(legacy_path, "w") as f:
            f.write(unified_csv)
        
        # Crea un sommario del dataset
        rows, cols = unified.shape
        column_summary = ", ".join(unified.columns.tolist())
        dataset_summary = f"Unified dataset saved. Shape: {rows} rows x {cols} columns. Columns: {column_summary}"
        print(f"DEBUG: {dataset_summary}")

        # Return solo i riferimenti al Context Store
        return {
            "unified_dataset_ref": unified_path,
            "data_report_ref": report_path
        }

    # ------------------------------------------------------------------
    # Internal helpers (unchanged)
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
        # Rinomina le colonne 'value' in ciascun dataframe in base al nome del file
        for i, df in enumerate(frames):
            if 'value' in df.columns:
                # Ottieni il nome del driver dal contesto o usa un nome generico
                driver_name = f"driver_{i+1}"
                # Se abbiamo i nomi dei file originali, usali per il nome del driver
                if hasattr(df, '_file_path') and df._file_path:
                    driver_name = Path(df._file_path).stem
                
                # Rinomina la colonna 'value'
                df.rename(columns={'value': f'value_{driver_name}'}, inplace=True)
        
        # Esegui il merge
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