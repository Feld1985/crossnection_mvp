"""Modulo di supporto per memorizzare e recuperare lo stato del flusso tra agenti."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Directory per salvare lo stato temporaneo
FLOW_STATE_DIR = Path("./flow_state")
FLOW_STATE_DIR.mkdir(exist_ok=True)

def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    """Salva un DataFrame in CSV per condivisione tra agenti."""
    file_path = FLOW_STATE_DIR / f"{name}.csv" 
    df.to_csv(file_path, index=False)
    return file_path

def load_dataframe(name: str) -> Optional[pd.DataFrame]:
    """Carica un DataFrame salvato precedentemente."""
    file_path = FLOW_STATE_DIR / f"{name}.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

def save_json(data: Dict[str, Any], name: str) -> Path:
    """Salva dati JSON per condivisione tra agenti."""
    file_path = FLOW_STATE_DIR / f"{name}.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

def load_json(name: str) -> Optional[Dict[str, Any]]:
    """Carica dati JSON salvati precedentemente."""
    file_path = FLOW_STATE_DIR / f"{name}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def get_unified_dataset() -> Optional[pd.DataFrame]:
    """Helper per ottenere il dataset unificato più recente."""
    # Cerca prima nella directory di stato del flusso
    df = load_dataframe("unified_dataset")
    if df is not None:
        return df
    
    # Fallback: cerca in examples/driver_csvs
    path = Path("examples/driver_csvs/unified_dataset.csv")
    if path.exists():
        return pd.read_csv(path)
    
    # Ultimo tentativo: crea un dataset dai file originali
    try:
        from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool
        tool = CrossDataProfilerTool()
        result = tool.run(csv_folder="examples/driver_csvs", kpi="value_speed", mode="full_pipeline")
        
        # Estrai il CSV dal risultato e convertilo in DataFrame
        if isinstance(result, dict) and "unified_dataset_csv" in result:
            from io import StringIO
            df = pd.read_csv(StringIO(result["unified_dataset_csv"]))
            save_dataframe(df, "unified_dataset")  # Salva per uso futuro
            return df
    except Exception as e:
        print(f"ERROR creating unified dataset: {e}")
    
    return None