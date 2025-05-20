# Salva questo come debug_context_store.py
import json
import pandas as pd
from pathlib import Path

def inspect_context_store():
    """Ispeziona i contenuti del Context Store."""
    # Cerca in diverse possibili directory
    possible_dirs = [
        Path("flow_context"),
        Path("flow_data"),  # Controlla questa alternativa
        Path("flow_state"),  # E questa
        Path("./flow_context"),
        Path("./flow_data"),
        Path("./flow_state")
    ]
    
    base_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            base_dir = dir_path
            print(f"Found Context Store at: {base_dir}")
            break
    
    if base_dir is None:
        # Cerca ovunque nella directory corrente
        all_dirs = [d for d in Path(".").glob("**/20*Z") if d.is_dir()]
        if all_dirs:
            base_dir = all_dirs[0].parent
            print(f"Found possible Context Store at: {base_dir}")
        else:
            print("Context Store directory not found in any expected location")
            return
        
    # Trova la directory del Context Store
    base_dir = Path("flow_context")
    if not base_dir.exists():
        print(f"Context Store directory not found: {base_dir}")
        return
    
    # Trova l'ultima sessione
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, 
                      reverse=True)
    
    if not sessions:
        print(f"No sessions found in {base_dir}")
        return
    
    latest_session = sessions[0]
    print(f"Inspecting latest Context Store session: {latest_session}")
    
    # Esamina i file JSON
    json_files = list(latest_session.glob("*.json"))
    for json_file in json_files:
        print(f"\n--- {json_file.name} ---")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Stampa contenuti rilevanti in base al tipo di file
            if "impact_ranking" in json_file.name:
                print_impact_ranking(data)
            elif "outlier_report" in json_file.name:
                print_outlier_report(data)
            elif "metadata" in json_file.name:
                print_metadata(data)
            else:
                # Stampa solo le chiavi principali per file generici
                print(f"Keys: {list(data.keys())}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Esamina i file CSV
    csv_files = list(latest_session.glob("*.csv"))
    for csv_file in csv_files:
        print(f"\n--- {csv_file.name} ---")
        try:
            df = pd.read_csv(csv_file)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 2 rows:")
            print(df.head(2))
        except Exception as e:
            print(f"Error reading CSV file: {e}")

def print_impact_ranking(data):
    """Stampa informazioni dal impact_ranking in formato leggibile."""
    if isinstance(data, dict) and "ranking" in data:
        ranking = data["ranking"]
        print(f"Impact Ranking - {len(ranking)} items")
        for i, item in enumerate(ranking):
            driver = item.get("driver_name", item.get("driver", "Unknown"))
            r = item.get("r", item.get("correlation", "N/A"))
            p = item.get("p_value", "N/A")
            print(f"  {i+1}. {driver}: r={r}, p-value={p}")
    else:
        print(f"Unexpected impact_ranking format: {data}")

def print_outlier_report(data):
    """Stampa informazioni dal outlier_report in formato leggibile."""
    if isinstance(data, dict) and "outliers" in data:
        outliers = data["outliers"]
        print(f"Outliers - {len(outliers)} items")
        for i, item in enumerate(outliers[:5]):  # Mostra solo i primi 5
            print(f"  {i+1}. {item}")
        if len(outliers) > 5:
            print(f"  ...and {len(outliers) - 5} more")
    else:
        print(f"Unexpected outlier_report format: {data}")

def print_metadata(data):
    """Stampa informazioni dal metadata in formato leggibile."""
    if isinstance(data, dict):
        print(f"Session ID: {data.get('session_id', 'N/A')}")
        print(f"Created At: {data.get('created_at', 'N/A')}")
        artifacts = data.get("artifacts", {})
        print(f"Artifacts ({len(artifacts)} items):")
        for name, info in artifacts.items():
            art_type = info.get("type", "unknown")
            path = info.get("path", "N/A")
            print(f"  - {name}: type={art_type}, path={path}")
    else:
        print(f"Unexpected metadata format: {data}")

if __name__ == "__main__":
    inspect_context_store()