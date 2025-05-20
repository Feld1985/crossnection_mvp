# Salva questo come debug_inspect_data.py nella cartella principale del progetto
import pandas as pd
from pathlib import Path

def inspect_dataset(file_path=None):
    """Ispeziona un dataset CSV o cerca il dataset unificato standard."""
    # Se non fornito, cerca il dataset predefinito
    if file_path is None:
        # Cerca in diverse posizioni possibili
        possible_paths = [
            "examples/driver_csvs/unified_dataset.csv",
            "flow_context/latest/unified_dataset.v1.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                file_path = path
                break
    
    if file_path is None or not Path(file_path).exists():
        print(f"ERROR: Unified dataset not found.")
        return
    
    print(f"Inspecting dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data (3 rows):")
    print(df.head(3))
    print("\nColumn statistics:")
    for col in df.columns:
        print(f"  {col}: {len(df[col].dropna())} non-null values, dtype={df[col].dtype}")
    
    # Verifica colonna KPI
    if "value_speed" in df.columns:
        print("\nKPI 'value_speed' found!")
        print(f"value_speed stats: min={df['value_speed'].min()}, max={df['value_speed'].max()}, mean={df['value_speed'].mean()}")
    else:
        print("\nWARNING: KPI 'value_speed' NOT found!")
        # Cerca colonne simili
        similar_cols = [col for col in df.columns if "speed" in col.lower()]
        if similar_cols:
            print(f"Found similar columns: {similar_cols}")

if __name__ == "__main__":
    # Puoi specificare un percorso come argomento
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    inspect_dataset(file_path)