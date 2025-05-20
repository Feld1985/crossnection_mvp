# Salva come add_debug_logging.py
import re
from pathlib import Path

def add_debugging_to_cross_stat_engine():
    """Aggiunge debugging temporaneo al cross_stat_engine.py"""
    file_path = Path('src/crossnection_mvp/tools/cross_stat_engine.py')
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return
    
    # Leggi il file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup del file originale
    backup_path = file_path.with_suffix('.py.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup saved to {backup_path}")
    
    # Aggiungi logging prima del controllo KPI
    modified_content = re.sub(
        r'if unified_dataset is not None and kpi not in unified_dataset\.columns:',
        r'''if unified_dataset is not None and kpi not in unified_dataset.columns:
            print(f"\\n----------------------------------------------------------------")
            print(f"DEBUG KPI CHECK: KPI='{kpi}'")
            print(f"DEBUG KPI CHECK: Columns={unified_dataset.columns.tolist()}")
            print(f"DEBUG KPI CHECK: Column types={unified_dataset.dtypes}")
            print(f"DEBUG KPI CHECK: Is KPI in columns? {kpi in unified_dataset.columns}")
            print(f"DEBUG KPI CHECK: Lowercase checks: {[c.lower() for c in unified_dataset.columns]}")
            print(f"DEBUG KPI CHECK: Looking for exact match with '{kpi}'")
            print(f"----------------------------------------------------------------\\n")''',
        content
    )
    
    # Aggiungi una normalizzazione dei nomi delle colonne prima del controllo KPI
    modified_content = re.sub(
        r'kpi_ok = True\n(\s+)if unified_dataset is not None',
        r'''kpi_ok = True
\1# Normalizza i nomi delle colonne per evitare problemi di spazi
\1if unified_dataset is not None:
\1    try:
\1        # Converti tutti i nomi di colonna a stringhe e rimuovi spazi
\1        unified_dataset.columns = [str(col).strip() for col in unified_dataset.columns]
\1        print(f"DEBUG: Normalized column names: {unified_dataset.columns.tolist()}")
\1    except Exception as e:
\1        print(f"ERROR normalizing columns: {e}")
\1
\1if unified_dataset is not None''',
        modified_content
    )
    
    # Aggiungi logging al metodo di correlazione
    modified_content = re.sub(
        r'def correlation_matrix\(df: pd\.DataFrame, \*, kpi: str\) -> pd\.DataFrame:',
        r'''def correlation_matrix(df: pd.DataFrame, *, kpi: str) -> pd.DataFrame:
    """Compute r & p per ogni colonna numerica vs kpi."""
    print(f"\\n----------------------------------------------------------------")
    print(f"DEBUG CORRELATION: Called with kpi='{kpi}'")
    print(f"DEBUG CORRELATION: df.shape={df.shape}")
    print(f"DEBUG CORRELATION: df.columns={df.columns.tolist()}")
    print(f"DEBUG CORRELATION: df.dtypes=\\n{df.dtypes}")
    
    # Normalizza i nomi delle colonne anche qui
    try:
        # Converti tutti i nomi di colonna a stringhe e rimuovi spazi
        df.columns = [str(col).strip() for col in df.columns]
        print(f"DEBUG CORRELATION: Normalized columns={df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR normalizing correlation columns: {e}")
        
    print(f"DEBUG CORRELATION: Is KPI in normalized columns? {kpi in df.columns}")
    print(f"----------------------------------------------------------------\\n")''',
        modified_content
    )
    
    # Scrivi il file modificato
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Added debugging to {file_path}")

def add_debugging_to_stats_agent():
    """Aggiunge debugging temporaneo allo stats_agent.py"""
    file_path = Path('src/crossnection_mvp/agents/stats_agent.py')
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return
    
    # Leggi il file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup del file originale
    backup_path = file_path.with_suffix('.py.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup saved to {backup_path}")
    
    # Aggiungi logging al metodo run_stats_pipeline
    modified_content = re.sub(
        r'def run_stats_pipeline\(',
        r'''def run_stats_pipeline(''',
        content
    )
    
    # Aggiungi logging dopo il caricamento del dataset
    modified_content = re.sub(
        r'if kpi not in df\.columns:',
        r'''# Aggiungi debug per verificare il dataframe
        print(f"\\n----------------------------------------------------------------")
        print(f"DEBUG STATS_AGENT: Dataset loaded, shape={df.shape}")
        print(f"DEBUG STATS_AGENT: Columns={df.columns.tolist()}")
        print(f"DEBUG STATS_AGENT: Looking for KPI='{kpi}'")
        print(f"DEBUG STATS_AGENT: KPI in columns? {kpi in df.columns}")
        print(f"----------------------------------------------------------------\\n")
        
        if kpi not in df.columns:''',
        modified_content
    )
    
    # Scrivi il file modificato
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Added debugging to {file_path}")

if __name__ == "__main__":
    add_debugging_to_cross_stat_engine()
    add_debugging_to_stats_agent()
    print("Debug logging added! Run your application and check the output.")
    print("Remember to restore the .py.bak files when you're done debugging.")