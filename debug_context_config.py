# Salva come debug_context_config.py
import importlib
import inspect
from pathlib import Path

def find_context_store_usage():
    """Cerca e mostra come viene configurato il Context Store."""
    # Cerca file che potrebbero contenere la configurazione
    config_files = []
    for path in Path('src/crossnection_mvp').rglob('*.py'):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            if 'ContextStore' in content and ('__init__' in content or 'get_instance' in content):
                config_files.append(path)
    
    print(f"Found {len(config_files)} files with Context Store configuration:")
    for file_path in config_files:
        print(f"\n--- {file_path} ---")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
            
        # Cerca righe rilevanti
        for i, line in enumerate(content):
            if 'ContextStore' in line and ('__init__' in line or 'get_instance' in line or 'base_dir' in line):
                # Mostra alcune righe di contesto
                start = max(0, i-2)
                end = min(len(content), i+3)
                for j in range(start, end):
                    print(f"{j+1:4d}: {content[j].rstrip()}")

if __name__ == "__main__":
    find_context_store_usage()