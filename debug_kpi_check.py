# Salva come debug_kpi_check.py
from pathlib import Path

def find_kpi_checks():
    """Trova dove viene verificata la presenza del KPI nel codice."""
    # Cerca file che potrebbero contenere la verifica del KPI
    kpi_files = []
    for path in Path('src/crossnection_mvp').rglob('*.py'):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            if 'kpi' in content.lower() and ('not found' in content.lower() or 'not in' in content.lower()):
                kpi_files.append(path)
    
    print(f"Found {len(kpi_files)} files with KPI validation:")
    for file_path in kpi_files:
        print(f"\n--- {file_path} ---")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
            
        # Cerca righe rilevanti
        for i, line in enumerate(content):
            if 'kpi' in line.lower() and ('not found' in line.lower() or 'not in' in line.lower() or 'in df.columns' in line.lower()):
                # Mostra alcune righe di contesto
                start = max(0, i-5)
                end = min(len(content), i+5)
                for j in range(start, end):
                    print(f"{j+1:4d}: {content[j].rstrip()}")

if __name__ == "__main__":
    find_kpi_checks()