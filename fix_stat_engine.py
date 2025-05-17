# fix_stat_engine.py
from pathlib import Path

# Percorso al file da modificare
file_path = Path("src/crossnection_mvp/tools/cross_stat_engine.py")

# Leggi il contenuto del file
content = file_path.read_text(encoding="utf-8")

# Aggiungi l'import necessario
if "from pathlib import Path" not in content:
    # Trova il primo blocco di import
    import_block_end = content.find("\n\n", content.find("import "))
    if import_block_end != -1:
        content = content[:import_block_end] + "\nfrom pathlib import Path" + content[import_block_end:]
    else:
        # Fallback: aggiungi semplicemente all'inizio
        content = "from pathlib import Path\n" + content

# Scrivi il file aggiornato
file_path.write_text(content, encoding="utf-8")

print(f"File {file_path} aggiornato con successo!")