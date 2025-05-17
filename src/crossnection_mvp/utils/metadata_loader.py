"""Utility per caricare e gestire i metadati dei driver."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Directory di default per i metadati
DEFAULT_METADATA_PATH = Path("examples/drivers_metadata.json")

def load_driver_metadata(
    metadata_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Carica i metadati dei driver dal file JSON specificato.
    
    Parameters
    ----------
    metadata_path : Path, optional
        Percorso al file JSON dei metadati. Se None, usa il percorso di default.
        
    Returns
    -------
    Dict[str, Any]
        Dizionario con i metadati dei driver.
    """
    path = metadata_path or DEFAULT_METADATA_PATH
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        print(f"WARNING: File metadati non trovato: {path}")
        # Restituisci metadati vuoti come fallback
        return {"drivers": {}}
    except json.JSONDecodeError:
        print(f"ERROR: File metadati non valido: {path}")
        return {"drivers": {}}

def get_driver_metadata(
    driver_name: str,
    metadata_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ottiene i metadati per un driver specifico.
    
    Parameters
    ----------
    driver_name : str
        Nome del driver (es. "speed", "temperature").
    metadata_path : Path, optional
        Percorso al file JSON dei metadati.
        
    Returns
    -------
    Dict[str, Any]
        Metadati del driver o dizionario vuoto se non trovato.
    """
    all_metadata = load_driver_metadata(metadata_path)
    return all_metadata.get("drivers", {}).get(driver_name, {})

def enrich_driver_names(
    driver_names: list[str],
    metadata_path: Optional[Path] = None
) -> Dict[str, str]:
    """
    Arricchisce i nomi dei driver con le loro descrizioni.
    
    Parameters
    ----------
    driver_names : list[str]
        Lista di nomi di driver.
    metadata_path : Path, optional
        Percorso al file JSON dei metadati.
        
    Returns
    -------
    Dict[str, str]
        Dizionario {nome_driver: descrizione}.
    """
    all_metadata = load_driver_metadata(metadata_path)
    drivers_metadata = all_metadata.get("drivers", {})
    
    enriched = {}
    for name in driver_names:
        # Estrai nome base (es. da "value_speed" a "speed")
        base_name = name
        if name.startswith("value_"):
            base_name = name[6:]  # Rimuovi "value_"
            
        metadata = drivers_metadata.get(base_name, {})
        description = metadata.get("description", f"Driver {base_name}")
        unit = metadata.get("unit", "")
        
        if unit:
            enriched[name] = f"{description} ({unit})"
        else:
            enriched[name] = description
            
    return enriched