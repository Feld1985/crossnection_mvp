# src/crossnection_mvp/utils/context_store.py
"""Utility per il Context Store centralizzato."""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

class ContextStore:
    """Gestore centralizzato per i dati intermedi tra task e agenti."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None):
        """Ottiene l'istanza singleton del Context Store."""
        if cls._instance is None:
            cls._instance = ContextStore(base_dir=base_dir)
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None):
        """Inizializza il Context Store."""
        self.base_dir = Path(base_dir or "flow_context")
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.session_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.metadata = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "artifacts": {}
        }
        self._save_metadata()
        
        print(f"Context Store initialized: session_id={self.session_id}, base_dir={self.base_dir}")
    
    def _save_metadata(self):
        """Salva i metadati in un file JSON."""
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _register_artifact(self, name: str, artifact_type: str, path: Path, **metadata):
        """Registra un artefatto nei metadati."""
        self.metadata["artifacts"][name] = {
            "type": artifact_type,
            "path": str(path.relative_to(self.base_dir)),
            "created_at": datetime.now().isoformat(),
            **metadata
        }
        self._save_metadata()
    
    def save_dataframe(self, name: str, df: pd.DataFrame, version: Optional[int] = None) -> str:
        """Salva un DataFrame nel Context Store."""
        # Gestione versioni
        if version is None:
            # Trova l'ultima versione
            existing_versions = [
                int(p.stem.split('.')[-1][1:])
                for p in self.session_dir.glob(f"{name}.v*.csv")
                if p.stem.split('.')[-1].startswith('v') and p.stem.split('.')[-1][1:].isdigit()
            ]
            version = 1 if not existing_versions else max(existing_versions) + 1
        
        filename = f"{name}.v{version}.csv"
        path = self.session_dir / filename
        
        # Salva DataFrame
        df.to_csv(path, index=False)
        
        # Registra nei metadati
        self._register_artifact(
            name, 
            "dataframe", 
            path, 
            version=version,
            shape=df.shape,
            columns=df.columns.tolist()
        )
        
        return str(path.relative_to(self.base_dir))
    
    def load_dataframe(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        """Carica un DataFrame dal Context Store."""
        # Implementazione del caricamento con supporto versioni
        if version is not None:
            path = self.session_dir / f"{name}.v{version}.csv"
            if not path.exists():
                raise ValueError(f"Version {version} of DataFrame '{name}' not found")
        else:
            # Trova l'ultima versione
            existing_versions = [
                int(p.stem.split('.')[-1][1:])
                for p in self.session_dir.glob(f"{name}.v*.csv")
                if p.stem.split('.')[-1].startswith('v') and p.stem.split('.')[-1][1:].isdigit()
            ]
            if not existing_versions:
                raise ValueError(f"No versions found for DataFrame '{name}'")
            version = max(existing_versions)
            path = self.session_dir / f"{name}.v{version}.csv"
        
        return pd.read_csv(path)
    
    def save_json(self, name: str, data: Dict[str, Any], version: Optional[int] = None) -> str:
        """Salva dati JSON nel Context Store."""
        # Simile a save_dataframe ma per JSON
        if version is None:
            existing_versions = [
                int(p.stem.split('.')[-1][1:])
                for p in self.session_dir.glob(f"{name}.v*.json")
                if p.stem.split('.')[-1].startswith('v') and p.stem.split('.')[-1][1:].isdigit()
            ]
            version = 1 if not existing_versions else max(existing_versions) + 1
        
        filename = f"{name}.v{version}.json"
        path = self.session_dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self._register_artifact(
            name, 
            "json", 
            path, 
            version=version
        )
        
        return str(path.relative_to(self.base_dir))
    
    def load_json(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Carica dati JSON dal Context Store."""
        # Simile a load_dataframe ma per JSON
        if version is not None:
            path = self.session_dir / f"{name}.v{version}.json"
            if not path.exists():
                raise ValueError(f"Version {version} of JSON '{name}' not found")
        else:
            # Trova l'ultima versione
            existing_versions = [
                int(p.stem.split('.')[-1][1:])
                for p in self.session_dir.glob(f"{name}.v*.json")
                if p.stem.split('.')[-1].startswith('v') and p.stem.split('.')[-1][1:].isdigit()
            ]
            if not existing_versions:
                raise ValueError(f"No versions found for JSON '{name}'")
            version = max(existing_versions)
            path = self.session_dir / f"{name}.v{version}.json"
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[str]:
        """Elenca tutti gli artefatti di un determinato tipo."""
        if artifact_type is None:
            return list(self.metadata["artifacts"].keys())
        return [
            name for name, info in self.metadata["artifacts"].items()
            if info["type"] == artifact_type
        ]