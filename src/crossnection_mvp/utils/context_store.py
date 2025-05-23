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
                # Prova a cercare nomi simili
                for pattern in [f"{name}.csv", f"{name}*.csv", f"*{name}*.csv"]:
                    matches = list(self.session_dir.glob(pattern))
                    if matches:
                        path = matches[0]
                        print(f"Found alternative file for '{name}': {path}")
                        break
                else:
                    # Cerca nel fallback
                    fallback = Path("examples/driver_csvs/unified_dataset.csv")
                    if fallback.exists() and name == "unified_dataset":
                        print(f"Using fallback for {name}: {fallback}")
                        return pd.read_csv(fallback)
                    raise ValueError(f"No versions found for DataFrame '{name}'")
            else:
                version = max(existing_versions)
                path = self.session_dir / f"{name}.v{version}.csv"
        
        try:
            df = pd.read_csv(path)
            
            # Verifica se è un DataFrame valido
            # Verifica che non sia un path invece del contenuto
            if df.shape[1] == 1:
                col_name = df.columns[0]
                first_value = df.iloc[0, 0] if len(df) > 0 else None
                
                # Se l'unica colonna è una stringa che sembra un percorso file
                if isinstance(first_value, str) and ('/' in first_value or '\\' in first_value or first_value.endswith('.csv')):
                    try:
                        # Prova a caricare il file effettivo
                        actual_path = Path(first_value)
                        if actual_path.exists():
                            print(f"WARNING: Loaded DataFrame contains file path. Loading actual file: {actual_path}")
                            return pd.read_csv(actual_path)
                        else:
                            print(f"WARNING: DataFrame points to file that doesn't exist: {actual_path}")
                            # Cerca il file standard come fallback
                            fallback = Path("examples/driver_csvs/unified_dataset.csv")
                            if fallback.exists():
                                print(f"Using fallback: {fallback}")
                                return pd.read_csv(fallback)
                    except Exception as e:
                        print(f"Error trying to load actual file: {e}")
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to load DataFrame '{name}': {e}")
    
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
    
    def extract_artifact_name(self, ref_path: str) -> str:
        """Estrae il nome base da un path di riferimento."""
        if not ref_path:
            return ""
            
        # Converti in Path per gestione più facile
        path = Path(ref_path)
        
        # Estrai il nome file senza estensione
        base_name = path.stem
        
        # Rimuovi la parte di versione se presente (es: "data_report.v1" → "data_report")
        if '.' in base_name:
            base_name = base_name.split('.')[0]
            
        return base_name
    
    def validate_json_structure(self, name: str, expected_keys: List[str]) -> bool:
        """Validate that a saved JSON has the expected structure."""
        try:
            data = self.load_json(name)
            return all(key in data for key in expected_keys)
        except Exception:
            return False

    def get_normalized_impact_ranking(self) -> Dict[str, Any]:
        """Get a normalized impact ranking with guaranteed structure."""
        try:
            data = self.load_json("impact_ranking")
            if not isinstance(data, dict):
                return {"kpi_name": "Default KPI", "ranking": []}
            if "ranking" not in data:
                data["ranking"] = []
            if "kpi_name" not in data:
                data["kpi_name"] = "Default KPI"
            return data
        except Exception:
            return {"kpi_name": "Default KPI", "ranking": []}
            
    def get_normalized_outlier_report(self) -> Dict[str, Any]:
        """Get a normalized outlier report with guaranteed structure."""
        try:
            data = self.load_json("outlier_report")
            if not isinstance(data, dict):
                return {"outliers": []}
                
            if "outliers" not in data:
                data["outliers"] = []
            return data
        except Exception:
            return {"outliers": []}
            
    def ensure_artifact_exists(self, name: str, default_value: Any) -> bool:
        """Ensure that an artifact exists, creating it if needed."""
        try:
            self.load_json(name)
            return True
        except Exception:
            if isinstance(default_value, pd.DataFrame):
                self.save_dataframe(name, default_value)
            else:
                self.save_json(name, default_value)
            return False