"""Utility per il debug di Crossnection."""

import json
import logging
import pprint
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def log_structure(obj: Any, name: str = "object", level: str = "INFO") -> None:
    """Log la struttura di un oggetto in modo chiaro."""
    log_method = getattr(logger, level.lower(), logger.info)
    formatted = pprint.pformat(obj, indent=2, width=100)
    log_method(f"Structure of {name}:\n{formatted}")

def inspect_context_store(artifact_name: str, context_store=None) -> None:
    """Ispeziona e logga un artefatto dal Context Store."""
    if context_store is None:
        from crossnection_mvp.utils.context_store import ContextStore
        context_store = ContextStore.get_instance()
    
    try:
        if artifact_name.endswith(".csv") or "_dataframe" in artifact_name:
            # È probabilmente un DataFrame
            df = context_store.load_dataframe(artifact_name.split(".")[0])
            logger.info(f"DataFrame '{artifact_name}' shape: {df.shape}")
            logger.info(f"DataFrame '{artifact_name}' columns: {df.columns.tolist()}")
            logger.info(f"DataFrame '{artifact_name}' preview:\n{df.head(3)}")
        else:
            # È probabilmente JSON
            data = context_store.load_json(artifact_name.split(".")[0])
            log_structure(data, name=f"JSON '{artifact_name}'")
    except Exception as e:
        logger.error(f"Failed to inspect {artifact_name}: {e}")

def dump_context_state(output_file: str = "context_dump.json") -> None:
    """Dumps the current state of the Context Store to a file."""
    from crossnection_mvp.utils.context_store import ContextStore
    store = ContextStore.get_instance()
    
    state = {
        "session_id": store.session_id,
        "artifacts": {}
    }
    
    # Lista tutti gli artefatti
    for artifact_name in store.list_artifacts():
        try:
            # Prova a caricare come JSON
            state["artifacts"][artifact_name] = store.load_json(artifact_name)
        except:
            try:
                # Prova a caricare come DataFrame
                df = store.load_dataframe(artifact_name)
                state["artifacts"][artifact_name] = {
                    "_type": "dataframe",
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "preview": df.head(3).to_dict(orient="records")
                }
            except:
                state["artifacts"][artifact_name] = {
                    "_type": "unknown",
                    "error": "Failed to load artifact"
                }
    
    # Salva su file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Context Store state dumped to {output_file}")