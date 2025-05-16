"""crossnection_mvp/crew.py

Centralised builder for the Crossnection Root‑Cause Discovery MVP.

• Carica agent e task YAML da `src/crossnection/config/`.
• Costruisce la crew (Flow + agenti) una sola volta e la riutilizza.
• Espone metodi helper per `run`, `train`, `test`, `replay` richiesti
  dall’interfaccia CLI (vedi `main.py`) o da notebook / test.

In questo modo separiamo la logica di orchestrazione applicativa dalla
semplice interfaccia a linea di comando.
"""
from pathlib import Path
from typing import Any, Dict

import crewai as cr

# ----------------------------------------------------------------------------
# Percorsi di configurazione
# ----------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent / "config"
AGENTS_FILE = CONFIG_DIR / "agents.yaml"
TASKS_FILE = CONFIG_DIR / "tasks.yaml"
# Updated fully-qualified Flow reference
FLOW_NAME = "crossnection_mvp.flows.root_cause_flow:root_cause_flow"


class CrossnectionMvpCrew:
    """Factory + facade per la crew MVP."""

    def __init__(self) -> None:
        self._crew: cr.Crew | None = None

    # ---------------------------------------------------------------------
    # Costruzione / caching
    # ---------------------------------------------------------------------

    def crew(self) -> cr.Crew:
        """Restituisce un'istanza di `cr.Crew`, creandola se necessario."""
        if self._crew is None:
            self._crew = cr.build_crew_from_yaml(
                agents_yaml=AGENTS_FILE,
                tasks_yaml=TASKS_FILE,
                flow=FLOW_NAME,
            )
        return self._crew

    # ---------------------------------------------------------------------
    # Operazioni di alto livello
    # ---------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Esegue la crew sul dataset/input fornito dall'utente."""
        return self.crew().run(inputs=inputs)

    def train(self) -> None:
        """Allena prompt, memory o agenti come previsto da CrewAI."""
        self.crew().train()

    def test(self) -> None:
        """Esegue test automatici definiti nel flusso o in tasks di test."""
        self.crew().test()

    def replay(self, session_id: str) -> Any:
        """Riesegue una sessione salvata per debug o regression."""
        return self.crew().replay(session_id)


# ---------------------------------------------------------------------------
# Debug rapido da riga di comando
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    print("[DEBUG] Avvio rapido CrossnectionMvpCrew…")

    # Accetta path ad un JSON con gli input oppure un dizionario inline
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            user_inputs = json.load(fh)
    else:
        # Mini‑stub d’esempio
        user_inputs = {
            "kpi": "First Pass Yield",
            "process_map": "./examples/process_map.json",
            "drivers_dir": "./examples/driver_csvs",
        }

    output = CrossnectionMvpCrew().run(inputs=user_inputs)
    print("=== ROOT‑CAUSE REPORT (truncated) ===")
    print(str(output)[:1000], "…")
