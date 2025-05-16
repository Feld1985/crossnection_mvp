"""crossnection_mvp/crew.py

Centralised builder for the Crossnection Root‑Cause Discovery MVP.
"""
from pathlib import Path
import yaml
from typing import Any, Dict, List

import crewai as cr

# ----------------------------------------------------------------------------
# Percorsi di configurazione
# ----------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent / "config"
AGENTS_FILE = CONFIG_DIR / "agents.yaml"
TASKS_FILE = CONFIG_DIR / "tasks.yaml"


class CrossnectionMvpCrew:
    """Factory + facade per la crew MVP."""

    def __init__(self) -> None:
        self._crew: cr.Crew | None = None
        self._agents: Dict[str, cr.Agent] = {}
        self._tasks: List[cr.Task] = []

    # ---------------------------------------------------------------------
    # Costruzione / caching
    # ---------------------------------------------------------------------

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Carica un file YAML e restituisce un dizionario."""
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _register_tools(self):
        """Registra i tool personalizzati."""
        from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        from crossnection_mvp.tools.cross_insight_formatter import CrossInsightFormatterTool
        
        # Crea istanze dei tool
        return {
            "cross_data_profiler": CrossDataProfilerTool(),
            "cross_stat_engine": CrossStatEngineTool(),
            "cross_insight_formatter": CrossInsightFormatterTool()
        }

    def _build_agents(self) -> Dict[str, cr.Agent]:
        """Costruisce gli oggetti Agent dalle configurazioni YAML."""
        if self._agents:
            return self._agents
            
        agents_config = self._load_yaml(AGENTS_FILE)
        tools = self._register_tools()
        
        for name, config in agents_config.items():
            # Crea l'agente con i parametri della configurazione
            agent_tools = [tools[tool_name] for tool_name in config.get("tools", [])]
            
            # Crea l'agente usando i parametri disponibili nella tua versione
            # Potresti dover adattare questi parametri in base alla versione di CrewAI
            agent = cr.Agent(
                name=name,
                role=config.get("role", ""),
                goal=config.get("goal", ""),
                backstory=config.get("description", ""),  # Usa description come backstory se disponibile
                verbose=True,
                allow_delegation=True,
                tools=agent_tools,
                llm=config.get("llm", {}),
            )
            self._agents[name] = agent
            
        return self._agents

    def _build_tasks(self) -> List[cr.Task]:
        """Costruisce gli oggetti Task dalle configurazioni YAML."""
        if self._tasks:
            return self._tasks
            
        tasks_config = self._load_yaml(TASKS_FILE)
        agents = self._build_agents()
        
        tasks = []
        for name, config in tasks_config.items():
            agent_name = config.get("agent", "")
            agent = agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found for task '{name}'")
            
            # Crea l'oggetto Task con i parametri disponibili nella tua versione
            task = cr.Task(
                description=config.get("description", ""),
                agent=agent,
                expected_output=config.get("expected_output", ""),
                human_input=config.get("human_input", False),
            )
            tasks.append(task)
        
        self._tasks = tasks
        return tasks

    def crew(self) -> cr.Crew:
        """Restituisce un'istanza di `cr.Crew`, creandola se necessario."""
        if self._crew is None:
            # Costruisci gli oggetti
            tasks = self._build_tasks()
            
            # Crea l'equipaggio - questo dovrebbe funzionare con la maggior parte delle versioni di CrewAI
            self._crew = cr.Crew(
                tasks=tasks,
                verbose=True,
                # Potrebbe essere necessario adattare questi parametri
                process=cr.Process.sequential  # Usa cr.Process.sequential se disponibile
            )
        return self._crew

    # ---------------------------------------------------------------------
    # Operazioni di alto livello
    # ---------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Esegue la crew sul dataset/input fornito dall'utente."""
        # Usa il metodo corretto per eseguire la crew
        try:
            # Prova prima kickoff (versioni più recenti)
            return self.crew().kickoff(inputs=inputs)
        except AttributeError:
            # Fallback a run per versioni precedenti
            return self.crew().run(inputs=inputs)

    def train(self) -> None:
        """Allena prompt, memory o agenti come previsto da CrewAI."""
        try:
            self.crew().train()
        except AttributeError:
            print("Il metodo 'train' non è disponibile nella versione di CrewAI installata.")

    def test(self) -> None:
        """Esegue test automatici definiti nel flusso o in tasks di test."""
        try:
            self.crew().test()
        except AttributeError:
            print("Il metodo 'test' non è disponibile nella versione di CrewAI installata.")

    def replay(self, session_id: str = None) -> Any:
        """Riesegue una sessione salvata per debug o regression."""
        try:
            return self.crew().replay(session_id)
        except AttributeError:
            print("Il metodo 'replay' non è disponibile nella versione di CrewAI installata.")
            return None


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
        # Mini‑stub d'esempio
        user_inputs = {
            "kpi": "First Pass Yield",
            "process_map": "./examples/process_map.json",
            "drivers_dir": "./examples/driver_csvs",
        }

    output = CrossnectionMvpCrew().run(inputs=user_inputs)
    print("=== ROOT‑CAUSE REPORT (truncated) ===")
    print(str(output)[:1000], "…")