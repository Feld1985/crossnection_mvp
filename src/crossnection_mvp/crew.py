"""crossnection_mvp/crew.py

Centralised builder for the Crossnection Root‑Cause Discovery MVP.
"""
from pathlib import Path
import yaml
from typing import Any, Dict, List

import crewai as cr
from crewai.tools import BaseTool
from crossnection_mvp.utils.token_counter import TokenCounterLLM
from crossnection_mvp.utils.context_store import ContextStore

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
        self._store = ContextStore.get_instance(base_dir="./flow_data")

    # ---------------------------------------------------------------------
    # Costruzione / caching
    # ---------------------------------------------------------------------

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Carica un file YAML e restituisce un dizionario."""
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _create_tools(self) -> Dict[str, BaseTool]:
        """Crea le istanze dei tool e le restituisce."""
        from crossnection_mvp.tools.cross_data_profiler import CrossDataProfilerTool
        from crossnection_mvp.tools.cross_stat_engine import CrossStatEngineTool
        from crossnection_mvp.tools.cross_insight_formatter import CrossInsightFormatterTool
        
        # Crea e restituisci le istanze degli strumenti
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
        tools_instances = self._create_tools()
        
        for name, config in agents_config.items():
            # Prepara i tool richiesti dall'agente
            agent_tools = []
            if "tools" in config:
                for tool_name in config["tools"]:
                    if tool_name in tools_instances:
                        tool_instance = tools_instances[tool_name]
                        # Verifica che sia un'istanza di BaseTool
                        if not isinstance(tool_instance, BaseTool):
                            raise TypeError(f"Tool '{tool_name}' is not a valid BaseTool instance")
                        agent_tools.append(tool_instance)
            
            # Crea l'agente con i parametri disponibili
            agent_params = {
                "name": name,
                "role": config.get("role", ""),
                "goal": config.get("goal", ""),
                "backstory": config.get("description", ""),
                "verbose": True,
                "tools": agent_tools,  # Passa le istanze direttamente
            }
            
            # Aggiungi LLM se presente nella configurazione
            if "llm" in config:
                llm_config = config["llm"]
                model = llm_config.get("model", "gpt-4o-mini")
                temperature = llm_config.get("temperature", 0.0)
                
                try:
                    # Utilizza LangChain per creare un LLM compatibile con CrewAI
                    from langchain_openai import ChatOpenAI
                    
                    # Crea l'istanza di ChatOpenAI
                    llm = ChatOpenAI(model=model, temperature=temperature)
                    
                    # Avvolgi con il TokenCounterLLM
                    wrapped_llm = TokenCounterLLM(llm, agent_name=name)
                    
                    # Imposta l'LLM nell'agente
                    agent_params["llm"] = wrapped_llm
                    
                    print(f"[INFO] Agent {name} using token-counted {model}, temp={temperature}")
                except ImportError as e:
                    print(f"[WARNING] LangChain not available ({e}), token counting disabled")
                    # Fallback al metodo standard
                    agent_params["llm_config"] = {
                        "model": model,
                        "temperature": temperature
                    }
                except Exception as e:
                    print(f"[WARNING] Failed to create token-counted LLM: {e}")
                    # Fallback al metodo standard
                    agent_params["llm_config"] = {
                        "model": model,
                        "temperature": temperature
                    }
            
            agent = cr.Agent(**agent_params)
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
            
            # Costruisci i parametri del task
            task_params = {
                "description": config.get("description", ""),
                "agent": agent,
                "human_input": config.get("human_input", False),
            }
            
            # Aggiungi expected_output se presente
            if "expected_output" in config:
                task_params["expected_output"] = config["expected_output"]
                
            task = cr.Task(**task_params)
            tasks.append(task)
        
        self._tasks = tasks
        return tasks

    def crew(self) -> cr.Crew:
        """Restituisce un'istanza di `cr.Crew`, creandola se necessario."""
        if self._crew is None:
            # Costruisci gli oggetti
            tasks = self._build_tasks()
            
            # Crea l'equipaggio
            crew_params = {
                "tasks": tasks,
                "verbose": True,
            }
            
            # Aggiungi process se disponibile
            try:
                crew_params["process"] = cr.Process.sequential
            except AttributeError:
                # Se Process.sequential non è disponibile, prova con una stringa
                crew_params["process"] = "sequential"
            
            self._crew = cr.Crew(**crew_params)
        return self._crew

    # ---------------------------------------------------------------------
    # Operazioni di alto livello
    # ---------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Esegue la crew sul dataset/input fornito dall'utente."""
        # Aggiunge la session_id agli input
        inputs["context_session_id"] = self._store.session_id
        
        # Debugging e validazione input
        print(f"Input parameters received: {inputs}")
        
        # Validazione percorso CSV
        if "csv_folder" in inputs:
            csv_path = Path(inputs["csv_folder"])
            if not csv_path.is_absolute():
                csv_path = Path.cwd() / csv_path
            
            if csv_path.is_dir():
                print(f"Validated csv_folder exists: {inputs['csv_folder']}")
            else:
                print(f"WARNING: csv_folder '{inputs['csv_folder']}' is not a valid directory")
                # Cerca di normalizzare il percorso
                if Path("examples/driver_csvs").is_dir():
                    print(f"Using fallback examples/driver_csvs as csv_folder")
                    inputs["csv_folder"] = str(Path("examples/driver_csvs").absolute())
        
        # Validazione process_map
        if "process_map_file" in inputs:
            map_path = Path(inputs["process_map_file"])
            if not map_path.is_absolute():
                map_path = Path.cwd() / map_path
            
            if not map_path.exists():
                print(f"WARNING: process_map_file '{inputs['process_map_file']}' not found")
        
        # Usa il metodo corretto per eseguire la crew
        try:
            # Prova prima kickoff (versioni più recenti)
            result = self.crew().kickoff(inputs=inputs)
            
            # Stampa il riepilogo dei token utilizzati
            try:
                TokenCounterLLM.print_usage_summary()
            except Exception as e:
                print(f"[WARNING] Failed to print token usage summary: {e}")
            
            # Stampa il riepilogo dell'utilizzo OpenAI
            try:
                from crossnection_mvp.utils.openai_logger import get_logger
                logger = get_logger()
                logger.print_summary()
            except Exception as e:
                print(f"[WARNING] Failed to print OpenAI usage summary: {e}")
            
            return result
        except AttributeError:
            # Fallback a run per versioni precedenti
            result = self.crew().run(inputs=inputs)
            
            # Stampa il riepilogo dei token utilizzati
            try:
                TokenCounterLLM.print_usage_summary()
            except Exception as e:
                print(f"[WARNING] Failed to print token usage summary: {e}")
            
            # Stampa il riepilogo dell'utilizzo OpenAI
            try:
                from crossnection_mvp.utils.openai_logger import get_logger
                logger = get_logger()
                logger.print_summary()
            except Exception as e:
                print(f"[WARNING] Failed to print OpenAI usage summary: {e}")
            
            return result

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