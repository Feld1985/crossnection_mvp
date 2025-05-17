"""
Logger personalizzato per tracciare le chiamate all'API OpenAI.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class OpenAILogger:
    """
    Logger per le chiamate all'API OpenAI.
    """
    
    def __init__(self, log_dir: str = "openai_logs"):
        """
        Inizializza il logger.
        
        Parameters
        ----------
        log_dir : str
            Directory dove salvare i log.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        self.log_file = self.log_dir / f"openai-log-{datetime.now().strftime('%Y-%m-%d')}.json"
        self.entries = []
        
        print(f"[INFO] OpenAI logger initialized. Logs will be saved to: {self.log_file}")
        
        # Carica le entries esistenti se il file esiste
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not parse existing log file: {self.log_file}")
                self.entries = []
    
    def log_api_call(self, model: str, prompt_tokens: int, completion_tokens: int, 
                     total_tokens: int, agent_name: Optional[str] = None):
        """
        Registra una chiamata all'API OpenAI.
        
        Parameters
        ----------
        model : str
            Nome del modello utilizzato.
        prompt_tokens : int
            Numero di token di input.
        completion_tokens : int
            Numero di token di output.
        total_tokens : int
            Numero totale di token.
        agent_name : Optional[str]
            Nome dell'agente che ha fatto la chiamata.
        """
        # Calcola il costo approssimativo (aggiorna con i prezzi corretti)
        cost = 0
        if model == "gpt-4o-mini":
            cost = (prompt_tokens / 1000 * 0.00015) + (completion_tokens / 1000 * 0.0002)
        elif model == "gpt-4o":
            cost = (prompt_tokens / 1000 * 0.0005) + (completion_tokens / 1000 * 0.0015)
        else:
            cost = (prompt_tokens / 1000 * 0.0001) + (completion_tokens / 1000 * 0.0002)
        
        # Crea l'entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "agent": agent_name
        }
        
        # Aggiungi l'entry
        self.entries.append(entry)
        
        # Scrivi il file di log
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to write log file: {e}")
    
    def print_summary(self):
        """
        Stampa un riepilogo dell'utilizzo.
        """
        if not self.entries:
            print("No API calls recorded.")
            return
        
        # Raggruppa per modello e agente
        by_model = {}
        by_agent = {}
        
        total_tokens = 0
        total_cost = 0
        
        for entry in self.entries:
            model = entry.get("model", "unknown")
            agent = entry.get("agent", "unknown")
            tokens = entry.get("total_tokens", 0)
            cost = entry.get("cost", 0)
            
            # Aggiorna totali
            total_tokens += tokens
            total_cost += cost
            
            # Aggiorna per modello
            if model not in by_model:
                by_model[model] = {"calls": 0, "tokens": 0, "cost": 0}
            by_model[model]["calls"] += 1
            by_model[model]["tokens"] += tokens
            by_model[model]["cost"] += cost
            
            # Aggiorna per agente
            if agent not in by_agent:
                by_agent[agent] = {"calls": 0, "tokens": 0, "cost": 0}
            by_agent[agent]["calls"] += 1
            by_agent[agent]["tokens"] += tokens
            by_agent[agent]["cost"] += cost
        
        # Stampa il riepilogo
        print("\n" + "=" * 50)
        print("OPENAI API USAGE SUMMARY")
        print("=" * 50)
        
        print(f"\nTotal API calls: {len(self.entries)}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.5f}")
        
        print("\nBy model:")
        for model, stats in by_model.items():
            print(f"  {model}:")
            print(f"    - Calls: {stats['calls']}")
            print(f"    - Tokens: {stats['tokens']:,}")
            print(f"    - Cost: ${stats['cost']:.5f}")
        
        print("\nBy agent:")
        for agent, stats in by_agent.items():
            print(f"  {agent}:")
            print(f"    - Calls: {stats['calls']}")
            print(f"    - Tokens: {stats['tokens']:,}")
            print(f"    - Cost: ${stats['cost']:.5f}")
        
        print("=" * 50 + "\n")


# Singleton pattern
_logger = None

def get_logger():
    """
    Restituisce l'istanza singleton del logger.
    """
    global _logger
    if _logger is None:
        _logger = OpenAILogger()
    return _logger