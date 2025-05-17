"""
Utility per contare e monitorare l'utilizzo di token nelle chiamate LLM.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class TokenCounterLLM:
    """
    Wrapper per LLM che conta i token utilizzati e registra le statistiche.
    Compatibile con LangChain e altri provider LLM.
    """
    
    # Registro globale di tutte le istanze per le statistiche aggregate
    _instances = []
    
    def __init__(self, llm, agent_name=None, task_name=None):
        """
        Inizializza il wrapper.
        
        Parameters
        ----------
        llm : Any
            L'istanza LLM da wrappare.
        agent_name : str, optional
            Nome dell'agente che utilizza questo LLM.
        task_name : str, optional
            Nome del task che utilizza questo LLM.
        """
        self.llm = llm
        self.agent_name = agent_name or "unknown"
        self.task_name = task_name or "unknown"
        
        # Statistiche di utilizzo
        self.calls = 0
        self.tokens_used = {"input": 0, "output": 0, "total": 0}
        self.history = []
        
        # Registra questa istanza nel registro globale
        TokenCounterLLM._instances.append(self)
        
        # Log file
        self.log_dir = Path("token_usage_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        
        print(f"[INFO] Initialized TokenCounterLLM for {agent_name} (logs in {self.log_dir})")
    
    def _estimate_tokens(self, text):
        """
        Stima approssimativa del numero di token in un testo.
        
        Parameters
        ----------
        text : str
            Testo da stimare.
            
        Returns
        -------
        int
            Numero stimato di token.
        """
        # Stima basica: ~4 caratteri = 1 token per l'inglese
        # Per l'italiano, potrebbe essere leggermente diverso
        return len(text) // 4
    
    def _log_usage(self, input_text, output_text, input_tokens, output_tokens):
        """
        Registra l'utilizzo in un file di log.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "task": self.task_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            # Tronca i testi lunghi per non appesantire il log
            "input_text_sample": input_text[:500] + "..." if len(input_text) > 500 else input_text,
            "output_text_sample": output_text[:500] + "..." if len(output_text) > 500 else output_text
        }
        
        self.history.append(entry)
        
        # Aggiorna il file di log
        log_file = self.log_dir / f"token_usage_{self.session_id}.json"
        try:
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            else:
                log_data = {
                    "session_id": self.session_id,
                    "start_time": datetime.now().isoformat(),
                    "entries": []
                }
            
            log_data["entries"].append(entry)
            log_data["last_update"] = datetime.now().isoformat()
            
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Failed to update token usage log: {e}")
    
    def __getattr__(self, name):
        """
        Passa tutti gli attributi non trovati al LLM sottostante.
        """
        return getattr(self.llm, name)
    
    async def agenerate(self, *args, **kwargs):
        """
        Wrapper per diverse chiamate LLM, compatibile sia con LangChain che altri.
        """
        self.calls += 1
        start_time = time.time()
        
        # Ottieni il prompt - diverso in base al tipo di LLM
        prompt = ""
        if len(args) > 0:
            prompt = str(args[0])
        elif "messages" in kwargs:
            # LangChain-style
            messages = kwargs.get("messages", [])
            prompt = "\n".join([f"{m.type}: {m.content}" for m in messages if hasattr(m, "content")])
        
        # Stima token input
        input_tokens = self._estimate_tokens(prompt)
        self.tokens_used["input"] += input_tokens
        
        # Chiamata effettiva LLM
        response = await self.llm.agenerate(*args, **kwargs)
        
        # Estrai il testo della risposta - diverso in base al tipo di risposta
        if hasattr(response, "text"):
            # CrewAI-style response
            output_text = response.text
        elif hasattr(response, "generations"):
            # LangChain-style response
            output_text = "".join([gen.text for generation in response.generations for gen in generation])
        else:
            output_text = str(response)
        
        output_tokens = self._estimate_tokens(output_text)
        self.tokens_used["output"] += output_tokens
        self.tokens_used["total"] = self.tokens_used["input"] + self.tokens_used["output"]
        
        # Calcola durata
        duration = time.time() - start_time
        
        # Log su console
        print(f"[{self.agent_name}][{self.task_name}] Call #{self.calls}: {input_tokens} in, {output_tokens} out ({duration:.2f}s)")
        
        # Log su file
        self._log_usage(prompt, output_text, input_tokens, output_tokens)
        
        return response

    def generate(self, *args, **kwargs):
        """
        Wrapper per diverse chiamate LLM sincrone.
        """
        self.calls += 1
        start_time = time.time()
        
        # Ottieni il prompt - diverso in base al tipo di LLM
        prompt = ""
        if len(args) > 0:
            prompt = str(args[0])
        elif "messages" in kwargs:
            # LangChain-style
            messages = kwargs.get("messages", [])
            prompt = "\n".join([f"{m.type}: {m.content}" for m in messages if hasattr(m, "content")])
        
        # Stima token input
        input_tokens = self._estimate_tokens(prompt)
        self.tokens_used["input"] += input_tokens
        
        # Chiamata effettiva LLM
        response = self.llm.generate(*args, **kwargs)
        
        # Estrai il testo della risposta - diverso in base al tipo di risposta
        if hasattr(response, "text"):
            # CrewAI-style response
            output_text = response.text
        elif hasattr(response, "generations"):
            # LangChain-style response
            output_text = "".join([gen.text for generation in response.generations for gen in generation])
        else:
            output_text = str(response)
        
        output_tokens = self._estimate_tokens(output_text)
        self.tokens_used["output"] += output_tokens
        self.tokens_used["total"] = self.tokens_used["input"] + self.tokens_used["output"]
        
        # Calcola durata
        duration = time.time() - start_time
        
        # Log su console
        print(f"[{self.agent_name}][{self.task_name}] Call #{self.calls}: {input_tokens} in, {output_tokens} out ({duration:.2f}s)")
        
        # Log su file
        self._log_usage(prompt, output_text, input_tokens, output_tokens)
        
        return response
    
    @classmethod
    def print_usage_summary(cls):
        """
        Stampa un riepilogo dell'utilizzo dei token per tutti gli agenti.
        """
        if not cls._instances:
            print("No token usage data available.")
            return
        
        # Calcola statistiche per agente
        stats_by_agent = {}
        for instance in cls._instances:
            agent = instance.agent_name
            if agent not in stats_by_agent:
                stats_by_agent[agent] = {"calls": 0, "input": 0, "output": 0, "total": 0}
            
            stats_by_agent[agent]["calls"] += instance.calls
            stats_by_agent[agent]["input"] += instance.tokens_used["input"]
            stats_by_agent[agent]["output"] += instance.tokens_used["output"]
            stats_by_agent[agent]["total"] += instance.tokens_used["total"]
        
        # Calcola totale
        total_tokens = sum(stats["total"] for stats in stats_by_agent.values())
        
        # Stima costo (assumendo gpt-4o-mini a $0.0005 per 1K token - adatta secondo il tuo modello)
        cost_per_1k = 0.0005
        
        print("\n" + "=" * 50)
        print("TOKEN USAGE SUMMARY")
        print("=" * 50)
        
        for agent, stats in stats_by_agent.items():
            cost = stats["total"] / 1000 * cost_per_1k
            print(f"\nAgent: {agent}")
            print(f"  - Total calls: {stats['calls']}")
            print(f"  - Input tokens: {stats['input']:,}")
            print(f"  - Output tokens: {stats['output']:,}")
            print(f"  - Total tokens: {stats['total']:,}")
            print(f"  - Estimated cost (${cost_per_1k}/1K tokens): ${cost:.5f}")
        
        total_cost = total_tokens / 1000 * cost_per_1k
        print("\n" + "-" * 50)
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total estimated cost: ${total_cost:.5f}")
        print("=" * 50 + "\n")
        
        # Salva anche il riepilogo in un file JSON
        summary_file = Path("token_usage_logs") / "usage_summary.json"
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "by_agent": stats_by_agent,
                "total_tokens": total_tokens,
                "estimated_cost_usd": total_cost
            }
            
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                
            print(f"Summary saved to {summary_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save summary: {e}")