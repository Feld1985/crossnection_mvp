"""crossnection_mvp/agents/explain_agent.py

Concrete **ExplainAgent** for the Crossnection MVP.
Transforms statistical artifacts produced by StatsAgent into human‑readable
narratives. Delegates templating and layout to `CrossInsightFormatterTool`,
leaving natural‑language generation to the LLM configured in CrewAI.

This class defines two public methods matching the task names:
- `draft_root_cause_narrative` for human‑in‑the‑loop review.
- `finalize_root_cause_report` for producing the final report after feedback.
"""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, Optional

import crewai as cr
from crossnection_mvp.tools.cross_insight_formatter import CrossInsightFormatterTool
from crossnection_mvp.utils.context_decorators import with_context_io
from crossnection_mvp.utils.context_store import ContextStore

logger = logging.getLogger(__name__)

# Costanti per nomi di chiavi standard
ERROR_STATE_KEY = "error_state"
ERROR_MESSAGE_KEY = "error_message"
USER_MESSAGE_KEY = "user_message"


def safe_json_loads(json_string: str, default_value: Any = None) -> Any:
    """
    Carica in modo sicuro una stringa JSON, restituendo un valore di default in caso di errore.
    """
    if not json_string:
        return default_value
        
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_string[:100]}...")
        return default_value


def handle_error_result(result: Dict[str, Any]) -> bool:
    """
    Controlla se un risultato contiene un errore e lo gestisce appropriatamente.
    """
    if not isinstance(result, dict):
        return False
        
    if not result.get(ERROR_STATE_KEY, False):
        return False
        
    # Log dell'errore
    error_message = result.get(ERROR_MESSAGE_KEY, "Unknown error")
    user_message = result.get(USER_MESSAGE_KEY, "Si è verificato un errore")
    
    logger.error(f"Error detected: {error_message}")
    logger.info(f"User message: {user_message}")
    
    return True


class ExplainAgent(cr.BaseAgent):
    """Agent that builds and validates root‑cause narratives."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._formatter = CrossInsightFormatterTool()

@with_context_io(
    input_keys={
        "impact_ranking": "impact_ranking",
        "outlier_report": "outlier_report"
    },
    output_key="narrative_draft",
    output_type="json"
)
def draft_root_cause_narrative(
    self,
    impact_ranking=None,
    outlier_report=None,
    **kwargs
):
    """
    Generate the draft narrative of root causes, combining impact ranking
    and outlier report. This output is intended for business‑sense validation.
    """
    try:
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "draft_root_cause_narrative"
            
        # Log di debug dei parametri ricevuti
        logger.info(f"draft_root_cause_narrative called with: impact_ranking={type(impact_ranking)}, outlier_report={type(outlier_report)}")
        
        # Tentativo di recuperare i dati dal Context Store se non forniti direttamente
        store = ContextStore.get_instance()
        
        if impact_ranking is None:
            try:
                impact_ranking = store.load_json("impact_ranking")
                logger.info("Successfully loaded impact_ranking from Context Store")
            except Exception as e:
                logger.error(f"Failed to load impact_ranking from Context Store: {e}")
                impact_ranking = {"kpi_name": "Default KPI", "ranking": []}
                
        if outlier_report is None:
            try:
                outlier_report = store.load_json("outlier_report")
                logger.info("Successfully loaded outlier_report from Context Store")
            except Exception as e:
                logger.error(f"Failed to load outlier_report from Context Store: {e}")
                outlier_report = {"outliers": []}
        
        # Controlla se input contiene errori
        if isinstance(impact_ranking, dict) and impact_ranking.get(ERROR_STATE_KEY, False):
            logger.warning("Impact ranking contains error state, passing through")
            return {
                "markdown": f"# ⚠️ Errore nell'analisi di impatto\n\n{impact_ranking.get(USER_MESSAGE_KEY, 'Si è verificato un errore durante l\'analisi di impatto.')}",
                ERROR_STATE_KEY: True,
                "error_source": "impact_ranking"
            }
                
        if isinstance(outlier_report, dict) and outlier_report.get(ERROR_STATE_KEY, False):
            logger.warning("Outlier report contains error state, passing through")
            return {
                "markdown": f"# ⚠️ Errore nell'analisi degli outlier\n\n{outlier_report.get(USER_MESSAGE_KEY, 'Si è verificato un errore durante la rilevazione degli outlier.')}",
                ERROR_STATE_KEY: True,
                "error_source": "outlier_report"
            }
        
        # Assicurati che impact_ranking e outlier_report siano dizionari
        if isinstance(impact_ranking, str):
            impact_ranking = safe_json_loads(impact_ranking, {})
        
        if isinstance(outlier_report, str):
            outlier_report = safe_json_loads(outlier_report, {})
        
        # Verifica se i dati hanno la struttura corretta
        if not isinstance(impact_ranking, dict) or "ranking" not in impact_ranking:
            logger.warning(f"Invalid impact_ranking format: {impact_ranking}")
            # Crea un formato di fallback
            impact_ranking = {"kpi_name": "Default KPI", "ranking": []}
            
        if not isinstance(outlier_report, dict) or "outliers" not in outlier_report:
            logger.warning(f"Invalid outlier_report format: {outlier_report}")
            # Crea un formato di fallback
            outlier_report = {"outliers": []}
        
        # Usa il formatter per generare il report
        logger.info(f"Calling CrossInsightFormatterTool with impact_ranking={len(impact_ranking.get('ranking', []))} items, outlier_report={len(outlier_report.get('outliers', []))} items")
        
        result = self._formatter.run(
            impact_ranking=impact_ranking,
            outlier_report=outlier_report,
            mode="draft",
        )
        
        # Gestisci risultato stringa o dizionario
        if isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return {"markdown": result}
        return result
    except Exception as e:
        logger.error(f"Error in draft_root_cause_narrative: {e}", exc_info=True)
        return {
            "markdown": f"# ⚠️ Errore nella generazione della bozza\n\nSi è verificato un errore durante la generazione della bozza: {str(e)}",
            ERROR_STATE_KEY: True,
            ERROR_MESSAGE_KEY: str(e),
            USER_MESSAGE_KEY: "Non è stato possibile generare la bozza del report. Verifica che i dati di input siano nel formato corretto."
        }

    @with_context_io(
        input_keys={"narrative_draft": "narrative_draft"},
        output_key="root_cause_report",
        output_type="json"
    )
    def finalize_root_cause_report(
        self,
        narrative_draft: Any,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produce the final root‑cause report after merging user feedback.
        """
        try:
            # Imposta il task_name nel TokenCounterLLM se presente
            if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
                self.llm.task_name = "finalize_root_cause_report"
                
            # Controlla se narrative_draft contiene errori
            if isinstance(narrative_draft, dict) and narrative_draft.get(ERROR_STATE_KEY, False):
                logger.warning("Narrative draft contains error state, passing through")
                error_md = narrative_draft.get("markdown", "# ⚠️ Errore nella generazione della bozza\n\nNon è stato possibile generare la bozza del report.")
                # Aggiungi sezione feedback se presente
                if feedback:
                    error_md += f"\n\n## Feedback dell'utente\n\n{feedback}"
                return {
                    "markdown": error_md,
                    ERROR_STATE_KEY: True,
                    "error_source": "narrative_draft"
                }
                
            # Recupera impact_ranking e outlier_report dal Context Store
            store = ContextStore.get_instance()
            try:
                impact_ranking = store.load_json("impact_ranking")
                outlier_report = store.load_json("outlier_report")
                
                # Controlla errori nei dati recuperati
                if handle_error_result(impact_ranking) or handle_error_result(outlier_report):
                    error_md = "# ⚠️ Errore nella finalizzazione del report\n\nI dati necessari contengono errori che impediscono la finalizzazione del report."
                    # Aggiungi sezione feedback se presente
                    if feedback:
                        error_md += f"\n\n## Feedback dell'utente\n\n{feedback}"
                    return {
                        "markdown": error_md,
                        ERROR_STATE_KEY: True,
                        "error_source": "data_sources"
                    }
            except Exception as e:
                logger.error(f"Failed to load data from Context Store: {e}")
                error_md = "# ⚠️ Errore nel caricamento dei dati\n\nNon è stato possibile caricare i dati necessari per il report."
                # Aggiungi sezione feedback se presente
                if feedback:
                    error_md += f"\n\n## Feedback dell'utente\n\n{feedback}"
                return {
                    "markdown": error_md,
                    ERROR_STATE_KEY: True,
                    "error_source": "context_store"
                }
            
            # Se narrative_draft è una stringa, parsa il JSON
            if isinstance(narrative_draft, str):
                narrative_draft = safe_json_loads(narrative_draft, {"markdown": narrative_draft})
            
            result = self._formatter.run(
                impact_ranking=impact_ranking,
                outlier_report=outlier_report,
                feedback=feedback,
                mode="final",
            )
            
            # Gestisci risultato stringa o dizionario
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except:
                    return {"markdown": result}
            return result
        except Exception as e:
            logger.error(f"Error in finalize_root_cause_report: {e}", exc_info=True)
            error_md = f"# ⚠️ Errore nella finalizzazione del report\n\nSi è verificato un errore durante la finalizzazione del report: {str(e)}"
            # Aggiungi sezione feedback se presente
            if feedback:
                error_md += f"\n\n## Feedback dell'utente\n\n{feedback}"
            return {
                "markdown": error_md,
                ERROR_STATE_KEY: True,
                ERROR_MESSAGE_KEY: str(e),
                USER_MESSAGE_KEY: "Non è stato possibile finalizzare il report. Verifica che i dati di input siano nel formato corretto."
            }

    def __repr__(self) -> str:
        return f"<ExplainAgent id={self.id}>"