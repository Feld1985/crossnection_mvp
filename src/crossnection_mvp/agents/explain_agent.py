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
        impact_ranking: Any,
        outlier_report: Any,
    ) -> Dict[str, Any]:
        """
        Generate the draft narrative of root causes, combining impact ranking
        and outlier report. This output is intended for business‑sense validation.
        """
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "draft_root_cause_narrative"
            
        try:
            # Assicurati che impact_ranking e outlier_report siano dizionari
            if isinstance(impact_ranking, str):
                try:
                    impact_ranking = json.loads(impact_ranking)
                except:
                    logger.warning(f"Could not parse impact_ranking as JSON: {impact_ranking[:100]}...")
            
            if isinstance(outlier_report, str):
                try:
                    outlier_report = json.loads(outlier_report)
                except:
                    logger.warning(f"Could not parse outlier_report as JSON: {outlier_report[:100]}...")
            
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
            logger.error("Error generating draft narrative: %s", e, exc_info=True)
            # Fornisci un fallback in caso di errore
            return {
                "markdown": f"""
# 📊 Error in Draft Root-Cause Narrative

An error occurred while generating the narrative: {str(e)}

## Validation Instructions

Please review the input data and ensure:
- The KPI is correctly specified
- There is sufficient driver data to analyze
- The data format is correct

*You can provide feedback on this error to help improve the process.*
                """
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
        # Imposta il task_name nel TokenCounterLLM se presente
        if hasattr(self, "llm") and hasattr(self.llm, "task_name"):
            self.llm.task_name = "finalize_root_cause_report"
            
        try:
            # Recupera impact_ranking e outlier_report dal Context Store
            store = ContextStore.get_instance()
            impact_ranking = store.load_json("impact_ranking")
            outlier_report = store.load_json("outlier_report")
            
            # Se narrative_draft è una stringa, parsa il JSON
            if isinstance(narrative_draft, str):
                try:
                    narrative_draft = json.loads(narrative_draft)
                except:
                    logger.warning(f"Could not parse narrative_draft as JSON: {narrative_draft[:100]}...")
                    narrative_draft = {"markdown": narrative_draft}
            
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
            logger.error("Error generating final narrative: %s", e, exc_info=True)
            # Fornisci un fallback in caso di errore
            return {
                "markdown": f"""
# 📘 Error in Final Root-Cause Report

An error occurred while generating the final report: {str(e)}

The draft narrative was provided, but there was an issue incorporating the feedback. 
Please try again or contact support for assistance.
                """
            }

    def __repr__(self) -> str:
        return f"<ExplainAgent id={self.id}>"