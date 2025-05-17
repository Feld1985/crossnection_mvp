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
from typing import Any, Dict, Optional

import crewai as cr
from crossnection_mvp.tools.cross_insight_formatter import CrossInsightFormatterTool

logger = logging.getLogger(__name__)


class ExplainAgent(cr.BaseAgent):
    """Agent that builds and validates root‑cause narratives."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._formatter = CrossInsightFormatterTool()

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
            result = self._formatter.run(
                impact_ranking=impact_ranking,
                outlier_report=outlier_report,
                mode="draft",
            )
            return result
        except Exception as e:
            logger.error("Error generating draft narrative: %s", e, exc_info=True)
            raise

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
            result = self._formatter.run(
                impact_ranking=None,
                outlier_report=None,
                feedback=feedback,
                mode="final",
            )
            return result
        except Exception as e:
            logger.error("Error generating final narrative: %s", e, exc_info=True)
            raise

    def __repr__(self) -> str:
        return f"<ExplainAgent id={self.id}>"