"""crossnection_mvp/tools/cross_insight_formatter.py

Custom CrewAI Tool: CrossInsightFormatterTool
============================================
Transforms raw statistical outputs into human-readable narratives.

Responsibilities
----------------
1. Build a *draft* root-cause narrative in Markdown that the user reviews.
2. Merge optional user feedback, regenerating text where needed.
3. Return a final report (Markdown string OR JSON dict ready for UI export).

Design notes
------------
* Heavy natural-language generation is delegated to the LLM (agent prompt);
  this tool focuses on structuring inputs/outputs and lightweight templating.
* Uses `jinja2` to keep templates readable and configurable.
* Converts Markdown → HTML only if required by downstream UI (optional arg).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from pydantic import BaseModel

import markdown  # pip install markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape
from crewai.tools import BaseTool

# ---------------------------------------------------------------------------#
# Jinja environment & templates
# ---------------------------------------------------------------------------#

TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    trim_blocks=True,
    lstrip_blocks=True,
)

_DRAFT_TEMPLATE = _ENV.get_template("draft_narrative.md.j2")
_FINAL_TEMPLATE = _ENV.get_template("final_narrative.md.j2")

# Markdown → HTML (optional)
def _md_to_html(md_text: str) -> str:
    return markdown.markdown(md_text, extensions=["tables", "fenced_code"])


# ---------------------------------------------------------------------------#
# Helper utils
# ---------------------------------------------------------------------------#


def _load_json_like(blob: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    """Accept dict, JSON string, or file-path and return dict."""
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, Path):
        return json.loads(Path(blob).read_text())
    return json.loads(blob)


def _top_drivers(impact_ranking: Sequence[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k drivers by composite score (already sorted)."""
    return list(impact_ranking[:k])


def _outlier_summary(outliers: Sequence[Dict[str, Any]]) -> str:
    if not outliers:
        return "No significant outliers were detected."
    cols = {row["driver"] for row in outliers}
    rows = len(outliers)
    return (
        f"{rows} outlying data points were flagged across "
        f"{len(cols)} driver(s): {', '.join(sorted(cols))}."
    )


# ---------------------------------------------------------------------------#
# Main tool class
# ---------------------------------------------------------------------------#

from typing import Annotated
class CrossInsightFormatterToolSchema(BaseModel):
    input: Annotated[Union[str, Dict[str, Any]], "Markdown config or feedback blob"]

class CrossInsightFormatterTool(BaseTool):
    """Tool entry-point to be registered in CrewAI."""

    name: str = "cross_insight_formatter"
    description: str = (
        "Builds draft and final root-cause narratives from statistical JSON results."
    )
    args_schema = CrossInsightFormatterToolSchema

    def _run(self, input: Union[str, Dict[str, Any]]) -> str:
        """
        Main entry point required by BaseTool.
        """
        print(f"DEBUG: CrossInsightFormatterTool received raw input: {input}")
        
        # Parse input if it's a string
        if isinstance(input, str):
            try:
                input_data = json.loads(input)
                print(f"DEBUG: Parsed JSON string to dict: {input_data}")
            except json.JSONDecodeError:
                # Caso in cui l'input è una stringa semplice (non JSON)
                print(f"DEBUG: Input is not JSON, treating as general feedback")
                input_data = {
                    "mode": "draft",
                    "feedback": input,
                    "impact_ranking": {"kpi_name": "Default KPI", "ranking": []},
                    "outlier_report": {"outliers": []}
                }
        else:
            input_data = input
        
        # Estrai parameters con valori predefiniti
        impact_ranking = input_data.get("impact_ranking", {"kpi_name": "Default KPI", "ranking": []})
        outlier_report = input_data.get("outlier_report", {"outliers": []})
        feedback = input_data.get("feedback")
        mode = input_data.get("mode", "draft")
        k_top = input_data.get("k_top", 5)
        output_html = input_data.get("output_html", False)
        
        # Se impact_ranking è None o vuoto, crea un dizionario di default
        if not impact_ranking:
            impact_ranking = {"kpi_name": "Default KPI", "ranking": []}
        
        # Se outlier_report è None o vuoto, crea un dizionario di default
        if not outlier_report:
            outlier_report = {"outliers": []}
        
        print(f"DEBUG: CrossInsightFormatter using mode={mode}, k_top={k_top}")
        
        try:
            # Run the original implementation
            result = self.run(
                impact_ranking=impact_ranking,
                outlier_report=outlier_report,
                feedback=feedback,
                mode=mode,
                k_top=k_top,
                output_html=output_html
            )
            
            # Convert result to string if needed by CrewAI
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return result
        except Exception as e:
            error_msg = f"ERROR in CrossInsightFormatterTool: {e}"
            print(error_msg)
            # Fornisci un output minimo che non interrompa il flusso
            markdown = f"""
    # 📊 Error in Root-Cause Narrative

    An error occurred while generating the narrative: {e}

    ## Validation Instructions

    Please review the input data and ensure:
    - The KPI is correctly specified
    - There is sufficient driver data to analyze
    - The data format is correct

    *You can provide feedback on this error to help improve the process.*
    """
            return json.dumps({"markdown": markdown, "error": str(e)})

    # ---------------------------------------------------------------------#
    # Original implementation
    # ---------------------------------------------------------------------#

    def run(
        self,
        *,
        impact_ranking: str | Path | Dict[str, Any] | None = None,
        outlier_report: str | Path | Dict[str, Any] | None = None,
        feedback: str | None = None,
        mode: str = "draft",
        k_top: int = 5,
        output_html: bool = False,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        impact_ranking
            JSON (dict / str / file) produced by `rank_impact`.
        outlier_report
            JSON produced by `detect_outliers`.
        feedback
            Optional JSON string from the human-in-the-loop step.
        mode
            'draft'  – generate draft narrative.\n
            'final'  – merge feedback & output final narrative.
        k_top
            How many top drivers to show in the summary table.
        output_html
            If True, returns HTML in addition to Markdown.

        Returns
        -------
        dict with keys:
            * 'markdown'  – narrative text
            * 'html' (optional) – html version
        """
        if mode not in {"draft", "final"}:
            raise ValueError("mode must be 'draft' or 'final'")

        impact_data = _load_json_like(impact_ranking) if impact_ranking else {}
        outlier_data = _load_json_like(outlier_report) if outlier_report else {}
        feedback_data = json.loads(feedback) if feedback else {}

        if mode == "draft":
            md = self._draft_markdown(impact_data, outlier_data, k_top=k_top)
        else:  # final
            md = self._final_markdown(impact_data, outlier_data, feedback_data, k_top=k_top)

        result: Dict[str, Any] = {"markdown": md}
        if output_html:
            result["html"] = _md_to_html(md)
        return result

    # ------------------------------------------------------------------#
    # Internal renderers
    # ------------------------------------------------------------------#

    def _draft_markdown(
        self,
        impact_ranking: Dict[str, Any],
        outlier_report: Dict[str, Any],
        *,
        k_top: int,
    ) -> str:
        """Render first draft to be validated by user."""
        top = _top_drivers(impact_ranking.get("ranking", []), k=k_top)
        context = {
            "top_drivers": top,
            "kpi": impact_ranking.get("kpi_name", "KPI"),
            "outlier_summary": _outlier_summary(outlier_report.get("outliers", [])),
            "validation_instructions": """
            Please review each driver and mark them as follows:
            - RELEVANT: Business-critical correlation worth investigating
            - OBVIOUS: Expected relationship, not a surprise
            - IRRELEVANT: Statistical noise or spurious correlation
            
            Add comments for any specific insights or context.
            """
        }
        return _DRAFT_TEMPLATE.render(**context)

    def _final_markdown(
        self,
        impact_ranking: Dict[str, Any],
        outlier_report: Dict[str, Any],
        feedback: Dict[str, Any],
        *,
        k_top: int,
    ) -> str:
        """Render final narrative, merging user feedback."""
        # Simple merge: mark each driver as accepted/rejected/edited
        ranking = impact_ranking.get("ranking", [])
        for driver in ranking:
            fb = feedback.get("drivers", {}).get(driver["driver_name"])
            if fb:
                driver["feedback"] = fb  # could include 'comment', 'status', etc.

        context = {
            "top_drivers": _top_drivers(ranking, k=k_top),
            "kpi": impact_ranking.get("kpi_name", "KPI"),
            "outlier_summary": _outlier_summary(outlier_report.get("outliers", [])),
            "user_notes": feedback.get("general_comment", ""),
        }
        return _FINAL_TEMPLATE.render(**context)