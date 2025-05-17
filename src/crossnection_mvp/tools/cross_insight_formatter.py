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
from typing import Any, Dict, List, Sequence, Union, Optional
from pydantic import BaseModel, Field

import markdown  # pip install markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape
from crewai.tools import BaseTool
from crossnection_mvp.utils.context_store import ContextStore

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
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse JSON: {blob[:100]}...")
            return {"error": "Failed to parse JSON", "text": blob}
    return blob


def _top_drivers(impact_ranking: Any, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k drivers by composite score (already sorted)."""
    # Handle different input formats
    if isinstance(impact_ranking, dict):
        if "ranking" in impact_ranking:
            # Standard format with "ranking" key
            ranking = impact_ranking["ranking"]
        elif all(isinstance(key, (int, str)) for key in impact_ranking.keys()):
            # Dict with numeric keys
            ranking = [value for key, value in sorted(impact_ranking.items())]
        else:
            # Fallback - treat as direct ranking
            ranking = impact_ranking
    else:
        # Direct sequence/list
        ranking = impact_ranking
    
    # Ensure we have a list before slicing
    if not isinstance(ranking, list):
        ranking = list(ranking) if hasattr(ranking, '__iter__') else [ranking]
    
    # Now slice the list safely
    return ranking[:k]


def _outlier_summary(outliers: Sequence[Dict[str, Any]]) -> str:
    if not outliers:
        return "No significant outliers were detected."
    
    # Extract driver names, handling different possible structures
    cols = set()
    for row in outliers:
        if "driver" in row:
            cols.add(row["driver"])
        else:
            # If no "driver" key, collect all keys that might be drivers
            for key in row:
                if key not in ["join_key", "row"]:
                    cols.add(key)
    
    rows = len(outliers)
    return (
        f"{rows} outlying data points were flagged across "
        f"{len(cols)} driver(s): {', '.join(sorted(cols))}."
    )


# ---------------------------------------------------------------------------#
# Main tool class
# ---------------------------------------------------------------------------#

class CrossInsightFormatterToolSchema(BaseModel):
    """Schema per il tool CrossInsightFormatter."""
    description: Optional[str] = None
    impact_ranking: Optional[Dict[str, Any]] = None
    outlier_report: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None
    mode: Optional[str] = "draft"
    k_top: Optional[int] = 5
    output_html: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = {}

class CrossInsightFormatterTool(BaseTool):
    """Tool entry-point to be registered in CrewAI."""

    name: str = "cross_insight_formatter"
    description: str = (
        "Builds draft and final root-cause narratives from statistical JSON results."
    )
    args_schema = CrossInsightFormatterToolSchema

    def _run(self, 
             description: Optional[str] = None,
             impact_ranking: Optional[Dict[str, Any]] = None,
             outlier_report: Optional[Dict[str, Any]] = None,
             feedback: Optional[str] = None,
             mode: str = "draft",
             k_top: int = 5,
             output_html: bool = False,
             metadata: Optional[Dict[str, Any]] = None,
             **kwargs) -> str:
        """
        Main entry point required by BaseTool.
        """
        print(f"DEBUG: CrossInsightFormatterTool received params: description={description is not None}, impact_ranking={impact_ranking is not None}, outlier_report={outlier_report is not None}")
        
        # Se non abbiamo i dati direttamente, prova a caricarli dal Context Store
        store = ContextStore.get_instance()
        
        if impact_ranking is None:
            try:
                impact_ranking = store.load_json("impact_ranking")
                print(f"DEBUG: Loaded impact_ranking from Context Store")
            except Exception as e:
                print(f"WARNING: Failed to load impact_ranking from Context Store: {e}")
                
        if outlier_report is None:
            try:
                outlier_report = store.load_json("outlier_report")
                print(f"DEBUG: Loaded outlier_report from Context Store")
            except Exception as e:
                print(f"WARNING: Failed to load outlier_report from Context Store: {e}")
        
        # Gestione dello scenario in cui l'agente passa direttamente un testo narrativo in 'description'
        if description and (not impact_ranking or not outlier_report):
            print(f"DEBUG: Using description as content: {description[:100]}...")
            result = {"markdown": description, "status": "success"}
            
            # Salva nel Context Store
            if mode == "draft":
                store.save_json("narrative_draft", result)
            else:
                store.save_json("root_cause_report", result)
            
            return json.dumps(result)
        
        # Default per impact_ranking
        if not impact_ranking:
            impact_ranking = {"kpi_name": "Default KPI", "ranking": []}
        
        # Default per outlier_report
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
            
            # Salva nel Context Store
            if mode == "draft":
                store.save_json("narrative_draft", result)
            else:
                store.save_json("root_cause_report", result)
            
            # Convert result to string if needed by CrewAI
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return result
        except Exception as e:
            error_msg = f"ERROR in CrossInsightFormatterTool: {e}"
            print(error_msg)
            # Fornisci un output minimo che non interrompa il flusso
            error_result = {
                "markdown": f"""
# 📊 Error in Root-Cause Narrative

An error occurred while generating the narrative: {e}

## Validation Instructions

Please review the input data and ensure:
- The KPI is correctly specified
- There is sufficient driver data to analyze
- The data format is correct

*You can provide feedback on this error to help improve the process.*
""",
                "error": str(e)
            }
            
            # Salva anche l'errore nel Context Store
            if mode == "draft":
                store.save_json("narrative_draft", error_result)
            else:
                store.save_json("root_cause_report", error_result)
                
            return json.dumps(error_result)

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

        # Se non abbiamo i dati direttamente, prova a caricarli dal Context Store
        store = ContextStore.get_instance()
        
        if impact_ranking is None:
            try:
                impact_ranking = store.load_json("impact_ranking")
                print(f"DEBUG: Loaded impact_ranking from Context Store")
            except Exception as e:
                print(f"WARNING: Failed to load impact_ranking from Context Store: {e}")
                impact_ranking = {"kpi_name": "Default KPI", "ranking": []}
                
        if outlier_report is None:
            try:
                outlier_report = store.load_json("outlier_report")
                print(f"DEBUG: Loaded outlier_report from Context Store")
            except Exception as e:
                print(f"WARNING: Failed to load outlier_report from Context Store: {e}")
                outlier_report = {"outliers": []}

        impact_data = _load_json_like(impact_ranking)
        outlier_data = _load_json_like(outlier_report)
        feedback_data = json.loads(feedback) if feedback else {}

        if mode == "draft":
            md = self._draft_markdown(impact_data, outlier_data, k_top=k_top)
        else:  # final
            md = self._final_markdown(impact_data, outlier_data, feedback_data, k_top=k_top)

        result: Dict[str, Any] = {"markdown": md}
        if output_html:
            result["html"] = _md_to_html(md)
            
        # Salva nel Context Store
        if mode == "draft":
            store.save_json("narrative_draft", result)
        else:
            store.save_json("root_cause_report", result)
            
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
        # Extract KPI name from various possible structures
        kpi_name = "KPI"
        if isinstance(impact_ranking, dict):
            if "kpi_name" in impact_ranking:
                kpi_name = impact_ranking["kpi_name"]
        
        # Get top drivers using the robust helper function
        top = _top_drivers(impact_ranking, k=k_top)
                
        context = {
            "top_drivers": top,
            "kpi": kpi_name,
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
        # Extract KPI name from various possible structures
        kpi_name = "KPI"
        if isinstance(impact_ranking, dict):
            if "kpi_name" in impact_ranking:
                kpi_name = impact_ranking["kpi_name"]
        
        # Get top drivers using the robust helper function
        top = _top_drivers(impact_ranking, k=k_top)
        
        # Simple merge: mark each driver as accepted/rejected/edited
        for driver in top:
            if "driver_name" in driver:
                driver_name = driver["driver_name"]
                fb = feedback.get("drivers", {}).get(driver_name)
                if fb:
                    driver["feedback"] = fb  # could include 'comment', 'status', etc.

        context = {
            "top_drivers": top,
            "kpi": kpi_name,
            "outlier_summary": _outlier_summary(outlier_report.get("outliers", [])),
            "user_notes": feedback.get("general_comment", ""),
        }
        return _FINAL_TEMPLATE.render(**context)