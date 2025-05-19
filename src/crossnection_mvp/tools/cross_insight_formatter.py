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
    # Verifica input e fornisci dati di default se necessario
    if impact_ranking is None:
        print("WARNING: impact_ranking is None, returning empty list")
        return []
        
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
        try:
            ranking = list(ranking) if hasattr(ranking, '__iter__') else [ranking]
        except Exception as e:
            print(f"WARNING: Error converting ranking to list: {e}")
            ranking = []
    
    # Se la lista è vuota, restituisci vuota
    if not ranking:
        return []
        
    try:
        # Verifica che ogni elemento sia un dizionario con i campi necessari
        for i, item in enumerate(ranking):
            if not isinstance(item, dict):
                print(f"WARNING: Ranking item {i} is not a dictionary: {item}")
                ranking[i] = {"driver_name": f"Unknown driver {i}", "score": 0, "p_value": 1.0, "r": 0}
            elif "driver_name" not in item:
                print(f"WARNING: Ranking item {i} missing driver_name: {item}")
                item["driver_name"] = f"Unknown driver {i}"
            
            # Assicurati che ci siano valori per score, p_value e r
            if "score" not in item and "effect_size" not in item:
                item["score"] = 0
            if "p_value" not in item:
                item["p_value"] = 1.0
            if "r" not in item:
                item["r"] = 0
    except Exception as e:
        print(f"WARNING: Error checking ranking items: {e}")
    
    # Now slice the list safely
    return ranking[:k] if k < len(ranking) else ranking


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
    
        # Migliora il debug per mostrare più dettagli sui dati ricevuti
        if impact_ranking is not None:
            if isinstance(impact_ranking, dict):
                ranking_list = impact_ranking.get('ranking', [])
                print(f"DEBUG: impact_ranking contains {len(ranking_list)} items")
                # Mostra un esempio degli elementi se presenti
                if ranking_list:
                    print(f"DEBUG: First ranking item example: {ranking_list[0]}")
            else:
                print(f"DEBUG: impact_ranking is not a dict, but a {type(impact_ranking)}")
                
        if outlier_report is not None:
            if isinstance(outlier_report, dict):
                outliers = outlier_report.get('outliers', [])
                print(f"DEBUG: outlier_report contains {len(outliers)} outliers")
            else:
                print(f"DEBUG: outlier_report is not a dict, but a {type(outlier_report)}")
        
        print(f"DEBUG: CrossInsightFormatter using mode={mode}, k_top={k_top}")
        
        # Se non abbiamo i dati direttamente, prova a caricarli dal Context Store
        store = ContextStore.get_instance()
        
        if impact_ranking is None:
            try:
                impact_ranking = store.load_json("impact_ranking")
                print(f"DEBUG: Loaded impact_ranking from Context Store")
                # Verifica il contenuto caricato
                if isinstance(impact_ranking, dict) and 'ranking' in impact_ranking:
                    print(f"DEBUG: impact_ranking has {len(impact_ranking['ranking'])} items")
                else:
                    print(f"DEBUG: impact_ranking from Context Store has unexpected format: {impact_ranking}")
            except Exception as e:
                print(f"WARNING: Failed to load impact_ranking from Context Store: {e}")
                    
        if outlier_report is None:
            try:
                outlier_report = store.load_json("outlier_report")
                print(f"DEBUG: Loaded outlier_report from Context Store")
                # Verifica il contenuto caricato
                if isinstance(outlier_report, dict) and 'outliers' in outlier_report:
                    print(f"DEBUG: outlier_report has {len(outlier_report['outliers'])} items")
                else:
                    print(f"DEBUG: outlier_report from Context Store has unexpected format: {outlier_report}")
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
        
        # Verifica la struttura corretta degli input
        valid_impact_ranking = isinstance(impact_ranking, dict) and 'ranking' in impact_ranking
        valid_outlier_report = isinstance(outlier_report, dict) and 'outliers' in outlier_report
        
        if not valid_impact_ranking or not valid_outlier_report:
            print(f"DEBUG: Invalid input structure: valid_impact_ranking={valid_impact_ranking}, valid_outlier_report={valid_outlier_report}")
            error_result = {
                "markdown": """
                # 📊 Problema con il formato dei dati

                L'analisi non può essere completata perché i dati di input non sono nel formato atteso.

                ## Problemi rilevati:
                - {"Impact ranking" if not valid_impact_ranking else ""} {"Outlier report" if not valid_outlier_report else ""}

                Si consiglia di verificare che:
                1. Lo StatsAgent abbia completato correttamente la sua analisi
                2. Il Context Store contenga i dati nel formato corretto
                3. Non ci siano stati errori nelle fasi precedenti

                *L'amministratore può controllare i log per maggiori dettagli.*
                """,
                "status": "error"
            }
            
            # Salva nel Context Store
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
        # Verifica se ci sono stati di errore
        error_state = False
        error_messages = []
        
        if isinstance(impact_ranking, dict) and impact_ranking.get("error_state", False):
            error_state = True
            error_messages.append(impact_ranking.get("user_message", "Errore nell'analisi di impatto"))
        
        if isinstance(outlier_report, dict) and outlier_report.get("error_state", False):
            error_state = True
            error_messages.append(outlier_report.get("user_message", "Errore nell'analisi degli outlier"))
        
        # Verifica se i ranking sono vuoti
        ranking = impact_ranking.get('ranking', [])
        if len(ranking) == 0:
            error_state = True
            error_messages.append("Nessun driver trovato nell'analisi di impatto")
        
        # Se ci sono errori, genera un report che li segnala
        if error_state:
            # Preparare la lista degli errori prima della f-string
            error_list = "- " + "\n- ".join(error_messages)
            
            return f"""
                    # 📊 Draft Root-Cause Narrative

                    ## Attenzione: Problemi rilevati

                    Sono stati riscontrati alcuni problemi durante l'analisi:

                    {error_list}

                    ## Suggerimenti per risolvere

                    - Verifica che i file CSV contengano dati validi e nel formato corretto
                    - Controlla che il KPI selezionato sia presente nei dati
                    - Assicurati che ci siano sufficienti dati numerici per l'analisi statistica

                    *Puoi procedere con una revisione parziale dei risultati disponibili o caricare nuovi dati per riprovare l'analisi.*
                    """
        
        # Extract KPI name from various possible structures
        kpi_name = "KPI"
        if isinstance(impact_ranking, dict):
            if "kpi_name" in impact_ranking:
                kpi_name = impact_ranking["kpi_name"]
        
        # Get top drivers using the robust helper function
        top = _top_drivers(impact_ranking, k=k_top)
        
        # Se non ci sono driver, mostra un errore
        if not top:
            return f"""
            # 📊 Draft Root-Cause Narrative for {kpi_name}

            ## Attenzione: Nessun driver significativo trovato

            L'analisi non ha identificato driver con correlazione statisticamente significativa rispetto al KPI.

            Questo potrebbe essere dovuto a:
            - Dati insufficienti per l'analisi
            - Mancanza di correlazioni reali tra i driver e il KPI
            - Problemi nel formato o nella qualità dei dati

            ## Suggerimenti

            - Valuta l'inclusione di altri driver potenzialmente rilevanti
            - Verifica che il periodo di analisi sia sufficientemente lungo
            - Controlla la qualità dei dati nei driver esistenti

            ## Validation Instructions

            Si prega di confermare se questa assenza di correlazioni è in linea con le aspettative di business o se è necessario rivedere l'approccio di analisi.
            """
                
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