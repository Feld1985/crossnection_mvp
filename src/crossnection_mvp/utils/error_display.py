"""Utilità per la visualizzazione degli errori all'utente."""

from typing import Dict, Any, List

def format_error_report(errors: List[Dict[str, Any]]) -> str:
    """
    Formatta un report di errore user-friendly in Markdown.
    
    Parameters
    ----------
    errors : List[Dict[str, Any]]
        Lista di dizionari di errore con campi 'stage', 'message', 'suggestions'
        
    Returns
    -------
    str
        Report di errore formattato in Markdown
    """
    if not errors:
        return ""
    
    report = "# ⚠️ Problemi rilevati durante l'analisi\n\n"
    
    for i, error in enumerate(errors, 1):
        stage = error.get("stage", "Fase sconosciuta")
        message = error.get("message", "Si è verificato un errore sconosciuto")
        suggestions = error.get("suggestions", ["Riprova l'operazione"])
        
        report += f"## Problema {i}: {stage}\n\n"
        report += f"{message}\n\n"
        
        if suggestions:
            report += "### Suggerimenti\n\n"
            for suggestion in suggestions:
                report += f"- {suggestion}\n"
            report += "\n"
    
    report += "---\n\n"
    report += "*Se i problemi persistono, contatta il supporto tecnico.*"
    
    return report

def create_error_artifact(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crea un artefatto di errore per l'ExplainAgent.
    
    Parameters
    ----------
    errors : List[Dict[str, Any]]
        Lista di dizionari di errore
        
    Returns
    -------
    Dict[str, Any]
        Artefatto di errore
    """
    return {
        "markdown": format_error_report(errors),
        "error_state": True,
        "errors": errors
    }

def extract_error_data(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Estrae informazioni di errore dai risultati.
    
    Parameters
    ----------
    result : Dict[str, Any]
        Risultato da una fase di elaborazione
        
    Returns
    -------
    List[Dict[str, Any]]
        Lista di errori estratti
    """
    errors = []
    
    if isinstance(result, dict):
        # Controlla se c'è un errore esplicito
        if result.get("error_state", False):
            error = {
                "stage": result.get("stage", "Analisi"),
                "message": result.get("user_message", "Si è verificato un errore"),
                "suggestions": result.get("suggestions", [
                    "Verifica che i file CSV siano nel formato corretto",
                    "Assicurati che il KPI sia presente nei dati"
                ])
            }
            errors.append(error)
    
    return errors