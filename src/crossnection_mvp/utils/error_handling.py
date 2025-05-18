"""
Utilities per la gestione unificata degli errori in Crossnection.

Fornisce un decoratore che può essere applicato ai metodi per
standardizzare la gestione degli errori in tutto il codebase.
"""
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union, List

# Chiavi standard per i messaggi di errore
ERROR_STATE_KEY = "error_state"
ERROR_MESSAGE_KEY = "error_message" 
USER_MESSAGE_KEY = "user_message"
TECHNICAL_DETAILS_KEY = "technical_details"
STAGE_KEY = "stage"
SUGGESTIONS_KEY = "suggestions"

# Mappatura personalizzabile di eccezioni a messaggi utente
EXCEPTION_MESSAGES = {
    "ValueError": "I valori forniti non sono validi. Verifica i dati di input.",
    "KeyError": "Chiave non trovata. Verifica che i nomi delle colonne siano corretti.",
    "TypeError": "Tipo di dato non corretto. Verifica che i dati siano nel formato atteso.",
    "FileNotFoundError": "File non trovato. Verifica che i percorsi dei file siano corretti.",
    "ConnectionError": "Errore di connessione. Verifica la connessione di rete.",
    "IndexError": "Indice fuori range. Potrebbe mancare dei dati.",
    "AttributeError": "Attributo non trovato. Verifica la struttura dei dati.",
    "ImportError": "Errore di importazione. Verifica che tutte le dipendenze siano installate.",
    "JSONDecodeError": "Errore nella decodifica JSON. Il formato del file potrebbe essere invalido.",
    "ZeroDivisionError": "Divisione per zero. Verifica i dati numerici.",
    "default": "Si è verificato un errore inaspettato. Riprova l'operazione."
}

# Suggerimenti comuni per gli errori
DEFAULT_SUGGESTIONS = [
    "Verifica che i dati di input siano nel formato corretto",
    "Assicurati che tutti i file necessari siano presenti",
    "Controlla i log per dettagli tecnici più specifici"
]

def with_robust_error_handling(
    return_fallback: bool = True,
    log_level: str = "ERROR",
    stage_name: Optional[str] = None,
    custom_exceptions: Optional[Dict[Type[Exception], str]] = None,
    custom_fallback: Optional[Dict[str, Any]] = None,
    store_error_key: Optional[str] = None
):
    """
    Decoratore per una gestione consistente degli errori in tutto il progetto.
    
    Parameters
    ----------
    return_fallback : bool, default=True
        Se True, restituisce un valore di fallback anziché rilanciare l'eccezione
    log_level : str, default="ERROR"
        Livello di logging da usare per gli errori ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    stage_name : str, optional
        Nome della fase o del componente per identificare l'origine dell'errore
    custom_exceptions : Dict[Type[Exception], str], optional
        Mappatura personalizzata di tipi di eccezione -> messaggi utente
    custom_fallback : Dict[str, Any], optional
        Valore di fallback personalizzato
    store_error_key : str, optional
        Se fornito, salva l'errore nel ContextStore con questa chiave
        
    Returns
    -------
    Callable
        Decoratore che incapsula la funzione target
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Configura logger
            logger = logging.getLogger(fn.__module__)
            stage = stage_name or fn.__name__
            
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                # Log dell'errore con livello appropriato
                log_method = getattr(logger, log_level.lower(), logger.error)
                error_details = f"Error in {stage} ({fn.__name__}): {e}"
                log_method(error_details, exc_info=True)
                
                # Ottieni il tipo di eccezione
                exc_type = type(e).__name__
                
                # Determina il messaggio utente
                if custom_exceptions and exc_type in custom_exceptions:
                    user_message = custom_exceptions[exc_type]
                elif exc_type in EXCEPTION_MESSAGES:
                    user_message = EXCEPTION_MESSAGES[exc_type]
                else:
                    user_message = EXCEPTION_MESSAGES["default"]
                
                # Suggerimenti specifici per il tipo di errore
                suggestions = DEFAULT_SUGGESTIONS.copy()
                
                # Se richiesto, salva l'errore nel ContextStore
                if store_error_key:
                    try:
                        from crossnection_mvp.utils.context_store import ContextStore
                        store = ContextStore.get_instance()
                        error_data = {
                            ERROR_STATE_KEY: True,
                            ERROR_MESSAGE_KEY: str(e),
                            USER_MESSAGE_KEY: user_message,
                            STAGE_KEY: stage,
                            SUGGESTIONS_KEY: suggestions,
                            TECHNICAL_DETAILS_KEY: traceback.format_exc()
                        }
                        store.save_json(store_error_key, error_data)
                    except Exception as store_error:
                        logger.error(f"Failed to save error to ContextStore: {store_error}")
                
                # Se richiesto, restituisci un fallback invece di rilanciare l'eccezione
                if return_fallback:
                    # Determina il fallback in base al tipo di funzione
                    if custom_fallback:
                        fallback = custom_fallback
                    else:
                        # Deduci il tipo di fallback dal nome della funzione
                        name = fn.__name__.lower()
                        if any(term in name for term in ["correlation", "matrix"]):
                            fallback = {
                                ERROR_STATE_KEY: True,
                                ERROR_MESSAGE_KEY: str(e),
                                USER_MESSAGE_KEY: user_message,
                                STAGE_KEY: stage,
                                SUGGESTIONS_KEY: suggestions,
                                "drivers": []
                            }
                        elif any(term in name for term in ["rank", "impact"]):
                            fallback = {
                                ERROR_STATE_KEY: True,
                                ERROR_MESSAGE_KEY: str(e),
                                USER_MESSAGE_KEY: user_message,
                                STAGE_KEY: stage,
                                SUGGESTIONS_KEY: suggestions,
                                "ranking": []
                            }
                        elif any(term in name for term in ["outlier", "anomaly"]):
                            fallback = {
                                ERROR_STATE_KEY: True,
                                ERROR_MESSAGE_KEY: str(e),
                                USER_MESSAGE_KEY: user_message,
                                STAGE_KEY: stage,
                                SUGGESTIONS_KEY: suggestions,
                                "outliers": []
                            }
                        elif any(term in name for term in ["narrative", "report", "draft"]):
                            fallback = {
                                ERROR_STATE_KEY: True,
                                ERROR_MESSAGE_KEY: str(e),
                                USER_MESSAGE_KEY: user_message,
                                STAGE_KEY: stage,
                                SUGGESTIONS_KEY: suggestions,
                                "markdown": f"# Errore durante la generazione del report\n\n{user_message}\n\n## Suggerimenti\n\n" + "\n".join([f"- {s}" for s in suggestions])
                            }
                        else:
                            # Fallback generico
                            fallback = {
                                ERROR_STATE_KEY: True,
                                ERROR_MESSAGE_KEY: str(e),
                                USER_MESSAGE_KEY: user_message,
                                STAGE_KEY: stage,
                                SUGGESTIONS_KEY: suggestions
                            }
                    
                    return fallback
                else:
                    # Rilancia l'eccezione originale
                    raise
        
        return wrapper
    
    return decorator

def safe_json_loads(json_string: str, default_value: Any = None) -> Any:
    """
    Carica in modo sicuro una stringa JSON, restituendo un valore di default in caso di errore.
    
    Parameters
    ----------
    json_string : str
        La stringa JSON da caricare
    default_value : Any, optional
        Il valore da restituire in caso di errore
        
    Returns
    -------
    Any
        Il contenuto JSON deserializzato o default_value
    """
    import json
    
    if not json_string:
        return default_value
        
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON: {json_string[:100]}...")
        return default_value

def handle_error_result(result: Dict[str, Any]) -> bool:
    """
    Controlla se un risultato contiene un errore e lo gestisce appropriatamente.
    
    Parameters
    ----------
    result : Dict[str, Any]
        Il risultato da controllare
        
    Returns
    -------
    bool
        True se il risultato è un errore, False altrimenti
    """
    if not isinstance(result, dict):
        return False
        
    if not result.get(ERROR_STATE_KEY, False):
        return False
        
    # Log dell'errore
    error_message = result.get(ERROR_MESSAGE_KEY, "Unknown error")
    user_message = result.get(USER_MESSAGE_KEY, "Si è verificato un errore")
    stage = result.get(STAGE_KEY, "Unknown stage")
    
    logging.error(f"Error in {stage}: {error_message}")
    logging.info(f"User message: {user_message}")
    
    return True

def format_error_for_user(error: Dict[str, Any]) -> str:
    """
    Formatta un dizionario di errore in un messaggio leggibile per l'utente.
    
    Parameters
    ----------
    error : Dict[str, Any]
        Dizionario di errore
        
    Returns
    -------
    str
        Messaggio formattato
    """
    user_message = error.get(USER_MESSAGE_KEY, "Si è verificato un errore")
    suggestions = error.get(SUGGESTIONS_KEY, DEFAULT_SUGGESTIONS)
    
    message = f"# ⚠️ {user_message}\n\n"
    
    if suggestions:
        message += "## Suggerimenti\n\n"
        for suggestion in suggestions:
            message += f"- {suggestion}\n"
            
    return message