def with_context_io(input_keys=None, output_key=None, output_type="json"):
    """Decorator per gestire I/O con Context Store nei metodi degli agenti.
    
    Parameters
    ----------
    input_keys : List[str] or Dict[str, str], optional
        Chiavi da caricare dal Context Store. 
        Se dict, mappa il nome del parametro alla chiave nello store.
    output_key : str, optional
        Nome della chiave per salvare l'output.
    output_type : str, optional
        Tipo di output ('json' o 'dataframe').
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, **kwargs):
            store = ContextStore.get_instance()
            
            # Gestisci input
            if input_keys:
                if isinstance(input_keys, dict):
                    # Map param_name -> store_key
                    for param_name, store_key in input_keys.items():
                        # Load only if the parameter was actually passed
                        if param_name in kwargs and kwargs[param_name]:
                            # Gestione migliorata dei riferimenti
                            if isinstance(kwargs[param_name], str):
                                # Se è un riferimento a un file
                                if kwargs[param_name].endswith('.json') or kwargs[param_name].endswith('.csv'):
                                    try:
                                        # Estrai il nome base senza estensione e versione
                                        ref_path = kwargs[param_name]
                                        base_name = store.extract_artifact_name(ref_path)
                                        
                                        if base_name:
                                            if kwargs[param_name].endswith('.json'):
                                                kwargs[param_name] = store.load_json(base_name)
                                            elif kwargs[param_name].endswith('.csv'):
                                                kwargs[param_name] = store.load_dataframe(base_name)
                                    except Exception as e:
                                        print(f"WARNING: Failed to load from reference '{kwargs[param_name]}': {e}")
                elif isinstance(input_keys, list):
                    # Load each key with the same name
                    for key in input_keys:
                        if key in kwargs and kwargs[key]:
                            if isinstance(kwargs[key], str):
                                if kwargs[key].endswith('.json'):
                                    try:
                                        base_name = store.extract_artifact_name(kwargs[key])
                                        if base_name:
                                            kwargs[key] = store.load_json(base_name)
                                    except Exception as e:
                                        print(f"WARNING: Failed to load JSON from reference '{kwargs[key]}': {e}")
                                elif kwargs[key].endswith('.csv'):
                                    try:
                                        base_name = store.extract_artifact_name(kwargs[key])
                                        if base_name:
                                            kwargs[key] = store.load_dataframe(base_name)
                                    except Exception as e:
                                        print(f"WARNING: Failed to load DataFrame from reference '{kwargs[key]}': {e}")
            
            # Esegui la funzione originale
            result = fn(self, **kwargs)
            
            # Gestisci output
            if output_key and result is not None:
                if output_type == 'json':
                    path = store.save_json(output_key, result)
                elif output_type == 'dataframe':
                    import pandas as pd
                    if isinstance(result, pd.DataFrame):
                        path = store.save_dataframe(output_key, result)
                    else:
                        raise TypeError(f"Expected DataFrame result for {output_key}, got {type(result)}")
                else:
                    raise ValueError(f"Unsupported output_type: {output_type}")
                
                # Restituisci il path invece del risultato completo
                return {"path": str(path), "type": output_type}
            
            return result
        
        return wrapper
    
    return decorator