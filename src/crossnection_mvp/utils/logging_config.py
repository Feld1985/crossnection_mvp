"""Configurazione centralizzata del logging per Crossnection."""

import logging
import sys
from pathlib import Path

def configure_logging():
    """
    Configura il logging per l'applicazione con output formattato
    e gestione dei file di log.
    """
    # Crea directory per i log
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurazione generale
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Formattatore dettagliato
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler per console con formato base
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Handler per file con formato dettagliato
    file_handler = logging.FileHandler(log_dir / 'crossnection.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Handler per errori con formato dettagliato
    error_handler = logging.FileHandler(log_dir / 'crossnection_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Aggiungi gli handler al logger root
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Imposta livelli specifici per moduli
    logging.getLogger('crossnection_mvp').setLevel(logging.DEBUG)
    
    # Riduci verbosità di librerie esterne
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return root_logger