# debug_app.py
import streamlit as st
import subprocess
import json
import os
import sys
import time
from pathlib import Path
import threading

st.set_page_config(page_title="Crossnection Debug", layout="wide")
st.title("Crossnection Debug")

# Usa dati di esempio
process_map_path = "examples/process_map.json"
drivers_dir_path = "examples/driver_csvs"
kpi = "value_speed"

# Verifica che i file esistano
process_map_exists = os.path.exists(process_map_path)
drivers_dir_exists = os.path.exists(drivers_dir_path)

st.info(f"Process Map: {process_map_path} - Exists: {process_map_exists}")
st.info(f"Drivers Dir: {drivers_dir_path} - Exists: {drivers_dir_exists}")

# Funzione per eseguire in un thread separato
def run_process():
    cmd = [
        sys.executable,
        "-m", "crossnection_mvp.main", "run",
        "--kpi", kpi,
        "--process-map", process_map_path,
        "--drivers-dir", drivers_dir_path
    ]
    
    st.session_state.process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_lines = []
    error_lines = []
    
    while True:
        # Verifica se il processo è ancora in esecuzione
        if st.session_state.process.poll() is not None:
            break
            
        # Leggi output
        stdout_line = st.session_state.process.stdout.readline()
        if stdout_line:
            output_lines.append(stdout_line.strip())
            st.session_state.output = output_lines
        
        # Leggi errori
        stderr_line = st.session_state.process.stderr.readline()
        if stderr_line:
            error_lines.append(stderr_line.strip())
            st.session_state.errors = error_lines
        
        # Breve pausa
        time.sleep(0.1)
    
    # Leggi eventuali output rimasti
    for line in st.session_state.process.stdout:
        output_lines.append(line.strip())
    for line in st.session_state.process.stderr:
        error_lines.append(line.strip())
    
    st.session_state.output = output_lines
    st.session_state.errors = error_lines
    st.session_state.completed = True
    st.session_state.returncode = st.session_state.process.returncode

# Inizializzazione delle variabili di sessione
if 'started' not in st.session_state:
    st.session_state.started = False
if 'output' not in st.session_state:
    st.session_state.output = []
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'completed' not in st.session_state:
    st.session_state.completed = False
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'returncode' not in st.session_state:
    st.session_state.returncode = None

# Pulsante per avviare l'esecuzione
if not st.session_state.started:
    if st.button("Avvia Processo", disabled=not (process_map_exists and drivers_dir_exists)):
        st.session_state.started = True
        st.session_state.output = []
        st.session_state.errors = []
        st.session_state.completed = False
        st.session_state.returncode = None
        
        # Avvia in un thread separato
        st.session_state.thread = threading.Thread(target=run_process)
        st.session_state.thread.daemon = True
        st.session_state.thread.start()
        
        st.rerun()

# Visualizza stato e output
if st.session_state.started:
    # Status
    if st.session_state.completed:
        if st.session_state.returncode == 0:
            st.success(f"Processo completato con successo (code: {st.session_state.returncode})")
        else:
            st.error(f"Processo terminato con errore (code: {st.session_state.returncode})")
    else:
        st.info("Processo in esecuzione...")
    
    # Output
    st.subheader("Output del Processo")
    output_tab, error_tab, files_tab = st.tabs(["Output", "Errori", "File generati"])
    
    with output_tab:
        # Aggiungi un pulsante per aggiornare manualmente
        if st.button("Aggiorna output"):
            st.rerun()
            
        output_text = "\n".join(st.session_state.output)
        st.code(output_text)
    
    with error_tab:
        error_text = "\n".join(st.session_state.errors)
        if error_text:
            st.code(error_text, language="bash")
        else:
            st.info("Nessun errore rilevato")
    
    with files_tab:
        # Mostra i file generati nella directory flow_data
        flow_data_dir = Path("flow_data")
        if flow_data_dir.exists():
            sessions = [p for p in flow_data_dir.iterdir() if p.is_dir()]
            
            if sessions:
                latest_session = sorted(sessions)[-1]
                st.write(f"Ultima sessione: {latest_session.name}")
                
                # Lista dei file
                files = list(latest_session.glob("*.*"))
                if files:
                    for file in files:
                        st.write(f"- {file.name}")
                        
                        # Mostra contenuto per file JSON
                        if file.suffix == ".json":
                            with st.expander(f"Contenuto di {file.name}"):
                                try:
                                    with open(file, "r") as f:
                                        content = json.load(f)
                                        st.json(content)
                                except Exception as e:
                                    st.error(f"Errore nella lettura del file: {e}")
                else:
                    st.warning("Nessun file trovato nella sessione")
            else:
                st.warning("Nessuna sessione trovata")
        else:
            st.warning("Directory flow_data non trovata")
    
    # Pulsante per terminare il processo
    if not st.session_state.completed:
        if st.button("Termina Processo"):
            if hasattr(st.session_state, 'process'):
                st.session_state.process.terminate()
                st.warning("Processo terminato dall'utente")
                st.session_state.completed = True
                st.rerun()
    
    # Pulsante per riavviare
    if st.session_state.completed:
        if st.button("Riavvia"):
            st.session_state.started = False
            st.rerun()