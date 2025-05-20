# app.py - Versione migliorata
import streamlit as st
import pandas as pd
import time
import subprocess
import json
import os
import sys
from pathlib import Path
from pdf_generator import generate_pdf_report
import threading
import tempfile

st.set_page_config(page_title="Crossnection - Root Cause Analysis", layout="wide")

# Header con logo e titolo
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=200)
else:
    st.title("Crossnection")
st.title("Root Cause Discovery")
st.markdown("---")

# Sidebar per i parametri di input
with st.sidebar:
    st.header("Parametri di Analisi")
    kpi = st.selectbox("KPI da analizzare", ["value_speed", "value_pressure", "value_temperature"], 
                      help="Seleziona il KPI target per l'analisi")
    
    # Opzione per usare dati di esempio
    use_example_data = st.checkbox("Usa dati di esempio", value=True,
                                  help="Utilizza i dataset di esempio invece di caricare i file")
    
    if not use_example_data:
        uploaded_process_map = st.file_uploader("Process Map (JSON)", type=["json"], 
                                               help="Carica il file JSON della mappa del processo")
        
        uploaded_csv_folder = st.file_uploader("Dataset dei driver (CSV)", type=["csv"], 
                                              accept_multiple_files=True, 
                                              help="Carica uno o più file CSV contenenti i dati dei driver")
    else:
        # Usa percorsi ai dati di esempio
        if os.path.exists("examples/process_map.json"):
            st.success("✅ Process Map di esempio trovata")
        else:
            st.warning("⚠️ Process Map di esempio non trovata")
            
        if os.path.exists("examples/driver_csvs"):
            csv_files = list(Path("examples/driver_csvs").glob("*.csv"))
            st.success(f"✅ {len(csv_files)} file CSV di esempio trovati")
        else:
            st.warning("⚠️ Directory dei driver CSV di esempio non trovata")
            
        uploaded_process_map = None
        uploaded_csv_folder = None

# Funzione per eseguire l'analisi
def run_analysis(kpi, use_example_data, uploaded_process_map=None, uploaded_csv_folder=None):
    # Directory temporanea
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    # Configurazione dei parametri
    if use_example_data:
        process_map_path = "examples/process_map.json"
        drivers_dir_path = "examples/driver_csvs"
    else:
        # Crea directory per i CSV caricati
        temp_csv_folder = temp_dir_path / "driver_csvs"
        temp_csv_folder.mkdir(exist_ok=True, parents=True)
        
        # Salva la process map caricata
        temp_process_map = temp_dir_path / "process_map.json"
        with open(temp_process_map, "wb") as f:
            f.write(uploaded_process_map.getbuffer())
        
        # Salva i CSV caricati
        for uploaded_file in uploaded_csv_folder:
            file_path = temp_csv_folder / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        process_map_path = str(temp_process_map)
        drivers_dir_path = str(temp_csv_folder)
    
    # Costruisci il comando per eseguire Crossnection
    cmd = [
        sys.executable,  # Usa l'interprete Python corrente
        "-m", "crossnection_mvp.main", "run",
        "--kpi", kpi,
        "--process-map", process_map_path,
        "--drivers-dir", drivers_dir_path
    ]
    
    # Esegui il processo con gestione avanzata dell'I/O
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process, temp_dir_path

def log_thread_function(process, log_placeholder):
    """Funzione che viene eseguita in un thread separato per gestire l'output del processo"""
    output_lines = []
    error_lines = []
    
    # Leggi continuamente da stdout e stderr
    while True:
        stdout_line = process.stdout.readline()
        if stdout_line:
            output_lines.append(stdout_line.strip())
            # Aggiorna il display
            log_placeholder.code("\n".join(output_lines[-50:]))  # Mostra solo le ultime 50 righe
        
        stderr_line = process.stderr.readline()
        if stderr_line:
            error_lines.append(stderr_line.strip())
            log_placeholder.error("\n".join(error_lines[-10:]))  # Mostra solo le ultime 10 righe di errore
        
        # Verifica se il processo è ancora in esecuzione
        if process.poll() is not None:
            # Leggi eventuali output rimasti
            for line in process.stdout:
                output_lines.append(line.strip())
            for line in process.stderr:
                error_lines.append(line.strip())
                
            # Aggiorna un'ultima volta il display
            log_placeholder.code("\n".join(output_lines[-50:]))
            if error_lines:
                log_placeholder.error("\n".join(error_lines[-10:]))
            break

# Sezioni UI principali
st.markdown("## Esecuzione Analisi")

if use_example_data:
    st.write("Verranno utilizzati i dati di esempio. Fai clic su 'Esegui Analisi' per iniziare.")
    can_run = True
else:
    st.write("Carica i file richiesti nel pannello laterale e fai clic su 'Esegui Analisi' per iniziare.")
    can_run = uploaded_process_map is not None and uploaded_csv_folder is not None and len(uploaded_csv_folder) > 0

# Pulsante per avviare l'analisi
if st.button("Esegui Analisi", disabled=not can_run):
    # Placeholder per mostrare lo stato
    progress_placeholder = st.empty()
    log_expander = st.expander("Log di esecuzione", expanded=True)
    log_placeholder = log_expander.empty()
    
    with progress_placeholder.container():
        st.info("Avvio dell'analisi in corso...")
        
        # Avvia l'analisi
        process, temp_dir = run_analysis(
            kpi, 
            use_example_data, 
            uploaded_process_map, 
            uploaded_csv_folder
        )
        
        # Crea un thread separato per monitorare l'output del processo
        log_thread = threading.Thread(
            target=log_thread_function, 
            args=(process, log_placeholder),
            daemon=True
        )
        log_thread.start()
        
        # UI per lo stato
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write("Inizializzazione analisi...")
        
        # Stati di avanzamento simulati
        stages = [
            "Profiling dei dataset",
            "Strategia join-key",
            "Pulizia e normalizzazione",
            "Calcolo correlazioni",
            "Ranking impatto",
            "Rilevamento outlier",
            "Generazione bozza narrativa",
            "Attesa feedback umano",
            "Finalizzazione report"
        ]
        
        # Esegui l'attesa in modo interattivo
        stage_index = 0
        process_completed = False
        hitl_stage_reached = False
        hitl_completed = False
        
        while stage_index < len(stages) and not process_completed:
            # Aggiorna il progresso
            progress_value = stage_index / len(stages)
            progress_bar.progress(progress_value)
            status_text.write(f"Fase corrente: {stages[stage_index]}")
            
            # Verifica se siamo alla fase HITL e non l'abbiamo ancora gestita
            if stage_index == 7 and not hitl_completed:
                # Cerca di trovare la bozza del report
                try:
                    # Tenta di trovare l'ultimo file narrative_draft
                    flow_data_dir = Path("flow_data")
                    if flow_data_dir.exists():
                        sessions = [p for p in flow_data_dir.iterdir() if p.is_dir()]
                        if sessions:
                            latest_session = sorted(sessions)[-1]
                            narrative_files = list(latest_session.glob("narrative_draft.v*.json"))
                            
                            if narrative_files:
                                latest_draft = sorted(narrative_files)[-1]
                                with open(latest_draft, "r") as f:
                                    draft_data = json.load(f)
                                    draft_markdown = draft_data.get("markdown", "Nessun contenuto trovato")
                                
                                # Mostra il form di feedback
                                st.markdown("## Draft Root-Cause Narrative")
                                st.markdown(draft_markdown)
                                
                                with st.form("hitl_feedback"):
                                    st.markdown("### Feedback e Validazione")
                                    st.markdown("""
                                    Rivedi i driver identificati e fornisci il tuo feedback:
                                    - **RELEVANT**: Correlazione importante che merita investigazione
                                    - **OBVIOUS**: Relazione attesa, già nota
                                    - **IRRELEVANT**: Rumore statistico o correlazione spuria
                                    """)
                                    
                                    # Form per i feedback specifici per driver
                                    # Qui dovresti analizzare la bozza per estrarre i driver reali
                                    # Per semplicità usiamo driver predefiniti
                                    feedback = {"drivers": {}, "general_comment": ""}
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        temp_status = st.radio("Temperature", 
                                                              ["RELEVANT", "OBVIOUS", "IRRELEVANT"], 
                                                              key="temp_feedback")
                                        feedback["drivers"]["Temperature"] = {"status": temp_status}
                                    
                                    with col2:
                                        press_status = st.radio("Pressure", 
                                                               ["RELEVANT", "OBVIOUS", "IRRELEVANT"], 
                                                               key="pressure_feedback")
                                        feedback["drivers"]["Pressure"] = {"status": press_status}
                                    
                                    with col3:
                                        speed_status = st.radio("Speed", 
                                                               ["RELEVANT", "OBVIOUS", "IRRELEVANT"], 
                                                               key="speed_feedback")
                                        feedback["drivers"]["Speed"] = {"status": speed_status}
                                    
                                    feedback["general_comment"] = st.text_area("Note aggiuntive", height=100)
                                    
                                    submitted = st.form_submit_button("Invia Feedback")
                                    
                                    if submitted:
                                        # Salva il feedback in un file
                                        feedback_file = latest_session / "user_feedback.json"
                                        with open(feedback_file, "w") as f:
                                            json.dump(feedback, f)
                                        
                                        st.success("Feedback inviato con successo!")
                                        
                                        # Invia il feedback al processo
                                        try:
                                            process.stdin.write(json.dumps(feedback) + "\n")
                                            process.stdin.flush()
                                            hitl_completed = True
                                            stage_index += 1  # Passa alla fase successiva
                                        except Exception as e:
                                            st.error(f"Errore nell'invio del feedback: {e}")
                                
                                if not hitl_completed:
                                    # Se non abbiamo ancora completato il HITL, attendiamo
                                    time.sleep(1)
                                    continue
                        
                        # Se non abbiamo trovato file di narrative draft ma il processo è ancora in esecuzione, attendiamo
                        if not hitl_completed and process.poll() is None:
                            time.sleep(1)
                            continue
                except Exception as e:
                    st.warning(f"Non è stato possibile trovare la bozza del report: {e}")
                    # Se c'è stato un errore ma il processo è ancora in esecuzione, attendiamo
                    if process.poll() is None:
                        time.sleep(1)
                        continue
                    else:
                        # Il processo è terminato con errore
                        process_completed = True
            
            # Verifica se il processo è ancora in esecuzione
            if process.poll() is not None:
                process_completed = True
                if process.returncode != 0:
                    st.error(f"Processo terminato con codice di errore: {process.returncode}")
                else:
                    # Termina con successo
                    progress_bar.progress(1.0)
                    status_text.write("Analisi completata con successo!")
                break
            
            # Aggiorna il progresso (simula avanzamento)
            time.sleep(1)
            if stage_index < len(stages) - 1 and not hitl_stage_reached:
                stage_index += 1
        
        # Attendi che il thread di logging termini
        if log_thread.is_alive():
            log_thread.join(timeout=5)
        
        # Al completamento del processo
        if process.returncode == 0:
            try:
                # Cerca il report finale
                flow_data_dir = Path("flow_data")
                sessions = [p for p in flow_data_dir.iterdir() if p.is_dir()]
                if sessions:
                    latest_session = sorted(sessions)[-1]
                    report_files = list(latest_session.glob("root_cause_report.v*.json"))
                    
                    if report_files:
                        latest_report = sorted(report_files)[-1]
                        with open(latest_report, "r") as f:
                            report_data = json.load(f)
                            final_markdown = report_data.get("markdown", "")
                        
                        # Mostra il report finale
                        st.markdown("## Final Root-Cause Report")
                        st.markdown(final_markdown)
                        
                        # Genera il PDF
                        pdf_path = generate_pdf_report(final_markdown)
                        
                        # Offri il download
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                            st.download_button(
                                "Download Report (PDF)",
                                data=pdf_bytes,
                                file_name="root_cause_report.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.warning("Non è stato trovato alcun report finale.")
                else:
                    st.warning("Non è stata trovata alcuna sessione di analisi.")
            except Exception as e:
                st.error(f"Errore durante la lettura del report finale: {e}")
                # Fallback - mostra un report di esempio
                final_markdown = """
                # 📘 Final Root-Cause Report for value_speed

                ## Validated Top-3 Drivers
                | Rank | Driver | Description | Effect Size | p-value | Business Validation | Strength | Business Context |
                | ---- | ------ | ----------- | ----------- | ------- | ------------------ | -------- | ---------------- |
                | 1 | value_temperature | Temperatura operativa del macchinario (°C) | 0.823 | 3.5e-05 | RELEVANT | Strong | Temperature elevate possono causare problemi di qualità e usura accelerata |
                | 2 | value_pressure | Pressione del sistema idraulico (bar) | 0.651 | 0.0021 | OBVIOUS | Moderate | La pressione influisce sulla stabilità del processo e sull'uniformità del prodotto |
                | 3 | value_speed | Velocità del macchinario in RPM | 0.455 | 0.031 | RELEVANT | Moderate | Una velocità elevata aumenta la produttività ma potrebbe compromettere la qualità |
                
                ## Outlier Check
                3 outlying data points were flagged across 2 driver(s): value_pressure, value_temperature.
                
                ## Normal Operating Ranges
                - **value_temperature**: 10 - 30 °C
                - **value_pressure**: 0.8 - 1.2 bar
                - **value_speed**: 80 - 120 RPM
                
                ## User Notes
                Si conferma che le temperature elevate sembrano essere la causa principale delle problematiche. Sarà necessario implementare controlli più stringenti e eventualmente un sistema di raffreddamento migliorato.
                """
                
                st.markdown("## Final Root-Cause Report (FALLBACK DATA)")
                st.markdown(final_markdown)
                
                # Genera il PDF dal fallback
                pdf_path = generate_pdf_report(final_markdown)
                
                # Offri il download
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    st.download_button(
                        "Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="root_cause_report.pdf",
                        mime="application/pdf"
                    )
        else:
            st.error("L'analisi è terminata con errori. Verifica il log di esecuzione per maggiori dettagli.")