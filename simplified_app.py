# simplified_app.py
import streamlit as st
import subprocess
import json
import os
import sys
import time
from pathlib import Path
import tempfile
import pandas as pd
from pdf_generator import generate_pdf_report

st.set_page_config(page_title="Crossnection - Root Cause Analysis", layout="wide")

# Header con logo e titolo
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=200)
else:
    st.title("Crossnection")
st.title("Root Cause Discovery")
st.markdown("---")

# Inizializza variabili di sessione se non esistono
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 'input'  # 'input', 'running', 'hitl', 'completed'
if 'process_output' not in st.session_state:
    st.session_state.process_output = []
if 'process_completed' not in st.session_state:
    st.session_state.process_completed = False
if 'hitl_submitted' not in st.session_state:
    st.session_state.hitl_submitted = False
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# Funzione per avanzare allo stage successivo
def advance_stage(new_stage):
    st.session_state.current_stage = new_stage
    st.rerun()  # Versione aggiornata di experimental_rerun

# Funzione per eseguire l'analisi
def run_analysis(kpi, process_map_path, drivers_dir_path):
    # Costruisci il comando
    cmd = [
        sys.executable,  # Usa l'interprete Python corrente
        "-m", "crossnection_mvp.main", "run",
        "--kpi", kpi,
        "--process-map", process_map_path,
        "--drivers-dir", drivers_dir_path
    ]
    
    # Esegui il processo
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Raccogli output
        output_lines = []
        error_lines = []
        
        while process.poll() is None:
            # Leggi stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                output_lines.append(stdout_line.strip())
                st.session_state.process_output = output_lines[-50:]  # Keep only last 50 lines
            
            # Leggi stderr
            stderr_line = process.stderr.readline()
            if stderr_line:
                error_lines.append(stderr_line.strip())
            
            # Aggiorniamo la UI ogni pochi secondi
            time.sleep(0.1)
            
        # Leggi eventuali output rimasti
        for line in process.stdout:
            output_lines.append(line.strip())
            st.session_state.process_output = output_lines[-50:]
        
        for line in process.stderr:
            error_lines.append(line.strip())
        
        # Imposta lo stato completato
        st.session_state.process_completed = True
        
        # Gestisci errori
        if process.returncode != 0:
            st.session_state.error_message = "\n".join(error_lines[-10:])
            return False
        
        return True
    except Exception as e:
        st.session_state.error_message = str(e)
        return False

def get_mock_draft_narrative():
    """Return a mock draft narrative for testing."""
    return """
# 📊 Draft Root-Cause Narrative for value_speed

## Top-3 Influencing Drivers

| Rank | Driver | Description | Effect Size | p-value | Strength | Normal Range | Business Context |
| ---- | ------ | ----------- | ----------- | ------- | -------- | ------------ | ---------------- |
| 1 | value_temperature | Temperatura operativa del macchinario | 0.823 | 3.5e-05 | Strong | 10 - 30 °C | Temperature elevate possono causare problemi di qualità e usura accelerata |
| 2 | value_pressure | Pressione del sistema idraulico | 0.651 | 0.0021 | Moderate | 0.8 - 1.2 bar | La pressione influisce sulla stabilità del processo e sull'uniformità del prodotto |
| 3 | value_speed | Velocità del macchinario in RPM | 0.455 | 0.031 | Moderate | 80 - 120 RPM | Una velocità elevata aumenta la produttività ma potrebbe compromettere la qualità |

## Outlier Check

3 outlying data points were flagged across 2 driver(s): value_pressure, value_temperature.

## Nota sull'interpretazione statistica

I risultati presentati mostrano **correlazioni**, non necessariamente **causalità**. Un valore di effect size alto indica una forte relazione tra driver e KPI, mentre un p-value basso (<0.05) indica che tale relazione è statisticamente significativa.

## Validation Instructions

Please mark each driver as **RELEVANT**, **OBVIOUS**, or **IRRELEVANT** and add comments if needed.
"""

def get_mock_final_report():
    """Return a mock final report for testing."""
    return """
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

def find_narrative_draft():
    """Tenta di trovare la bozza del report creata da Crossnection."""
    try:
        # Cerca nella directory flow_data
        flow_data_dir = Path("flow_data")
        if flow_data_dir.exists():
            sessions = [p for p in flow_data_dir.iterdir() if p.is_dir()]
            if sessions:
                latest_session = sorted(sessions)[-1]
                draft_files = list(latest_session.glob("narrative_draft.v*.json"))
                
                if draft_files:
                    latest_draft = sorted(draft_files)[-1]
                    with open(latest_draft, "r") as f:
                        draft_data = json.load(f)
                        return draft_data.get("markdown", None), latest_session
        
        return None, None
    except Exception as e:
        st.error(f"Error finding narrative draft: {e}")
        return None, None

def find_final_report():
    """Tenta di trovare il report finale creato da Crossnection."""
    try:
        # Cerca nella directory flow_data
        flow_data_dir = Path("flow_data")
        if flow_data_dir.exists():
            sessions = [p for p in flow_data_dir.iterdir() if p.is_dir()]
            if sessions:
                latest_session = sorted(sessions)[-1]
                report_files = list(latest_session.glob("root_cause_report.v*.json"))
                
                if report_files:
                    latest_report = sorted(report_files)[-1]
                    with open(latest_report, "r") as f:
                        report_data = json.load(f)
                        return report_data.get("markdown", None)
        
        return None
    except Exception as e:
        st.error(f"Error finding final report: {e}")
        return None

# Sidebar per i parametri di input
with st.sidebar:
    st.header("Parametri di Analisi")
    
    # Input form - visibile solo nella fase di input
    if st.session_state.current_stage == 'input':
        kpi = st.selectbox("KPI da analizzare", ["value_speed", "value_pressure", "value_temperature"])
        
        # Opzione per usare dati di esempio
        use_example_data = st.checkbox("Usa dati di esempio", value=True)
        
        if not use_example_data:
            process_map_file = st.file_uploader("Process Map (JSON)", type=["json"])
            driver_files = st.file_uploader("Driver CSV Files", type=["csv"], accept_multiple_files=True)
            
            can_run = process_map_file is not None and driver_files and len(driver_files) > 0
        else:
            # Verifica disponibilità dati di esempio
            if not os.path.exists("examples/process_map.json"):
                st.warning("⚠️ Process Map di esempio non trovata in examples/process_map.json")
                can_run = False
            elif not os.path.exists("examples/driver_csvs"):
                st.warning("⚠️ Directory driver_csvs di esempio non trovata in examples/")
                can_run = False
            else:
                csv_files = list(Path("examples/driver_csvs").glob("*.csv"))
                if not csv_files:
                    st.warning("⚠️ Nessun file CSV trovato in examples/driver_csvs/")
                    can_run = False
                else:
                    st.success(f"✅ Trovati {len(csv_files)} file CSV di esempio")
                    can_run = True
            
            # Placeholder per uniformità dell'interfaccia
            process_map_file = None
            driver_files = None
        
        if st.button("Esegui Analisi", disabled=not can_run):
            # Setup temporaneo se necessario
            if not use_example_data:
                # Crea directory temporanea
                temp_dir = tempfile.mkdtemp()
                temp_dir_path = Path(temp_dir)
                
                # Salva i file
                temp_map_path = temp_dir_path / "process_map.json"
                with open(temp_map_path, "wb") as f:
                    f.write(process_map_file.getbuffer())
                
                temp_csv_dir = temp_dir_path / "driver_csvs"
                temp_csv_dir.mkdir()
                
                for file in driver_files:
                    file_path = temp_csv_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                process_map_path = str(temp_map_path)
                drivers_dir_path = str(temp_csv_dir)
                
                # Salva il percorso temporaneo in session_state
                st.session_state.temp_dir = temp_dir
            else:
                # Usa i percorsi di esempio
                process_map_path = "examples/process_map.json"
                drivers_dir_path = "examples/driver_csvs"
            
            # Salva parametri in session_state
            st.session_state.kpi = kpi
            st.session_state.process_map_path = process_map_path
            st.session_state.drivers_dir_path = drivers_dir_path
            st.session_state.use_example_data = use_example_data
            
            # Passa allo stato running
            advance_stage('running')
    
    # Visualizzazione parametri - visibile in tutte le fasi successive
    else:
        st.info(f"KPI: {st.session_state.get('kpi', 'value_speed')}")
        if st.session_state.get('use_example_data', True):
            st.info("Dati: Esempi predefiniti")
        else:
            st.info(f"Process Map: {Path(st.session_state.process_map_path).name}")
            st.info(f"Driver Dir: {Path(st.session_state.drivers_dir_path).name}")
        
        # Pulsante per ricominciare
        if st.button("Nuova Analisi"):
            # Pulisci la sessione
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_stage = 'input'
            st.rerun()

# Contenuto principale basato sullo stage corrente
if st.session_state.current_stage == 'input':
    st.markdown("## Configurazione Analisi")
    st.write("Configura i parametri nel pannello laterale e clicca 'Esegui Analisi' per iniziare.")
    
    # Mostra una preview dei dati di esempio se selezionati
    if 'use_example_data' in st.session_state and st.session_state.use_example_data:
        with st.expander("👁️ Preview dei dati di esempio"):
            try:
                # Process Map
                process_map_path = "examples/process_map.json"
                if os.path.exists(process_map_path):
                    with open(process_map_path, "r") as f:
                        process_map = json.load(f)
                    st.json(process_map)
                
                # CSV Files
                csv_dir = Path("examples/driver_csvs")
                if csv_dir.exists():
                    csv_files = list(csv_dir.glob("*.csv"))
                    for file in csv_files[:3]:  # Mostra solo i primi 3 file
                        st.subheader(f"File: {file.name}")
                        df = pd.read_csv(file)
                        st.dataframe(df.head())
            except Exception as e:
                st.error(f"Errore nel caricamento dei dati di esempio: {e}")

elif st.session_state.current_stage == 'running':
    st.markdown("## Analisi in corso")
    
    # Placeholder per la barra di progresso
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    log_expander = st.expander("Log di esecuzione", expanded=True)
    
    # Avvia l'analisi in un container di progresso
    with progress_placeholder.container():
        # Mostra barra di progresso
        progress_bar = st.progress(0)
        
        # Stati di avanzamento simulati
        stages = [
            "Profiling dei dataset",
            "Strategia join-key",
            "Pulizia e normalizzazione",
            "Calcolo correlazioni",
            "Ranking impatto",
            "Rilevamento outlier",
            "Generazione bozza narrativa"
        ]
        
        # Esegui l'analisi e mostra progresso
        success = run_analysis(
            st.session_state.kpi,
            st.session_state.process_map_path,
            st.session_state.drivers_dir_path
        )
        
        # Nella finestra del log, mostra l'output del processo
        with log_expander:
            st.code("\n".join(st.session_state.process_output))
        
        # Simulazione del progresso mentre il processo è in esecuzione
        stage_index = 0
        while not st.session_state.process_completed and stage_index < len(stages):
            # Aggiorna barra di progresso
            progress = stage_index / len(stages)
            progress_bar.progress(progress)
            status_placeholder.write(f"Fase: {stages[stage_index]}")
            
            # Attendiamo un po' prima di avanzare
            time.sleep(1)
            stage_index = min(stage_index + 1, len(stages) - 1)
        
        # Aggiorna barra alla fine
        progress_bar.progress(1.0)
        
        # Verifica se ci sono stati errori
        if st.session_state.error_message:
            status_placeholder.error(f"Errore: {st.session_state.error_message}")
            # Mostra opzione per riprovare
            if st.button("Riprova"):
                # Reset stati di errore
                st.session_state.error_message = None
                st.session_state.process_completed = False
                st.rerun()
        else:
            status_placeholder.success("Analisi completata!")
            
            # Tenta di trovare il report draft
            draft_markdown, session_dir = find_narrative_draft()
            
            if draft_markdown and session_dir:
                # Passa alla fase HITL
                st.session_state.draft_markdown = draft_markdown
                st.session_state.session_dir = session_dir
                advance_stage('hitl')
            else:
                # Fallback a una bozza simulata
                st.session_state.draft_markdown = get_mock_draft_narrative()
                advance_stage('hitl')

elif st.session_state.current_stage == 'hitl':
    st.markdown("## Human-in-the-Loop Validation")
    
    # Mostra la bozza del report
    st.markdown(st.session_state.draft_markdown)
    
    # Form per il feedback
    with st.form("hitl_feedback"):
        st.markdown("### Feedback sui Driver")
        st.markdown("""
        Rivedi i driver identificati e fornisci il tuo feedback:
        - **RELEVANT**: Correlazione importante che merita investigazione
        - **OBVIOUS**: Relazione attesa, già nota
        - **IRRELEVANT**: Rumore statistico o correlazione spuria
        """)
        
        # Form per i feedback specifici per driver
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
            try:
                # Salva il feedback in un file
                if hasattr(st.session_state, 'session_dir') and st.session_state.session_dir:
                    # Usa il percorso della sessione reale
                    feedback_file = st.session_state.session_dir / "user_feedback.json"
                    with open(feedback_file, "w") as f:
                        json.dump(feedback, f)
                else:
                    # Fallback
                    feedback_file = Path("flow_data") / "user_feedback.json"
                    os.makedirs(feedback_file.parent, exist_ok=True)
                    with open(feedback_file, "w") as f:
                        json.dump(feedback, f)
                
                st.success("Feedback inviato con successo!")
                
                # Salva il feedback in session_state
                st.session_state.feedback = feedback
                st.session_state.hitl_submitted = True
                
                # Simuliamo il completamento del processo
                time.sleep(2)
                
                # Cerca report finale o usa mock
                final_report = find_final_report()
                if final_report:
                    st.session_state.final_report = final_report
                else:
                    st.session_state.final_report = get_mock_final_report()
                
                # Avanza alla fase completata
                advance_stage('completed')
            except Exception as e:
                st.error(f"Errore nell'invio del feedback: {e}")

elif st.session_state.current_stage == 'completed':
    st.markdown("## Final Root-Cause Report")
    
    # Mostra il report finale
    st.markdown(st.session_state.final_report)
    
    # Genera il PDF
    try:
        pdf_path = generate_pdf_report(st.session_state.final_report)
        
        # Offri il download
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                "Download Report (PDF)",
                data=pdf_bytes,
                file_name="root_cause_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Errore nella generazione del PDF: {e}")
        
    # Mostra anche la panoramica del feedback
    if hasattr(st.session_state, 'feedback'):
        with st.expander("📋 Riepilogo Feedback"):
            st.json(st.session_state.feedback)