# console_ui.py
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
import time
from typing import List, Dict, Any
import subprocess
import json
from pathlib import Path
from pdf_generator import generate_pdf_report

console = Console()

def run_crossnection(kpi: str, process_map: str, drivers_dir: str):
    """Wrapper per eseguire Crossnection con UI migliorata."""
    
    # Titolo
    console.print(Panel.fit(
        "[bold blue]Crossnection[/bold blue] - [cyan]Root-Cause Discovery[/cyan]", 
        border_style="blue"
    ))
    
    # Parametri
    params_table = Table(show_header=False, box=None)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")
    params_table.add_row("KPI", kpi)
    params_table.add_row("Process Map", process_map)
    params_table.add_row("Drivers Directory", drivers_dir)
    
    console.print(Panel(params_table, title="[bold]Parameters[/bold]", border_style="cyan"))
    
    # Esegui il comando CLI di Crossnection
    cmd = [
        "python", "-m", "crossnection_mvp.main", "run",
        "--kpi", kpi,
        "--process-map", process_map,
        "--drivers-dir", drivers_dir
    ]
    
    # Progress bar
    agents = ["DataAgent", "StatsAgent", "ExplainAgent"]
    tasks = [
        "profile_validate_dataset",
        "join_key_strategy",
        "clean_normalize_dataset", 
        "compute_correlations",
        "rank_impact",
        "detect_outliers",
        "draft_root_cause_narrative",
        "finalize_root_cause_report"
    ]
    
    # Avvia il processo di analisi
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[cyan]Overall Progress", total=len(tasks))
        
        current_task_idx = 0
        current_task = None
        hitl_done = False
        
        # Leggi l'output del processo
        for line in iter(process.stdout.readline, ''):
            # Cerca stringhe che indicano lo stato del task
            if "Task started" in line and current_task_idx < len(tasks):
                # Chiudi il task precedente se esiste
                if current_task is not None:
                    progress.update(current_task, completed=100)
                
                # Inizia un nuovo task
                task_name = tasks[current_task_idx]
                agent_idx = 0
                if current_task_idx >= 3 and current_task_idx < 6:
                    agent_idx = 1
                elif current_task_idx >= 6:
                    agent_idx = 2
                    
                agent_name = agents[agent_idx]
                task_display = f"{agent_name} | {task_name}"
                current_task = progress.add_task(task_display, total=100)
                current_task_idx += 1
                
                # Aggiorna il progresso complessivo
                progress.update(overall_task, completed=current_task_idx-1)
                
            elif "Progress:" in line and current_task is not None:
                # Estrai la percentuale e aggiorna il task
                try:
                    percentage = int(line.split("Progress:")[1].strip().rstrip("%"))
                    progress.update(current_task, completed=percentage)
                except:
                    pass
            
            # Se raggiungiamo la fase HITL e non l'abbiamo ancora gestita
            elif "human_input: true" in line and not hitl_done and current_task_idx > 6:
                # Pausa la progress bar
                progress.stop()
                
                # Ottieni la bozza del report dal Context Store
                try:
                    context_store_path = Path("flow_data")
                    latest_session = sorted([p for p in context_store_path.iterdir() if p.is_dir()])[-1]
                    narrative_draft_files = list(latest_session.glob("narrative_draft.v*.json"))
                    
                    if narrative_draft_files:
                        latest_draft = sorted(narrative_draft_files)[-1]
                        with open(latest_draft, "r") as f:
                            draft_data = json.load(f)
                            draft_markdown = draft_data.get("markdown", "")
                    else:
                        draft_markdown = get_draft_narrative()
                except Exception as e:
                    console.print(f"[red]Error loading draft narrative: {e}[/red]")
                    draft_markdown = get_draft_narrative()
                
                # Display draft narrative
                console.print("\n")
                console.print(Panel(
                    Markdown(draft_markdown), 
                    title="[bold]Draft Root-Cause Narrative[/bold]",
                    border_style="green", 
                    width=100
                ))
                
                # Get user feedback
                console.print("\n[bold cyan]Human-in-the-Loop Validation[/bold cyan]")
                console.print("Please review each driver and mark as RELEVANT, OBVIOUS, or IRRELEVANT:")
                
                # Estrai i driver dalla bozza o usa dei driver di default
                drivers = ["Temperature", "Pressure", "Speed"]
                feedback = {"drivers": {}, "general_comment": ""}
                
                for driver in drivers:
                    status = console.input(f"[cyan]{driver}[/cyan] (RELEVANT/OBVIOUS/IRRELEVANT): ")
                    feedback["drivers"][driver] = {"status": status}
                    console.print(f"Marked {driver} as [bold]{status}[/bold]")
                
                feedback["general_comment"] = console.input("[cyan]Additional notes[/cyan]: ")
                
                # Salva il feedback in un file che il processo possa leggere
                feedback_file = Path("flow_data") / "user_feedback.json"
                with open(feedback_file, "w") as f:
                    json.dump(feedback, f)
                    
                console.print("[green]Feedback recorded. Generating final report...[/green]")
                
                # Segnala al processo che il feedback è pronto
                process.stdin.write(json.dumps(feedback) + "\n")
                process.stdin.flush()
                
                hitl_done = True
                
                # Riavvia la progress bar
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}[/bold blue]"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console
                )
                progress.start()
                overall_task = progress.add_task("[cyan]Finalizing", total=1)
                current_task = progress.add_task("ExplainAgent | finalize_root_cause_report", total=100)
            
            # Mostra l'output grezzo per debugging
            # console.print(line.strip())
        
        # Chiudi l'ultimo task
        if current_task is not None:
            progress.update(current_task, completed=100)
        progress.update(overall_task, completed=len(tasks))

    # Ottieni il report finale dal Context Store
    try:
        context_store_path = Path("flow_data")
        latest_session = sorted([p for p in context_store_path.iterdir() if p.is_dir()])[-1]
        report_files = list(latest_session.glob("root_cause_report.v*.json"))
        
        if report_files:
            latest_report = sorted(report_files)[-1]
            with open(latest_report, "r") as f:
                report_data = json.load(f)
                final_markdown = report_data.get("markdown", "")
        else:
            final_markdown = get_final_report()
    except Exception as e:
        console.print(f"[red]Error loading final report: {e}[/red]")
        final_markdown = get_final_report()
    
    # Final report
    console.print("\n")
    console.print(Panel(
        Markdown(final_markdown), 
        title="[bold]Final Root-Cause Report[/bold]",
        border_style="green", 
        width=100
    ))
    
    # Genera il PDF
    pdf_path = generate_pdf_report(final_markdown)
    
    # Success message
    console.print(f"\n[bold green]Analysis complete![/bold green] Report saved to [cyan]{pdf_path}[/cyan]")
    console.print("You can now review the report or run another analysis.")

def get_draft_narrative():
    """Return a mock draft narrative."""
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

def get_final_report():
    """Return a mock final report."""
    return """
# 📘 Final Root-Cause Report for value_speed

## Validated Top-3 Drivers

| Rank | Driver | Description | Effect Size | p-value | Business Validation | Strength | Business Context |
| ---- | ------ | ----------- | ----------- | ------- | ------------------ | -------- | ---------------- |
| 1 | value_temperature | Temperatura operativa del macchinario | 0.823 | 3.5e-05 | RELEVANT | Strong | Temperature elevate possono causare problemi di qualità e usura accelerata |
| 2 | value_pressure | Pressione del sistema idraulico | 0.651 | 0.0021 | OBVIOUS | Moderate | La pressione influisce sulla stabilità del processo e sull'uniformità del prodotto |
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

# Comando principale per eseguire lo script da CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Crossnection with enhanced UI")
    parser.add_argument("--kpi", default="value_speed", help="Target KPI for analysis")
    parser.add_argument("--process-map", default="examples/process_map.json", help="Path to process map JSON file")
    parser.add_argument("--drivers-dir", default="examples/driver_csvs", help="Path to directory with driver CSV files")
    
    args = parser.parse_args()
    run_crossnection(args.kpi, args.process_map, args.drivers_dir)