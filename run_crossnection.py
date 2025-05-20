# run_crossnection.py
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
import time
from typing import List, Dict, Any
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from pdf_generator import generate_pdf_report

console = Console()

def run_crossnection(kpi: str, process_map: str, drivers_dir: str):
    """Esegue Crossnection con UI migliorata da terminale e genera PDF finale."""
    
    # Crea la directory di output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Genera un identificatore unico per questo lancio (timestamp)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Titolo
    console.print(Panel.fit(
        f"[bold blue]Crossnection[/bold blue] - [cyan]Root-Cause Discovery[/cyan] - Run ID: {run_id}", 
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
        sys.executable,  # Usa l'interprete Python corrente
        "-m", "crossnection_mvp.main", "run",
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
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        stdin=subprocess.PIPE,
        text=True, 
        bufsize=1
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
            # Mostra la linea grezza per debugging se necessario
            # console.print(f"DEBUG: {line.strip()}")
            
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
                        draft_markdown = "Bozza non trovata."
                except Exception as e:
                    console.print(f"[red]Error loading draft narrative: {e}[/red]")
                    draft_markdown = "Errore nel caricamento della bozza."
                
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
                feedback_file = latest_session / "user_feedback.json"
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
            
        # Chiudi l'ultimo task
        if current_task is not None:
            progress.update(current_task, completed=100)
        progress.update(overall_task, completed=len(tasks))
    
    # Verifica se il processo è terminato correttamente
    if process.returncode is None:
        process.wait()
    
    if process.returncode != 0:
        console.print(f"[red]Process terminated with error code: {process.returncode}[/red]")
        return
    
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
            console.print("[yellow]No final report found.[/yellow]")
            return
    except Exception as e:
        console.print(f"[red]Error loading final report: {e}[/red]")
        return
    
    # Final report
    console.print("\n")
    console.print(Panel(
        Markdown(final_markdown), 
        title="[bold]Final Root-Cause Report[/bold]",
        border_style="green", 
        width=100
    ))
    
    # Genera il PDF con nome univoco
    pdf_filename = f"root_cause_report_{kpi}_{run_id}.pdf"
    pdf_path = output_dir / pdf_filename
    
    try:
        pdf_path = generate_pdf_report(final_markdown, str(pdf_path))
        
        # Success message
        console.print(f"\n[bold green]Analysis complete![/bold green] Report saved to [cyan]{pdf_path}[/cyan]")
        console.print("You can now review the report or run another analysis.")
    except Exception as e:
        console.print(f"[red]Error generating PDF: {e}[/red]")

# Comando principale per eseguire lo script da CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Crossnection with enhanced UI")
    parser.add_argument("--kpi", default="value_speed", help="Target KPI for analysis")
    parser.add_argument("--process-map", default="examples/process_map.json", help="Path to process map JSON file")
    parser.add_argument("--drivers-dir", default="examples/driver_csvs", help="Path to directory with driver CSV files")
    
    args = parser.parse_args()
    run_crossnection(args.kpi, args.process_map, args.drivers_dir)