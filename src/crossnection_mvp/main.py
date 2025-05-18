"""crossnection_mvp/main.py

Command‑line interface for the Crossnection Root‑Cause Discovery MVP.
Uses **Typer** (https://typer.tiangolo.com/) to expose four sub‑commands that
map directly to the helper methods of `CrossnectionMvpCrew`.

Usage examples:
    # Run full analysis
    crossnection run --kpi "First Pass Yield" --process-map ./inputs/process_map.json \
                    --drivers-dir ./inputs/driver_csvs

    # Train prompts / memory
    crossnection train

    # Run tests
    crossnection test

    # Replay a session ID obtained from logs
    crossnection replay --session-id 20250515T101500Z

The CLI is also invocable via:
    python -m crossnection_mvp.main run --kpi "..." --process-map ...
"""
from crossnection_mvp.utils.error_handling import with_robust_error_handling
# Importa e configura il logger OpenAI (aggiungi questa parte)
try:
    import openai
    
    # Crea la directory per i logs
    from pathlib import Path
    log_dir = Path("openai_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Importa e inizializza il logger
    from crossnection_mvp.utils.openai_logger import get_logger
    logger = get_logger()
    
    # Salva la funzione di chiamata originale
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        original_chat_completions_create = openai.chat.completions.create
        
        # Funzione di intercettazione
        def patched_chat_completions_create(*args, **kwargs):
            # Chiamata originale
            response = original_chat_completions_create(*args, **kwargs)
            
            # Log dell'utilizzo
            model = kwargs.get("model", "unknown")
            usage = getattr(response, "usage", None)
            if usage:
                logger.log_api_call(
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                    agent_name="unknown"  # Non abbiamo questa info qui
                )
            
            return response
        
        # Sostituisci la funzione originale con quella intercettata
        openai.chat.completions.create = patched_chat_completions_create
        
        print("[INFO] OpenAI API logger installed")
except Exception as e:
    print(f"[WARNING] Failed to install OpenAI API logger: {e}")
    
    
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.traceback import install as rich_install

from crossnection_mvp.crew import CrossnectionMvpCrew

# Pretty tracebacks for easier debugging
rich_install(show_locals=True)
console = Console()
app = typer.Typer(help="Crossnection Root‑Cause Discovery MVP – CLI")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

CREW = CrossnectionMvpCrew()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
@with_robust_error_handling(
    return_fallback=False,  # Rilancia l'eccezione per mostrare l'errore all'utente
    log_level="ERROR",
    stage_name="main_run"
)
def run(
    kpi: str = typer.Option(..., help="Target KPI to analyse (e.g., FPY)"),
    process_map: Path = typer.Option(
        ..., exists=True, readable=True, help="JSON file describing phases → sub‑phases → drivers"
    ),
    drivers_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, readable=True, help="Directory containing CSV files for each driver"
    ),
    metadata_file: Path = typer.Option(
        None, exists=True, readable=True, help="Optional JSON file with driver metadata"
    ),
):
    """Execute the full Root‑Cause Flow on user‑supplied data."""
    console.rule("[bold green]Crossnection – RUN")

    inputs = {
        "kpi": kpi,
        "process_map_file": str(process_map),
        "csv_folder": str(drivers_dir),
    }

    try:
        result = CREW.run(inputs)
        console.print("[bold green]✔ Analysis completed. Check output directory for the root‑cause report.")
        return result
    except Exception as exc:
        console.print(f"[bold red]✖ Error:[/bold red] {exc}")
        console.print(f"[yellow]Stack trace:[/yellow]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from exc


@app.command()
def train():
    """(Optional) Fine‑tune prompts / memory via CrewAI train()."""
    console.rule("[bold cyan]Crossnection – TRAIN")
    try:
        CREW.train()
        console.print("[bold green]✔ Training completed.")
    except Exception as exc:
        console.print(f"[bold red]✖ Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def test():
    """Run automated tests (CrewAI test or pytest wrappers)."""
    console.rule("[bold cyan]Crossnection – TEST")
    try:
        CREW.test()
        console.print("[bold green]✔ Tests passed.")
    except Exception as exc:
        console.print(f"[bold red]✖ Test failure:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def replay(session_id: Optional[str] = typer.Option(None, help="Session ID to replay")):
    """Replay a stored session for debugging or demonstration."""
    console.rule("[bold cyan]Crossnection – REPLAY")
    try:
        CREW.replay(session_id=session_id)
        console.print("[bold green]✔ Replay finished.")
    except Exception as exc:
        console.print(f"[bold red]✖ Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Debug entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app()
