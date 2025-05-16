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
def run(
    kpi: str = typer.Option(..., help="Target KPI to analyse (e.g., FPY)"),
    process_map: Path = typer.Option(
        ..., exists=True, readable=True, help="JSON file describing phases → sub‑phases → drivers"
    ),
    drivers_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, readable=True, help="Directory containing CSV files for each driver"
    ),
):
    """Execute the full Root‑Cause Flow on user‑supplied data."""
    console.rule("[bold green]Crossnection – RUN")

    inputs = {
        "kpi": kpi,
        "process_map_file": str(process_map),
        "drivers_folder": str(drivers_dir),
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
