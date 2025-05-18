"""crossnection_mvp/flows/root_cause_flow.py

Definition of **RootCauseFlow** – the linear Flow that orchestrates all
three agents (Data, Stats, Explain) for the Crossnection MVP.

The Flow is intentionally minimal: it simply enumerates the eight tasks in
order, relying on CrewAI’s YAML loader to resolve agent bindings,
`input_key` / `output_key` hand‑offs and the `human_input` pause in the
Explain stage.

If you change *tasks.yaml* names or their sequence, update the lists below.
"""
from __future__ import annotations

from typing import List

import crewai as cr

# ---------------------------------------------------------------------------
# Helper – ordered lists of task identifiers as declared in tasks.yaml
# ---------------------------------------------------------------------------

DATA_STAGE_TASKS: List[str] = [
    "profile_validate_dataset",
    "join_key_strategy",
    "clean_normalize_dataset",
]

STATS_STAGE_TASKS: List[str] = [
    ["compute_correlations", "detect_outliers"],  # Esecuzione parallela
    "rank_impact",  # Dipende da compute_correlations, quindi rimane sequenziale
]

EXPLAIN_STAGE_TASKS: List[str] = [
    "draft_root_cause_narrative",
    "finalize_root_cause_report",
]


def build_flow() -> cr.Flow:  # noqa: D401 – simple builder
    """Return a *cr.Flow* instance wired with the ordered stages."""

    return cr.Flow(
        name="RootCauseFlow",
        description="Linear three‑stage Flow that produces a human‑validated root‑cause report.",
        stages=[
            cr.Stage(name="data_stage", tasks=DATA_STAGE_TASKS),
            cr.Stage(name="stats_stage", tasks=STATS_STAGE_TASKS),
            cr.Stage(name="explain_stage", tasks=EXPLAIN_STAGE_TASKS),
        ],
    )


# ---------------------------------------------------------------------------
# Module‑level constant so that `crew.py` can reference it by name.
# ---------------------------------------------------------------------------

root_cause_flow: cr.Flow = build_flow()

__all__ = ["root_cause_flow", "build_flow"]
