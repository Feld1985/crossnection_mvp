# Crossnection ‑ Root‑Cause Discovery MVP

## Executive Summary

Crossnection è un **motore di Root‑Cause Discovery** progettato per aiutare team operativi, analyst e continuous‑improvement manager a **identificare velocemente le cause reali** che determinano anomalie o scostamenti rispetto ai KPI di processo.
L’MVP si basa su un’architettura **multi‑agent** implementata con CrewAI:

* **DataAgent** integra e normalizza dataset eterogenei caricati dall’utente, individuando o generando automaticamente una **Join‑Key univoca** che renda relazionabili tutti i driver numerici raccolti lungo le fasi di processo.
* **StatsAgent** applica analisi statistiche multivariate (correlazioni, test d’ipotesi, ranking d’impatto) per mappare, in termini quantitativi, il legame tra driver e KPI.
* **ExplainAgent** traduce i risultati numerici in **narrazioni intelligibili** per il business e, grazie ad una fase "Human‑in‑the‑Loop", chiede all’utente di convalidare il **senso economico** degli insight prima di produrre il report finale.

Questo approccio riduce drasticamente il *time‑to‑insight*: l’utente carica dati grezzi, seleziona il KPI, conferma/filtra le correlazioni proposte e ottiene un **Root‑Cause Report** prioritizzato, pronto per l’azione.

---

## Table of Contents

1. [Value Proposition](#value-proposition)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Agent & Task Details](#agent--task-details)
6. [Flow Lifecycle](#flow-lifecycle)
7. [Human‑in‑the‑Loop](#human-in-the-loop)
8. [Custom Tools](#custom-tools)
9. [User Inputs](#user-inputs)
10. [Roadmap](#roadmap)
11. [License](#license)

---

## Value Proposition

> **“From raw process data to validated root‑causes in hours, not weeks.”**

* Elimina la dipendenza da data‑science specialistici nelle fasi preliminari di analisi.
* Centralizza dataset disparati grazie al rilevamento intelligente della Join‑Key.
* Combina solidi metodi statistici con la convalida di buon senso del team di business.
* Restituisce report azionabili che alimentano cicli di miglioramento continuo.

---

## Architecture Overview

```
User Input ─► DataAgent ─► StatsAgent ─► ExplainAgent(draft) ─► User ✓ ─► ExplainAgent(final)
                                ▲                                         │
                                └────────────── Flow State ────────────────┘
```

| Agent            | Core Tasks                                                                 | Output                              |
| ---------------- | -------------------------------------------------------------------------- | ----------------------------------- |
| **DataAgent**    | 1. Profile & Validate  2. Discover/Generate Join‑Key  3. Clean & Normalize | Integrated dataset + profile report |
| **StatsAgent**   | 1. Compute Correlations  2. Rank Impact  3. Detect Outliers                | Impact matrix + statistics          |
| **ExplainAgent** | 1. Draft Narrative (`human_input`)  2. Final Narrative                     | Root‑Cause Report (Markdown/PDF)    |

The agents are orchestrated by **RootCauseFlow**, a linear Flow consisting of three stages—`data_stage`, `stats_stage`, `explain_stage`—with context passed automatically between steps.

---

## Project Structure

```
crossnection/
├── pyproject.toml            # or requirements.txt
├── .env                      # API keys & secrets
├── README.md                 # You are here
└── src/
    └── crossnection/
        ├── __init__.py
        ├── main.py           # CLI entrypoint
        ├── config/
        │   ├── agents.yaml   # DataAgent, StatsAgent, ExplainAgent
        │   └── tasks.yaml    # 8 atomic tasks with human_input flag
        ├── flows/
        │   └── root_cause_flow.py
        └── tools/
            ├── cross_data_profiler.py
            ├── cross_stat_engine.py
            └── cross_insight_formatter.py
```

## Agent & Task Details

### DataAgent

* **Profile & Validate CSV** – schema check, missing values, outlier scan.
* **Discover / Generate Join‑Key** – automatic detection; proposes composite or fuzzy matching if needed.
* **Clean & Normalize** – scaling, date alignment, type coercion.

### StatsAgent

* **Compute Correlations & Significance** – Pearson, Spearman, χ², ANOVA.
* **Rank Impact** – effect size & p‑value blended score.
* **Detect Outliers** – Z‑score & IQR methods.

### ExplainAgent

* **Draft Narrative** – generates preliminary insights and correlational reasoning; flagged with `human_input: true`.
* **Final Narrative** – integrates user feedback; outputs Markdown and optional PDF.

---

## Flow Lifecycle

1. **Data Stage** – returns validated dataset and profile.
2. **Stats Stage** – processes dataset, yields impact matrix.
3. **Explain Stage** – produces draft; waits for **business confirmation**; emits final report.

Error handling and retries are managed per‑stage; any blocking data issue loops back to the user with clear remediation steps.

---

## Human‑in‑the‑Loop

The draft output of ExplainAgent is surfaced via CrewAI’s `human_input` mechanism.
You can:

* Mark correlations as *relevant*, *obvious*, or *irrelevant*.
* Add domain annotations or suspected lurking variables.
* Trigger a re‑run of the narrative generation.

The workflow resumes only after explicit validation.

---

## Custom Tools

| Tool                          | Purpose                                 | Dependencies                   |
| ----------------------------- | --------------------------------------- | ------------------------------ |
| **CrossDataProfilerTool**     | Profiling, join‑key discovery, cleaning | `pandas`, `great_expectations` |
| **CrossStatEngineTool**       | Statistical pipeline (corr, ranking)    | `scipy`, `statsmodels`         |
| **CrossInsightFormatterTool** | Prompt templates, Markdown rendering    | OpenAI LLM, `markdown2`        |

---

## User Inputs

1. **Problem / KPI**
2. **Process Map** – phases (A, B, C…) and sub‑phases (1, 2, 3…)
3. **Driver Assignment** – numeric indicators mapped to sub‑phases
4. **Driver Description** – plain‑language meaning of each driver
5. **CSV Upload** – one file per driver containing:

   * join‑key (if present)
   * optional timestamp
   * driver value

---

## Roadmap

* [ ] Add visualization layer (charts) after ExplainAgent.
* [ ] Support time‑series forecasting for proactive alerting.
* [ ] Integrate knowledge base for industry‑specific heuristics.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.