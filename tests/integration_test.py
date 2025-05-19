"""Test di integrazione per verificare il flusso Crossnection."""

import sys
import logging
from pathlib import Path

# Aggiungi il percorso src alla path di Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from crossnection_mvp.agents.data_agent import DataAgent
from crossnection_mvp.agents.stats_agent import StatsAgent
from crossnection_mvp.agents.explain_agent import ExplainAgent
from crossnection_mvp.utils.debug_helpers import log_structure, inspect_context_store
from crossnection_mvp.utils.context_store import ContextStore

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Testa l'intero pipeline di Crossnection usando gli agenti direttamente."""
    logger.info("Starting integration test")
    
    # Inizializza il Context Store
    store = ContextStore.get_instance(base_dir="test_flow_context")
    
    # Fase 1: DataAgent
    logger.info("=== STEP 1: DataAgent ===")
    data_agent = DataAgent()
    csv_dir = Path("examples/driver_csvs")
    unified_df, data_report = data_agent.run_data_pipeline(csv_dir, kpi="value_speed")
    
    logger.info(f"Unified dataset shape: {unified_df.shape}")
    log_structure(data_report, name="data_report")
    
    # Fase 2: StatsAgent
    logger.info("=== STEP 2: StatsAgent ===")
    stats_agent = StatsAgent()
    corr, ranking, outliers = stats_agent.run_stats_pipeline(
        unified_df,
        kpi="value_speed",
    )
    
    logger.info(f"Correlation matrix: {len(corr)} drivers")
    logger.info(f"Impact ranking: {len(ranking)} drivers")
    logger.info(f"Outlier report: {len(outliers.get('outliers', []))} outliers")
    
    # Fase 3: ExplainAgent
    logger.info("=== STEP 3: ExplainAgent ===")
    explain_agent = ExplainAgent()
    
    # Verifica che i dati siano nel Context Store
    inspect_context_store("impact_ranking")
    inspect_context_store("outlier_report")
    
    # Genera la bozza
    draft_result = explain_agent.draft_root_cause_narrative()
    
    logger.info("Draft narrative generated")
    if isinstance(draft_result, dict) and "markdown" in draft_result:
        logger.info(f"Draft content preview: {draft_result['markdown'][:200]}...")
    else:
        log_structure(draft_result, name="draft_result")
    
    # Simula feedback utente
    feedback = """
    {
        "general_comment": "I agree with most insights but would prioritize temperature impact.",
        "drivers": {
            "value_temperature": {
                "status": "RELEVANT",
                "comment": "This matches our operational experience."
            },
            "value_pressure": {
                "status": "OBVIOUS",
                "comment": "This was expected."
            }
        }
    }
    """
    
    # Genera il report finale
    final_result = explain_agent.finalize_root_cause_report(
        narrative_draft=draft_result,
        feedback=feedback
    )
    
    logger.info("Final report generated")
    if isinstance(final_result, dict) and "markdown" in final_result:
        logger.info(f"Final report preview: {final_result['markdown'][:200]}...")
    else:
        log_structure(final_result, name="final_result")
    
    logger.info("=== TEST COMPLETED ===")
    return True

if __name__ == "__main__":
    test_pipeline()