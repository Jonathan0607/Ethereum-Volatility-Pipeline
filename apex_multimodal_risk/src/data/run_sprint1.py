"""
Sprint 1 End-to-End Orchestrator and Quantitative Audit CLI.
Runs Ingestion -> 4-Stage Noise Funnel -> Semantic Dedup -> Causal Target Alignment.
Prints standardized Audit Summary table.
"""

import sys
import time
from pathlib import Path

from apex_multimodal_risk.src.utils import setup_logger, PROJECT_ROOT
from apex_multimodal_risk.src.data.ingest_news import ingest_historical_news
from apex_multimodal_risk.src.data.noise_filter import apply_4stage_noise_funnel
from apex_multimodal_risk.src.data.align_targets import align_causal_targets

logger = setup_logger("run_sprint1")


def run_pipeline() -> dict:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING SPRINT 1: INGESTION, NOISE FILTERING & CAUSAL ALIGNMENT")
    logger.info("=" * 80)
    
    # 1. Ingestion
    logger.info("STEP 1: Ingesting Historical DLT-Sentiment-News...")
    raw_df = ingest_historical_news()
    raw_count = len(raw_df)
    
    # 2. 4-Stage Noise Funnel & Deduplication
    logger.info("STEP 2: Executing 4-Stage Noise Rejection Funnel...")
    dedup_df, filter_audit = apply_4stage_noise_funnel()
    
    # 3. Causal Alignment & Target Generation
    logger.info("STEP 3: Aligning Causal Market Targets (Zero-Lookahead)...")
    aligned_df, align_audit = align_causal_targets()
    
    elapsed = time.time() - t0
    
    # Format Audit Summary Table
    print("\n" + "=" * 80)
    print("APEX MULTIMODAL RISK: SPRINT 1 QUANTITATIVE AUDIT SUMMARY")
    print("=" * 80)
    
    print("\n[1] 4-STAGE NOISE REJECTION FUNNEL THROUGHPUT")
    print("-" * 65)
    print(f"  Raw Ingested Rows:              {raw_count:>8,}")
    print(f"  Post-Stage 1 (Domain Allowlist): {filter_audit['post_stage1']:>8,}  (Dropped: {filter_audit['dropped_stage1']:>6,})")
    print(f"  Post-Stage 2 (Clickbait Regex):  {filter_audit['post_stage2']:>8,}  (Dropped: {filter_audit['dropped_stage2']:>6,})")
    print(f"  Post-Stage 3 (Semantic Dedup):   {filter_audit['post_stage3_clusters']:>8,}  (Dedup Drop %: {filter_audit['dedup_drop_pct']:>5.2f}%)")
    
    print("\n[2] ENTITY LINKING DISTRIBUTION (SURVIVING CLUSTERS)")
    print("-" * 65)
    for ent, cnt in filter_audit["entity_counts"].items():
        print(f"  {ent:<8}: {cnt:>6,} clusters ({cnt / filter_audit['post_stage3_clusters']:>6.2%})")
        
    print("\n[3] WEAK RISK-CLASS SUPERVISION BREAKDOWN")
    print("-" * 65)
    for r_cls, cnt in filter_audit["risk_class_counts"].items():
        print(f"  {r_cls:<22}: {cnt:>6,} ({cnt / filter_audit['post_stage3_clusters']:>6.2%})")
        
    print("\n[4] CAUSAL MARKET ALIGNMENT & FORWARD TARGETS")
    print("-" * 65)
    print(f"  Market-News Overlap Period:     {align_audit['overlap_start']} to {align_audit['overlap_end']}")
    print(f"  Total Aligned Hourly Records:   {align_audit['total_aligned_rows']:,}")
    print("\n  Tail Crash (6h) Base Rates (Threshold: < -2.0 * 24h Realized Vol):")
    for asset, stats in align_audit["base_rates"].items():
        print(f"    - {asset:<10}: Base Rate = {stats['base_rate']:>6.2%} ({stats['positive_count']:,} / {stats['total_valid']:,} valid bars)")
        
    print(f"\nPipeline completed in {elapsed:.2f} seconds.")
    print("=" * 80 + "\n")
    
    return {
        "raw_count": raw_count,
        "filter_audit": filter_audit,
        "align_audit": align_audit,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_pipeline()
