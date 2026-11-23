"""
4-Stage Noise Rejection Funnel & Semantic Deduplication Module.
Stage 1: Source Allowlist & Tiering
Stage 2: Regex Clickbait Stripper
Stage 3: Rolling 6-Hour Semantic Deduplication with BAAI/bge-small-en-v1.5 (384-dim)
Stage 4: Entity Linker (SOL, AVAX, LINK, MACRO) & Weak Risk Labels
Persists deduplicated clusters and embeddings to Parquet via DuckDB.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta

import numpy as np
import pandas as pd
import duckdb
from sentence_transformers import SentenceTransformer

from apex_multimodal_risk.src.utils import (
    setup_logger,
    load_config,
    get_device,
    get_duckdb_connection,
    enforce_utc_columns,
    PROJECT_ROOT,
)

logger = setup_logger("noise_filter")


def apply_stage1_source_filter(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Stage 1: Extracts domain frequencies, then filters against allowlist with tier mapping.
    """
    allowlist_cfg = config["noise_filter"]["allowlist_domains"]
    tier1_domains = set(allowlist_cfg.get("tier_1", []))
    tier2_domains = set(allowlist_cfg.get("tier_2", []))
    all_allowed = tier1_domains | tier2_domains
    
    # Domain frequency audit
    domain_counts = df["domain"].value_counts().to_dict()
    top_5 = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 raw domains before filter: {top_5}")
    
    initial_count = len(df)
    is_allowed = df["domain"].isin(all_allowed)
    s1_df = df[is_allowed].copy()
    
    # Assign source tier
    s1_df["source_tier"] = np.where(s1_df["domain"].isin(tier1_domains), 1, 2)
    dropped = initial_count - len(s1_df)
    
    logger.info(
        f"Stage 1 (Source Allowlist): {initial_count:,} -> {len(s1_df):,} rows "
        f"(Dropped {dropped:,} non-whitelisted rows | Tier 1: {(s1_df['source_tier'] == 1).sum():,}, Tier 2: {(s1_df['source_tier'] == 2).sum():,})"
    )
    audit = {"initial": initial_count, "post_stage1": len(s1_df), "dropped_stage1": dropped}
    return s1_df.reset_index(drop=True), audit


def apply_stage2_clickbait_stripper(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Stage 2: Drops SEO spam and price-prediction clickbait using regex.
    """
    patterns = config["noise_filter"].get("clickbait_regex", [])
    clean_patterns = [re.sub(r"^\(\?i\)", "", p) for p in patterns]
    combined_regex = re.compile("|".join(clean_patterns), re.IGNORECASE)
    
    initial_count = len(df)
    # Match on title and description
    is_clickbait = df["title"].apply(lambda t: bool(combined_regex.search(str(t)))) | \
                   df["description"].apply(lambda d: bool(combined_regex.search(str(d)[:100])))
                   
    s2_df = df[~is_clickbait].copy()
    dropped = initial_count - len(s2_df)
    
    logger.info(
        f"Stage 2 (Clickbait Stripper): {initial_count:,} -> {len(s2_df):,} rows "
        f"(Dropped {dropped:,} clickbait/spam rows, {dropped / max(1, initial_count):.2%})"
    )
    audit = {"post_stage2": len(s2_df), "dropped_stage2": dropped}
    return s2_df.reset_index(drop=True), audit


def apply_stage3_semantic_dedup(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Stage 3: Embeds titles using BAAI/bge-small-en-v1.5 and applies rolling 6-hour
    semantic deduplication. Drops duplicates from inference feed while tracking confirmation counts.
    """
    dedup_cfg = config["noise_filter"]["deduplication"]
    model_name = dedup_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    sim_threshold = float(dedup_cfg.get("similarity_threshold", 0.85))
    window_hours = float(dedup_cfg.get("window_hours", 6.0))
    window_delta = timedelta(hours=window_hours)
    
    device = get_device()
    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
    embedder = SentenceTransformer(model_name, device=str(device))
    
    # Ensure chronological order
    df = df.sort_values(by="ingested_at", ascending=True).reset_index(drop=True)
    titles = df["title"].tolist()
    
    logger.info(f"Encoding {len(titles):,} titles into 384-dim L2-normalized embeddings...")
    embeddings = embedder.encode(
        titles,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    logger.info(f"Applying chronological rolling {window_hours}h semantic deduplication (threshold > {sim_threshold})...")
    n_items = len(df)
    timestamps = df["ingested_at"].tolist()
    
    is_duplicate = np.zeros(n_items, dtype=bool)
    cluster_ids = np.zeros(n_items, dtype=int)
    confirmation_counts = np.ones(n_items, dtype=int)
    
    active_clusters: List[Dict[str, Any]] = []
    cluster_counter = 0
    
    for i in range(n_items):
        t = timestamps[i]
        emb = embeddings[i]
        
        # Prune active clusters outside rolling 6-hour window: [t - window, t]
        active_clusters = [c for c in active_clusters if c["time"] >= t - window_delta]
        
        matched_cluster = None
        if active_clusters:
            centroids = np.array([c["centroid"] for c in active_clusters], dtype=np.float32)
            # Cosine similarity is dot product because vectors are L2-normalized
            sims = np.dot(centroids, emb)
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > sim_threshold:
                matched_cluster = active_clusters[best_idx]
                
        if matched_cluster is not None:
            # Mark as duplicate
            is_duplicate[i] = True
            c_id = matched_cluster["id"]
            cluster_ids[i] = c_id
            
            # Increment confirmation count on the primary cluster record
            primary_idx = matched_cluster["primary_idx"]
            confirmation_counts[primary_idx] += 1
            
            # Update active cluster latest observation time
            matched_cluster["time"] = t
        else:
            # New unique story cluster
            cluster_counter += 1
            c_id = cluster_counter
            cluster_ids[i] = c_id
            is_duplicate[i] = False
            active_clusters.append({
                "id": c_id,
                "time": t,
                "centroid": emb,
                "primary_idx": i
            })
            
    df["is_duplicate"] = is_duplicate
    df["cluster_id"] = cluster_ids
    df["confirmation_count"] = confirmation_counts
    df["embedding"] = list(embeddings)
    
    # Primary deduplicated stories for inference feed
    dedup_df = df[~df["is_duplicate"]].copy().reset_index(drop=True)
    # Strictly causal: as-of primary story ingestion time, base confirmation count is 1
    # Prevents retrospective mutation when duplicate stories arrive up to 6 hours later
    dedup_df["confirmation_count"] = 1
    duplicates_dropped = int(is_duplicate.sum())
    
    logger.info(
        f"Stage 3 (Semantic Dedup): {n_items:,} -> {len(dedup_df):,} clusters "
        f"(Dropped {duplicates_dropped:,} duplicates, {duplicates_dropped / max(1, n_items):.2%})"
    )
    audit = {
        "post_stage3_clusters": len(dedup_df),
        "duplicates_dropped": duplicates_dropped,
        "dedup_drop_pct": (duplicates_dropped / max(1, n_items)) * 100.0,
    }
    return dedup_df, audit


def apply_stage4_entity_and_weak_risk(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stage 4: Entity Linker (SOL, AVAX, LINK, MACRO) & Weak-supervision Risk Labeler.
    Uses regex matching on title & description combined with content characteristics and vote counts.
    """
    entity_cfg = config["noise_filter"]["entity_linking"]
    sol_pat = re.compile(entity_cfg["sol"], re.IGNORECASE)
    avax_pat = re.compile(entity_cfg["avax"], re.IGNORECASE)
    link_pat = re.compile(entity_cfg["link"], re.IGNORECASE)
    macro_pat = re.compile(entity_cfg["macro"], re.IGNORECASE)
    
    risk_cfg = config["noise_filter"]["risk_classes"]
    exploit_pat = re.compile(risk_cfg["protocol_exploit"], re.IGNORECASE)
    outage_pat = re.compile(risk_cfg["network_outage"], re.IGNORECASE)
    reg_pat = re.compile(risk_cfg["regulatory_action"], re.IGNORECASE)
    insolv_pat = re.compile(risk_cfg["exchange_insolvency"], re.IGNORECASE)
    macro_shock_pat = re.compile(risk_cfg["macro_shock"], re.IGNORECASE)
    
    # Combined search corpus
    corpus = (df["title"] + " " + df["description"]).fillna("")
    
    # Entity flags (multi-hot)
    df["entity_sol"] = corpus.apply(lambda text: bool(sol_pat.search(text)))
    df["entity_avax"] = corpus.apply(lambda text: bool(avax_pat.search(text)))
    df["entity_link"] = corpus.apply(lambda text: bool(link_pat.search(text)))
    df["entity_macro"] = corpus.apply(lambda text: bool(macro_pat.search(text)))
    
    # Risk category flags
    df["risk_exploit"] = corpus.apply(lambda text: bool(exploit_pat.search(text)))
    df["risk_outage"] = corpus.apply(lambda text: bool(outage_pat.search(text)))
    df["risk_regulatory"] = corpus.apply(lambda text: bool(reg_pat.search(text)))
    df["risk_insolvency"] = corpus.apply(lambda text: bool(insolv_pat.search(text)))
    df["risk_macro_shock"] = corpus.apply(lambda text: bool(macro_shock_pat.search(text)))
    
    # Assign deterministic primary weak risk class
    # Priority: exploit > outage > insolvency > regulatory > macro_shock > non_event_noise
    def assign_weak_class(row) -> str:
        if row["risk_exploit"]:
            return "protocol_exploit"
        if row["risk_outage"]:
            return "network_outage"
        if row["risk_insolvency"]:
            return "exchange_insolvency"
        if row["risk_regulatory"]:
            return "regulatory_action"
        if row["risk_macro_shock"]:
            return "macro_shock"
        return "non_event_noise"
        
    df["weak_risk_class"] = df.apply(assign_weak_class, axis=1)
    
    # Weak supervision risk score combining content importance and vote sentiment
    is_important = (df["content_characteristics"] == 1) | (df["votes_important"] >= 1)
    bearish_tilt = (df["votes_bearish"] > df["votes_bullish"]) & (df["votes_bearish"] >= 2)
    df["is_high_severity"] = is_important & (df["weak_risk_class"] != "non_event_noise")
    df["is_bearish_voted"] = bearish_tilt
    
    entity_counts = {
        "SOL": int(df["entity_sol"].sum()),
        "AVAX": int(df["entity_avax"].sum()),
        "LINK": int(df["entity_link"].sum()),
        "MACRO": int(df["entity_macro"].sum()),
    }
    risk_class_counts = df["weak_risk_class"].value_counts().to_dict()
    
    logger.info(f"Stage 4 Entity Counts: {entity_counts}")
    logger.info(f"Stage 4 Weak Risk Classes: {risk_class_counts}")
    
    audit = {
        "entity_counts": entity_counts,
        "risk_class_counts": risk_class_counts,
    }
    return df, audit


def apply_4stage_noise_funnel(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the entire 4-stage noise rejection and semantic deduplication funnel.
    Persists data/processed/news_dedup.parquet via DuckDB.
    """
    config = load_config(config_path)
    raw_path = config["news_ingestion"]["raw_parquet_path"]
    if not os.path.isabs(raw_path):
        raw_path = str(PROJECT_ROOT / raw_path)
        
    proc_path = config["noise_filter"]["deduplication"]["processed_parquet_path"]
    if not os.path.isabs(proc_path):
        proc_path = str(PROJECT_ROOT / proc_path)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    
    logger.info(f"Loading raw news from '{raw_path}' via DuckDB...")
    con = get_duckdb_connection()
    raw_df = con.execute(f"SELECT * FROM '{raw_path}' ORDER BY published_at ASC").df()
    con.close()
    raw_df = enforce_utc_columns(raw_df)
    
    # Stage 1: Allowlist
    s1_df, audit1 = apply_stage1_source_filter(raw_df, config)
    # Stage 2: Clickbait
    s2_df, audit2 = apply_stage2_clickbait_stripper(s1_df, config)
    # Stage 3: Semantic Dedup & Embeddings
    s3_df, audit3 = apply_stage3_semantic_dedup(s2_df, config)
    # Stage 4: Entity Linker & Weak Labels
    final_df, audit4 = apply_stage4_entity_and_weak_risk(s3_df, config)
    final_df = enforce_utc_columns(final_df)
    
    # Save processed deduplicated news with embeddings to Parquet via DuckDB
    logger.info(f"Persisting {len(final_df):,} deduplicated clusters to '{proc_path}' via DuckDB...")
    con = get_duckdb_connection()
    con.register("dedup_news_df", final_df)
    con.execute(f"COPY dedup_news_df TO '{proc_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    
    audit_summary = {**audit1, **audit2, **audit3, **audit4}
    logger.info("4-Stage Noise Rejection Funnel complete.")
    return final_df, audit_summary


if __name__ == "__main__":
    apply_4stage_noise_funnel()
