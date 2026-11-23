"""
Leakage-Safe MiniLM-L12-H384-uncased Risk Classifier Module.
Replaces DeBERTa with microsoft/MiniLM-L12-H384-uncased (pre-2021 release, zero temporal leakage, 33M params).
Fine-tunes on 6 weak-supervision risk classes:
0: non_event_noise
1: regulatory_action
2: protocol_exploit
3: macro_shock
4: network_outage
5: exchange_insolvency

Optimizations:
- Freezes embeddings and bottom 10 layers; fine-tunes ONLY top 2 layers (10-11) + head.
- max_length=48, batch_size=32, epochs=3, num_workers=0.
- Sqrt-damped class weights: w_c = clip(sqrt(N / (6 * count_c)), 1.0, 8.0).
- Differential AdamW LRs: lr=5e-4 for head, lr=3e-5 for layers 10-11, grad_clip=1.0.
- Strict RAM cleanup after every split (gc.collect() + torch.mps.empty_cache()).
- Caches predictions to data/processed/text_encoder_cpcv_cache/split_{i}.parquet.
"""

import os
import gc
import sys
import time
import resource
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_curve, auc

from apex_multimodal_risk.src.utils import (
    setup_logger,
    PROJECT_ROOT,
    get_device,
    enforce_utc_columns,
)
from apex_multimodal_risk.src.train.cpcv_runner import CPCVSplit

logger = setup_logger("text_encoder")

MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
DEFAULT_CACHE_DIR = "apex_multimodal_risk/data/processed/text_encoder_cpcv_cache"

RISK_CLASS_NAMES = [
    "non_event_noise",
    "regulatory_action",
    "protocol_exploit",
    "macro_shock",
    "network_outage",
    "exchange_insolvency",
]

RISK_CLASS_TO_ID = {name: i for i, name in enumerate(RISK_CLASS_NAMES)}


def get_peak_rss_mb() -> float:
    """Returns peak Resident Set Size memory in MB for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


class FastNewsDataset(Dataset):
    """Memory-efficient PyTorch Dataset for pre-tokenized news."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def build_minilm_classifier(
    model_name: str = MODEL_NAME,
    num_labels: int = 6,
    freeze_bottom_layers: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Initializes MiniLM-L12 model with frozen embeddings and bottom 10 layers.
    Fine-tunes ONLY layers 10-11 and classification head.
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model = model.float()

    # Freeze embeddings and bottom 10 layers
    for name, param in model.named_parameters():
        if "embeddings" in name or any(f"encoder.layer.{i}." in name for i in range(freeze_bottom_layers)):
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"MiniLM parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%) on {device}")
    return model, tokenizer


def compute_sqrt_damped_weights(labels: np.ndarray, num_classes: int = 6) -> torch.Tensor:
    """
    Computes square-root damped class weights:
      w_c = clip(sqrt(N / (6 * count_c)), 1.0, 8.0)
    Normalized so mean(weights) = 1.0 to preserve loss scale.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(len(labels))
    sqrt_w = np.clip(np.sqrt(total / (num_classes * counts)), 1.0, 8.0)
    sqrt_norm = sqrt_w / np.mean(sqrt_w)
    return torch.tensor(sqrt_norm, dtype=torch.float32)


def tokenize_news_clusters(
    news_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = 48,
) -> Dict[str, torch.Tensor]:
    """Pre-tokenizes all news titles + descriptions with max_length=48."""
    titles = news_df["title"].fillna("").astype(str).tolist()
    descs = news_df["description"].fillna("").astype(str).tolist()
    texts = [f"{t} - {d[:80]}" if d else t for t, d in zip(titles, descs)]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def train_minilm_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Trains MiniLM for 1 epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def infer_minilm(
    model: nn.Module,
    dataset: FastNewsDataset,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infers 6-class softmax probabilities and 384-dim pooled CLS embeddings.
    """
    if device is None:
        device = get_device()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_cls = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        # Extract CLS token from last hidden state (384-dim)
        last_hidden = outputs.hidden_states[-1]
        cls_embs = last_hidden[:, 0, :].cpu().numpy()
        all_cls.append(cls_embs)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_cls, axis=0)


def evaluate_text_classifier(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Computes Macro F1 and per-class PR-AUC."""
    preds = np.argmax(probs, axis=1)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    metrics = {"macro_f1": macro_f1}
    for c_id, c_name in enumerate(RISK_CLASS_NAMES):
        binary_y = (labels == c_id).astype(int)
        if np.sum(binary_y) > 0 and np.sum(binary_y) < len(binary_y):
            p, r, _ = precision_recall_curve(binary_y, probs[:, c_id])
            pr_auc = float(auc(r, p))
        else:
            pr_auc = 0.0
        metrics[f"pr_auc_{c_name}"] = pr_auc

    return metrics


def train_single_split(
    split: CPCVSplit,
    all_input_ids: torch.Tensor,
    all_attention_mask: torch.Tensor,
    all_labels: torch.Tensor,
    ingested_times: pd.Series,
    news_df: pd.DataFrame,
    device: torch.device,
    cache_file: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr_head: float = 5e-4,
    lr_backbone: float = 3e-5,
) -> Tuple[Dict[str, float], List[float], float]:
    """
    Trains MiniLM on a single CPCV split, evaluates OOS metrics, and caches predictions.
    Returns: (metrics, epoch_losses, elapsed_seconds).
    """
    start_time = time.time()
    train_mask = split.get_news_train_mask(ingested_times)
    test_mask = split.get_news_test_mask(ingested_times)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) == 0 or len(test_indices) == 0:
        logger.warning(f"Split {split.split_id}: Empty train or test set. Skipping.")
        return {}, [], 0.0

    split_model, _ = build_minilm_classifier(device=device)
    train_labels_np = all_labels[train_indices].numpy()
    class_weights = compute_sqrt_damped_weights(train_labels_np).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    head_params = [
        p for n, p in split_model.named_parameters()
        if p.requires_grad and ("classifier" in n or "pooler" in n)
    ]
    layer_params = [
        p for n, p in split_model.named_parameters()
        if p.requires_grad and not ("classifier" in n or "pooler" in n)
    ]

    optimizer = torch.optim.AdamW([
        {"params": layer_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)

    train_dataset = FastNewsDataset(
        input_ids=all_input_ids[train_indices],
        attention_mask=all_attention_mask[train_indices],
        labels=all_labels[train_indices],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    epoch_losses = []
    for epoch in range(epochs):
        loss = train_minilm_epoch(split_model, train_loader, optimizer, loss_fn, device, grad_clip=1.0)
        epoch_losses.append(loss)
        logger.info(f"Split {split.split_id} | Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    # Infer on full dataset for caching
    full_dataset = FastNewsDataset(
        input_ids=all_input_ids,
        attention_mask=all_attention_mask,
    )
    probs, cls_embs = infer_minilm(split_model, full_dataset, batch_size=128, device=device)

    # Evaluate OOS metrics on test set
    test_probs = probs[test_indices]
    test_labels = all_labels[test_indices].numpy()
    metrics = evaluate_text_classifier(test_probs, test_labels)
    metrics["split_id"] = split.split_id
    metrics["final_train_loss"] = epoch_losses[-1]

    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = elapsed

    # Save to parquet cache
    out_df = pd.DataFrame({
        "cluster_id": news_df["cluster_id"].to_numpy(),
        "ingested_at": news_df["ingested_at"].to_numpy(),
        "weak_risk_class": news_df["weak_risk_class"].to_numpy(),
        "label": news_df["label"].to_numpy(),
        "is_test": test_mask,
        "pred_class": np.argmax(probs, axis=1),
        "max_risk_prob": 1.0 - probs[:, 0],
    })
    for c_id, c_name in enumerate(RISK_CLASS_NAMES):
        out_df[f"p_{c_name}"] = probs[:, c_id]
    out_df["minilm_cls_embedding"] = [v for v in cls_embs]

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    out_df.to_parquet(cache_file, index=False)
    logger.info(f"Split {split.split_id}: Cached predictions to '{cache_file}' ({elapsed:.1f}s)")

    # Strict memory guardrails: delete model, optimizer, clean GPU cache
    del split_model, optimizer, train_dataset, train_loader, full_dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return metrics, epoch_losses, elapsed


def run_split0_sanity_gate(
    news_parquet_path: str = "apex_multimodal_risk/data/processed/news_dedup.parquet",
    aligned_parquet_path: str = "apex_multimodal_risk/data/processed/aligned_hourly.parquet",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Fast Split-0 Sanity Gate:
    Runs Split 0 and verifies:
    1. Wall-clock runtime < 90 seconds.
    2. Training loss drops cleanly below 1.10.
    Prints per-epoch loss, macro F1, elapsed seconds, and peak RSS memory.
    """
    logger.info("=" * 80)
    logger.info("RUNNING SPLIT-0 FAST SANITY GATE")
    logger.info("=" * 80)

    device = get_device()
    if not os.path.isabs(news_parquet_path):
        news_parquet_path = str(PROJECT_ROOT / news_parquet_path)
    if not os.path.isabs(aligned_parquet_path):
        aligned_parquet_path = str(PROJECT_ROOT / aligned_parquet_path)
    if not os.path.isabs(cache_dir):
        cache_dir = str(PROJECT_ROOT / cache_dir)

    # 1. Load panel timestamps and generate splits
    panel = pd.read_parquet(aligned_parquet_path, columns=["timestamp"])
    unique_ts = np.array(sorted(panel["timestamp"].unique()))
    from apex_multimodal_risk.src.train.cpcv_runner import generate_cpcv_splits
    splits = generate_cpcv_splits(unique_ts, n_blocks=5, k_test=2, purge_embargo_hours=504)
    split_0 = splits[0]

    # 2. Load news
    news_df = pd.read_parquet(
        news_parquet_path,
        columns=["cluster_id", "title", "description", "ingested_at", "weak_risk_class"],
    )
    news_df = enforce_utc_columns(news_df)
    news_df["label"] = news_df["weak_risk_class"].map(RISK_CLASS_TO_ID).fillna(0).astype(int)

    # 3. Pre-tokenize
    _, tokenizer = build_minilm_classifier(device=device)
    tokens = tokenize_news_clusters(news_df, tokenizer, max_length=48)
    all_input_ids = tokens["input_ids"]
    all_attention_mask = tokens["attention_mask"]
    all_labels = torch.tensor(news_df["label"].to_numpy(), dtype=torch.long)
    ingested_times = pd.to_datetime(news_df["ingested_at"], utc=True)

    cache_file = os.path.join(cache_dir, "split_0.parquet")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    metrics, losses, elapsed = train_single_split(
        split=split_0,
        all_input_ids=all_input_ids,
        all_attention_mask=all_attention_mask,
        all_labels=all_labels,
        ingested_times=ingested_times,
        news_df=news_df,
        device=device,
        cache_file=cache_file,
        epochs=3,
        batch_size=32,
        lr_head=5e-4,
        lr_backbone=3e-5,
    )

    peak_rss = get_peak_rss_mb()

    print("\n" + "=" * 60)
    print("SPLIT-0 SANITY GATE RESULTS:")
    print("=" * 60)
    for ep, l in enumerate(losses):
        print(f"  Epoch {ep+1}/3 Loss:         {l:.4f}")
    print(f"  OOS Macro F1:             {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  Wall-Clock Time:          {elapsed:.2f} seconds (Gate: < 90.0s)")
    print(f"  Final Epoch Loss:         {losses[-1]:.4f} (Gate: < 1.10)")
    print(f"  Peak RSS Memory:          {peak_rss:.1f} MB (Budget: < 3,000 MB)")
    print("=" * 60)

    time_passed = elapsed < 90.0
    loss_passed = losses[-1] < 1.10
    gate_passed = time_passed and loss_passed

    if gate_passed:
        logger.info(">>> SPLIT-0 SANITY GATE PASSED! <<<")
    else:
        logger.error(f">>> SPLIT-0 SANITY GATE FAILED! (time_passed={time_passed}, loss_passed={loss_passed}) <<<")

    return {
        "gate_passed": gate_passed,
        "elapsed": elapsed,
        "final_loss": losses[-1],
        "macro_f1": metrics.get("macro_f1", 0.0),
        "peak_rss_mb": peak_rss,
        "losses": losses,
    }


def train_and_cache_cpcv_splits(
    news_parquet_path: str = "apex_multimodal_risk/data/processed/news_dedup.parquet",
    cpcv_splits: Optional[List[CPCVSplit]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    epochs: int = 3,
    batch_size: int = 32,
    lr_head: float = 5e-4,
    lr_backbone: float = 3e-5,
) -> Dict[str, Any]:
    """
    Executes leak-free MiniLM fine-tuning across all 10 CPCV splits.
    Caches fold predictions and returns summary metrics table.
    """
    if not os.path.isabs(news_parquet_path):
        news_parquet_path = str(PROJECT_ROOT / news_parquet_path)
    if not os.path.isabs(cache_dir):
        cache_dir = str(PROJECT_ROOT / cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Loading deduplicated news from '{news_parquet_path}'...")
    news_df = pd.read_parquet(
        news_parquet_path,
        columns=["cluster_id", "title", "description", "ingested_at", "weak_risk_class"],
    )
    news_df = enforce_utc_columns(news_df)
    news_df["label"] = news_df["weak_risk_class"].map(RISK_CLASS_TO_ID).fillna(0).astype(int)

    logger.info("Building MiniLM model and tokenizer...")
    _, tokenizer = build_minilm_classifier(device=device)

    logger.info("Pre-tokenizing news corpus (max_length=48)...")
    tokens = tokenize_news_clusters(news_df, tokenizer, max_length=48)
    all_input_ids = tokens["input_ids"]
    all_attention_mask = tokens["attention_mask"]
    all_labels = torch.tensor(news_df["label"].to_numpy(), dtype=torch.long)
    ingested_times = pd.to_datetime(news_df["ingested_at"], utc=True)

    split_metrics = []

    for split in cpcv_splits:
        split_id = split.split_id
        cache_file = os.path.join(cache_dir, f"split_{split_id}.parquet")

        if os.path.exists(cache_file):
            logger.info(f"Split {split_id}: Cached predictions found at '{cache_file}'. Loading...")
            cached_df = pd.read_parquet(cache_file)
            test_mask = cached_df["is_test"].to_numpy()
            if np.sum(test_mask) > 0:
                y_test = cached_df.loc[test_mask, "label"].to_numpy()
                prob_cols = [f"p_{c}" for c in RISK_CLASS_NAMES]
                test_probs = cached_df.loc[test_mask, prob_cols].to_numpy()
                m = evaluate_text_classifier(test_probs, y_test)
                m["split_id"] = split_id
                split_metrics.append(m)
            continue

        logger.info(f"--- Training MiniLM on CPCV Split {split_id}/9 ---")
        m, _, _ = train_single_split(
            split=split,
            all_input_ids=all_input_ids,
            all_attention_mask=all_attention_mask,
            all_labels=all_labels,
            ingested_times=ingested_times,
            news_df=news_df,
            device=device,
            cache_file=cache_file,
            epochs=epochs,
            batch_size=batch_size,
            lr_head=lr_head,
            lr_backbone=lr_backbone,
        )
        if m:
            split_metrics.append(m)
            logger.info(
                f"Split {split_id} OOS: Macro F1={m['macro_f1']:.3f}, "
                f"PR-AUC(exploit)={m.get('pr_auc_protocol_exploit', 0):.3f}, "
                f"Time={m.get('elapsed_seconds', 0):.1f}s"
            )

    summary_df = pd.DataFrame(split_metrics)
    mean_f1 = float(summary_df["macro_f1"].mean()) if not summary_df.empty else 0.0
    return {
        "summary": summary_df,
        "mean_macro_f1": mean_f1,
    }


if __name__ == "__main__":
    run_split0_sanity_gate()
