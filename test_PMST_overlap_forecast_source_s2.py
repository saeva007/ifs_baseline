#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired test for the S2 Tianji-vs-IFS overlap PMST experiments.

This script evaluates the two models trained by train_PMST_overlap_baseline_s2.py:
  1) data_source=tianji, using ml_dataset_overlap_tianji_12h_pm10_pm25_baseline
  2) data_source=ifs,    using ml_dataset_overlap_ifs_12h_pm10_pm25_baseline

It is intended as a controlled data-source experiment for the paper:
same PMST architecture, same overlap variable layout, same observed
500/1000 m labels, paired test samples by (time, station_id), and
validation-only calibration/threshold selection.

Optionally, it also matches the raw IFS diagnostic visibility product
(VIS_IDW_KDTree_*.nc by default) to the same test rows, following the
loader used by vis_eval/run_paper_eval_pm10_pm25_11_s2.ipynb.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import average_precision_score
except Exception:  # pragma: no cover - sklearn exists in the training env.
    average_precision_score = None


VIS_MLP_ROOT = os.environ.get("VIS_MLP_ROOT", "/public/home/putianshu/vis_mlp")
IFS_BASELINE_ROOT = os.environ.get(
    "IFS_BASELINE_ROOT", os.path.join(VIS_MLP_ROOT, "ifs_baseline")
)
DEFAULT_CKPT_DIR = os.path.join(IFS_BASELINE_ROOT, "checkpoints")
DEFAULT_TIANJI_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_baseline"
)
DEFAULT_IFS_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"
)
DEFAULT_IFS_FORECAST_NC = os.path.join(VIS_MLP_ROOT, "VIS_IDW_KDTree_20250101_20251231.nc")
DEFAULT_OUT_DIR = os.path.join(
    IFS_BASELINE_ROOT, "paper_eval_overlap_forecast_source_s2"
)

CLASS_NAMES = {0: "fog_0_500m", 1: "mist_500_1000m", 2: "clear_ge_1000m"}
SOURCE_LABELS = {
    "tianji": "Tianji-trained/Tianji-input PMST",
    "ifs": "IFS-trained/IFS-input PMST",
    "ifs_diagnostic": "IFS diagnostic visibility",
}
HIGHER_IS_BETTER = {
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "fog_precision",
    "fog_pod",
    "fog_f1",
    "fog_csi",
    "mist_precision",
    "mist_pod",
    "mist_f1",
    "mist_csi",
    "clear_precision",
    "clear_recall",
    "clear_f1",
    "clear_csi",
    "low_vis_precision",
    "low_vis_pod",
    "low_vis_recall",
    "low_vis_f1",
    "low_vis_csi",
    "fog_ap",
    "mist_ap",
    "low_vis_ap",
    "target_achievement",
}
LOWER_IS_BETTER = {
    "fog_far",
    "mist_far",
    "clear_far",
    "low_vis_far",
    "low_vis_fpr",
    "multiclass_brier",
    "low_vis_brier",
    "ece_multiclass",
    "ece_low_vis",
}
BOOTSTRAP_DEFAULT_METRICS = [
    "fog_csi",
    "fog_pod",
    "fog_precision",
    "fog_f1",
    "fog_far",
    "mist_csi",
    "mist_pod",
    "mist_precision",
    "mist_f1",
    "mist_far",
    "low_vis_csi",
    "low_vis_precision",
    "low_vis_recall",
    "low_vis_f1",
    "low_vis_fpr",
    "accuracy",
]


@dataclass
class SourceSpec:
    name: str
    data_dir: str
    ckpt_path: str
    scaler_path: str


@dataclass
class SourceEval:
    source: str
    spec: SourceSpec
    feature_dim: int
    extra_feat_dim: int
    dyn_vars_count: int
    temperature: float
    thresholds: Dict[str, float]
    val_metrics: Dict[str, float]
    test_probs: np.ndarray
    test_preds: np.ndarray
    test_targets: np.ndarray
    test_raw_vis: np.ndarray
    test_meta: Optional[pd.DataFrame]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate paired Tianji-vs-IFS overlap S2 PMST models on the held-out "
            "month-tail test split."
        )
    )
    ap.add_argument("--tianji_data_dir", default=os.environ.get("OVERLAP_TIANJI_DATA_DIR", DEFAULT_TIANJI_DIR))
    ap.add_argument("--ifs_data_dir", default=os.environ.get("OVERLAP_IFS_DATA_DIR", DEFAULT_IFS_DIR))
    ap.add_argument("--ckpt_dir", default=os.environ.get("OVERLAP_CKPT_DIR", DEFAULT_CKPT_DIR))
    ap.add_argument("--tianji_ckpt", default="")
    ap.add_argument("--ifs_ckpt", default="")
    ap.add_argument("--tianji_scaler", default="")
    ap.add_argument("--ifs_scaler", default="")
    ap.add_argument("--ifs_forecast_nc", default=os.environ.get("IFS_FORECAST_NC", DEFAULT_IFS_FORECAST_NC))
    ap.add_argument("--ifs_forecast_var", default=os.environ.get("IFS_FORECAST_VAR", "VIS"))
    ap.add_argument("--checkpoint_tag", default="S2_PhaseB_best_score")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=27)
    ap.add_argument("--expected_extra_dim", type=int, default=36)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument(
        "--threshold_mode",
        choices=["val_search", "argmax", "fixed"],
        default="val_search",
        help="Use validation threshold search, argmax, or fixed fog/mist thresholds.",
    )
    ap.add_argument("--fog_threshold", type=float, default=0.5)
    ap.add_argument("--mist_threshold", type=float, default=0.5)
    ap.add_argument("--no_temp_scaling", action="store_true")
    ap.add_argument("--temp_lr", type=float, default=0.01)
    ap.add_argument("--temp_max_iter", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap_size", type=int, default=200000)
    ap.add_argument("--bootstrap_seed", type=int, default=20250424)
    ap.add_argument("--skip_bootstrap", action="store_true")
    ap.add_argument("--strict_meta", action="store_true", help="Fail unless test metadata are identical and in order.")
    ap.add_argument("--allow_partial_load", action="store_true")
    ap.add_argument("--limit_samples", type=int, default=0, help="Smoke-test limit for val/test rows; 0 means all.")
    ap.add_argument("--no_per_sample_csv", action="store_true")
    ap.add_argument("--no_figures", action="store_true", help="Skip publication-style summary figures.")
    ap.add_argument(
        "--skip_ifs_forecast_baseline",
        action="store_true",
        help="Skip the raw IFS diagnostic-visibility baseline matched from --ifs_forecast_nc.",
    )
    return ap.parse_args()


def default_ckpt_path(source: str, ckpt_dir: str, checkpoint_tag: str) -> str:
    run_exp_id = f"exp_overlap_pmst_baseline_s2_{source}_pm10_pm25"
    return os.path.join(ckpt_dir, f"{run_exp_id}_{checkpoint_tag}.pt")


def default_scaler_path(source: str, ckpt_dir: str, window: int, dyn_vars_count: int) -> str:
    return os.path.join(
        ckpt_dir,
        f"robust_scaler_w{window}_dyn{dyn_vars_count}_overlap_baseline_{source}.pkl",
    )


def import_training_module():
    module_path = Path(__file__).resolve().parent / "train_PMST_overlap_baseline_s2.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find training module: {module_path}")

    old_argv = sys.argv[:]
    try:
        # The training script parses known args at import time. Keep it isolated
        # from this evaluator's CLI.
        sys.argv = [str(module_path)]
        spec = importlib.util.spec_from_file_location("pmst_overlap_train_s2", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def read_build_config(data_dir: str) -> Dict:
    path = os.path.join(data_dir, "dataset_build_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def infer_feature_layout(
    data_dir: str,
    split: str,
    window: int,
    dyn_vars_count: int,
    expected_extra_dim: int,
) -> Tuple[int, int]:
    x_path = os.path.join(data_dir, f"X_{split}.npy")
    require_file(x_path, f"X_{split}.npy")
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) != 2:
        raise ValueError(f"{x_path} must be 2D, got shape={shape}")
    feature_dim = int(shape[1])
    base_dim = window * dyn_vars_count + 5 + 1
    extra_dim = feature_dim - base_dim
    if extra_dim <= 0:
        raise ValueError(
            f"Invalid feature layout for {x_path}: feature_dim={feature_dim}, "
            f"base_dim={base_dim}, extra_dim={extra_dim}"
        )
    if expected_extra_dim > 0 and extra_dim != expected_extra_dim:
        print(
            f"[WARN] {data_dir} extra feature dim is {extra_dim}, "
            f"expected {expected_extra_dim}. Using inferred value.",
            flush=True,
        )
    return feature_dim, int(extra_dim)


def build_y_cls_raw(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_raw = np.asarray(y_raw, dtype=np.float32).copy()
    if np.nanmax(y_raw) < 100:
        y_raw *= 1000.0
    y_cls = np.zeros(len(y_raw), dtype=np.int64)
    y_cls[y_raw >= 500.0] = 1
    y_cls[y_raw >= 1000.0] = 2
    return y_raw, y_cls


def load_meta(data_dir: str, split: str, indices: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
    path = os.path.join(data_dir, f"meta_{split}.csv")
    if not os.path.isfile(path):
        return None
    meta = pd.read_csv(path)
    if indices is not None:
        meta = meta.iloc[indices].reset_index(drop=True)
    return meta


def make_dataset(
    train_mod,
    data_dir: str,
    split: str,
    scaler,
    window: int,
    dyn_vars_count: int,
    limit_samples: int,
):
    x_path = os.path.join(data_dir, f"X_{split}.npy")
    y_path = os.path.join(data_dir, f"y_{split}.npy")
    require_file(x_path, f"X_{split}.npy")
    require_file(y_path, f"y_{split}.npy")

    y_raw, y_cls = build_y_cls_raw(np.load(y_path))
    indices = None
    if limit_samples and limit_samples > 0:
        indices = np.arange(min(limit_samples, len(y_cls)), dtype=np.int64)

    ds = train_mod.PMSTDataset(
        x_path,
        y_cls,
        y_raw,
        scaler,
        window_size=window,
        use_fe=True,
        indices=indices,
        dyn_vars_count=dyn_vars_count,
    )
    meta = load_meta(data_dir, split, indices)
    return ds, meta


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.X = None


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kwargs.update(num_workers=num_workers, persistent_workers=True, worker_init_fn=worker_init_fn)
    else:
        kwargs.update(num_workers=0)
    return DataLoader(dataset, **kwargs)


def load_model(
    train_mod,
    ckpt_path: str,
    device: torch.device,
    window: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    allow_partial_load: bool,
) -> nn.Module:
    require_file(ckpt_path, "checkpoint")
    model = train_mod.ImprovedDualStreamPMSTNet(
        window_size=window,
        hidden_dim=train_mod.CONFIG.get("MODEL_HIDDEN_DIM", 512),
        num_classes=3,
        extra_feat_dim=extra_feat_dim,
        dyn_vars_count=dyn_vars_count,
    ).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    clean_state = {}
    for k, v in state.items():
        clean_state[k[7:] if k.startswith("module.") else k] = v

    result = model.load_state_dict(clean_state, strict=not allow_partial_load)
    if allow_partial_load:
        print(
            f"[WARN] partial load for {ckpt_path}: "
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    return model


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_l: List[np.ndarray] = []
    targets_l: List[np.ndarray] = []
    raw_l: List[np.ndarray] = []
    model.eval()
    for bx, by, _, braw in loader:
        bx = bx.to(device, non_blocking=True)
        logits, _, _ = model(bx)
        logits_l.append(logits.detach().cpu().numpy().astype(np.float32))
        targets_l.append(by.numpy().astype(np.int64))
        raw_l.append(braw.numpy().astype(np.float32))
    return np.concatenate(logits_l, axis=0), np.concatenate(targets_l), np.concatenate(raw_l)


def calibrate_temperature_from_logits(
    logits_np: np.ndarray,
    targets_np: np.ndarray,
    device: torch.device,
    lr: float,
    max_iter: int,
) -> float:
    logits = torch.as_tensor(logits_np, dtype=torch.float32, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
    log_temp = nn.Parameter(torch.log(torch.tensor([1.5], dtype=torch.float32, device=device)))
    opt = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_temp).detach().clamp(0.05, 20.0).item())


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = logits.astype(np.float64) / temp
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(np.float32)


def predict_from_probs(
    probs: np.ndarray,
    mode: str,
    fog_threshold: float,
    mist_threshold: float,
) -> np.ndarray:
    if mode == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64)

    preds = np.full(len(probs), 2, dtype=np.int64)
    fog_conf = (probs[:, 0] > fog_threshold) & (probs[:, 0] > probs[:, 1])
    mist_conf = (probs[:, 1] > mist_threshold) & (probs[:, 1] > probs[:, 0])
    preds[fog_conf] = 0
    preds[mist_conf] = 1
    return preds


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_matrix_3(targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    valid = (targets >= 0) & (targets <= 2) & (preds >= 0) & (preds <= 2)
    np.add.at(cm, (targets[valid].astype(int), preds[valid].astype(int)), 1)
    return cm


def ece_binary(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(prob)
    if total == 0:
        return math.nan
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        n = int(mask.sum())
        if n:
            ece += n / total * abs(float(prob[mask].mean()) - float(outcome[mask].mean()))
    return float(ece)


def ece_multiclass(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    return ece_binary(conf, (pred == targets).astype(np.float32), n_bins=n_bins)


def average_precision_safe(targets_binary: np.ndarray, scores: np.ndarray) -> float:
    if average_precision_score is None:
        return math.nan
    if len(np.unique(targets_binary)) < 2:
        return math.nan
    return float(average_precision_score(targets_binary, scores))


def compute_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray] = None,
    prefix_counts: bool = True,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = confusion_matrix_3(targets, preds)
    n = int(cm.sum())
    metrics: Dict[str, float] = {"n": float(n)}

    f1_values = []
    weighted_f1_num = 0.0
    for cid, cname in CLASS_NAMES.items():
        short = cname.split("_")[0]
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        support = float(cm[cid, :].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        csi = safe_div(tp, tp + fp + fn)
        far = safe_div(fp, tp + fp)
        metrics[f"{short}_precision"] = precision
        if short == "clear":
            metrics[f"{short}_recall"] = recall
        else:
            metrics[f"{short}_pod"] = recall
        metrics[f"{short}_f1"] = f1
        metrics[f"{short}_csi"] = csi
        metrics[f"{short}_far"] = far
        if prefix_counts:
            metrics[f"{short}_support"] = support
            metrics[f"pred_{short}"] = float(cm[:, cid].sum())
        f1_values.append(f1)
        weighted_f1_num += f1 * support

    metrics["accuracy"] = safe_div(float(np.trace(cm)), float(n))
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else math.nan
    metrics["weighted_f1"] = safe_div(weighted_f1_num, float(n))

    pred_low = preds <= 1
    true_low = targets <= 1
    is_clear = targets == 2
    low_tp = float((pred_low & true_low).sum())
    low_fp = float((pred_low & ~true_low).sum())
    low_fn = float((~pred_low & true_low).sum())
    low_precision = safe_div(low_tp, low_tp + low_fp)
    low_pod = safe_div(low_tp, low_tp + low_fn)
    metrics["low_vis_precision"] = low_precision
    metrics["low_vis_pod"] = low_pod
    metrics["low_vis_recall"] = low_pod
    metrics["low_vis_f1"] = safe_div(2.0 * low_precision * low_pod, low_precision + low_pod)
    metrics["low_vis_csi"] = safe_div(low_tp, low_tp + low_fp + low_fn)
    metrics["low_vis_far"] = safe_div(low_fp, low_tp + low_fp)
    metrics["low_vis_fpr"] = safe_div(float((pred_low & is_clear).sum()), float(is_clear.sum()))
    metrics["recall_500"] = metrics["fog_pod"]
    metrics["recall_1000"] = metrics["mist_pod"]
    metrics["false_positive_rate"] = metrics["low_vis_fpr"]

    if probs is not None:
        one_hot = np.eye(3, dtype=np.float32)[targets]
        metrics["multiclass_brier"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        low_prob = probs[:, 0] + probs[:, 1]
        metrics["low_vis_brier"] = float(np.mean((low_prob - true_low.astype(np.float32)) ** 2))
        metrics["ece_multiclass"] = ece_multiclass(probs, targets)
        metrics["ece_low_vis"] = ece_binary(low_prob, true_low.astype(np.float32))
        metrics["fog_ap"] = average_precision_safe((targets == 0).astype(np.int32), probs[:, 0])
        metrics["mist_ap"] = average_precision_safe((targets == 1).astype(np.int32), probs[:, 1])
        metrics["low_vis_ap"] = average_precision_safe(true_low.astype(np.int32), low_prob)

    metrics["target_achievement"] = compute_target_achievement(metrics)
    return metrics


def compute_target_achievement(metrics: Dict[str, float]) -> float:
    cfg = {
        "TARGET_RECALL_500_GOAL": 0.65,
        "TARGET_RECALL_1000_GOAL": 0.75,
        "TARGET_ACCURACY_GOAL": 0.95,
        "TARGET_LOW_VIS_PREC_GOAL": 0.20,
        "TARGET_FPR_GOAL": 0.40,
        "TARGET_W_RECALL_500": 0.30,
        "TARGET_W_RECALL_1000": 0.30,
        "TARGET_W_ACCURACY": 0.25,
        "TARGET_W_LOW_VIS_PREC": 0.10,
        "TARGET_W_FPR": 0.05,
    }
    return float(
        min(metrics["recall_500"] / cfg["TARGET_RECALL_500_GOAL"], 1.0)
        * cfg["TARGET_W_RECALL_500"]
        + min(metrics["recall_1000"] / cfg["TARGET_RECALL_1000_GOAL"], 1.0)
        * cfg["TARGET_W_RECALL_1000"]
        + min(metrics["accuracy"] / cfg["TARGET_ACCURACY_GOAL"], 1.0)
        * cfg["TARGET_W_ACCURACY"]
        + min(metrics["low_vis_precision"] / cfg["TARGET_LOW_VIS_PREC_GOAL"], 1.0)
        * cfg["TARGET_W_LOW_VIS_PREC"]
        + min((1.0 - metrics["false_positive_rate"]) / (1.0 - cfg["TARGET_FPR_GOAL"]), 1.0)
        * cfg["TARGET_W_FPR"]
    )


def threshold_grid() -> np.ndarray:
    low_part = np.arange(0.10, 0.50, 0.04)
    high_part = np.arange(0.50, 0.96, 0.03)
    return np.unique(np.concatenate([low_part, high_part]))


def search_thresholds_on_val(probs: np.ndarray, targets: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_score = -np.inf
    best_metrics: Optional[Dict[str, float]] = None
    best_th = {"fog": 0.5, "mist": 0.5}
    grid = threshold_grid()

    def try_search(min_prec: float, min_clear_recall: float, penalty: bool = False) -> None:
        nonlocal best_score, best_metrics, best_th
        for f_th in grid:
            for m_th in grid:
                preds = predict_from_probs(probs, "fixed", float(f_th), float(m_th))
                metrics = compute_metrics(targets, preds, probs=None, prefix_counts=False)
                if (
                    metrics["fog_precision"] >= min_prec
                    and metrics["mist_precision"] >= min_prec
                    and metrics["clear_recall"] >= min_clear_recall
                ):
                    score = metrics["target_achievement"]
                    if penalty:
                        score -= max(0.0, 0.10 - metrics["fog_precision"])
                        score -= max(0.0, 0.10 - metrics["mist_precision"])
                    if score > best_score:
                        best_score = score
                        best_metrics = metrics
                        best_th = {"fog": float(f_th), "mist": float(m_th)}

    try_search(min_prec=0.10, min_clear_recall=0.90, penalty=False)
    if best_metrics is None:
        try_search(min_prec=0.05, min_clear_recall=0.88, penalty=True)
    if best_metrics is None:
        preds = np.argmax(probs, axis=1).astype(np.int64)
        best_metrics = compute_metrics(targets, preds, probs=None, prefix_counts=False)
        best_th = {"fog": math.nan, "mist": math.nan}
    return best_th, best_metrics


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    return "higher"


def rows_from_metrics(source: str, metrics: Dict[str, float], extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    row: Dict[str, object] = {"source": source, "source_label": SOURCE_LABELS.get(source, source)}
    if extra:
        row.update(extra)
    row.update(metrics)
    return row


def classify_visibility_values(
    vis_values: np.ndarray,
    fog_threshold: float = 500.0,
    mist_threshold: float = 1000.0,
) -> np.ndarray:
    """Map continuous visibility in meters to the 0/1/2 paper classes."""
    vis = np.asarray(vis_values, dtype=np.float64)
    cls = np.full(vis.shape, 2, dtype=np.int64)
    cls[vis < mist_threshold] = 1
    cls[vis < fog_threshold] = 0
    return cls


def normalize_station_ids_for_lookup(station_values) -> pd.Series:
    station = pd.Series(station_values)
    numeric = pd.to_numeric(station, errors="coerce")
    out = station.astype(str)
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(np.int64).astype(str)
    return out


def station_ids_for_ifs_lookup(meta: pd.DataFrame) -> pd.Series:
    out = normalize_station_ids_for_lookup(meta["station_id"].to_numpy())
    out.index = meta.index
    return out


def load_ifs_forecast_baseline(
    meta: pd.DataFrame,
    ifs_nc_path: str,
    vis_var: str = "VIS",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match raw IFS diagnostic visibility to the evaluation samples.

    This mirrors vis_eval.plot_spatial.load_ifs_baseline: match by exact
    (time, station_id), then classify VIS with the paper's 500/1000 m thresholds.
    """
    if "time" not in meta.columns or "station_id" not in meta.columns:
        raise KeyError("IFS diagnostic matching requires meta columns: time, station_id")
    if not ifs_nc_path or not os.path.exists(ifs_nc_path):
        raise FileNotFoundError(f"IFS diagnostic NetCDF not found: {ifs_nc_path}")

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to read the IFS diagnostic NetCDF.") from exc

    ds_ifs = xr.open_dataset(ifs_nc_path)
    try:
        if vis_var not in ds_ifs:
            raise KeyError(f"Variable '{vis_var}' not found in {ifs_nc_path}")
        if "time" not in ds_ifs.coords or "station" not in ds_ifs.coords:
            raise KeyError("IFS diagnostic dataset must provide 'time' and 'station' coordinates.")

        vis_da = ds_ifs[vis_var]
        if "time" in vis_da.dims and "station" in vis_da.dims:
            vis_da = vis_da.squeeze().transpose("time", "station", ...)
        if vis_da.ndim != 2:
            raise ValueError(f"IFS diagnostic variable '{vis_var}' must be 2D time x station, got {vis_da.shape}")

        ifs_vis = np.asarray(vis_da.values, dtype=np.float64)
        ifs_times = pd.to_datetime(ds_ifs["time"].values)
        ifs_stations = pd.Index(normalize_station_ids_for_lookup(ds_ifs["station"].values))

        time_lookup = pd.Series(np.arange(len(ifs_times), dtype=np.int64), index=pd.Index(ifs_times))
        station_lookup = pd.Series(np.arange(len(ifs_stations), dtype=np.int64), index=ifs_stations)

        meta_times = pd.to_datetime(meta["time"], errors="coerce")
        meta_stations = station_ids_for_ifs_lookup(meta)
        time_idx = meta_times.map(time_lookup)
        station_idx = meta_stations.map(station_lookup)
        key_valid = time_idx.notna() & station_idx.notna()

        ifs_vis_raw = np.full(len(meta), np.nan, dtype=np.float64)
        ifs_preds = np.full(len(meta), -1, dtype=np.int64)
        ifs_valid = np.zeros(len(meta), dtype=bool)

        if key_valid.any():
            key_valid_pos = np.flatnonzero(key_valid.to_numpy())
            t_idx = time_idx.iloc[key_valid_pos].astype(np.int64).to_numpy()
            s_idx = station_idx.iloc[key_valid_pos].astype(np.int64).to_numpy()
            matched_vis = np.asarray(ifs_vis[t_idx, s_idx], dtype=np.float64)
            finite = np.isfinite(matched_vis)
            matched_pos = key_valid_pos[finite]
            ifs_vis_raw[matched_pos] = matched_vis[finite]
            ifs_preds[matched_pos] = classify_visibility_values(matched_vis[finite])
            ifs_valid[matched_pos] = True

        print(
            f"[IFS diagnostic] Matched {int(ifs_valid.sum())}/{len(meta)} finite samples "
            f"from {os.path.basename(ifs_nc_path)}::{vis_var}",
            flush=True,
        )
        return ifs_preds, ifs_vis_raw, ifs_valid
    finally:
        ds_ifs.close()


def normalize_key_frame(meta: pd.DataFrame) -> pd.DataFrame:
    if "time" not in meta.columns or "station_id" not in meta.columns:
        raise KeyError("meta csv must contain time and station_id columns for paired alignment")
    out = pd.DataFrame(index=meta.index)
    time_parsed = pd.to_datetime(meta["time"], errors="coerce")
    out["time_key"] = time_parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    missing_time = out["time_key"].isna()
    if missing_time.any():
        out.loc[missing_time, "time_key"] = meta.loc[missing_time, "time"].astype(str)
    out["station_key"] = meta["station_id"].astype(str)
    out["dup"] = out.groupby(["time_key", "station_key"]).cumcount()
    out["row_idx"] = np.arange(len(meta), dtype=np.int64)
    return out


def align_test_outputs(tianji: SourceEval, ifs: SourceEval, strict_meta: bool) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if tianji.test_meta is not None and ifs.test_meta is not None:
        left = normalize_key_frame(tianji.test_meta).rename(columns={"row_idx": "idx_tianji"})
        right = normalize_key_frame(ifs.test_meta).rename(columns={"row_idx": "idx_ifs"})
        joined = left.merge(right, on=["time_key", "station_key", "dup"], how="inner", sort=False)
        if joined.empty:
            raise RuntimeError("No common (time, station_id) rows between Tianji and IFS test metadata.")

        idx_t = joined["idx_tianji"].to_numpy(dtype=np.int64)
        idx_i = joined["idx_ifs"].to_numpy(dtype=np.int64)
        same_order = (
            len(idx_t) == len(tianji.test_targets)
            and len(idx_i) == len(ifs.test_targets)
            and np.array_equal(idx_t, np.arange(len(idx_t)))
            and np.array_equal(idx_i, np.arange(len(idx_i)))
        )
        if strict_meta and not same_order:
            raise RuntimeError(
                "Test metadata are not identical/in-order. Rerun without --strict_meta "
                "to use the paired intersection."
            )
        meta_common = tianji.test_meta.iloc[idx_t].reset_index(drop=True).copy()
        return idx_t, idx_i, meta_common

    n = min(len(tianji.test_targets), len(ifs.test_targets))
    idx_t = np.arange(n, dtype=np.int64)
    idx_i = np.arange(n, dtype=np.int64)
    meta_common = pd.DataFrame({"row": np.arange(n, dtype=np.int64)})
    if strict_meta and len(tianji.test_targets) != len(ifs.test_targets):
        raise RuntimeError("No metadata and test lengths differ under --strict_meta.")
    return idx_t, idx_i, meta_common


def validate_paired_labels(tianji: SourceEval, ifs: SourceEval, idx_t: np.ndarray, idx_i: np.ndarray) -> None:
    yt = tianji.test_targets[idx_t]
    yi = ifs.test_targets[idx_i]
    if not np.array_equal(yt, yi):
        mismatch = int((yt != yi).sum())
        raise RuntimeError(
            f"Paired test labels differ in {mismatch} rows. This experiment must use "
            "the same observed 500/1000 m labels for both data sources."
        )
    rt = tianji.test_raw_vis[idx_t]
    ri = ifs.test_raw_vis[idx_i]
    finite = np.isfinite(rt) & np.isfinite(ri)
    if finite.any() and float(np.max(np.abs(rt[finite] - ri[finite]))) > 1e-3:
        raise RuntimeError(
            "Paired raw visibility values differ between Tianji and IFS datasets. "
            "Check meta alignment and y_test.npy before using this comparison in the paper."
        )


def evaluate_one_source(
    source: str,
    spec: SourceSpec,
    train_mod,
    args: argparse.Namespace,
    device: torch.device,
) -> SourceEval:
    print(f"[{source}] data_dir={spec.data_dir}", flush=True)
    print(f"[{source}] ckpt={spec.ckpt_path}", flush=True)
    print(f"[{source}] scaler={spec.scaler_path}", flush=True)

    feature_dim, extra_dim = infer_feature_layout(
        spec.data_dir, "test", args.window, args.dyn_vars_count, args.expected_extra_dim
    )
    val_feature_dim, val_extra_dim = infer_feature_layout(
        spec.data_dir, "val", args.window, args.dyn_vars_count, args.expected_extra_dim
    )
    if val_feature_dim != feature_dim or val_extra_dim != extra_dim:
        raise ValueError(
            f"{source}: val/test feature layout differs: "
            f"val=({val_feature_dim},{val_extra_dim}) test=({feature_dim},{extra_dim})"
        )

    require_file(spec.scaler_path, "RobustScaler")
    scaler = joblib.load(spec.scaler_path)
    model = load_model(
        train_mod,
        spec.ckpt_path,
        device,
        args.window,
        args.dyn_vars_count,
        extra_dim,
        args.allow_partial_load,
    )

    val_ds, _ = make_dataset(
        train_mod, spec.data_dir, "val", scaler, args.window, args.dyn_vars_count, args.limit_samples
    )
    test_ds, test_meta = make_dataset(
        train_mod, spec.data_dir, "test", scaler, args.window, args.dyn_vars_count, args.limit_samples
    )
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    print(f"[{source}] running validation inference: N={len(val_ds)}", flush=True)
    val_logits, val_targets, _ = collect_logits(model, val_loader, device)
    if args.no_temp_scaling:
        temperature = 1.0
    else:
        temperature = calibrate_temperature_from_logits(
            val_logits, val_targets, device, args.temp_lr, args.temp_max_iter
        )
    val_probs = softmax_np(val_logits, temperature)

    if args.threshold_mode == "val_search":
        thresholds, val_metrics = search_thresholds_on_val(val_probs, val_targets)
        threshold_mode_for_pred = "fixed"
    elif args.threshold_mode == "fixed":
        thresholds = {"fog": float(args.fog_threshold), "mist": float(args.mist_threshold)}
        val_preds = predict_from_probs(val_probs, "fixed", thresholds["fog"], thresholds["mist"])
        val_metrics = compute_metrics(val_targets, val_preds, probs=val_probs)
        threshold_mode_for_pred = "fixed"
    else:
        thresholds = {"fog": math.nan, "mist": math.nan}
        val_preds = predict_from_probs(val_probs, "argmax", 0.5, 0.5)
        val_metrics = compute_metrics(val_targets, val_preds, probs=val_probs)
        threshold_mode_for_pred = "argmax"

    print(
        f"[{source}] temperature={temperature:.4f}, thresholds={thresholds}, "
        f"val target_achievement={val_metrics.get('target_achievement', math.nan):.4f}",
        flush=True,
    )

    print(f"[{source}] running test inference: N={len(test_ds)}", flush=True)
    test_logits, test_targets, test_raw = collect_logits(model, test_loader, device)
    test_probs = softmax_np(test_logits, temperature)
    test_preds = predict_from_probs(
        test_probs,
        threshold_mode_for_pred,
        thresholds.get("fog", 0.5),
        thresholds.get("mist", 0.5),
    )

    return SourceEval(
        source=source,
        spec=spec,
        feature_dim=feature_dim,
        extra_feat_dim=extra_dim,
        dyn_vars_count=args.dyn_vars_count,
        temperature=float(temperature),
        thresholds=thresholds,
        val_metrics=val_metrics,
        test_probs=test_probs,
        test_preds=test_preds,
        test_targets=test_targets,
        test_raw_vis=test_raw,
        test_meta=test_meta,
    )


def compare_metric_sets(
    metrics_left: Dict[str, float],
    metrics_right: Dict[str, float],
    left_name: str,
    right_name: str,
    metric_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if metric_names is None:
        metric_names = sorted((set(metrics_left) & set(metrics_right)) - {"n"})
    rows = []
    delta_col = f"delta_{left_name}_minus_{right_name}"
    better_col = f"{left_name}_better"
    for m in metric_names:
        if m not in metrics_left or m not in metrics_right:
            continue
        left_val = metrics_left[m]
        right_val = metrics_right[m]
        if not isinstance(left_val, (int, float, np.floating)) or not isinstance(right_val, (int, float, np.floating)):
            continue
        direction = metric_direction(m)
        delta = float(left_val) - float(right_val)
        if direction == "lower":
            left_better = delta < 0
        else:
            left_better = delta > 0
        rows.append(
            {
                "metric": m,
                left_name: float(left_val),
                right_name: float(right_val),
                delta_col: delta,
                "preferred_direction": direction,
                better_col: bool(left_better),
            }
        )
    return pd.DataFrame(rows)


def compare_overall(
    metrics_t: Dict[str, float],
    metrics_i: Dict[str, float],
    metric_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return compare_metric_sets(metrics_t, metrics_i, "tianji", "ifs", metric_names)


def bootstrap_delta_ci(
    targets: np.ndarray,
    t_probs: np.ndarray,
    t_preds: np.ndarray,
    i_probs: np.ndarray,
    i_preds: np.ndarray,
    metrics: Sequence[str],
    n_bootstrap: int,
    bootstrap_size: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    n = len(targets)
    bs = n if bootstrap_size <= 0 else min(int(bootstrap_size), n)
    values: Dict[str, List[float]] = {m: [] for m in metrics}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=bs, endpoint=False)
        mt = compute_metrics(targets[idx], t_preds[idx], probs=None, prefix_counts=False)
        mi = compute_metrics(targets[idx], i_preds[idx], probs=None, prefix_counts=False)
        for m in metrics:
            values[m].append(float(mt[m]) - float(mi[m]))
        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"[bootstrap] {b + 1}/{n_bootstrap}", flush=True)

    rows = []
    for m, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "metric": m,
                "bootstrap_reps": n_bootstrap,
                "bootstrap_size": bs,
                "delta_mean": float(np.mean(arr)),
                "delta_ci95_low": float(np.percentile(arr, 2.5)),
                "delta_ci95_high": float(np.percentile(arr, 97.5)),
                "preferred_direction": metric_direction(m),
            }
        )
    return pd.DataFrame(rows)


def scenario_masks(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {"All": np.ones(len(meta), dtype=bool)}
    if "time" not in meta.columns:
        return masks
    time = pd.to_datetime(meta["time"], errors="coerce")
    valid = ~time.isna()
    if valid.any():
        hour = time.dt.hour.to_numpy()
        month = time.dt.month.to_numpy()
        masks["Day_hour_06_18"] = valid.to_numpy() & (hour >= 6) & (hour < 18)
        masks["Night_hour_18_06"] = valid.to_numpy() & ~((hour >= 6) & (hour < 18))
        masks["DJF"] = valid.to_numpy() & np.isin(month, [12, 1, 2])
        masks["MAM"] = valid.to_numpy() & np.isin(month, [3, 4, 5])
        masks["JJA"] = valid.to_numpy() & np.isin(month, [6, 7, 8])
        masks["SON"] = valid.to_numpy() & np.isin(month, [9, 10, 11])
    return {k: v for k, v in masks.items() if int(v.sum()) > 0}


def write_confusion_csv(path: str, targets: np.ndarray, preds: np.ndarray) -> None:
    cm = confusion_matrix_3(targets, preds)
    df = pd.DataFrame(cm, index=[CLASS_NAMES[i] for i in range(3)], columns=[CLASS_NAMES[i] for i in range(3)])
    df.index.name = "truth_pred"
    df.to_csv(path)


def write_report(
    path: str,
    args: argparse.Namespace,
    specs: Dict[str, SourceSpec],
    build_configs: Dict[str, Dict],
    overall_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    n_pair: int,
    ifs_diagnostic_df: Optional[pd.DataFrame] = None,
    ifs_diagnostic_delta_df: Optional[pd.DataFrame] = None,
    n_ifs_diagnostic: int = 0,
) -> None:
    def fmt(metric: str, source: str) -> str:
        row = overall_df.loc[overall_df["source"] == source]
        if row.empty or metric not in row.columns:
            return "NA"
        val = row.iloc[0][metric]
        return "NA" if pd.isna(val) else f"{float(val):.4f}"

    def fmt_from(df: pd.DataFrame, metric: str, source: str) -> str:
        if df is None or df.empty:
            return "NA"
        row = df.loc[df["source"] == source]
        if row.empty or metric not in row.columns:
            return "NA"
        val = row.iloc[0][metric]
        return "NA" if pd.isna(val) else f"{float(val):.4f}"

    with open(path, "w", encoding="utf-8") as f:
        f.write("PMST overlap S2 forecast-source paired evaluation\n")
        f.write("=" * 58 + "\n\n")
        f.write("Purpose: controlled comparison of two train_PMST_overlap_baseline_s2.py models.\n")
        f.write("Interpretation: this tests forecast-field source quality under the same PMST\n")
        f.write("architecture, same observed 0-500 m / 500-1000 m / >=1000 m labels, and\n")
        f.write("paired test samples. A separate matched section compares against the raw\n")
        f.write("IFS diagnostic-visibility product when --ifs_forecast_nc is available.\n\n")
        f.write(f"Paired test rows: {n_pair}\n")
        f.write(f"Threshold mode: {args.threshold_mode}\n")
        f.write(f"Temperature scaling: {not args.no_temp_scaling}\n")
        f.write(f"Window: {args.window}; dyn_vars_count: {args.dyn_vars_count}\n\n")
        for source in ("tianji", "ifs"):
            f.write(f"[{source}]\n")
            f.write(f"data_dir: {specs[source].data_dir}\n")
            f.write(f"checkpoint: {specs[source].ckpt_path}\n")
            f.write(f"scaler: {specs[source].scaler_path}\n")
            cfg = build_configs.get(source) or {}
            if cfg:
                f.write(
                    "dataset_build_config: "
                    + json.dumps(
                        {
                            k: cfg.get(k)
                            for k in [
                                "dataset",
                                "overlap_vars",
                                "dyn_layout",
                                "fe_dim",
                                "window",
                                "split",
                                "val_last_days",
                                "test_last_days",
                                "gap_hours",
                                "max_vis_threshold",
                            ]
                            if k in cfg
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.write("\n")

        f.write("Key test metrics\n")
        f.write(
            "metric,tianji,ifs,delta_tianji_minus_ifs,preferred_direction,tianji_better\n"
        )
        key_metrics = [
            "fog_csi",
            "fog_pod",
            "fog_precision",
            "fog_f1",
            "fog_far",
            "mist_csi",
            "mist_pod",
            "mist_precision",
            "mist_f1",
            "mist_far",
            "low_vis_csi",
            "low_vis_precision",
            "low_vis_recall",
            "low_vis_f1",
            "low_vis_fpr",
            "accuracy",
            "multiclass_brier",
            "ece_low_vis",
        ]
        delta_lookup = delta_df.set_index("metric") if not delta_df.empty else pd.DataFrame()
        for m in key_metrics:
            if m not in delta_lookup.index:
                continue
            r = delta_lookup.loc[m]
            f.write(
                f"{m},{fmt(m, 'tianji')},{fmt(m, 'ifs')},"
                f"{float(r['delta_tianji_minus_ifs']):.4f},"
                f"{r['preferred_direction']},{bool(r['tianji_better'])}\n"
            )
        if not boot_df.empty:
            f.write("\nPaired bootstrap 95% CI for delta_tianji_minus_ifs\n")
            f.write(boot_df.to_csv(index=False))

        if ifs_diagnostic_df is not None and not ifs_diagnostic_df.empty:
            f.write("\nIFS diagnostic-visibility matched comparison\n")
            f.write("-" * 52 + "\n")
            f.write("Scope: only test rows with matched finite IFS VIS values.\n")
            f.write(f"IFS diagnostic file: {args.ifs_forecast_nc}\n")
            f.write(f"IFS diagnostic variable: {args.ifs_forecast_var}\n")
            f.write(f"Matched rows: {n_ifs_diagnostic}\n")
            f.write(
                "metric,tianji_pmst,ifs_input_pmst,ifs_diagnostic,"
                "delta_tianji_minus_ifs_diagnostic,preferred_direction,tianji_better\n"
            )
            diag_delta_lookup = (
                ifs_diagnostic_delta_df.set_index("metric")
                if ifs_diagnostic_delta_df is not None and not ifs_diagnostic_delta_df.empty
                else pd.DataFrame()
            )
            for m in key_metrics:
                if m not in ifs_diagnostic_df.columns:
                    continue
                diag_val = fmt_from(ifs_diagnostic_df, m, "ifs_diagnostic")
                if diag_val == "NA":
                    continue
                delta_val = "NA"
                direction = metric_direction(m)
                better = "NA"
                if m in diag_delta_lookup.index:
                    r = diag_delta_lookup.loc[m]
                    delta_val = f"{float(r['delta_tianji_minus_ifs_diagnostic']):.4f}"
                    direction = str(r["preferred_direction"])
                    better = str(bool(r["tianji_better"]))
                f.write(
                    f"{m},{fmt_from(ifs_diagnostic_df, m, 'tianji')},"
                    f"{fmt_from(ifs_diagnostic_df, m, 'ifs')},"
                    f"{diag_val},"
                    f"{delta_val},{direction},{better}\n"
                )


def plot_key_metrics_figure(overall_df: pd.DataFrame, out_dir: Path) -> List[str]:
    """One publication-style figure for Fog/Mist/low-vis key metrics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting env dependent.
        print(f"[WARN] matplotlib unavailable; skip key-metrics figure: {exc}", flush=True)
        return []

    source_order = [s for s in ("tianji", "ifs", "ifs_diagnostic") if s in set(overall_df["source"].astype(str))]
    if not source_order:
        return []

    row_by_source = {
        str(row["source"]): row
        for _, row in overall_df.iterrows()
        if str(row.get("source", "")) in source_order
    }
    source_labels = {
        "tianji": "Tianji-trained",
        "ifs": "IFS-trained",
        "ifs_diagnostic": "IFS diagnostic VIS",
    }
    source_colors = {
        "tianji": "#2E5A87",
        "ifs": "#6C6C6C",
        "ifs_diagnostic": "#E69F00",
    }

    panels = [
        (
            "Fog (0-500 m)",
            [
                ("fog_precision", "Precision"),
                ("fog_pod", "Recall"),
                ("fog_f1", "F1"),
                ("fog_csi", "CSI"),
            ],
        ),
        (
            "Mist (500-1000 m)",
            [
                ("mist_precision", "Precision"),
                ("mist_pod", "Recall"),
                ("mist_f1", "F1"),
                ("mist_csi", "CSI"),
            ],
        ),
        (
            "Low visibility (<1000 m)",
            [
                ("low_vis_precision", "Precision"),
                ("low_vis_recall", "Recall"),
                ("low_vis_f1", "F1"),
                ("low_vis_csi", "CSI"),
                ("low_vis_fpr", "FPR"),
            ],
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_sources = len(source_order)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=False, constrained_layout=True)
    panel_letters = ["a", "b", "c"]

    def _adaptive_score_ylim(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.20
        vmax = float(np.nanmax(arr))
        if vmax <= 0:
            return 0.10
        padded = min(1.0, vmax + max(0.025, 0.10 * vmax))
        step = 0.02 if padded <= 0.20 else 0.05
        upper = max(step * 3, math.ceil(padded / step) * step)
        if vmax < 0.80:
            upper = min(upper, 0.85)
        elif vmax < 0.90:
            upper = min(upper, 0.95)
        return min(1.0, upper)

    for ax_idx, (ax, (title, metrics)) in enumerate(zip(axes, panels)):
        x = np.arange(len(metrics), dtype=np.float64)
        width = min(0.34, 0.78 / max(n_sources, 1))
        panel_values: List[float] = []
        for source in source_order:
            row = row_by_source[source]
            for metric, _ in metrics:
                try:
                    panel_values.append(float(row.get(metric, np.nan)))
                except Exception:
                    panel_values.append(math.nan)
        y_max = _adaptive_score_ylim(panel_values)
        for src_idx, source in enumerate(source_order):
            row = row_by_source[source]
            vals: List[float] = []
            finite_flags: List[bool] = []
            for metric, _ in metrics:
                val = row.get(metric, np.nan)
                try:
                    val = float(val)
                except Exception:
                    val = math.nan
                finite_flags.append(bool(np.isfinite(val)))
                vals.append(val if np.isfinite(val) else 0.0)

            offset = (src_idx - (n_sources - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                vals,
                width * 0.92,
                label=source_labels.get(source, source) if ax_idx == 0 else None,
                color=source_colors.get(source, "#7F7F7F"),
                edgecolor="white",
                linewidth=0.45,
                alpha=0.96,
            )
            for bar, val, ok in zip(bars, vals, finite_flags):
                if ok:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(val + y_max * 0.025, y_max * 0.98),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.text(
            -0.13,
            1.04,
            f"({panel_letters[ax_idx]})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
        if ax_idx == 0:
            ax.set_ylabel("Score")
        if title.startswith("Low visibility"):
            ax.text(
                0.98,
                0.96,
                "FPR lower is better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#444444",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(handles), frameon=False)

    out_paths = [
        out_dir / "fig_forecast_source_key_metrics.png",
        out_dir / "fig_forecast_source_key_metrics.pdf",
    ]
    for path in out_paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  [Fig] Saved -> {path}", flush=True)
    plt.close(fig)
    return [str(p) for p in out_paths]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "tianji": SourceSpec(
            name="tianji",
            data_dir=args.tianji_data_dir,
            ckpt_path=args.tianji_ckpt
            or default_ckpt_path("tianji", args.ckpt_dir, args.checkpoint_tag),
            scaler_path=args.tianji_scaler
            or default_scaler_path("tianji", args.ckpt_dir, args.window, args.dyn_vars_count),
        ),
        "ifs": SourceSpec(
            name="ifs",
            data_dir=args.ifs_data_dir,
            ckpt_path=args.ifs_ckpt or default_ckpt_path("ifs", args.ckpt_dir, args.checkpoint_tag),
            scaler_path=args.ifs_scaler
            or default_scaler_path("ifs", args.ckpt_dir, args.window, args.dyn_vars_count),
        ),
    }

    for source, spec in specs.items():
        if not os.path.isdir(spec.data_dir):
            raise FileNotFoundError(f"{source} data_dir does not exist: {spec.data_dir}")
        require_file(spec.ckpt_path, f"{source} checkpoint")
        require_file(spec.scaler_path, f"{source} scaler")

    train_mod = import_training_module()
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)

    build_configs = {source: read_build_config(spec.data_dir) for source, spec in specs.items()}
    evals = {
        source: evaluate_one_source(source, spec, train_mod, args, device)
        for source, spec in specs.items()
    }

    idx_t, idx_i, meta_common = align_test_outputs(evals["tianji"], evals["ifs"], args.strict_meta)
    validate_paired_labels(evals["tianji"], evals["ifs"], idx_t, idx_i)

    y = evals["tianji"].test_targets[idx_t]
    raw_vis = evals["tianji"].test_raw_vis[idx_t]
    t_probs = evals["tianji"].test_probs[idx_t]
    i_probs = evals["ifs"].test_probs[idx_i]
    t_preds = evals["tianji"].test_preds[idx_t]
    i_preds = evals["ifs"].test_preds[idx_i]

    metrics_t = compute_metrics(y, t_preds, probs=t_probs)
    metrics_i = compute_metrics(y, i_preds, probs=i_probs)
    overall_df = pd.DataFrame(
        [
            rows_from_metrics(
                "tianji",
                metrics_t,
                {
                    "temperature": evals["tianji"].temperature,
                    "fog_threshold": evals["tianji"].thresholds.get("fog"),
                    "mist_threshold": evals["tianji"].thresholds.get("mist"),
                    "feature_dim": evals["tianji"].feature_dim,
                    "extra_feat_dim": evals["tianji"].extra_feat_dim,
                    "checkpoint": evals["tianji"].spec.ckpt_path,
                    "data_dir": evals["tianji"].spec.data_dir,
                },
            ),
            rows_from_metrics(
                "ifs",
                metrics_i,
                {
                    "temperature": evals["ifs"].temperature,
                    "fog_threshold": evals["ifs"].thresholds.get("fog"),
                    "mist_threshold": evals["ifs"].thresholds.get("mist"),
                    "feature_dim": evals["ifs"].feature_dim,
                    "extra_feat_dim": evals["ifs"].extra_feat_dim,
                    "checkpoint": evals["ifs"].spec.ckpt_path,
                    "data_dir": evals["ifs"].spec.data_dir,
                },
            ),
        ]
    )
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)

    validation_df = pd.DataFrame(
        [
            rows_from_metrics(
                "tianji",
                evals["tianji"].val_metrics,
                {
                    "temperature": evals["tianji"].temperature,
                    "fog_threshold": evals["tianji"].thresholds.get("fog"),
                    "mist_threshold": evals["tianji"].thresholds.get("mist"),
                },
            ),
            rows_from_metrics(
                "ifs",
                evals["ifs"].val_metrics,
                {
                    "temperature": evals["ifs"].temperature,
                    "fog_threshold": evals["ifs"].thresholds.get("fog"),
                    "mist_threshold": evals["ifs"].thresholds.get("mist"),
                },
            ),
        ]
    )
    validation_df.to_csv(out_dir / "validation_metrics.csv", index=False)

    delta_df = compare_overall(metrics_t, metrics_i)
    delta_df.to_csv(out_dir / "metric_deltas_tianji_minus_ifs.csv", index=False)

    write_confusion_csv(out_dir / "confusion_tianji.csv", y, t_preds)
    write_confusion_csv(out_dir / "confusion_ifs.csv", y, i_preds)

    ifs_diag_preds: Optional[np.ndarray] = None
    ifs_diag_vis_raw: Optional[np.ndarray] = None
    ifs_diag_valid: Optional[np.ndarray] = None
    ifs_diag_metrics_df: Optional[pd.DataFrame] = None
    ifs_diag_delta_df: Optional[pd.DataFrame] = None
    if not args.skip_ifs_forecast_baseline:
        try:
            ifs_diag_preds, ifs_diag_vis_raw, ifs_diag_valid = load_ifs_forecast_baseline(
                meta_common,
                args.ifs_forecast_nc,
                args.ifs_forecast_var,
            )
        except Exception as exc:
            print(f"[WARN] IFS diagnostic baseline skipped: {exc}", flush=True)
            ifs_diag_valid = np.zeros(len(y), dtype=bool)

        if ifs_diag_valid is not None and int(ifs_diag_valid.sum()) > 0 and ifs_diag_preds is not None:
            valid_diag = ifs_diag_valid
            y_diag = y[valid_diag]
            t_preds_diag = t_preds[valid_diag]
            i_preds_diag = i_preds[valid_diag]
            d_preds_diag = ifs_diag_preds[valid_diag]
            t_probs_diag = t_probs[valid_diag]
            i_probs_diag = i_probs[valid_diag]

            metrics_t_diag = compute_metrics(y_diag, t_preds_diag, probs=t_probs_diag)
            metrics_i_diag = compute_metrics(y_diag, i_preds_diag, probs=i_probs_diag)
            metrics_d_diag = compute_metrics(y_diag, d_preds_diag, probs=None)

            ifs_diag_metrics_df = pd.DataFrame(
                [
                    rows_from_metrics(
                        "tianji",
                        metrics_t_diag,
                        {
                            "sample_scope": "ifs_diagnostic_matched_test",
                            "matched_rows": int(valid_diag.sum()),
                            "temperature": evals["tianji"].temperature,
                            "fog_threshold": evals["tianji"].thresholds.get("fog"),
                            "mist_threshold": evals["tianji"].thresholds.get("mist"),
                        },
                    ),
                    rows_from_metrics(
                        "ifs",
                        metrics_i_diag,
                        {
                            "sample_scope": "ifs_diagnostic_matched_test",
                            "matched_rows": int(valid_diag.sum()),
                            "temperature": evals["ifs"].temperature,
                            "fog_threshold": evals["ifs"].thresholds.get("fog"),
                            "mist_threshold": evals["ifs"].thresholds.get("mist"),
                        },
                    ),
                    rows_from_metrics(
                        "ifs_diagnostic",
                        metrics_d_diag,
                        {
                            "sample_scope": "ifs_diagnostic_matched_test",
                            "matched_rows": int(valid_diag.sum()),
                            "ifs_forecast_nc": args.ifs_forecast_nc,
                            "ifs_forecast_var": args.ifs_forecast_var,
                        },
                    ),
                ]
            )
            ifs_diag_metrics_df.to_csv(out_dir / "ifs_diagnostic_matched_metrics.csv", index=False)
            ifs_diag_delta_df = compare_metric_sets(
                metrics_t_diag,
                metrics_d_diag,
                "tianji",
                "ifs_diagnostic",
            )
            ifs_diag_delta_df.to_csv(
                out_dir / "metric_deltas_tianji_minus_ifs_diagnostic.csv",
                index=False,
            )
            compare_metric_sets(metrics_i_diag, metrics_d_diag, "ifs", "ifs_diagnostic").to_csv(
                out_dir / "metric_deltas_ifs_pmst_minus_ifs_diagnostic.csv",
                index=False,
            )
            write_confusion_csv(out_dir / "confusion_tianji_ifs_diagnostic_matched.csv", y_diag, t_preds_diag)
            write_confusion_csv(out_dir / "confusion_ifs_pmst_ifs_diagnostic_matched.csv", y_diag, i_preds_diag)
            write_confusion_csv(out_dir / "confusion_ifs_diagnostic.csv", y_diag, d_preds_diag)
        elif not args.skip_ifs_forecast_baseline:
            print("[WARN] IFS diagnostic baseline matched 0 finite test samples.", flush=True)

    scenario_rows = []
    for scenario, mask in scenario_masks(meta_common).items():
        mt = compute_metrics(y[mask], t_preds[mask], probs=t_probs[mask])
        mi = compute_metrics(y[mask], i_preds[mask], probs=i_probs[mask])
        scenario_rows.append(rows_from_metrics("tianji", mt, {"scenario": scenario}))
        scenario_rows.append(rows_from_metrics("ifs", mi, {"scenario": scenario}))
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)

    scenario_delta_rows = []
    for scenario, mask in scenario_masks(meta_common).items():
        mt = compute_metrics(y[mask], t_preds[mask], probs=t_probs[mask])
        mi = compute_metrics(y[mask], i_preds[mask], probs=i_probs[mask])
        d = compare_overall(mt, mi)
        d.insert(0, "scenario", scenario)
        scenario_delta_rows.append(d)
    if scenario_delta_rows:
        pd.concat(scenario_delta_rows, ignore_index=True).to_csv(
            out_dir / "scenario_metric_deltas_tianji_minus_ifs.csv", index=False
        )

    if ifs_diag_valid is not None and ifs_diag_preds is not None and int(ifs_diag_valid.sum()) > 0:
        diag_scenario_rows = []
        for scenario, mask in scenario_masks(meta_common).items():
            diag_mask = mask & ifs_diag_valid
            if int(diag_mask.sum()) == 0:
                continue
            mt = compute_metrics(y[diag_mask], t_preds[diag_mask], probs=t_probs[diag_mask])
            mi = compute_metrics(y[diag_mask], i_preds[diag_mask], probs=i_probs[diag_mask])
            md = compute_metrics(y[diag_mask], ifs_diag_preds[diag_mask], probs=None)
            extra = {"scenario": scenario, "sample_scope": "ifs_diagnostic_matched_test"}
            diag_scenario_rows.append(rows_from_metrics("tianji", mt, extra))
            diag_scenario_rows.append(rows_from_metrics("ifs", mi, extra))
            diag_scenario_rows.append(rows_from_metrics("ifs_diagnostic", md, extra))
        if diag_scenario_rows:
            pd.DataFrame(diag_scenario_rows).to_csv(
                out_dir / "ifs_diagnostic_scenario_metrics.csv",
                index=False,
            )

    boot_df = pd.DataFrame()
    if not args.skip_bootstrap and args.bootstrap > 0:
        boot_df = bootstrap_delta_ci(
            y,
            t_probs,
            t_preds,
            i_probs,
            i_preds,
            BOOTSTRAP_DEFAULT_METRICS,
            args.bootstrap,
            args.bootstrap_size,
            args.bootstrap_seed,
        )
        boot_df.to_csv(out_dir / "paired_bootstrap_delta_ci.csv", index=False)

    if not args.no_per_sample_csv:
        sample_df = meta_common.reset_index(drop=True).copy()
        sample_df["y_true"] = y
        sample_df["vis_raw_m"] = raw_vis
        for source, probs, preds in (("tianji", t_probs, t_preds), ("ifs", i_probs, i_preds)):
            sample_df[f"{source}_pred"] = preds
            sample_df[f"{source}_p_fog"] = probs[:, 0]
            sample_df[f"{source}_p_mist"] = probs[:, 1]
            sample_df[f"{source}_p_clear"] = probs[:, 2]
            sample_df[f"{source}_correct"] = preds == y
        if ifs_diag_preds is not None and ifs_diag_vis_raw is not None and ifs_diag_valid is not None:
            sample_df["ifs_diagnostic_valid"] = ifs_diag_valid
            sample_df["ifs_diagnostic_vis_m"] = ifs_diag_vis_raw
            sample_df["ifs_diagnostic_pred"] = ifs_diag_preds
            sample_df["ifs_diagnostic_correct"] = ifs_diag_valid & (ifs_diag_preds == y)
        sample_df["tianji_wins"] = sample_df["tianji_correct"] & ~sample_df["ifs_correct"]
        sample_df["ifs_wins"] = sample_df["ifs_correct"] & ~sample_df["tianji_correct"]
        if "ifs_diagnostic_correct" in sample_df.columns:
            sample_df["tianji_wins_vs_ifs_diagnostic"] = (
                sample_df["ifs_diagnostic_valid"]
                & sample_df["tianji_correct"]
                & ~sample_df["ifs_diagnostic_correct"]
            )
            sample_df["ifs_diagnostic_wins_vs_tianji"] = (
                sample_df["ifs_diagnostic_valid"]
                & sample_df["ifs_diagnostic_correct"]
                & ~sample_df["tianji_correct"]
            )
        sample_df.to_csv(out_dir / "per_sample_paired_eval.csv", index=False)

    run_config = {
        "args": vars(args),
        "specs": {k: vars(v) for k, v in specs.items()},
        "build_configs": build_configs,
        "paired_rows": int(len(y)),
        "evaluation_scope": "test split; validation split is used only for calibration/threshold selection",
        "ifs_diagnostic": {
            "enabled": bool(not args.skip_ifs_forecast_baseline),
            "path": args.ifs_forecast_nc,
            "variable": args.ifs_forecast_var,
            "matched_rows": int(ifs_diag_valid.sum()) if ifs_diag_valid is not None else 0,
        },
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    write_report(
        str(out_dir / "paired_forecast_source_report.txt"),
        args,
        specs,
        build_configs,
        overall_df,
        delta_df,
        boot_df,
        len(y),
        ifs_diagnostic_df=ifs_diag_metrics_df,
        ifs_diagnostic_delta_df=ifs_diag_delta_df,
        n_ifs_diagnostic=int(ifs_diag_valid.sum()) if ifs_diag_valid is not None else 0,
    )

    if not args.no_figures:
        plot_key_metrics_figure(
            ifs_diag_metrics_df if ifs_diag_metrics_df is not None and not ifs_diag_metrics_df.empty else overall_df,
            out_dir,
        )

    print("\n[OK] wrote paired evaluation outputs to:", out_dir, flush=True)
    print(delta_df[delta_df["metric"].isin(BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False), flush=True)
    if ifs_diag_delta_df is not None and not ifs_diag_delta_df.empty:
        print(
            "\n[IFS diagnostic matched deltas: Tianji PMST minus IFS diagnostic VIS]",
            flush=True,
        )
        print(
            ifs_diag_delta_df[ifs_diag_delta_df["metric"].isin(BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
