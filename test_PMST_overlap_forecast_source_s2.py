#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired test for the S2 Tianji-vs-IFS overlap low-visibility experiments.

By default this script evaluates the current Static-MLP + RNN model trained by
train_static_rnn_overlap_baseline_s2.py:
  1) data_source=tianji, using ml_dataset_overlap_tianji_12h_pm10_pm25_baseline
  2) data_source=ifs,    using ml_dataset_overlap_ifs_12h_pm10_pm25_baseline

It is intended as a controlled data-source experiment for the paper:
same model architecture, same overlap variable layout, same observed
500/1000 m labels, paired test samples by (time, station_id), and
validation-only calibration/threshold selection.

The legacy PMST architecture remains available with --model_arch pmst for
backward-compatible audits.

Optionally, it also matches the raw IFS diagnostic visibility product
(VIS_IDW_KDTree_*.nc by default) to the same test rows, following the
loader used by vis_eval/run_paper_eval_pm10_pm25_11_s2.ipynb.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

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
DEFAULT_TIANJI_T2ND_RH2M_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m"
)
DEFAULT_TIANJI_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_common_core"
)
DEFAULT_TIANJI_T2ND_RH2M_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_common_core"
)
DEFAULT_TIANJI_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_compact_common_core_no_rh2m"
)
DEFAULT_TIANJI_T2ND_RH2M_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_compact_common_core_no_rh2m"
)
DEFAULT_IFS_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"
)
DEFAULT_IFS_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_common_core"
)
DEFAULT_IFS_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_compact_common_core_no_rh2m"
)
DEFAULT_PANGU2021_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2021_12h_pm10_pm25_common_core"
)
DEFAULT_PANGU2021_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2021_12h_pm10_pm25_compact_common_core_no_rh2m"
)
DEFAULT_ERA5_2025_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_era5_2025_12h_pm10_pm25_common_core"
)
DEFAULT_ERA5_2025_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_era5_2025_12h_pm10_pm25_compact_common_core_no_rh2m"
)
DEFAULT_IFS_FORECAST_NC = os.path.join(VIS_MLP_ROOT, "VIS_IDW_KDTree_20250101_20251231.nc")
DEFAULT_OUT_DIR = os.path.join(
    IFS_BASELINE_ROOT, "paper_eval_overlap_forecast_source_s2"
)
DEFAULT_STATIC_RNN_TRAIN_DIR = os.environ.get(
    "STATIC_RNN_TRAIN_DIR", os.path.join(VIS_MLP_ROOT, "train")
)
DEFAULT_STATIC_RNN_S1_RUN_ID = "exp_overlap_static_rnn_s1_common_core_pm10_pm25"
DEFAULT_STATIC_RNN_S1_COMPACT_RUN_ID = "exp_overlap_static_rnn_s1_compact_common_core_no_rh2m_pm10_pm25"
DEFAULT_STATIC_RNN_S1_CKPT = os.path.join(
    DEFAULT_CKPT_DIR, f"{DEFAULT_STATIC_RNN_S1_RUN_ID}_S1_best_score.pt"
)
DEFAULT_STATIC_RNN_S1_SCALER = os.path.join(
    DEFAULT_CKPT_DIR, f"robust_scaler_{DEFAULT_STATIC_RNN_S1_RUN_ID}_s1_w12_dyn19_pm.pkl"
)
DEFAULT_STATIC_RNN_S1_COMPACT_CKPT = os.path.join(
    DEFAULT_CKPT_DIR, f"{DEFAULT_STATIC_RNN_S1_COMPACT_RUN_ID}_S1_best_score.pt"
)
DEFAULT_STATIC_RNN_S1_COMPACT_SCALER = os.path.join(
    DEFAULT_CKPT_DIR, f"robust_scaler_{DEFAULT_STATIC_RNN_S1_COMPACT_RUN_ID}_s1_w12_dyn18_pm.pkl"
)

CLASS_NAMES = {0: "fog_0_500m", 1: "mist_500_1000m", 2: "clear_ge_1000m"}
SOURCE_LABELS = {
    "tianji": "Tianji-trained/Tianji-input model",
    "T2ND_rh2m": "Tianji-trained/T2ND-rh2m-input model",
    "T2ND_rh2m_common_core": "Tianji T2ND RH2M-trained",
    "tianji_compact_common_core": "Tianji compact no-RH2M-trained",
    "T2ND_rh2m_compact_common_core": "Tianji T2ND compact no-RH2M-trained",
    "ifs": "IFS-trained/IFS-input model",
    "T2ND_rh2m_source_full": "Tianji T2ND RH2M source-full-trained",
    "pangu2021_source_full": "Pangu-2021 source-full-trained",
    "pangu2025_source_full": "Pangu-2025 source-full-trained",
    "era5_2025_source_full": "ERA5-2025 source-full-trained",
    "ifs_compact_common_core": "IFS compact no-RH2M-trained",
    "pangu2021_common_core": "Pangu-2021-trained",
    "pangu2025_common_core": "Pangu-2025-trained",
    "pangu2021_compact_common_core": "Pangu-2021 compact no-RH2M-trained",
    "pangu2025_compact_common_core": "Pangu-2025 compact no-RH2M-trained",
    "era5_2025_common_core": "ERA5-2025-trained",
    "era5_2025_compact_common_core": "ERA5-2025 compact no-RH2M-trained",
    "ifs_diagnostic": "IFS diagnostic visibility",
}
ZERO_TRANSFER_SOURCE_LABELS = {
    "tianji": "S1 zero-transfer / Tianji",
    "ifs": "S1 zero-transfer / IFS",
    "T2ND_rh2m_source_full": "S1 zero-transfer / T2ND RH2M source-full",
    "pangu2021_source_full": "S1 zero-transfer / Pangu-2021 source-full",
    "pangu2025_source_full": "S1 zero-transfer / Pangu-2025 source-full",
    "era5_2025_source_full": "S1 zero-transfer / ERA5-2025 source-full",
    "T2ND_rh2m_common_core": "S1 zero-transfer / T2ND RH2M",
    "tianji_compact_common_core": "S1 compact zero-transfer / Tianji",
    "ifs_compact_common_core": "S1 compact zero-transfer / IFS",
    "T2ND_rh2m_compact_common_core": "S1 compact zero-transfer / T2ND RH2M",
    "pangu2021_common_core": "S1 zero-transfer / Pangu-2021",
    "pangu2025_common_core": "S1 zero-transfer / Pangu-2025",
    "pangu2021_compact_common_core": "S1 compact zero-transfer / Pangu-2021",
    "pangu2025_compact_common_core": "S1 compact zero-transfer / Pangu-2025",
    "era5_2025_common_core": "S1 zero-transfer / ERA5-2025",
    "era5_2025_compact_common_core": "S1 compact zero-transfer / ERA5-2025",
}

STATIC_RNN_DATASET_ARG_DEFAULTS = {
    "boundary_weight": 0.0,
    "boundary_fog_sigma": 100.0,
    "boundary_mist_sigma": 150.0,
    "physical_hard_weight": 0.0,
    "humid_rh_th": 90.0,
    "humid_dpd_th": 2.0,
    "humid_clear_vis_max": 3000.0,
    "aerosol_hard_weight": 0.0,
    "aerosol_rh_th": 85.0,
    "pm25_hard_th": 75.0,
    "pm10_hard_th": 150.0,
    "ordinal_cost_weight": 0.0,
    "sample_weight_cap": 4.0,
}

FEATURE_NAME_ALIASES = {
    "RH2M": "RH2M",
    "RH_2M": "RH2M",
    "Q1000": "Q_1000",
    "Q_1000": "Q_1000",
    "DP1000": "DP_1000",
    "DPT1000": "DP_1000",
    "DP_1000": "DP_1000",
    "RH925": "RH_925",
    "R925": "RH_925",
    "RH_925": "RH_925",
    "PRECIP": "PRECIP",
}

DEFAULT_FEATURE_SWAP_ORDER = ("RH2M", "Q_1000", "DP_1000", "RH_925", "PRECIP")
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
    dynamic_feature_order: Optional[List[str]]
    temperature: float
    thresholds: Dict[str, float]
    threshold_source: str
    val_metrics: Dict[str, float]
    test_probs: np.ndarray
    test_preds: np.ndarray
    test_targets: np.ndarray
    test_raw_vis: np.ndarray
    test_meta: Optional[pd.DataFrame]
    model: Optional[nn.Module] = None
    scaler: Optional[object] = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate paired Tianji-vs-IFS overlap S2 low-vis models on the held-out "
            "month-tail test split."
        )
    )
    ap.add_argument(
        "--tianji_source_tag",
        choices=["tianji", "T2ND_rh2m", "tianji_compact_common_core", "T2ND_rh2m_compact_common_core"],
        default=os.environ.get("OVERLAP_TIANJI_SOURCE_TAG", "tianji"),
        help="Which Tianji-input checkpoint/data naming family to use for defaults.",
    )
    ap.add_argument("--tianji_data_dir", default=os.environ.get("OVERLAP_TIANJI_DATA_DIR", ""))
    ap.add_argument("--ifs_data_dir", default=os.environ.get("OVERLAP_IFS_DATA_DIR", DEFAULT_IFS_DIR))
    ap.add_argument(
        "--extra_sources",
        default=os.environ.get("OVERLAP_EXTRA_SOURCES", ""),
        help=(
            "Semicolon-separated extra source specs: "
            "tag=data_dir|ckpt_path|scaler_path[|label]. Paths may be absolute or under VIS_MLP_ROOT."
        ),
    )
    ap.add_argument(
        "--source_subset",
        "--source-subset",
        default=os.environ.get("OVERLAP_SOURCE_SUBSET", ""),
        help="Comma/semicolon-separated source tags to evaluate after building all specs; empty means all.",
    )
    ap.add_argument(
        "--independent_sources",
        "--independent-sources",
        action="store_true",
        help="Evaluate each selected source independently and skip paired Tianji-vs-IFS diagnostics.",
    )
    ap.add_argument(
        "--zero_transfer_s1",
        "--zero-transfer-s1",
        action="store_true",
        help="Use the overlap Static-RNN S1 best checkpoint/scaler for every selected forecast source.",
    )
    ap.add_argument(
        "--allow_zero_transfer_checkpoint_thresholds",
        "--allow-zero-transfer-checkpoint-thresholds",
        action="store_true",
        help=(
            "Allow S1 zero-transfer runs to reuse checkpoint thresholds directly. "
            "By default this is blocked because it often collapses forecast-source "
            "predictions to clear under domain shift."
        ),
    )
    ap.add_argument(
        "--shared_ckpt",
        "--shared-ckpt",
        default=os.environ.get("OVERLAP_SHARED_CKPT", ""),
        help="Use this checkpoint for every selected source, overriding source-specific checkpoints.",
    )
    ap.add_argument(
        "--shared_scaler",
        "--shared-scaler",
        default=os.environ.get("OVERLAP_SHARED_SCALER", ""),
        help="Use this scaler for every selected source, overriding source-specific scalers.",
    )
    ap.add_argument(
        "--skip_validation_inference",
        "--skip-validation-inference",
        action="store_true",
        help="Skip validation inference when thresholds do not require validation search; useful for full-test audits.",
    )
    ap.add_argument("--ckpt_dir", default=os.environ.get("OVERLAP_CKPT_DIR", DEFAULT_CKPT_DIR))
    ap.add_argument("--tianji_ckpt", default="")
    ap.add_argument("--ifs_ckpt", default="")
    ap.add_argument("--tianji_scaler", default="")
    ap.add_argument("--ifs_scaler", default="")
    ap.add_argument(
        "--model_arch",
        choices=["static_rnn", "pmst"],
        default=os.environ.get("OVERLAP_MODEL_ARCH", "static_rnn"),
        help="Model family to evaluate. static_rnn is the current paper candidate; pmst keeps the legacy overlap audit path.",
    )
    ap.add_argument("--static_rnn_train_dir", default=DEFAULT_STATIC_RNN_TRAIN_DIR)
    ap.add_argument("--static_rnn_encoder", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--static_rnn_hidden_dim", type=int, default=256)
    ap.add_argument("--static_rnn_static_hidden_dim", type=int, default=96)
    ap.add_argument("--static_rnn_fe_hidden_dim", type=int, default=128)
    ap.add_argument("--static_rnn_fusion_hidden_dim", type=int, default=256)
    ap.add_argument("--static_rnn_veg_emb_dim", type=int, default=16)
    ap.add_argument("--static_rnn_rnn_layers", type=int, default=1)
    ap.add_argument("--static_rnn_dropout", type=float, default=0.2)
    ap.add_argument("--static_rnn_bidirectional", action="store_true")
    ap.add_argument("--static_rnn_pooling", choices=["mean", "last", "attention"], default="mean")
    ap.add_argument("--static_rnn_no_fe", action="store_true")
    ap.add_argument("--static_rnn_no_pm", action="store_true")
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
        choices=["checkpoint", "val_search", "argmax", "fixed"],
        default="checkpoint",
        help=(
            "Use thresholds stored in the selected checkpoint, rerun validation "
            "threshold search, argmax, or fixed fog/mist thresholds."
        ),
    )
    ap.add_argument(
        "--threshold_search_policy",
        "--threshold-search-policy",
        choices=["operational", "response"],
        default=os.environ.get("OVERLAP_THRESHOLD_SEARCH_POLICY", "operational"),
        help=(
            "Validation threshold-search objective. operational preserves precision/clear-recall "
            "guards for deployable S2 comparisons; response is a diagnostic for S1 zero-transfer "
            "that asks whether any low-visibility response can be recovered without weight transfer."
        ),
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
    ap.add_argument(
        "--local_time_offset_hours",
        type=int,
        default=8,
        help="Offset from UTC for day/night and seasonal scenario grouping.",
    )
    ap.add_argument(
        "--allow_legacy_time_alignment",
        action="store_true",
        help="Allow datasets without an accepted UTC build marker; intended only for legacy audits.",
    )
    ap.add_argument("--allow_partial_load", action="store_true")
    ap.add_argument("--limit_samples", type=int, default=0, help="Smoke-test limit for val/test rows; 0 means all.")
    ap.add_argument("--no_per_sample_csv", action="store_true")
    ap.add_argument("--no_figures", action="store_true", help="Skip publication-style summary figures.")
    ap.add_argument("--feature_importance_csv", default="", help="Optional feature-importance table used to choose replacement variables.")
    ap.add_argument("--feature_swap_top_k", type=int, default=0, help="Replace top-K dynamic variables from --feature_importance_csv; 0 disables unless --feature_swap_features is set.")
    ap.add_argument(
        "--feature_swap_features",
        default="RH2M,Q_1000,DP_1000,RH_925,PRECIP",
        help="Comma/semicolon-separated dynamic variables to replace, e.g. RH2M,Q_1000,DP_1000,RH_925.",
    )
    ap.add_argument("--feature_swap_metric", default="low_vis_recall", help="Metric highlighted in the feature-replacement figure.")
    ap.add_argument(
        "--skip_ifs_forecast_baseline",
        action="store_true",
        help="Skip the raw IFS diagnostic-visibility baseline matched from --ifs_forecast_nc.",
    )
    return ap.parse_args()


def default_run_id(source: str, model_arch: str) -> str:
    if model_arch == "static_rnn":
        return f"exp_overlap_static_rnn_s2_{source}_pm10_pm25"
    return f"exp_overlap_pmst_baseline_s2_{source}_pm10_pm25"


def default_tianji_data_dir(source_tag: str) -> str:
    if source_tag == "T2ND_rh2m":
        return DEFAULT_TIANJI_T2ND_RH2M_DIR
    if source_tag == "tianji_common_core":
        return DEFAULT_TIANJI_COMMON_CORE_DIR
    if source_tag == "T2ND_rh2m_common_core":
        return DEFAULT_TIANJI_T2ND_RH2M_COMMON_CORE_DIR
    if source_tag == "tianji_compact_common_core":
        return DEFAULT_TIANJI_COMPACT_COMMON_CORE_DIR
    if source_tag == "T2ND_rh2m_compact_common_core":
        return DEFAULT_TIANJI_T2ND_RH2M_COMPACT_COMMON_CORE_DIR
    return DEFAULT_TIANJI_DIR


def resolve_under_root(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return value
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str(Path(VIS_MLP_ROOT) / p)


def default_ckpt_path(source: str, ckpt_dir: str, checkpoint_tag: str, model_arch: str) -> str:
    run_exp_id = default_run_id(source, model_arch)
    return os.path.join(ckpt_dir, f"{run_exp_id}_{checkpoint_tag}.pt")


def default_scaler_path(
    source: str,
    ckpt_dir: str,
    window: int,
    dyn_vars_count: int,
    model_arch: str,
    static_no_pm: bool = False,
) -> str:
    if model_arch == "static_rnn":
        pm_tag = "nopm" if static_no_pm else "pm"
        run_id = default_run_id(source, model_arch)
        return os.path.join(ckpt_dir, f"robust_scaler_{run_id}_s2_w{window}_dyn{dyn_vars_count}_{pm_tag}.pkl")
    return os.path.join(ckpt_dir, f"robust_scaler_w{window}_dyn{dyn_vars_count}_overlap_baseline_{source}.pkl")


def read_dataset_build_config(data_dir: str) -> Dict[str, object]:
    cfg_path = os.path.join(str(data_dir), "dataset_build_config.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        print(f"[layout] warning: failed to read {cfg_path}: {exc}", flush=True)
        return {}


def dataset_layout_from_config(data_dir: str) -> Tuple[Optional[int], Optional[int], Optional[List[str]]]:
    cfg = read_dataset_build_config(data_dir)
    dyn = cfg.get("dyn_vars")
    fe = cfg.get("fe_dim")
    order = cfg.get("dynamic_feature_order")
    order_list = [str(v) for v in order] if isinstance(order, list) else None
    dyn_i = int(dyn) if dyn is not None else None
    fe_i = int(fe) if fe is not None else None
    if dyn_i is not None and order_list is not None and len(order_list) != dyn_i:
        raise ValueError(
            f"{data_dir}: dataset_build_config dynamic_feature_order length "
            f"{len(order_list)} != dyn_vars {dyn_i}"
        )
    return dyn_i, fe_i, order_list


def source_dyn_vars_count(data_dir: str, fallback: int) -> int:
    cfg_dyn, _, _ = dataset_layout_from_config(data_dir)
    return int(cfg_dyn if cfg_dyn is not None else fallback)


def checkpoint_run_id(ckpt_path: str, checkpoint_tag: str) -> str:
    name = Path(ckpt_path).name
    suffix = f"_{checkpoint_tag}.pt"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".pt"):
        return name[:-3]
    return Path(name).stem


def default_scaler_path_for_spec(
    spec: SourceSpec,
    source: str,
    args: argparse.Namespace,
    dyn_vars_count: int,
) -> str:
    if args.model_arch == "static_rnn":
        pm_tag = "nopm" if args.static_rnn_no_pm else "pm"
        run_id = checkpoint_run_id(spec.ckpt_path, args.checkpoint_tag)
        ckpt_parent = Path(spec.ckpt_path).expanduser().parent if spec.ckpt_path else Path(args.ckpt_dir)
        scaler_dir = str(ckpt_parent if str(ckpt_parent) not in {"", "."} else Path(args.ckpt_dir))
        return os.path.join(
            scaler_dir,
            f"robust_scaler_{run_id}_s2_w{args.window}_dyn{dyn_vars_count}_{pm_tag}.pkl",
        )
    return default_scaler_path(
        source,
        args.ckpt_dir,
        args.window,
        dyn_vars_count,
        args.model_arch,
        args.static_rnn_no_pm,
    )


def fill_auto_scaler_paths(specs: Dict[str, SourceSpec], args: argparse.Namespace) -> None:
    for source, spec in specs.items():
        if str(spec.scaler_path or "").strip().upper() not in {"", "AUTO"}:
            continue
        dyn_vars = source_dyn_vars_count(spec.data_dir, args.dyn_vars_count)
        spec.scaler_path = default_scaler_path_for_spec(spec, source, args, dyn_vars)


def parse_extra_source_specs(text: str) -> Dict[str, SourceSpec]:
    specs: Dict[str, SourceSpec] = {}
    for raw in str(text or "").split(";"):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --extra_sources item {item!r}; expected tag=data|ckpt|scaler[|label].")
        tag, payload = item.split("=", 1)
        tag = tag.strip()
        parts = [part.strip() for part in payload.split("|")]
        if len(parts) < 3:
            raise ValueError(f"Bad --extra_sources item {item!r}; expected tag=data|ckpt|scaler[|label].")
        if tag in {"tianji", "ifs", "ifs_diagnostic"}:
            raise ValueError(f"--extra_sources tag {tag!r} is reserved.")
        if len(parts) >= 4 and parts[3]:
            SOURCE_LABELS[tag] = parts[3]
        specs[tag] = SourceSpec(
            name=tag,
            data_dir=resolve_under_root(parts[0]),
            ckpt_path=resolve_under_root(parts[1]),
            scaler_path="" if parts[2].upper() == "AUTO" else resolve_under_root(parts[2]),
        )
    return specs


def split_source_tags(value: str) -> List[str]:
    aliases = {
        "tianji_common_core": "tianji",
        "ifs_common_core": "ifs",
        "T2ND_rh2m": "T2ND_rh2m_common_core",
        "pangu2021": "pangu2021_common_core",
        "pangu2025": "pangu2025_common_core",
        "era5_2025": "era5_2025_common_core",
        "tianji_compact": "tianji",
        "ifs_compact": "ifs",
        "T2ND_rh2m_compact": "T2ND_rh2m_compact_common_core",
        "pangu2021_compact": "pangu2021_compact_common_core",
        "pangu2025_compact": "pangu2025_compact_common_core",
        "era5_2025_compact": "era5_2025_compact_common_core",
    }
    tags: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        tag = chunk.strip()
        tag = aliases.get(tag, tag)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def filter_source_specs(specs: Dict[str, SourceSpec], subset_text: str) -> Dict[str, SourceSpec]:
    subset = split_source_tags(subset_text)
    if not subset:
        return specs
    missing = [tag for tag in subset if tag not in specs]
    if missing:
        raise KeyError(f"--source_subset contains unknown source tag(s): {missing}; available={list(specs)}")
    return {tag: specs[tag] for tag in subset}


def apply_shared_checkpoint_scaler(specs: Dict[str, SourceSpec], args: argparse.Namespace) -> None:
    shared_ckpt = str(args.shared_ckpt or "").strip()
    shared_scaler = str(args.shared_scaler or "").strip()
    if args.zero_transfer_s1:
        use_compact_s1 = int(args.dyn_vars_count) == 18 or any(
            "compact_common_core" in str(spec.data_dir) for spec in specs.values()
        )
        shared_ckpt = shared_ckpt or (
            DEFAULT_STATIC_RNN_S1_COMPACT_CKPT if use_compact_s1 else DEFAULT_STATIC_RNN_S1_CKPT
        )
        shared_scaler = shared_scaler or (
            DEFAULT_STATIC_RNN_S1_COMPACT_SCALER if use_compact_s1 else DEFAULT_STATIC_RNN_S1_SCALER
        )
    if not shared_ckpt and not shared_scaler:
        return
    if not shared_ckpt or not shared_scaler:
        raise ValueError("Shared source evaluation requires both --shared_ckpt and --shared_scaler.")
    shared_ckpt = resolve_under_root(shared_ckpt)
    shared_scaler = resolve_under_root(shared_scaler)
    for source, spec in specs.items():
        spec.ckpt_path = shared_ckpt
        spec.scaler_path = shared_scaler
        if args.zero_transfer_s1:
            SOURCE_LABELS[source] = ZERO_TRANSFER_SOURCE_LABELS.get(source, f"S1 zero-transfer / {source}")


def load_checkpoint_payload(ckpt_path: str, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def checkpoint_metadata(ckpt_path: str, device: torch.device) -> Dict[str, object]:
    payload = load_checkpoint_payload(ckpt_path, torch.device("cpu"))
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def checkpoint_thresholds(ckpt_meta: Dict[str, object]) -> Optional[Dict[str, float]]:
    raw = ckpt_meta.get("thresholds")
    if not isinstance(raw, dict):
        raw = {
            "fog": ckpt_meta.get("fog_threshold", ckpt_meta.get("fog_th")),
            "mist": ckpt_meta.get("mist_threshold", ckpt_meta.get("mist_th")),
        }
    try:
        fog = float(raw["fog"])
        mist = float(raw["mist"])
    except Exception:
        return None
    if not math.isfinite(fog) or not math.isfinite(mist):
        return None
    return {"fog": fog, "mist": mist}


def resolve_static_rnn_train_dir(train_dir: str) -> Path:
    candidates = [
        Path(train_dir).expanduser(),
        Path(__file__).resolve().parent.parent / "train",
        Path(__file__).resolve().parent.parent / "vis_mlp",
    ]
    for path in candidates:
        if (path / "train_static_rnn_lowvis.py").is_file():
            return path.resolve()
    checked = "\n  ".join(str(p / "train_static_rnn_lowvis.py") for p in candidates)
    raise FileNotFoundError(f"Cannot find train_static_rnn_lowvis.py. Checked:\n  {checked}")


def import_training_module(args: argparse.Namespace):
    if args.model_arch == "static_rnn":
        module_path = resolve_static_rnn_train_dir(args.static_rnn_train_dir) / "train_static_rnn_lowvis.py"
        module_name = "static_rnn_lowvis_train"
    else:
        module_path = Path(__file__).resolve().parent / "train_PMST_overlap_baseline_s2.py"
        module_name = "pmst_overlap_train_s2"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find training module: {module_path}")

    old_argv = sys.argv[:]
    try:
        # The training script parses known args at import time. Keep it isolated
        # from this evaluator's CLI.
        sys.argv = [str(module_path)]
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
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


def populated_overlap_features(data_dir: str) -> Set[str]:
    cfg = read_build_config(data_dir)
    explicit = cfg.get("overlap_vars") or cfg.get("dynamic_features") or cfg.get("feature_order")
    if not explicit:
        return set()
    available = {str(v) for v in explicit}
    if {"U10", "V10"}.issubset(available):
        available.add("WSPD10")
    return available


def validate_build_time_alignment(build_configs: Dict[str, Dict], allow_legacy: bool) -> None:
    expected = {"bjt_minus_8_to_utc", "raw_utc_no_shift"}
    for source, cfg in build_configs.items():
        marker = cfg.get("tianji_raw_time_alignment") if cfg else None
        if str(cfg.get("time_coordinate", "")).upper() == "UTC":
            continue
        if marker in expected:
            continue
        msg = (
            f"{source} dataset was not built with the required Tianji UTC marker "
            f"({sorted(expected)!r}); got {marker!r}. Rebuild the overlap dataset with the "
            "current builders before using this comparison in the paper."
        )
        if allow_legacy:
            print(f"[WARN] {msg}", flush=True)
        else:
            raise RuntimeError(msg)


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
    cfg_dyn, cfg_fe, _ = dataset_layout_from_config(data_dir)
    dyn_vars_count = int(cfg_dyn if cfg_dyn is not None else dyn_vars_count)
    base_dim = window * dyn_vars_count + 5 + 1
    extra_dim = feature_dim - base_dim
    if extra_dim <= 0:
        raise ValueError(
            f"Invalid feature layout for {x_path}: feature_dim={feature_dim}, "
            f"base_dim={base_dim}, extra_dim={extra_dim}"
        )
    if cfg_fe is not None and int(extra_dim) != int(cfg_fe):
        raise ValueError(
            f"{data_dir} extra feature dim is {extra_dim}, but dataset_build_config fe_dim is {cfg_fe}."
        )
    if cfg_dyn is None and expected_extra_dim > 0 and extra_dim != expected_extra_dim:
        raise ValueError(
            f"{data_dir} extra feature dim is {extra_dim}, expected {expected_extra_dim}. "
            "This usually means a stale PM10-only or wrong-layout overlap dataset."
        )
    return feature_dim, int(extra_dim)


def build_y_cls_raw(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_raw = np.asarray(y_raw, dtype=np.float32).copy()
    if len(y_raw) > 0 and np.nanmax(y_raw) < 100:
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


def ensure_static_rnn_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    for name, default in STATIC_RNN_DATASET_ARG_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def make_static_layout(
    train_mod,
    window: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    dynamic_feature_order: Optional[List[str]] = None,
):
    try:
        return train_mod.Layout(
            window_size=window,
            dyn_vars=dyn_vars_count,
            fe_dim=extra_feat_dim,
            dynamic_feature_order=dynamic_feature_order,
        )
    except TypeError:
        return train_mod.Layout(window_size=window, dyn_vars=dyn_vars_count, fe_dim=extra_feat_dim)


def make_dataset(
    train_mod,
    data_dir: str,
    split: str,
    scaler,
    window: int,
    dyn_vars_count: int,
    extra_feat_dim: int,
    limit_samples: int,
    model_arch: str,
    static_use_fe: bool,
    static_use_pm: bool,
    dataset_args: Optional[argparse.Namespace] = None,
    dynamic_feature_order: Optional[List[str]] = None,
):
    x_path = os.path.join(data_dir, f"X_{split}.npy")
    y_path = os.path.join(data_dir, f"y_{split}.npy")
    require_file(x_path, f"X_{split}.npy")
    require_file(y_path, f"y_{split}.npy")

    y_raw, y_cls = build_y_cls_raw(np.load(y_path))
    indices = None
    if limit_samples and limit_samples > 0:
        indices = np.arange(min(limit_samples, len(y_cls)), dtype=np.int64)

    if model_arch == "static_rnn":
        layout = make_static_layout(train_mod, window, dyn_vars_count, extra_feat_dim, dynamic_feature_order)
        ctor = inspect.signature(train_mod.LowVisDataset)
        kwargs = {"use_fe": static_use_fe, "use_pm": static_use_pm}
        if "args" in ctor.parameters:
            kwargs["args"] = ensure_static_rnn_dataset_args(dataset_args or argparse.Namespace())
        base_ds = train_mod.LowVisDataset(x_path, y_raw, y_cls, layout, scaler, **kwargs)
        ds = Subset(base_ds, indices.tolist()) if indices is not None else base_ds
    else:
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


def reset_dataset_cache(dataset) -> None:
    if hasattr(dataset, "X"):
        dataset.X = None
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "X"):
        dataset.dataset.X = None


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        reset_dataset_cache(worker_info.dataset)


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
    args: argparse.Namespace,
    dynamic_feature_order: Optional[List[str]] = None,
) -> nn.Module:
    require_file(ckpt_path, "checkpoint")
    if args.model_arch == "static_rnn":
        layout = make_static_layout(train_mod, window, dyn_vars_count, extra_feat_dim, dynamic_feature_order)
        model = train_mod.StaticRNNLowVisNet(
            layout=layout,
            encoder=args.static_rnn_encoder,
            hidden_dim=args.static_rnn_hidden_dim,
            static_hidden_dim=args.static_rnn_static_hidden_dim,
            fe_hidden_dim=args.static_rnn_fe_hidden_dim,
            fusion_hidden_dim=args.static_rnn_fusion_hidden_dim,
            veg_emb_dim=args.static_rnn_veg_emb_dim,
            rnn_layers=args.static_rnn_rnn_layers,
            dropout=args.static_rnn_dropout,
            bidirectional=args.static_rnn_bidirectional,
            pooling=args.static_rnn_pooling,
            use_fe=not args.static_rnn_no_fe,
        ).to(device)
    else:
        model = train_mod.ImprovedDualStreamPMSTNet(
            window_size=window,
            hidden_dim=train_mod.CONFIG.get("MODEL_HIDDEN_DIM", 512),
            num_classes=3,
            extra_feat_dim=extra_feat_dim,
            dyn_vars_count=dyn_vars_count,
        ).to(device)

    payload = load_checkpoint_payload(ckpt_path, device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    clean_state = {}
    for k, v in state.items():
        clean_state[k[7:] if k.startswith("module.") else k] = v

    if args.model_arch == "static_rnn" and hasattr(train_mod, "validate_pretrained_layout"):
        train_mod.validate_pretrained_layout(model, clean_state, metadata, ckpt_path, "strict")
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
    for batch in loader:
        bx, by, braw = batch[0], batch[1], batch[3]
        bx = bx.to(device, non_blocking=True)
        out = model(bx)
        logits = out[0] if isinstance(out, (tuple, list)) else out
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


def threshold_grid(policy: str = "operational") -> np.ndarray:
    if str(policy).lower() == "response":
        return np.unique(np.concatenate([np.arange(0.01, 0.50, 0.01), np.arange(0.50, 0.96, 0.03)]))
    low_part = np.arange(0.10, 0.50, 0.04)
    high_part = np.arange(0.50, 0.96, 0.03)
    return np.unique(np.concatenate([low_part, high_part]))


def search_thresholds_on_val(
    probs: np.ndarray,
    targets: np.ndarray,
    policy: str = "operational",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_score = -np.inf
    best_metrics: Optional[Dict[str, float]] = None
    best_th = {"fog": 0.5, "mist": 0.5}
    policy = str(policy or "operational").lower()
    grid = threshold_grid(policy)

    if policy == "response":
        for f_th in grid:
            for m_th in grid:
                preds = predict_from_probs(probs, "fixed", float(f_th), float(m_th))
                if not np.any(preds <= 1):
                    continue
                metrics = compute_metrics(targets, preds, probs=None, prefix_counts=False)
                score = (
                    0.30 * metrics["fog_pod"]
                    + 0.30 * metrics["mist_pod"]
                    + 0.25 * metrics["low_vis_csi"]
                    + 0.15 * metrics["low_vis_precision"]
                    - 0.10 * metrics["low_vis_fpr"]
                )
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    best_th = {"fog": float(f_th), "mist": float(m_th)}
        if best_metrics is not None:
            return best_th, best_metrics

        preds = np.argmax(probs, axis=1).astype(np.int64)
        best_metrics = compute_metrics(targets, preds, probs=None, prefix_counts=False)
        return {"fog": math.nan, "mist": math.nan}, best_metrics

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


def pred_mode_from_thresholds(thresholds: Dict[str, float]) -> str:
    try:
        fog = float(thresholds.get("fog", math.nan))
        mist = float(thresholds.get("mist", math.nan))
    except Exception:
        return "argmax"
    return "fixed" if math.isfinite(fog) and math.isfinite(mist) else "argmax"


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

    source_dyn_vars = source_dyn_vars_count(spec.data_dir, args.dyn_vars_count)
    _, _, dynamic_order = dataset_layout_from_config(spec.data_dir)
    feature_dim, extra_dim = infer_feature_layout(
        spec.data_dir, "test", args.window, source_dyn_vars, args.expected_extra_dim
    )
    if not args.skip_validation_inference:
        val_feature_dim, val_extra_dim = infer_feature_layout(
            spec.data_dir, "val", args.window, source_dyn_vars, args.expected_extra_dim
        )
        if val_feature_dim != feature_dim or val_extra_dim != extra_dim:
            raise ValueError(
                f"{source}: val/test feature layout differs: "
                f"val=({val_feature_dim},{val_extra_dim}) test=({feature_dim},{extra_dim})"
            )

    require_file(spec.scaler_path, "RobustScaler")
    scaler = joblib.load(spec.scaler_path)
    ckpt_meta = checkpoint_metadata(spec.ckpt_path, device)
    model = load_model(
        train_mod,
        spec.ckpt_path,
        device,
        args.window,
        source_dyn_vars,
        extra_dim,
        args.allow_partial_load,
        args,
        dynamic_order,
    )

    test_ds, test_meta = make_dataset(
        train_mod,
        spec.data_dir,
        "test",
        scaler,
        args.window,
        source_dyn_vars,
        extra_dim,
        args.limit_samples,
        args.model_arch,
        not args.static_rnn_no_fe,
        not args.static_rnn_no_pm,
        args,
        dynamic_order,
    )
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    threshold_source = args.threshold_mode
    if args.skip_validation_inference:
        if args.threshold_mode == "val_search":
            raise ValueError("--skip_validation_inference cannot be used with --threshold_mode val_search.")
        temperature = 1.0
        if args.threshold_mode == "checkpoint":
            thresholds = checkpoint_thresholds(ckpt_meta)
            if thresholds is None:
                raise RuntimeError(
                    f"{source}: checkpoint has no saved thresholds, so validation inference is required."
                )
            val_metrics = {"n": 0}
            threshold_source = "checkpoint_metadata_no_val"
            threshold_mode_for_pred = "fixed"
        elif args.threshold_mode == "fixed":
            thresholds = {"fog": float(args.fog_threshold), "mist": float(args.mist_threshold)}
            val_metrics = {"n": 0}
            threshold_mode_for_pred = "fixed"
        else:
            thresholds = {"fog": math.nan, "mist": math.nan}
            val_metrics = {"n": 0}
            threshold_mode_for_pred = "argmax"
    else:
        val_ds, _ = make_dataset(
            train_mod,
            spec.data_dir,
            "val",
            scaler,
            args.window,
            source_dyn_vars,
            extra_dim,
            args.limit_samples,
            args.model_arch,
            not args.static_rnn_no_fe,
            not args.static_rnn_no_pm,
            args,
            dynamic_order,
        )
        val_loader = make_loader(val_ds, args.batch_size, args.num_workers)
        print(f"[{source}] running validation inference: N={len(val_ds)}", flush=True)
        val_logits, val_targets, _ = collect_logits(model, val_loader, device)
        if args.threshold_mode == "checkpoint":
            # Checkpoint thresholds were selected from uncalibrated validation
            # probabilities by the trainer, so keep the probability scale unchanged.
            temperature = 1.0
        elif args.no_temp_scaling:
            temperature = 1.0
        else:
            temperature = calibrate_temperature_from_logits(
                val_logits, val_targets, device, args.temp_lr, args.temp_max_iter
            )
        val_probs = softmax_np(val_logits, temperature)

        if args.threshold_mode == "checkpoint":
            thresholds = checkpoint_thresholds(ckpt_meta)
            if thresholds is None:
                print(f"[{source}] checkpoint has no saved thresholds; falling back to validation search.", flush=True)
                thresholds, val_metrics = search_thresholds_on_val(
                    val_probs,
                    val_targets,
                    args.threshold_search_policy,
                )
                threshold_source = "val_search_fallback_no_checkpoint_thresholds"
            else:
                val_preds = predict_from_probs(val_probs, "fixed", thresholds["fog"], thresholds["mist"])
                val_metrics = compute_metrics(val_targets, val_preds, probs=val_probs)
                threshold_source = "checkpoint_metadata"
            threshold_mode_for_pred = pred_mode_from_thresholds(thresholds)
        elif args.threshold_mode == "val_search":
            thresholds, val_metrics = search_thresholds_on_val(
                val_probs,
                val_targets,
                args.threshold_search_policy,
            )
            threshold_mode_for_pred = pred_mode_from_thresholds(thresholds)
            threshold_source = f"val_search_{args.threshold_search_policy}"
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
        f"threshold_source={threshold_source}, "
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
        dyn_vars_count=source_dyn_vars,
        dynamic_feature_order=dynamic_order,
        temperature=float(temperature),
        thresholds=thresholds,
        threshold_source=threshold_source,
        val_metrics=val_metrics,
        test_probs=test_probs,
        test_preds=test_preds,
        test_targets=test_targets,
        test_raw_vis=test_raw,
        test_meta=test_meta,
        model=model,
        scaler=scaler,
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


def scenario_masks(meta: pd.DataFrame, local_time_offset_hours: int = 8) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {"All": np.ones(len(meta), dtype=bool)}
    if "time" not in meta.columns:
        return masks
    time = pd.to_datetime(meta["time"], errors="coerce") + pd.Timedelta(hours=local_time_offset_hours)
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
        f.write("Overlap S2 forecast-source paired evaluation\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Model architecture: {args.model_arch}\n")
        f.write("Purpose: controlled comparison of Tianji-trained and IFS-trained overlap models.\n")
        f.write("Interpretation: this tests forecast-field source quality under the same model\n")
        f.write("architecture, same observed 0-500 m / 500-1000 m / >=1000 m labels, and\n")
        f.write("paired test samples. A separate matched section compares against the raw\n")
        f.write("IFS diagnostic-visibility product when --ifs_forecast_nc is available.\n\n")
        f.write(f"Paired test rows: {n_pair}\n")
        f.write(f"Threshold mode: {args.threshold_mode}\n")
        f.write(f"Temperature scaling: {args.threshold_mode != 'checkpoint' and not args.no_temp_scaling}\n")
        f.write(f"Scenario local time offset: UTC+{args.local_time_offset_hours}\n")
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
                                "tianji_raw_time_alignment",
                                "time_coordinate",
                                "ifs_time_match",
                                "pm_time_match",
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
    """Publication-style split figures for Fog/Mist/low-vis key metrics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting env dependent.
        print(f"[WARN] matplotlib unavailable; skip key-metrics figure: {exc}", flush=True)
        return []

    available_sources = list(dict.fromkeys(overall_df["source"].astype(str).tolist()))
    available_set = set(available_sources)
    if not available_sources:
        return []

    row_by_source = {str(row["source"]): row for _, row in overall_df.iterrows()}
    source_labels = {
        "tianji": "Tianji-trained",
        "ifs": "IFS-trained",
        "T2ND_rh2m_source_full": "T2ND RH2M source-full",
        "T2ND_rh2m_common_core": "T2ND RH2M",
        "tianji_compact_common_core": "Tianji compact",
        "ifs_compact_common_core": "IFS compact",
        "T2ND_rh2m_compact_common_core": "T2ND RH2M compact",
        "pangu2021_source_full": "Pangu-2021 source-full",
        "pangu2025_source_full": "Pangu-2025 source-full",
        "pangu2021_common_core": "Pangu-2021",
        "pangu2025_common_core": "Pangu-2025",
        "pangu2021_compact_common_core": "Pangu-2021 compact",
        "pangu2025_compact_common_core": "Pangu-2025 compact",
        "era5_2025_source_full": "ERA5-2025 source-full",
        "era5_2025_common_core": "ERA5-2025",
        "era5_2025_compact_common_core": "ERA5-2025 compact",
        "ifs_diagnostic": "IFS diagnostic VIS",
    }
    for _, row in overall_df.iterrows():
        src = str(row.get("source", ""))
        label = str(row.get("source_label", "") or "").strip()
        if src and label and label != src:
            source_labels[src] = label
    source_colors = {
        "tianji": "#2E5A87",
        "ifs": "#6C6C6C",
        "T2ND_rh2m_source_full": "#1B9E77",
        "T2ND_rh2m_common_core": "#1B9E77",
        "tianji_compact_common_core": "#2E5A87",
        "ifs_compact_common_core": "#6C6C6C",
        "T2ND_rh2m_compact_common_core": "#1B9E77",
        "pangu2021_source_full": "#8E6BBE",
        "pangu2025_source_full": "#8E6BBE",
        "pangu2021_common_core": "#8E6BBE",
        "pangu2025_common_core": "#8E6BBE",
        "pangu2021_compact_common_core": "#8E6BBE",
        "pangu2025_compact_common_core": "#8E6BBE",
        "era5_2025_source_full": "#D95F02",
        "era5_2025_common_core": "#D95F02",
        "era5_2025_compact_common_core": "#D95F02",
        "ifs_diagnostic": "#E69F00",
    }
    fallback_colors = ["#4C78A8", "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2", "#E15759"]

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

    def _draw_group(source_order: List[str], stem: str) -> List[str]:
        source_order = [src for src in source_order if src in available_set and src in row_by_source]
        if not source_order:
            return []
        n_sources = len(source_order)
        fig_w = max(11.2, 10.2 + 0.74 * max(0, n_sources - 2))
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4.05), sharey=False, constrained_layout=True)

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
                    color=source_colors.get(source, fallback_colors[src_idx % len(fallback_colors)]),
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
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=min(len(handles), 5),
                frameon=False,
            )

        out_paths = [
            out_dir / f"{stem}.png",
            out_dir / f"{stem}.pdf",
            out_dir / f"{stem}.svg",
        ]
        for path in out_paths:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"  [Fig] Saved -> {path}", flush=True)
        plt.close(fig)
        return [str(p) for p in out_paths]

    written: List[str] = []
    if any("source_full" in src for src in available_sources) or any(
        "source_full" in str(row.get("data_dir", "")) for row in row_by_source.values()
    ):
        written.extend(
            _draw_group(
                [
                    "tianji",
                    "T2ND_rh2m_source_full",
                    "ifs",
                    "pangu2021_source_full",
                    "pangu2025_source_full",
                    "era5_2025_source_full",
                    "ifs_diagnostic",
                ],
                "fig_forecast_source_key_metrics_source_full",
            )
        )
    written.extend(
        _draw_group(
            ["tianji", "T2ND_rh2m_common_core", "ifs", "era5_2025_common_core", "ifs_diagnostic"],
            "fig_forecast_source_key_metrics_numerical_models",
        )
    )
    if any("compact_common_core" in src for src in available_sources):
        written.extend(
            _draw_group(
                [
                    "tianji",
                    "T2ND_rh2m_compact_common_core",
                    "ifs",
                    "era5_2025_compact_common_core",
                    "ifs_diagnostic",
                ],
                "fig_forecast_source_key_metrics_compact_common_core",
            )
        )
    written.extend(
        _draw_group(
            ["tianji", "pangu2021_common_core", "pangu2021_compact_common_core"],
            "fig_forecast_source_key_metrics_tianji_pangu",
        )
    )
    written.extend(
        _draw_group(
            ["tianji", "pangu2025_source_full", "pangu2025_common_core", "pangu2025_compact_common_core"],
            "fig_forecast_source_key_metrics_tianji_pangu2025",
        )
    )
    if not written:
        fallback_order = [
            "tianji",
            "T2ND_rh2m_source_full",
            "T2ND_rh2m_common_core",
            "T2ND_rh2m_compact_common_core",
            "ifs",
            "pangu2021_source_full",
            "pangu2025_source_full",
            "era5_2025_source_full",
            "era5_2025_common_core",
            "era5_2025_compact_common_core",
            "pangu2021_common_core",
            "pangu2025_common_core",
            "pangu2021_compact_common_core",
            "pangu2025_compact_common_core",
            "ifs_diagnostic",
        ]
        fallback_order.extend([s for s in available_sources if s not in set(fallback_order)])
        written.extend(_draw_group(fallback_order, "fig_forecast_source_key_metrics"))
    return written


def split_feature_names(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value or "").split(";"):
        for raw in chunk.split(","):
            name = raw.strip()
            if not name:
                continue
            key = name.upper().replace("-", "_").replace(" ", "_")
            canon = FEATURE_NAME_ALIASES.get(key, name)
            if canon not in out:
                out.append(canon)
    return out


def dynamic_feature_lookup(dyn_vars_count: int) -> Dict[str, int]:
    if int(dyn_vars_count) == 18:
        compact = [
            "T2M",
            "MSLP",
            "U10",
            "WSPD10",
            "V10",
            "WDIR10",
            "RH_925",
            "U_925",
            "WSPD925",
            "V_925",
            "DP_1000",
            "DP_925",
            "Q_1000",
            "Q_925",
            "DPD",
            "zenith",
            "PM10_ugm3",
            "PM25_ugm3",
        ]
        return {name: i for i, name in enumerate(compact)}
    if int(dyn_vars_count) == 19:
        legacy_compact = [
            "RH2M",
            "T2M",
            "MSLP",
            "U10",
            "WSPD10",
            "V10",
            "WDIR10",
            "RH_925",
            "U_925",
            "WSPD925",
            "V_925",
            "DP_1000",
            "DP_925",
            "Q_1000",
            "Q_925",
            "DPD",
            "zenith",
            "PM10_ugm3",
            "PM25_ugm3",
        ]
        return {name: i for i, name in enumerate(legacy_compact)}
    vis_eval_dir = Path(__file__).resolve().parent.parent / "vis_eval"
    if str(vis_eval_dir) not in sys.path:
        sys.path.insert(0, str(vis_eval_dir))
    try:
        from feature_catalog_pm10_pm25 import dynamic_features_for_count

        return {item["feature"]: i for i, item in enumerate(dynamic_features_for_count(dyn_vars_count))}
    except Exception:
        fallback = [
            "RH2M",
            "T2M",
            "PRECIP",
            "MSLP",
            "SW_RAD",
            "U10",
            "WSPD10",
            "V10",
            "WDIR10",
            "CAPE",
            "LCC",
            "T_925",
            "RH_925",
            "U_925",
            "WSPD925",
            "V_925",
            "DP_1000",
            "DP_925",
            "Q_1000",
            "Q_925",
            "W_925",
            "W_1000",
            "DPD",
            "INVERSION",
            "zenith",
            "PM10_ugm3",
            "PM25_ugm3",
        ]
        return {name: i for i, name in enumerate(fallback[:dyn_vars_count])}


def choose_replacement_features(args: argparse.Namespace, lookup: Dict[str, int]) -> List[str]:
    explicit = split_feature_names(args.feature_swap_features)
    selected: List[str] = [f for f in explicit if f in lookup]
    top_k = int(args.feature_swap_top_k or 0)
    if top_k > 0 and args.feature_importance_csv:
        path = Path(args.feature_importance_csv)
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "feature" in df:
                    importance_cols = [c for c in df.columns if c.startswith("importance_")]
                    sort_col = ""
                    preferred = [f"importance_{args.feature_swap_metric}", "importance_low_vis_recall", "importance_low_vis_csi"]
                    for col in preferred + importance_cols:
                        if col in df:
                            sort_col = col
                            break
                    if sort_col:
                        df = df.sort_values(sort_col, ascending=False)
                    for feat in df["feature"].astype(str):
                        if feat in lookup and feat not in selected:
                            selected.append(feat)
                        if len(selected) >= top_k:
                            break
            except Exception as exc:
                print(f"[feature-swap] Could not read importance table {path}: {exc}", flush=True)
    if top_k > 0:
        selected = selected[:top_k]
    if not selected and top_k > 0:
        for feat in DEFAULT_FEATURE_SWAP_ORDER:
            if feat in lookup:
                selected.append(feat)
            if len(selected) >= top_k:
                break
    return selected


def predict_static_rows_for_swap(
    rows: np.ndarray,
    source_eval: SourceEval,
    train_mod,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if args.model_arch != "static_rnn":
        raise NotImplementedError("Feature replacement currently supports --model_arch static_rnn.")
    if source_eval.model is None or source_eval.scaler is None:
        raise RuntimeError("SourceEval must retain model and scaler for feature replacement.")
    layout = make_static_layout(
        train_mod,
        args.window,
        source_eval.dyn_vars_count,
        source_eval.extra_feat_dim,
        source_eval.dynamic_feature_order,
    )
    log_mask = train_mod.build_dyn_log_mask(layout)
    out: List[np.ndarray] = []
    model = source_eval.model
    scaler = source_eval.scaler
    model.eval()
    for start in range(0, len(rows), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(rows))
        batch = rows[start:end].astype(np.float32, copy=True)
        core = batch[:, : layout.core_dim].astype(np.float32, copy=True)
        core = train_mod.apply_core_transform(core, layout, not args.static_rnn_no_pm, log_mask)
        if scaler is not None:
            core = (core - scaler.center_) / (scaler.scale_ + 1e-6)
        core = np.clip(core, -10.0, 10.0)
        veg = batch[:, layout.split_dyn + 5 : layout.split_dyn + 6].astype(np.float32, copy=False)
        parts = [core, veg]
        if not args.static_rnn_no_fe:
            fe = batch[:, layout.split_dyn + 6 : layout.split_dyn + 6 + source_eval.extra_feat_dim].astype(
                np.float32,
                copy=True,
            )
            parts.append(np.clip(fe, -10.0, 10.0))
        final = np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        bx = torch.from_numpy(final).float().to(device, non_blocking=(device.type == "cuda"))
        with torch.inference_mode():
            logits = model(bx)[0]
            probs = F.softmax(logits / max(source_eval.temperature, 1e-6), dim=1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)


def plot_feature_replacement(repl_df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    if repl_df.empty:
        return
    delta_col = f"delta_{metric}"
    if delta_col not in repl_df:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_df = repl_df[repl_df["variant"] != "baseline"].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(delta_col, ascending=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    y = np.arange(len(plot_df))
    vals = plot_df[delta_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, max(3.4, 0.38 * len(plot_df) + 1.2)))
    ax.barh(y, vals, color=["#18864B" if v > 0 else "#B45B43" for v in vals])
    ax.axvline(0, color="#222222", lw=0.8)
    max_abs = max(float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.0, 1.0e-4)
    pad = max_abs * 0.10
    ax.set_xlim(-max_abs - pad, max_abs + pad)
    for yi, v in zip(y, vals):
        if not np.isfinite(v):
            continue
        label = f"{v:+.2e}" if 0 < abs(v) < 1.0e-3 else f"{v:+.4f}"
        if abs(v) >= max_abs * 0.12:
            x = v * 0.5
            ha = "center"
            color = "white"
        elif abs(v) < max_abs * 0.02:
            x = pad * 0.35
            ha = "left"
            color = "#2F3437"
        else:
            x = v + (pad * 0.25 if v >= 0 else -pad * 0.25)
            ha = "left" if v >= 0 else "right"
            color = "#2F3437"
        ax.text(x, yi, label, va="center", ha=ha, fontsize=7.0, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["variant"].astype(str).str.replace("swap_", "", regex=False))
    ax.set_xlabel(f"Change after replacing IFS variable with Tianji ({metric})")
    ax.set_title("Counterfactual single-variable replacement")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"fig_feature_replacement_{metric}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  [Fig] Saved -> {path}", flush=True)
    plt.close(fig)


def run_feature_replacement_experiment(
    args: argparse.Namespace,
    evals: Dict[str, SourceEval],
    idx_t: np.ndarray,
    idx_i: np.ndarray,
    y: np.ndarray,
    train_mod,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    if args.model_arch != "static_rnn":
        print("[feature-swap] skipped: currently implemented only for static_rnn.", flush=True)
        return pd.DataFrame()

    base_source = "ifs"
    donor_source = "tianji"
    base_eval = evals[base_source]
    donor_eval = evals[donor_source]
    if (
        base_eval.dyn_vars_count != donor_eval.dyn_vars_count
        or (base_eval.dynamic_feature_order or []) != (donor_eval.dynamic_feature_order or [])
    ):
        print("[feature-swap] skipped: source layouts differ.", flush=True)
        return pd.DataFrame()
    lookup = dynamic_feature_lookup(base_eval.dyn_vars_count)
    features = choose_replacement_features(args, lookup)
    if not features:
        return pd.DataFrame()
    base_available = populated_overlap_features(base_eval.spec.data_dir)
    donor_available = populated_overlap_features(donor_eval.spec.data_dir)
    if base_available and donor_available:
        available = base_available & donor_available
        skipped = [f for f in features if f not in available]
        features = [f for f in features if f in available]
        if skipped:
            print(
                "[feature-swap] skipped unpopulated overlap feature(s): " + ",".join(skipped),
                flush=True,
            )
        if not features:
            return pd.DataFrame()
    x_base = np.load(base_eval.spec.data_dir + "/X_test.npy", mmap_mode="r")
    x_donor = np.load(donor_eval.spec.data_dir + "/X_test.npy", mmap_mode="r")
    base_idx = idx_i
    donor_idx = idx_t
    base_rows = np.asarray(x_base[base_idx], dtype=np.float32)
    donor_rows = np.asarray(x_donor[donor_idx], dtype=np.float32)

    base_probs = base_eval.test_probs[idx_i]
    base_preds = base_eval.test_preds[idx_i]
    baseline_metrics = compute_metrics(y, base_preds, probs=base_probs)
    records: List[Dict[str, object]] = [rows_from_metrics("ifs_feature_swap", baseline_metrics, {"variant": "baseline"})]

    for feature in features:
        col_idx = int(lookup[feature])
        cols = [t * int(base_eval.dyn_vars_count) + col_idx for t in range(int(args.window))]
        swapped = base_rows.copy()
        swapped[:, cols] = donor_rows[:, cols]
        probs = predict_static_rows_for_swap(swapped, base_eval, train_mod, args, device)
        fog_th = float(base_eval.thresholds.get("fog", 0.5))
        mist_th = float(base_eval.thresholds.get("mist", 0.5))
        pred_mode = "fixed" if np.isfinite(fog_th) and np.isfinite(mist_th) else "argmax"
        preds = predict_from_probs(probs, pred_mode, fog_th, mist_th)
        metrics = compute_metrics(y, preds, probs=probs)
        extra = {"variant": f"swap_{feature}", "replaced_features": feature, "n_replaced_columns": len(cols)}
        row = rows_from_metrics("ifs_feature_swap", metrics, extra)
        for metric_name, base_val in baseline_metrics.items():
            if isinstance(base_val, (int, float, np.floating)) and metric_name in metrics:
                row[f"delta_{metric_name}"] = float(metrics[metric_name]) - float(base_val)
        records.append(row)

    if len(features) > 1:
        all_cols: List[int] = []
        for feature in features:
            col_idx = int(lookup[feature])
            all_cols.extend([t * int(base_eval.dyn_vars_count) + col_idx for t in range(int(args.window))])
        swapped = base_rows.copy()
        swapped[:, all_cols] = donor_rows[:, all_cols]
        probs = predict_static_rows_for_swap(swapped, base_eval, train_mod, args, device)
        fog_th = float(base_eval.thresholds.get("fog", 0.5))
        mist_th = float(base_eval.thresholds.get("mist", 0.5))
        pred_mode = "fixed" if np.isfinite(fog_th) and np.isfinite(mist_th) else "argmax"
        preds = predict_from_probs(probs, pred_mode, fog_th, mist_th)
        metrics = compute_metrics(y, preds, probs=probs)
        extra = {
            "variant": "swap_all_selected",
            "replaced_features": ",".join(features),
            "n_replaced_columns": len(all_cols),
        }
        row = rows_from_metrics("ifs_feature_swap", metrics, extra)
        for metric_name, base_val in baseline_metrics.items():
            if isinstance(base_val, (int, float, np.floating)) and metric_name in metrics:
                row[f"delta_{metric_name}"] = float(metrics[metric_name]) - float(base_val)
        records.append(row)

    repl_df = pd.DataFrame(records)
    path = out_dir / "feature_replacement_metrics.csv"
    repl_df.to_csv(path, index=False)
    print(f"[feature-swap] wrote {path}", flush=True)
    if not args.no_figures:
        plot_feature_replacement(repl_df, out_dir, args.feature_swap_metric)
    return repl_df


def write_independent_source_outputs(
    args: argparse.Namespace,
    specs: Dict[str, SourceSpec],
    build_configs: Dict[str, Dict[str, object]],
    evals: Dict[str, SourceEval],
    out_dir: Path,
) -> None:
    overall_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    ifs_diagnostic_info: Dict[str, object] = {
        "enabled": bool(not args.skip_ifs_forecast_baseline),
        "path": args.ifs_forecast_nc,
        "variable": args.ifs_forecast_var,
        "reference_source": "",
        "matched_rows": 0,
    }
    for source, eval_obj in evals.items():
        test_metrics = compute_metrics(
            eval_obj.test_targets,
            eval_obj.test_preds,
            probs=eval_obj.test_probs,
        )
        common_extra = {
            "temperature": eval_obj.temperature,
            "fog_threshold": eval_obj.thresholds.get("fog"),
            "mist_threshold": eval_obj.thresholds.get("mist"),
            "threshold_source": eval_obj.threshold_source,
            "feature_dim": eval_obj.feature_dim,
            "dyn_vars_count": eval_obj.dyn_vars_count,
            "extra_feat_dim": eval_obj.extra_feat_dim,
            "model_arch": args.model_arch,
            "checkpoint": eval_obj.spec.ckpt_path,
            "scaler": eval_obj.spec.scaler_path,
            "data_dir": eval_obj.spec.data_dir,
            "evaluation_mode": "independent_source",
        }
        overall_rows.append(rows_from_metrics(source, test_metrics, common_extra))
        validation_rows.append(
            rows_from_metrics(
                source,
                eval_obj.val_metrics,
                {
                    "temperature": eval_obj.temperature,
                    "fog_threshold": eval_obj.thresholds.get("fog"),
                    "mist_threshold": eval_obj.thresholds.get("mist"),
                    "threshold_source": eval_obj.threshold_source,
                    "dyn_vars_count": eval_obj.dyn_vars_count,
                    "extra_feat_dim": eval_obj.extra_feat_dim,
                    "model_arch": args.model_arch,
                    "checkpoint": eval_obj.spec.ckpt_path,
                    "scaler": eval_obj.spec.scaler_path,
                    "data_dir": eval_obj.spec.data_dir,
                    "evaluation_mode": "independent_source",
                },
            )
        )
        if not args.no_per_sample_csv and eval_obj.test_meta is not None:
            sample = eval_obj.test_meta.reset_index(drop=True).copy()
            sample["y_cls"] = eval_obj.test_targets
            sample["vis_raw_m"] = eval_obj.test_raw_vis
            sample["pred"] = eval_obj.test_preds
            sample["p_fog"] = eval_obj.test_probs[:, 0]
            sample["p_mist"] = eval_obj.test_probs[:, 1]
            sample["p_clear"] = eval_obj.test_probs[:, 2]
            sample["correct"] = eval_obj.test_preds == eval_obj.test_targets
            sample.to_csv(out_dir / f"per_sample_{source}.csv", index=False)

    if not args.skip_ifs_forecast_baseline:
        ref_order = [
            "tianji",
            "ifs",
            "T2ND_rh2m_common_core",
            "T2ND_rh2m_compact_common_core",
            "era5_2025_common_core",
            "era5_2025_compact_common_core",
            *[
                source
                for source in evals
                if source
                not in {
                    "tianji",
                    "ifs",
                    "T2ND_rh2m_common_core",
                    "T2ND_rh2m_compact_common_core",
                    "era5_2025_common_core",
                    "era5_2025_compact_common_core",
                }
            ],
        ]
        seen_ref: Set[str] = set()
        for ref_source in ref_order:
            if ref_source in seen_ref or ref_source not in evals:
                continue
            seen_ref.add(ref_source)
            eval_obj = evals[ref_source]
            if eval_obj.test_meta is None:
                continue
            try:
                ifs_preds, ifs_vis_raw, ifs_valid = load_ifs_forecast_baseline(
                    eval_obj.test_meta,
                    args.ifs_forecast_nc,
                    args.ifs_forecast_var,
                )
            except Exception as exc:
                print(f"[WARN] independent IFS diagnostic baseline skipped for {ref_source}: {exc}", flush=True)
                continue
            matched = int(np.asarray(ifs_valid, dtype=bool).sum())
            if matched <= 0:
                continue
            valid = np.asarray(ifs_valid, dtype=bool)
            diag_metrics = compute_metrics(eval_obj.test_targets[valid], ifs_preds[valid], probs=None)
            overall_rows.append(
                rows_from_metrics(
                    "ifs_diagnostic",
                    diag_metrics,
                    {
                        "matched_rows": matched,
                        "reference_source": ref_source,
                        "ifs_forecast_nc": args.ifs_forecast_nc,
                        "ifs_forecast_var": args.ifs_forecast_var,
                        "evaluation_mode": "independent_source_diagnostic",
                    },
                )
            )
            ifs_diagnostic_info.update({"reference_source": ref_source, "matched_rows": matched})
            if not args.no_per_sample_csv:
                sample = eval_obj.test_meta.loc[valid].reset_index(drop=True).copy()
                sample["y_cls"] = eval_obj.test_targets[valid]
                sample["vis_raw_m"] = eval_obj.test_raw_vis[valid]
                sample["ifs_diagnostic_vis_m"] = np.asarray(ifs_vis_raw, dtype=np.float32)[valid]
                sample["ifs_diagnostic_pred"] = ifs_preds[valid]
                sample["correct"] = ifs_preds[valid] == eval_obj.test_targets[valid]
                sample.to_csv(out_dir / "per_sample_ifs_diagnostic.csv", index=False)
            break

    overall_df = pd.DataFrame(overall_rows)
    validation_df = pd.DataFrame(validation_rows)
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    validation_df.to_csv(out_dir / "validation_metrics.csv", index=False)

    run_config = {
        "args": vars(args),
        "specs": {k: vars(v) for k, v in specs.items()},
        "model_arch": args.model_arch,
        "build_configs": build_configs,
        "evaluation_scope": "independent full test split for each selected source",
        "source_thresholds": {
            source: {
                "threshold_source": eval_obj.threshold_source,
                "temperature": eval_obj.temperature,
                "fog_threshold": eval_obj.thresholds.get("fog"),
                "mist_threshold": eval_obj.thresholds.get("mist"),
                "checkpoint": eval_obj.spec.ckpt_path,
                "scaler": eval_obj.spec.scaler_path,
            }
            for source, eval_obj in evals.items()
        },
        "ifs_diagnostic": ifs_diagnostic_info,
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    if not args.no_figures:
        plot_key_metrics_figure(overall_df, out_dir)

    metric_cols = [
        "source",
        "source_label",
        "n",
        "fog_pod",
        "fog_csi",
        "mist_pod",
        "mist_csi",
        "low_vis_recall",
        "low_vis_csi",
        "low_vis_precision",
        "low_vis_fpr",
    ]
    existing = [c for c in metric_cols if c in overall_df.columns]
    print("\n[OK] wrote independent source evaluation outputs to:", out_dir, flush=True)
    print(overall_df[existing].to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if int(args.dyn_vars_count) == 18 and int(args.expected_extra_dim) == 36:
        args.expected_extra_dim = 31
    if int(args.dyn_vars_count) == 19 and int(args.expected_extra_dim) == 36:
        args.expected_extra_dim = 31
    if (
        args.zero_transfer_s1
        and args.threshold_mode == "checkpoint"
        and not args.allow_zero_transfer_checkpoint_thresholds
    ):
        raise ValueError(
            "S1 zero-transfer with --threshold_mode checkpoint is blocked because "
            "checkpoint thresholds were selected on the S1 station-analysis domain "
            "and can collapse forecast-source predictions to clear. Use "
            "--threshold_mode argmax for raw response, or --threshold_mode val_search "
            "to recalibrate thresholds on each source validation split."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "tianji": SourceSpec(
            name="tianji",
            data_dir=args.tianji_data_dir or default_tianji_data_dir(args.tianji_source_tag),
            ckpt_path=args.tianji_ckpt
            or default_ckpt_path(args.tianji_source_tag, args.ckpt_dir, args.checkpoint_tag, args.model_arch),
            scaler_path=args.tianji_scaler,
        ),
        "ifs": SourceSpec(
            name="ifs",
            data_dir=args.ifs_data_dir,
            ckpt_path=args.ifs_ckpt or default_ckpt_path("ifs", args.ckpt_dir, args.checkpoint_tag, args.model_arch),
            scaler_path=args.ifs_scaler,
        ),
    }
    specs.update(parse_extra_source_specs(args.extra_sources))
    specs = filter_source_specs(specs, args.source_subset)
    apply_shared_checkpoint_scaler(specs, args)
    fill_auto_scaler_paths(specs, args)
    if not args.independent_sources and not {"tianji", "ifs"}.issubset(specs):
        raise ValueError(
            "Paired mode requires both 'tianji' and 'ifs'. "
            "Use --independent_sources when evaluating a source subset."
        )

    for source, spec in specs.items():
        if not os.path.isdir(spec.data_dir):
            raise FileNotFoundError(f"{source} data_dir does not exist: {spec.data_dir}")
        require_file(spec.ckpt_path, f"{source} checkpoint")
        require_file(spec.scaler_path, f"{source} scaler")

    train_mod = import_training_module(args)
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)

    build_configs = {source: read_build_config(spec.data_dir) for source, spec in specs.items()}
    validate_build_time_alignment(build_configs, args.allow_legacy_time_alignment)
    evals = {
        source: evaluate_one_source(source, spec, train_mod, args, device)
        for source, spec in specs.items()
    }
    if args.independent_sources:
        write_independent_source_outputs(args, specs, build_configs, evals, out_dir)
        return

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
    extra_alignment: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    extra_metrics: Dict[str, Dict[str, float]] = {}
    alignment_rows: List[Dict[str, object]] = [
        {
            "source": "tianji",
            "reference_source": "tianji",
            "test_rows": int(len(evals["tianji"].test_targets)),
            "paired_rows": int(len(y)),
            "sample_scope": "tianji_ifs_paired_test",
            "alignment_status": "reference",
            "alignment_note": "",
        },
        {
            "source": "ifs",
            "reference_source": "tianji",
            "test_rows": int(len(evals["ifs"].test_targets)),
            "paired_rows": int(len(y)),
            "sample_scope": "tianji_ifs_paired_test",
            "alignment_status": "paired",
            "alignment_note": "",
        },
    ]
    for source in specs:
        if source in {"tianji", "ifs"}:
            continue
        try:
            idx_base, idx_src, _ = align_test_outputs(evals["tianji"], evals[source], args.strict_meta)
            validate_paired_labels(evals["tianji"], evals[source], idx_base, idx_src)
        except RuntimeError as exc:
            print(
                f"[WARN] {source} has no usable paired test intersection with Tianji; "
                "using its native full test split for source-level metrics only. "
                f"Reason: {exc}",
                flush=True,
            )
            extra_metrics[source] = compute_metrics(
                evals[source].test_targets,
                evals[source].test_preds,
                probs=evals[source].test_probs,
            )
            alignment_rows.append(
                {
                    "source": source,
                    "reference_source": "tianji",
                    "test_rows": int(len(evals[source].test_targets)),
                    "paired_rows": 0,
                    "sample_scope": "native_full_test",
                    "alignment_status": "not_paired",
                    "alignment_note": str(exc),
                }
            )
            continue
        extra_alignment[source] = (idx_base, idx_src)
        extra_metrics[source] = compute_metrics(
            evals["tianji"].test_targets[idx_base],
            evals[source].test_preds[idx_src],
            probs=evals[source].test_probs[idx_src],
        )
        alignment_rows.append(
            {
                "source": source,
                "reference_source": "tianji",
                "test_rows": int(len(evals[source].test_targets)),
                "paired_rows": int(len(idx_src)),
                "sample_scope": "paired_to_tianji_test",
                "alignment_status": "paired",
                "alignment_note": "",
            }
        )

    overall_rows = [
        rows_from_metrics(
                "tianji",
                metrics_t,
                {
                    "temperature": evals["tianji"].temperature,
                    "fog_threshold": evals["tianji"].thresholds.get("fog"),
                    "mist_threshold": evals["tianji"].thresholds.get("mist"),
                    "threshold_source": evals["tianji"].threshold_source,
                    "feature_dim": evals["tianji"].feature_dim,
                    "extra_feat_dim": evals["tianji"].extra_feat_dim,
                    "model_arch": args.model_arch,
                    "checkpoint": evals["tianji"].spec.ckpt_path,
                    "data_dir": evals["tianji"].spec.data_dir,
                    "matched_rows": int(len(y)),
                    "sample_scope": "tianji_ifs_paired_test",
                },
            ),
        rows_from_metrics(
                "ifs",
                metrics_i,
                {
                    "temperature": evals["ifs"].temperature,
                    "fog_threshold": evals["ifs"].thresholds.get("fog"),
                    "mist_threshold": evals["ifs"].thresholds.get("mist"),
                    "threshold_source": evals["ifs"].threshold_source,
                    "feature_dim": evals["ifs"].feature_dim,
                    "extra_feat_dim": evals["ifs"].extra_feat_dim,
                    "model_arch": args.model_arch,
                    "checkpoint": evals["ifs"].spec.ckpt_path,
                    "data_dir": evals["ifs"].spec.data_dir,
                    "matched_rows": int(len(y)),
                    "sample_scope": "tianji_ifs_paired_test",
                },
            ),
    ]
    for source, metrics in extra_metrics.items():
        idx_base = extra_alignment[source][0] if source in extra_alignment else np.arange(
            len(evals[source].test_targets), dtype=np.int64
        )
        sample_scope = "paired_to_tianji_test" if source in extra_alignment else "native_full_test"
        overall_rows.append(
            rows_from_metrics(
                source,
                metrics,
                {
                    "matched_rows": int(len(idx_base)),
                    "paired_rows": int(len(extra_alignment[source][0])) if source in extra_alignment else 0,
                    "sample_scope": sample_scope,
                    "temperature": evals[source].temperature,
                    "fog_threshold": evals[source].thresholds.get("fog"),
                    "mist_threshold": evals[source].thresholds.get("mist"),
                    "threshold_source": evals[source].threshold_source,
                    "feature_dim": evals[source].feature_dim,
                    "extra_feat_dim": evals[source].extra_feat_dim,
                    "model_arch": args.model_arch,
                    "checkpoint": evals[source].spec.ckpt_path,
                    "data_dir": evals[source].spec.data_dir,
                },
            )
        )
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(out_dir / "source_alignment_summary.csv", index=False)

    validation_rows = [
        rows_from_metrics(
                "tianji",
                evals["tianji"].val_metrics,
                {
                    "temperature": evals["tianji"].temperature,
                    "fog_threshold": evals["tianji"].thresholds.get("fog"),
                    "mist_threshold": evals["tianji"].thresholds.get("mist"),
                    "threshold_source": evals["tianji"].threshold_source,
                    "model_arch": args.model_arch,
                },
            ),
        rows_from_metrics(
                "ifs",
                evals["ifs"].val_metrics,
                {
                    "temperature": evals["ifs"].temperature,
                    "fog_threshold": evals["ifs"].thresholds.get("fog"),
                    "mist_threshold": evals["ifs"].thresholds.get("mist"),
                    "threshold_source": evals["ifs"].threshold_source,
                    "model_arch": args.model_arch,
                },
            ),
    ]
    for source in specs:
        if source in {"tianji", "ifs"}:
            continue
        validation_rows.append(
            rows_from_metrics(
                source,
                evals[source].val_metrics,
                {
                    "temperature": evals[source].temperature,
                    "fog_threshold": evals[source].thresholds.get("fog"),
                    "mist_threshold": evals[source].thresholds.get("mist"),
                    "threshold_source": evals[source].threshold_source,
                    "model_arch": args.model_arch,
                },
            )
        )
    validation_df = pd.DataFrame(validation_rows)
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

            diag_rows = [
                rows_from_metrics(
                        "tianji",
                        metrics_t_diag,
                        {
                            "sample_scope": "ifs_diagnostic_matched_test",
                            "matched_rows": int(valid_diag.sum()),
                            "temperature": evals["tianji"].temperature,
                            "fog_threshold": evals["tianji"].thresholds.get("fog"),
                            "mist_threshold": evals["tianji"].thresholds.get("mist"),
                            "threshold_source": evals["tianji"].threshold_source,
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
                            "threshold_source": evals["ifs"].threshold_source,
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
            common_pos_by_tianji_idx = {int(src_idx): pos for pos, src_idx in enumerate(idx_t.tolist())}
            for source, (idx_base, idx_src) in extra_alignment.items():
                keep_base: List[int] = []
                keep_src: List[int] = []
                keep_common: List[int] = []
                for base_idx, src_idx in zip(idx_base.tolist(), idx_src.tolist()):
                    common_pos = common_pos_by_tianji_idx.get(int(base_idx))
                    if common_pos is None or not bool(valid_diag[common_pos]):
                        continue
                    keep_base.append(int(base_idx))
                    keep_src.append(int(src_idx))
                    keep_common.append(int(common_pos))
                if not keep_src:
                    continue
                src_idx_arr = np.asarray(keep_src, dtype=np.int64)
                common_idx_arr = np.asarray(keep_common, dtype=np.int64)
                metrics_extra_diag = compute_metrics(
                    y[common_idx_arr],
                    evals[source].test_preds[src_idx_arr],
                    probs=evals[source].test_probs[src_idx_arr],
                )
                diag_rows.append(
                    rows_from_metrics(
                        source,
                        metrics_extra_diag,
                        {
                            "sample_scope": "ifs_diagnostic_matched_test",
                            "matched_rows": int(len(src_idx_arr)),
                            "temperature": evals[source].temperature,
                            "fog_threshold": evals[source].thresholds.get("fog"),
                            "mist_threshold": evals[source].thresholds.get("mist"),
                            "threshold_source": evals[source].threshold_source,
                        },
                    )
                )
            ifs_diag_metrics_df = pd.DataFrame(diag_rows)
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
    for scenario, mask in scenario_masks(meta_common, args.local_time_offset_hours).items():
        mt = compute_metrics(y[mask], t_preds[mask], probs=t_probs[mask])
        mi = compute_metrics(y[mask], i_preds[mask], probs=i_probs[mask])
        scenario_rows.append(rows_from_metrics("tianji", mt, {"scenario": scenario}))
        scenario_rows.append(rows_from_metrics("ifs", mi, {"scenario": scenario}))
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "scenario_metrics.csv", index=False)

    scenario_delta_rows = []
    for scenario, mask in scenario_masks(meta_common, args.local_time_offset_hours).items():
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
        for scenario, mask in scenario_masks(meta_common, args.local_time_offset_hours).items():
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

    feature_replacement_df = run_feature_replacement_experiment(
        args,
        evals,
        idx_t,
        idx_i,
        y,
        train_mod,
        device,
        out_dir,
    )

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
        "model_arch": args.model_arch,
        "build_configs": build_configs,
        "paired_rows": int(len(y)),
        "source_alignment": alignment_rows,
        "evaluation_scope": "test split; validation split is used only for calibration/threshold selection",
        "source_thresholds": {
            source: {
                "threshold_source": eval_obj.threshold_source,
                "temperature": eval_obj.temperature,
                "fog_threshold": eval_obj.thresholds.get("fog"),
                "mist_threshold": eval_obj.thresholds.get("mist"),
                "checkpoint": eval_obj.spec.ckpt_path,
            }
            for source, eval_obj in evals.items()
        },
        "ifs_diagnostic": {
            "enabled": bool(not args.skip_ifs_forecast_baseline),
            "path": args.ifs_forecast_nc,
            "variable": args.ifs_forecast_var,
            "matched_rows": int(ifs_diag_valid.sum()) if ifs_diag_valid is not None else 0,
        },
        "feature_replacement": {
            "enabled": bool(feature_replacement_df is not None and not feature_replacement_df.empty),
            "features": split_feature_names(args.feature_swap_features),
            "feature_importance_csv": args.feature_importance_csv,
            "top_k": int(args.feature_swap_top_k or 0),
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
        plot_df = overall_df.copy()
        if ifs_diag_metrics_df is not None and not ifs_diag_metrics_df.empty:
            diag_only = ifs_diag_metrics_df.loc[
                ifs_diag_metrics_df["source"].astype(str) == "ifs_diagnostic"
            ].copy()
            if not diag_only.empty:
                plot_df = pd.concat([plot_df, diag_only], ignore_index=True, sort=False)
        plot_key_metrics_figure(plot_df, out_dir)

    print("\n[OK] wrote paired evaluation outputs to:", out_dir, flush=True)
    print(delta_df[delta_df["metric"].isin(BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False), flush=True)
    if ifs_diag_delta_df is not None and not ifs_diag_delta_df.empty:
        print(
            "\n[IFS diagnostic matched deltas: Tianji model minus IFS diagnostic VIS]",
            flush=True,
        )
        print(
            ifs_diag_delta_df[ifs_diag_delta_df["metric"].isin(BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
