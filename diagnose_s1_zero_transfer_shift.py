#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit S1 zero-transfer input shift against one forecast-source dataset.

This script is intentionally torch-free.  It compares the overlap S1 training
distribution with a selected forecast-source dataset after the same core
preprocessing used by the Static-RNN trainer: log1p on skewed dynamic channels,
S1 RobustScaler, and clipping diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from pmst_overlap_common import (
    COMMON_CORE_PMST_FEATURES,
    FINAL_FEATURE_ORDER,
    OVERLAP_CANONICAL,
    TOTAL_DYN,
)


WINDOW = 12
DYN_VARS = TOTAL_DYN
BASE_DYN = WINDOW * DYN_VARS
STATIC_CONT = 5
CORE_DIM = BASE_DYN + STATIC_CONT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose S1 zero-transfer source shift without torch.")
    p.add_argument(
        "--s1_dir",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap",
    )
    p.add_argument(
        "--source_dir",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_era5_2025_12h_pm10_pm25_common_core",
    )
    p.add_argument("--source_name", default="era5_2025_common_core")
    p.add_argument(
        "--scaler",
        default="/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/robust_scaler_exp_overlap_static_rnn_s1_common_core_pm10_pm25_s1_w12_dyn27_pm.pkl",
    )
    p.add_argument(
        "--out_dir",
        default="/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/s1_zero_transfer_shift_diagnostics",
    )
    p.add_argument("--sample_rows", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scaler(path: Path):
    if not path.is_file():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def y_to_class(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(y, dtype=np.float32).copy()
    finite = raw[np.isfinite(raw)]
    if finite.size and float(np.nanmax(finite)) < 100.0:
        raw *= 1000.0
    cls = np.zeros(len(raw), dtype=np.int64)
    cls[raw >= 500.0] = 1
    cls[raw >= 1000.0] = 2
    return raw, cls


def class_rows(data_dir: Path, label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"y_{split}.npy"
        if not path.is_file():
            continue
        raw, cls = y_to_class(np.load(path))
        n = int(len(cls))
        rows.append(
            {
                "dataset": label,
                "split": split,
                "n": n,
                "fog_n": int(np.sum(cls == 0)),
                "mist_n": int(np.sum(cls == 1)),
                "clear_n": int(np.sum(cls == 2)),
                "fog_rate": float(np.mean(cls == 0)) if n else math.nan,
                "mist_rate": float(np.mean(cls == 1)) if n else math.nan,
                "low_vis_rate": float(np.mean(cls < 2)) if n else math.nan,
                "vis_p50": float(np.nanpercentile(raw, 50)) if n else math.nan,
                "vis_p90": float(np.nanpercentile(raw, 90)) if n else math.nan,
            }
        )
    return rows


def choose_indices(n: int, sample_rows: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if sample_rows <= 0 or n <= sample_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(sample_rows), replace=False)).astype(np.int64)


def load_sample(data_dir: Path, split: str, sample_rows: int, seed: int) -> np.ndarray:
    path = data_dir / f"X_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] < CORE_DIM:
        raise ValueError(f"{path}: expected at least {CORE_DIM} columns, got {arr.shape}")
    idx = choose_indices(int(arr.shape[0]), sample_rows, seed)
    return np.asarray(arr[idx, :CORE_DIM], dtype=np.float32)


def log1p_dyn_indices(dyn_vars: int) -> List[int]:
    idxs = [2, 4, 9]
    if dyn_vars >= 26:
        idxs.extend([dyn_vars - 2, dyn_vars - 1])
    return sorted(set(i for i in idxs if 0 <= i < dyn_vars))


def apply_core_transform(core: np.ndarray, window: int = WINDOW, dyn_vars: int = DYN_VARS) -> np.ndarray:
    out = core.astype(np.float32, copy=True)
    for t in range(window):
        base = t * dyn_vars
        for idx in log1p_dyn_indices(dyn_vars):
            col = base + idx
            out[:, col] = np.log1p(np.maximum(out[:, col], 0.0))
    return out


def finite_stats(values: np.ndarray, prefix: str) -> Dict[str, object]:
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    out: Dict[str, object] = {
        f"{prefix}_finite_rate": float(finite.size / len(vals)) if len(vals) else math.nan,
        f"{prefix}_zero_rate": float(np.mean(finite == 0.0)) if finite.size else math.nan,
    }
    for q in (1, 5, 50, 95, 99):
        out[f"{prefix}_p{q}"] = float(np.nanpercentile(finite, q)) if finite.size else math.nan
    return out


def feature_shift_rows(
    s1_core: np.ndarray,
    src_core: np.ndarray,
    scaler,
    source_cfg: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    source_overlap = set(source_cfg.get("overlap_vars") or source_cfg.get("populated_pmst_features") or [])
    source_feature_set = str(source_cfg.get("feature_set", ""))
    if not source_overlap and source_feature_set == "common_core":
        source_overlap = set(COMMON_CORE_PMST_FEATURES)
    if not source_overlap:
        source_overlap = set(source_cfg.get("available_pmst_features") or [])

    s1_trans = apply_core_transform(s1_core)
    src_trans = apply_core_transform(src_core)
    center = getattr(scaler, "center_", None) if scaler is not None else None
    scale = getattr(scaler, "scale_", None) if scaler is not None else None

    for feat_idx, feature in enumerate(FINAL_FEATURE_ORDER + ["zenith", "PM10_ugm3", "PM25_ugm3"]):
        col = (WINDOW - 1) * DYN_VARS + feat_idx
        s1_raw = s1_core[:, col]
        src_raw = src_core[:, col]
        row: Dict[str, object] = {
            "feature": feature,
            "dyn_index": feat_idx,
            "last_hour_col": col,
            "s1_overlap_channel": feature in set(OVERLAP_CANONICAL) or feature in {"zenith", "PM10_ugm3", "PM25_ugm3"},
            "source_filled_common_core": feature in source_overlap or feature in {"zenith", "PM10_ugm3", "PM25_ugm3"},
            "source_feature_set": source_feature_set,
            "log1p_transformed": feat_idx in log1p_dyn_indices(DYN_VARS),
        }
        row.update(finite_stats(s1_raw, "s1_raw"))
        row.update(finite_stats(src_raw, "source_raw"))
        if center is not None and scale is not None and col < len(center):
            denom = float(scale[col]) + 1e-6
            s1_scaled = (s1_trans[:, col] - float(center[col])) / denom
            src_scaled = (src_trans[:, col] - float(center[col])) / denom
            row["scaler_center"] = float(center[col])
            row["scaler_scale"] = float(scale[col])
            row.update(finite_stats(s1_scaled, "s1_scaled"))
            row.update(finite_stats(src_scaled, "source_scaled"))
            row["source_abs_scaled_gt3_rate"] = float(np.mean(np.abs(src_scaled[np.isfinite(src_scaled)]) > 3.0))
            row["source_abs_scaled_gt8_rate"] = float(np.mean(np.abs(src_scaled[np.isfinite(src_scaled)]) > 8.0))
        rows.append(row)
    return pd.DataFrame(rows)


def core_shift_summary(s1_core: np.ndarray, src_core: np.ndarray, scaler) -> pd.DataFrame:
    if scaler is None:
        return pd.DataFrame()
    center = getattr(scaler, "center_", None)
    scale = getattr(scaler, "scale_", None)
    if center is None or scale is None:
        return pd.DataFrame()
    rows = []
    for label, core in (("s1_train_sample", s1_core), ("source_test_sample", src_core)):
        transformed = apply_core_transform(core)
        scaled = (transformed - center[: CORE_DIM]) / (scale[: CORE_DIM] + 1e-6)
        finite = scaled[np.isfinite(scaled)]
        rows.append(
            {
                "dataset": label,
                "n_rows": int(core.shape[0]),
                "n_core_values": int(finite.size),
                "abs_scaled_gt3_rate": float(np.mean(np.abs(finite) > 3.0)) if finite.size else math.nan,
                "abs_scaled_gt8_rate": float(np.mean(np.abs(finite) > 8.0)) if finite.size else math.nan,
                "scaled_p01": float(np.nanpercentile(finite, 1)) if finite.size else math.nan,
                "scaled_p50": float(np.nanpercentile(finite, 50)) if finite.size else math.nan,
                "scaled_p99": float(np.nanpercentile(finite, 99)) if finite.size else math.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    s1_dir = Path(args.s1_dir)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_cfg = read_json(s1_dir / "dataset_build_config.json")
    source_cfg = read_json(source_dir / "dataset_build_config.json")
    scaler = load_scaler(Path(args.scaler))

    class_df = pd.DataFrame([*class_rows(s1_dir, "s1_overlap"), *class_rows(source_dir, args.source_name)])
    class_df.to_csv(out_dir / "class_distribution.csv", index=False)

    s1_core = load_sample(s1_dir, "train", args.sample_rows, args.seed)
    src_split = "test" if (source_dir / "X_test.npy").is_file() else "val"
    src_core = load_sample(source_dir, src_split, args.sample_rows, args.seed + 1)

    feature_df = feature_shift_rows(s1_core, src_core, scaler, source_cfg)
    feature_df.to_csv(out_dir / "dynamic_feature_shift_last_hour.csv", index=False)
    core_df = core_shift_summary(s1_core, src_core, scaler)
    if not core_df.empty:
        core_df.to_csv(out_dir / "core_scaled_shift_summary.csv", index=False)

    summary = {
        "s1_dir": str(s1_dir),
        "source_dir": str(source_dir),
        "source_name": args.source_name,
        "source_split_sampled": src_split,
        "sample_rows": int(args.sample_rows),
        "scaler": str(args.scaler),
        "s1_config": s1_cfg,
        "source_config": source_cfg,
        "s1_overlap_channels": list(OVERLAP_CANONICAL),
        "common_core_channels": list(COMMON_CORE_PMST_FEATURES),
        "common_core_removes_from_s1_overlap": [x for x in OVERLAP_CANONICAL if x not in COMMON_CORE_PMST_FEATURES],
    }
    with (out_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    keep_cols = [
        "feature",
        "s1_overlap_channel",
        "source_filled_common_core",
        "s1_raw_zero_rate",
        "source_raw_zero_rate",
        "s1_raw_p50",
        "source_raw_p50",
        "s1_raw_p95",
        "source_raw_p95",
        "source_scaled_p50",
        "source_scaled_p99",
        "source_abs_scaled_gt3_rate",
    ]
    print("[OK] wrote diagnostics to:", out_dir, flush=True)
    print("\n[class distribution]", flush=True)
    print(class_df.to_string(index=False), flush=True)
    print("\n[largest last-hour scaled shifts]", flush=True)
    if "source_abs_scaled_gt3_rate" in feature_df:
        top = feature_df.sort_values("source_abs_scaled_gt3_rate", ascending=False).head(12)
        print(top[[c for c in keep_cols if c in top.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
