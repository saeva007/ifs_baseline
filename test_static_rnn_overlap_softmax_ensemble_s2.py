#!/usr/bin/env python3
"""
Mean-softmax ensemble for paired Tianji/IFS overlap S2 models.

The experiment keeps the two trained overlap models unchanged. It runs each
model on the matching paired source dataset, averages the post-softmax class
probabilities row by row, selects ensemble decision thresholds from validation
data by default, and evaluates the final ensemble once on the held-out test
split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import test_PMST_overlap_forecast_source_s2 as paired


ENSEMBLE_SOURCE = "ensemble_mean_softmax"
ENSEMBLE_LABEL = "Mean softmax ensemble"
DEFAULT_ENSEMBLE_OUT_DIR = (
    "/public/home/putianshu/vis_mlp/"
    "paper_eval_results_pm10_pm25_journal/overlap_softmax_ensemble"
)


@dataclass
class SourceOutput:
    source: str
    spec: paired.SourceSpec
    feature_dim: int
    extra_feat_dim: int
    dyn_vars_count: int
    temperature: float
    thresholds: Dict[str, float]
    threshold_source: str
    val_probs: np.ndarray
    val_preds: np.ndarray
    val_targets: np.ndarray
    val_raw_vis: np.ndarray
    val_meta: Optional[pd.DataFrame]
    val_metrics: Dict[str, float]
    test_probs: np.ndarray
    test_preds: np.ndarray
    test_targets: np.ndarray
    test_raw_vis: np.ndarray
    test_meta: Optional[pd.DataFrame]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a paired Tianji/IFS overlap model ensemble by averaging "
            "post-softmax class probabilities."
        )
    )
    ap.add_argument(
        "--tianji_source_tag",
        choices=["tianji", "T2ND_rh2m"],
        default=os.environ.get("OVERLAP_TIANJI_SOURCE_TAG", "tianji"),
        help="Which Tianji-input checkpoint/data naming family to use for defaults.",
    )
    ap.add_argument("--tianji_data_dir", default=os.environ.get("OVERLAP_TIANJI_DATA_DIR", ""))
    ap.add_argument("--ifs_data_dir", default=os.environ.get("OVERLAP_IFS_DATA_DIR", paired.DEFAULT_IFS_DIR))
    ap.add_argument("--ckpt_dir", default=os.environ.get("OVERLAP_CKPT_DIR", paired.DEFAULT_CKPT_DIR))
    ap.add_argument("--tianji_ckpt", default="")
    ap.add_argument("--ifs_ckpt", default="")
    ap.add_argument("--tianji_scaler", default="")
    ap.add_argument("--ifs_scaler", default="")
    ap.add_argument(
        "--model_arch",
        choices=["static_rnn", "pmst"],
        default=os.environ.get("OVERLAP_MODEL_ARCH", "static_rnn"),
        help="Model family. static_rnn is the current overlap paper path.",
    )
    ap.add_argument("--static_rnn_train_dir", default=paired.DEFAULT_STATIC_RNN_TRAIN_DIR)
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
    ap.add_argument("--checkpoint_tag", default="S2_PhaseB_best_score")
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", DEFAULT_ENSEMBLE_OUT_DIR))
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--dyn_vars_count", type=int, default=27)
    ap.add_argument("--expected_extra_dim", type=int, default=36)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument(
        "--source_threshold_mode",
        choices=["checkpoint", "val_search", "argmax", "fixed"],
        default="checkpoint",
        help="Threshold rule used for the two single-model reference rows.",
    )
    ap.add_argument(
        "--ensemble_threshold_mode",
        choices=["val_search", "mean_checkpoint", "argmax", "fixed"],
        default="val_search",
        help=(
            "Threshold rule used after probability averaging. val_search uses only "
            "the paired validation split and is the recommended paper-safe default."
        ),
    )
    ap.add_argument("--fog_threshold", type=float, default=0.5)
    ap.add_argument("--mist_threshold", type=float, default=0.5)
    ap.add_argument("--no_temp_scaling", action="store_true")
    ap.add_argument("--temp_lr", type=float, default=0.01)
    ap.add_argument("--temp_max_iter", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap_size", type=int, default=200000)
    ap.add_argument("--bootstrap_seed", type=int, default=20250525)
    ap.add_argument("--skip_bootstrap", action="store_true")
    ap.add_argument("--strict_meta", action="store_true", help="Fail unless paired rows are identical and in order.")
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
    ap.add_argument("--no_figures", action="store_true")
    return ap.parse_args()


def build_specs(args: argparse.Namespace) -> Dict[str, paired.SourceSpec]:
    return {
        "tianji": paired.SourceSpec(
            name="tianji",
            data_dir=args.tianji_data_dir or paired.default_tianji_data_dir(args.tianji_source_tag),
            ckpt_path=args.tianji_ckpt
            or paired.default_ckpt_path(args.tianji_source_tag, args.ckpt_dir, args.checkpoint_tag, args.model_arch),
            scaler_path=args.tianji_scaler
            or paired.default_scaler_path(
                args.tianji_source_tag,
                args.ckpt_dir,
                args.window,
                args.dyn_vars_count,
                args.model_arch,
                args.static_rnn_no_pm,
            ),
        ),
        "ifs": paired.SourceSpec(
            name="ifs",
            data_dir=args.ifs_data_dir,
            ckpt_path=args.ifs_ckpt
            or paired.default_ckpt_path("ifs", args.ckpt_dir, args.checkpoint_tag, args.model_arch),
            scaler_path=args.ifs_scaler
            or paired.default_scaler_path(
                "ifs",
                args.ckpt_dir,
                args.window,
                args.dyn_vars_count,
                args.model_arch,
                args.static_rnn_no_pm,
            ),
        ),
    }


def choose_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    ckpt_meta: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, float], str, str, Dict[str, float]]:
    if mode == "checkpoint":
        thresholds = paired.checkpoint_thresholds(ckpt_meta or {})
        if thresholds is None:
            print("[threshold] checkpoint metadata missing; falling back to validation search.", flush=True)
            thresholds, metrics = paired.search_thresholds_on_val(probs, targets)
            return thresholds, "fixed", "val_search_fallback_no_checkpoint_thresholds", metrics
        preds = paired.predict_from_probs(probs, "fixed", thresholds["fog"], thresholds["mist"])
        metrics = paired.compute_metrics(targets, preds, probs=probs)
        return thresholds, "fixed", "checkpoint_metadata", metrics

    if mode == "val_search":
        thresholds, metrics = paired.search_thresholds_on_val(probs, targets)
        return thresholds, "fixed", "val_search", metrics

    if mode == "fixed":
        thresholds = {"fog": float(args.fog_threshold), "mist": float(args.mist_threshold)}
        preds = paired.predict_from_probs(probs, "fixed", thresholds["fog"], thresholds["mist"])
        metrics = paired.compute_metrics(targets, preds, probs=probs)
        return thresholds, "fixed", "fixed_cli", metrics

    thresholds = {"fog": math.nan, "mist": math.nan}
    preds = paired.predict_from_probs(probs, "argmax", 0.5, 0.5)
    metrics = paired.compute_metrics(targets, preds, probs=probs)
    return thresholds, "argmax", "argmax", metrics


def evaluate_source(
    source: str,
    spec: paired.SourceSpec,
    train_mod,
    args: argparse.Namespace,
    device: torch.device,
) -> SourceOutput:
    print(f"[{source}] data_dir={spec.data_dir}", flush=True)
    print(f"[{source}] ckpt={spec.ckpt_path}", flush=True)
    print(f"[{source}] scaler={spec.scaler_path}", flush=True)

    feature_dim, extra_dim = paired.infer_feature_layout(
        spec.data_dir, "test", args.window, args.dyn_vars_count, args.expected_extra_dim
    )
    val_feature_dim, val_extra_dim = paired.infer_feature_layout(
        spec.data_dir, "val", args.window, args.dyn_vars_count, args.expected_extra_dim
    )
    if val_feature_dim != feature_dim or val_extra_dim != extra_dim:
        raise ValueError(
            f"{source}: val/test feature layout differs: "
            f"val=({val_feature_dim},{val_extra_dim}) test=({feature_dim},{extra_dim})"
        )

    paired.require_file(spec.scaler_path, "RobustScaler")
    scaler = joblib.load(spec.scaler_path)
    ckpt_meta = paired.checkpoint_metadata(spec.ckpt_path, device)
    model = paired.load_model(
        train_mod,
        spec.ckpt_path,
        device,
        args.window,
        args.dyn_vars_count,
        extra_dim,
        args.allow_partial_load,
        args,
    )

    val_ds, val_meta = paired.make_dataset(
        train_mod,
        spec.data_dir,
        "val",
        scaler,
        args.window,
        args.dyn_vars_count,
        extra_dim,
        args.limit_samples,
        args.model_arch,
        not args.static_rnn_no_fe,
        not args.static_rnn_no_pm,
        args,
    )
    test_ds, test_meta = paired.make_dataset(
        train_mod,
        spec.data_dir,
        "test",
        scaler,
        args.window,
        args.dyn_vars_count,
        extra_dim,
        args.limit_samples,
        args.model_arch,
        not args.static_rnn_no_fe,
        not args.static_rnn_no_pm,
        args,
    )
    val_loader = paired.make_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = paired.make_loader(test_ds, args.batch_size, args.num_workers)

    print(f"[{source}] running validation inference: N={len(val_ds)}", flush=True)
    val_logits, val_targets, val_raw = paired.collect_logits(model, val_loader, device)
    if args.source_threshold_mode == "checkpoint" or args.no_temp_scaling:
        temperature = 1.0
    else:
        temperature = paired.calibrate_temperature_from_logits(
            val_logits, val_targets, device, args.temp_lr, args.temp_max_iter
        )
    val_probs = paired.softmax_np(val_logits, temperature)
    thresholds, pred_mode, threshold_source, val_metrics = choose_thresholds(
        val_probs, val_targets, args.source_threshold_mode, args, ckpt_meta
    )
    val_preds = paired.predict_from_probs(
        val_probs,
        pred_mode,
        thresholds.get("fog", 0.5),
        thresholds.get("mist", 0.5),
    )

    print(
        f"[{source}] temperature={temperature:.4f}, thresholds={thresholds}, "
        f"threshold_source={threshold_source}, "
        f"val target_achievement={val_metrics.get('target_achievement', math.nan):.4f}",
        flush=True,
    )

    print(f"[{source}] running test inference: N={len(test_ds)}", flush=True)
    test_logits, test_targets, test_raw = paired.collect_logits(model, test_loader, device)
    test_probs = paired.softmax_np(test_logits, temperature)
    test_preds = paired.predict_from_probs(
        test_probs,
        pred_mode,
        thresholds.get("fog", 0.5),
        thresholds.get("mist", 0.5),
    )

    return SourceOutput(
        source=source,
        spec=spec,
        feature_dim=feature_dim,
        extra_feat_dim=extra_dim,
        dyn_vars_count=args.dyn_vars_count,
        temperature=float(temperature),
        thresholds=thresholds,
        threshold_source=threshold_source,
        val_probs=val_probs,
        val_preds=val_preds,
        val_targets=val_targets,
        val_raw_vis=val_raw,
        val_meta=val_meta,
        val_metrics=val_metrics,
        test_probs=test_probs,
        test_preds=test_preds,
        test_targets=test_targets,
        test_raw_vis=test_raw,
        test_meta=test_meta,
    )


def split_arrays(source: SourceOutput, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    if split == "val":
        return source.val_targets, source.val_raw_vis, source.val_probs, source.val_meta
    if split == "test":
        return source.test_targets, source.test_raw_vis, source.test_probs, source.test_meta
    raise ValueError(f"Unknown split: {split}")


def align_split_outputs(
    left: SourceOutput,
    right: SourceOutput,
    split: str,
    strict_meta: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    left_targets, _, _, left_meta = split_arrays(left, split)
    right_targets, _, _, right_meta = split_arrays(right, split)

    if left_meta is not None and right_meta is not None:
        l_key = paired.normalize_key_frame(left_meta).rename(columns={"row_idx": "idx_left"})
        r_key = paired.normalize_key_frame(right_meta).rename(columns={"row_idx": "idx_right"})
        joined = l_key.merge(r_key, on=["time_key", "station_key", "dup"], how="inner", sort=False)
        if joined.empty:
            raise RuntimeError(f"No common (time, station_id) rows between paired {split} metadata.")
        idx_left = joined["idx_left"].to_numpy(dtype=np.int64)
        idx_right = joined["idx_right"].to_numpy(dtype=np.int64)
        same_order = (
            len(idx_left) == len(left_targets)
            and len(idx_right) == len(right_targets)
            and np.array_equal(idx_left, np.arange(len(idx_left)))
            and np.array_equal(idx_right, np.arange(len(idx_right)))
        )
        if strict_meta and not same_order:
            raise RuntimeError(
                f"{split} metadata are not identical/in-order. Rerun without "
                "--strict_meta to use the paired intersection."
            )
        meta_common = left_meta.iloc[idx_left].reset_index(drop=True).copy()
        return idx_left, idx_right, meta_common

    n = min(len(left_targets), len(right_targets))
    idx_left = np.arange(n, dtype=np.int64)
    idx_right = np.arange(n, dtype=np.int64)
    if strict_meta and len(left_targets) != len(right_targets):
        raise RuntimeError(f"No metadata and {split} lengths differ under --strict_meta.")
    return idx_left, idx_right, pd.DataFrame({"row": np.arange(n, dtype=np.int64)})


def validate_split_labels(
    left: SourceOutput,
    right: SourceOutput,
    split: str,
    idx_left: np.ndarray,
    idx_right: np.ndarray,
) -> None:
    left_targets, left_raw, _, _ = split_arrays(left, split)
    right_targets, right_raw, _, _ = split_arrays(right, split)
    yl = left_targets[idx_left]
    yr = right_targets[idx_right]
    if not np.array_equal(yl, yr):
        mismatch = int((yl != yr).sum())
        raise RuntimeError(
            f"Paired {split} labels differ in {mismatch} rows. "
            "The ensemble requires the same observed labels for both sources."
        )
    rl = left_raw[idx_left]
    rr = right_raw[idx_right]
    finite = np.isfinite(rl) & np.isfinite(rr)
    if finite.any() and float(np.max(np.abs(rl[finite] - rr[finite]))) > 1e-3:
        raise RuntimeError(
            f"Paired {split} raw visibility values differ. "
            "Check meta alignment and y files before using the ensemble metrics."
        )


def mean_probs(left_probs: np.ndarray, right_probs: np.ndarray) -> np.ndarray:
    if left_probs.shape != right_probs.shape:
        raise ValueError(f"Cannot average probability arrays with shapes {left_probs.shape} and {right_probs.shape}")
    probs = 0.5 * (left_probs.astype(np.float64) + right_probs.astype(np.float64))
    row_sums = probs.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probs)) or np.any(row_sums <= 0):
        raise ValueError("Ensemble probabilities contain non-finite values or non-positive row sums.")
    probs = probs / row_sums
    return probs.astype(np.float32)


def ensemble_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    args: argparse.Namespace,
    outputs: Dict[str, SourceOutput],
) -> Tuple[Dict[str, float], str, str, Dict[str, float]]:
    mode = args.ensemble_threshold_mode
    if mode == "mean_checkpoint":
        fogs = [outputs[s].thresholds.get("fog", math.nan) for s in ("tianji", "ifs")]
        mists = [outputs[s].thresholds.get("mist", math.nan) for s in ("tianji", "ifs")]
        if not all(math.isfinite(float(v)) for v in fogs + mists):
            raise RuntimeError(
                "--ensemble_threshold_mode mean_checkpoint requires finite source thresholds. "
                "Use --ensemble_threshold_mode val_search or fixed instead."
            )
        thresholds = {"fog": float(np.mean(fogs)), "mist": float(np.mean(mists))}
        preds = paired.predict_from_probs(val_probs, "fixed", thresholds["fog"], thresholds["mist"])
        metrics = paired.compute_metrics(val_targets, preds, probs=val_probs)
        return thresholds, "fixed", "mean_source_thresholds", metrics
    return choose_thresholds(val_probs, val_targets, mode, args, ckpt_meta=None)


def rows_for_source(
    source: str,
    metrics: Dict[str, float],
    output: Optional[SourceOutput],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = dict(extra or {})
    if source == ENSEMBLE_SOURCE:
        payload.setdefault("source_label", ENSEMBLE_LABEL)
    if output is not None:
        payload.update(
            {
                "temperature": output.temperature,
                "fog_threshold": output.thresholds.get("fog"),
                "mist_threshold": output.thresholds.get("mist"),
                "threshold_source": output.threshold_source,
                "feature_dim": output.feature_dim,
                "extra_feat_dim": output.extra_feat_dim,
                "checkpoint": output.spec.ckpt_path,
                "data_dir": output.spec.data_dir,
            }
        )
    return paired.rows_from_metrics(source, metrics, payload)


def bootstrap_delta_ci_named(
    targets: np.ndarray,
    left_preds: np.ndarray,
    right_preds: np.ndarray,
    metric_names: Sequence[str],
    n_bootstrap: int,
    bootstrap_size: int,
    seed: int,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    n = len(targets)
    bs = n if bootstrap_size <= 0 else min(int(bootstrap_size), n)
    values: Dict[str, List[float]] = {m: [] for m in metric_names}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=bs, endpoint=False)
        left_metrics = paired.compute_metrics(targets[idx], left_preds[idx], probs=None, prefix_counts=False)
        right_metrics = paired.compute_metrics(targets[idx], right_preds[idx], probs=None, prefix_counts=False)
        for m in metric_names:
            values[m].append(float(left_metrics[m]) - float(right_metrics[m]))
        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"[bootstrap {left_name}-{right_name}] {b + 1}/{n_bootstrap}", flush=True)

    rows = []
    delta_col = f"delta_{left_name}_minus_{right_name}"
    for m, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "metric": m,
                "left_source": left_name,
                "right_source": right_name,
                "bootstrap_reps": n_bootstrap,
                "bootstrap_size": bs,
                f"{delta_col}_mean": float(np.mean(arr)),
                f"{delta_col}_ci95_low": float(np.percentile(arr, 2.5)),
                f"{delta_col}_ci95_high": float(np.percentile(arr, 97.5)),
                "preferred_direction": paired.metric_direction(m),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    specs: Dict[str, paired.SourceSpec],
    build_configs: Dict[str, Dict],
    overall_df: pd.DataFrame,
    delta_tianji: pd.DataFrame,
    delta_ifs: pd.DataFrame,
    n_val: int,
    n_test: int,
    ensemble_thresholds_used: Dict[str, float],
    ensemble_threshold_source: str,
) -> None:
    def fmt(metric: str, source: str) -> str:
        row = overall_df.loc[overall_df["source"] == source]
        if row.empty or metric not in row.columns:
            return "NA"
        val = row.iloc[0][metric]
        return "NA" if pd.isna(val) else f"{float(val):.4f}"

    key_metrics = [
        "fog_csi",
        "fog_pod",
        "fog_precision",
        "mist_csi",
        "mist_pod",
        "mist_precision",
        "low_vis_csi",
        "low_vis_precision",
        "low_vis_recall",
        "low_vis_fpr",
        "accuracy",
        "multiclass_brier",
        "ece_low_vis",
    ]
    dt = delta_tianji.set_index("metric") if not delta_tianji.empty else pd.DataFrame()
    di = delta_ifs.set_index("metric") if not delta_ifs.empty else pd.DataFrame()

    with open(path, "w", encoding="utf-8") as f:
        f.write("Overlap S2 mean-softmax ensemble evaluation\n")
        f.write("=" * 56 + "\n\n")
        f.write("Purpose: test whether averaging post-softmax probabilities from the\n")
        f.write("Tianji-input and IFS-input overlap models improves held-out performance.\n")
        f.write("The ensemble uses paired rows only; validation rows are used only for\n")
        f.write("ensemble threshold selection, and test rows are used once for reporting.\n\n")
        f.write(f"Model architecture: {args.model_arch}\n")
        f.write(f"Paired validation rows: {n_val}\n")
        f.write(f"Paired test rows: {n_test}\n")
        f.write(f"Source threshold mode: {args.source_threshold_mode}\n")
        f.write(f"Ensemble threshold mode: {args.ensemble_threshold_mode}\n")
        f.write(f"Ensemble threshold source: {ensemble_threshold_source}\n")
        f.write(f"Ensemble thresholds: {json.dumps(ensemble_thresholds_used, ensure_ascii=False)}\n")
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
            "metric,tianji,ifs,ensemble,"
            "delta_ensemble_minus_tianji,delta_ensemble_minus_ifs,"
            "preferred_direction,ensemble_better_than_tianji,ensemble_better_than_ifs\n"
        )
        for m in key_metrics:
            if m not in overall_df.columns:
                continue
            r_t = dt.loc[m] if m in dt.index else None
            r_i = di.loc[m] if m in di.index else None
            if r_t is None or r_i is None:
                delta_t = "NA"
                delta_i = "NA"
                direction = "NA"
                better_t = "NA"
                better_i = "NA"
            else:
                delta_t = f"{float(r_t['delta_ensemble_mean_softmax_minus_tianji']):.4f}"
                delta_i = f"{float(r_i['delta_ensemble_mean_softmax_minus_ifs']):.4f}"
                direction = str(r_t["preferred_direction"])
                better_t = str(bool(r_t["ensemble_mean_softmax_better"]))
                better_i = str(bool(r_i["ensemble_mean_softmax_better"]))
            f.write(
                f"{m},{fmt(m, 'tianji')},{fmt(m, 'ifs')},{fmt(m, ENSEMBLE_SOURCE)},"
                f"{delta_t},{delta_i},{direction},{better_t},{better_i}\n"
            )


def plot_ensemble_key_metrics_figure(overall_df: pd.DataFrame, out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting env dependent.
        print(f"[WARN] matplotlib unavailable; skip ensemble figure: {exc}", flush=True)
        return []

    source_order = [s for s in ("tianji", "ifs", ENSEMBLE_SOURCE) if s in set(overall_df["source"].astype(str))]
    if not source_order:
        return []

    row_by_source = {
        str(row["source"]): row
        for _, row in overall_df.iterrows()
        if str(row.get("source", "")) in source_order
    }
    source_labels = {
        "tianji": "Tianji",
        "ifs": "IFS",
        ENSEMBLE_SOURCE: "Mean softmax",
    }
    source_colors = {
        "tianji": "#2E5A87",
        "ifs": "#6C6C6C",
        ENSEMBLE_SOURCE: "#18864B",
    }
    panels = [
        ("Fog (0-500 m)", [("fog_precision", "Precision"), ("fog_pod", "Recall"), ("fog_csi", "CSI")]),
        ("Mist (500-1000 m)", [("mist_precision", "Precision"), ("mist_pod", "Recall"), ("mist_csi", "CSI")]),
        (
            "Low visibility (<1000 m)",
            [("low_vis_precision", "Precision"), ("low_vis_recall", "Recall"), ("low_vis_csi", "CSI"), ("low_vis_fpr", "FPR")],
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

    def adaptive_ylim(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.20
        vmax = float(np.nanmax(arr))
        if vmax <= 0:
            return 0.10
        padded = min(1.0, vmax + max(0.025, 0.10 * vmax))
        step = 0.02 if padded <= 0.20 else 0.05
        return min(1.0, max(step * 3, math.ceil(padded / step) * step))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=False, constrained_layout=True)
    width = min(0.26, 0.78 / max(len(source_order), 1))
    for ax_idx, (ax, (title, metrics)) in enumerate(zip(axes, panels)):
        x = np.arange(len(metrics), dtype=np.float64)
        panel_values: List[float] = []
        for source in source_order:
            row = row_by_source[source]
            for metric, _ in metrics:
                try:
                    panel_values.append(float(row.get(metric, np.nan)))
                except Exception:
                    panel_values.append(math.nan)
        y_max = adaptive_ylim(panel_values)
        for src_idx, source in enumerate(source_order):
            row = row_by_source[source]
            vals: List[float] = []
            finite_flags: List[bool] = []
            for metric, _ in metrics:
                try:
                    value = float(row.get(metric, np.nan))
                except Exception:
                    value = math.nan
                finite_flags.append(bool(np.isfinite(value)))
                vals.append(value if np.isfinite(value) else 0.0)
            offset = (src_idx - (len(source_order) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                vals,
                width * 0.92,
                label=source_labels.get(source, source) if ax_idx == 0 else None,
                color=source_colors.get(source, "#7F7F7F"),
                edgecolor="white",
                linewidth=0.45,
            )
            for bar, value, ok in zip(bars, vals, finite_flags):
                if ok:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        min(value + y_max * 0.025, y_max * 0.98),
                        f"{value:.2f}",
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
        if ax_idx == 0:
            ax.set_ylabel("Score")
        if title.startswith("Low visibility"):
            ax.text(0.98, 0.96, "FPR lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(handles), frameon=False)

    out_paths = [
        out_dir / "fig_overlap_softmax_ensemble_key_metrics.png",
        out_dir / "fig_overlap_softmax_ensemble_key_metrics.pdf",
        out_dir / "fig_overlap_softmax_ensemble_key_metrics.svg",
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

    specs = build_specs(args)
    for source, spec in specs.items():
        if not os.path.isdir(spec.data_dir):
            raise FileNotFoundError(f"{source} data_dir does not exist: {spec.data_dir}")
        paired.require_file(spec.ckpt_path, f"{source} checkpoint")
        paired.require_file(spec.scaler_path, f"{source} scaler")

    train_mod = paired.import_training_module(args)
    device = paired.resolve_device(args.device)
    print(f"[device] {device}", flush=True)

    build_configs = {source: paired.read_build_config(spec.data_dir) for source, spec in specs.items()}
    paired.validate_build_time_alignment(build_configs, args.allow_legacy_time_alignment)

    outputs = {
        source: evaluate_source(source, spec, train_mod, args, device)
        for source, spec in specs.items()
    }

    idx_t_val, idx_i_val, meta_val = align_split_outputs(outputs["tianji"], outputs["ifs"], "val", args.strict_meta)
    idx_t_test, idx_i_test, meta_test = align_split_outputs(outputs["tianji"], outputs["ifs"], "test", args.strict_meta)
    validate_split_labels(outputs["tianji"], outputs["ifs"], "val", idx_t_val, idx_i_val)
    validate_split_labels(outputs["tianji"], outputs["ifs"], "test", idx_t_test, idx_i_test)

    y_val = outputs["tianji"].val_targets[idx_t_val]
    t_val_probs = outputs["tianji"].val_probs[idx_t_val]
    i_val_probs = outputs["ifs"].val_probs[idx_i_val]
    e_val_probs = mean_probs(t_val_probs, i_val_probs)

    e_thresholds, e_pred_mode, e_threshold_source, e_val_metrics = ensemble_thresholds(
        e_val_probs, y_val, args, outputs
    )
    e_val_preds = paired.predict_from_probs(
        e_val_probs,
        e_pred_mode,
        e_thresholds.get("fog", 0.5),
        e_thresholds.get("mist", 0.5),
    )
    e_val_metrics = paired.compute_metrics(y_val, e_val_preds, probs=e_val_probs)

    y = outputs["tianji"].test_targets[idx_t_test]
    raw_vis = outputs["tianji"].test_raw_vis[idx_t_test]
    t_probs = outputs["tianji"].test_probs[idx_t_test]
    i_probs = outputs["ifs"].test_probs[idx_i_test]
    e_probs = mean_probs(t_probs, i_probs)
    t_preds = outputs["tianji"].test_preds[idx_t_test]
    i_preds = outputs["ifs"].test_preds[idx_i_test]
    e_preds = paired.predict_from_probs(
        e_probs,
        e_pred_mode,
        e_thresholds.get("fog", 0.5),
        e_thresholds.get("mist", 0.5),
    )

    metrics_t = paired.compute_metrics(y, t_preds, probs=t_probs)
    metrics_i = paired.compute_metrics(y, i_preds, probs=i_probs)
    metrics_e = paired.compute_metrics(y, e_preds, probs=e_probs)

    common_extra = {"model_arch": args.model_arch}
    overall_df = pd.DataFrame(
        [
            rows_for_source("tianji", metrics_t, outputs["tianji"], common_extra),
            rows_for_source("ifs", metrics_i, outputs["ifs"], common_extra),
            rows_for_source(
                ENSEMBLE_SOURCE,
                metrics_e,
                None,
                {
                    **common_extra,
                    "source_label": ENSEMBLE_LABEL,
                    "temperature": "mean_of_source_probabilities",
                    "fog_threshold": e_thresholds.get("fog"),
                    "mist_threshold": e_thresholds.get("mist"),
                    "threshold_source": e_threshold_source,
                    "feature_dim": outputs["tianji"].feature_dim,
                    "extra_feat_dim": outputs["tianji"].extra_feat_dim,
                    "checkpoint": f"{outputs['tianji'].spec.ckpt_path};{outputs['ifs'].spec.ckpt_path}",
                    "data_dir": f"{outputs['tianji'].spec.data_dir};{outputs['ifs'].spec.data_dir}",
                },
            ),
        ]
    )
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)

    validation_df = pd.DataFrame(
        [
            rows_for_source("tianji", outputs["tianji"].val_metrics, outputs["tianji"], common_extra),
            rows_for_source("ifs", outputs["ifs"].val_metrics, outputs["ifs"], common_extra),
            rows_for_source(
                ENSEMBLE_SOURCE,
                e_val_metrics,
                None,
                {
                    **common_extra,
                    "source_label": ENSEMBLE_LABEL,
                    "fog_threshold": e_thresholds.get("fog"),
                    "mist_threshold": e_thresholds.get("mist"),
                    "threshold_source": e_threshold_source,
                },
            ),
        ]
    )
    validation_df.to_csv(out_dir / "validation_metrics.csv", index=False)

    delta_e_t = paired.compare_metric_sets(metrics_e, metrics_t, ENSEMBLE_SOURCE, "tianji")
    delta_e_i = paired.compare_metric_sets(metrics_e, metrics_i, ENSEMBLE_SOURCE, "ifs")
    delta_t_i = paired.compare_metric_sets(metrics_t, metrics_i, "tianji", "ifs")
    delta_e_t.to_csv(out_dir / "metric_deltas_ensemble_minus_tianji.csv", index=False)
    delta_e_i.to_csv(out_dir / "metric_deltas_ensemble_minus_ifs.csv", index=False)
    delta_t_i.to_csv(out_dir / "metric_deltas_tianji_minus_ifs.csv", index=False)

    paired.write_confusion_csv(str(out_dir / "confusion_tianji.csv"), y, t_preds)
    paired.write_confusion_csv(str(out_dir / "confusion_ifs.csv"), y, i_preds)
    paired.write_confusion_csv(str(out_dir / "confusion_ensemble_mean_softmax.csv"), y, e_preds)

    scenario_rows = []
    for scenario, mask in paired.scenario_masks(meta_test, args.local_time_offset_hours).items():
        scenario_rows.append(
            paired.rows_from_metrics("tianji", paired.compute_metrics(y[mask], t_preds[mask], probs=t_probs[mask]), {"scenario": scenario})
        )
        scenario_rows.append(
            paired.rows_from_metrics("ifs", paired.compute_metrics(y[mask], i_preds[mask], probs=i_probs[mask]), {"scenario": scenario})
        )
        scenario_rows.append(
            paired.rows_from_metrics(
                ENSEMBLE_SOURCE,
                paired.compute_metrics(y[mask], e_preds[mask], probs=e_probs[mask]),
                {"scenario": scenario, "source_label": ENSEMBLE_LABEL},
            )
        )
    pd.DataFrame(scenario_rows).to_csv(out_dir / "scenario_metrics.csv", index=False)

    scenario_delta_rows = []
    for scenario, mask in paired.scenario_masks(meta_test, args.local_time_offset_hours).items():
        me = paired.compute_metrics(y[mask], e_preds[mask], probs=e_probs[mask])
        mt = paired.compute_metrics(y[mask], t_preds[mask], probs=t_probs[mask])
        mi = paired.compute_metrics(y[mask], i_preds[mask], probs=i_probs[mask])
        d_et = paired.compare_metric_sets(me, mt, ENSEMBLE_SOURCE, "tianji")
        d_ei = paired.compare_metric_sets(me, mi, ENSEMBLE_SOURCE, "ifs")
        d_et.insert(0, "scenario", scenario)
        d_ei.insert(0, "scenario", scenario)
        d_et.insert(1, "comparison", "ensemble_minus_tianji")
        d_ei.insert(1, "comparison", "ensemble_minus_ifs")
        scenario_delta_rows.extend([d_et, d_ei])
    if scenario_delta_rows:
        pd.concat(scenario_delta_rows, ignore_index=True).to_csv(out_dir / "scenario_metric_deltas.csv", index=False)

    if not args.skip_bootstrap and args.bootstrap > 0:
        boot_frames = [
            bootstrap_delta_ci_named(
                y,
                e_preds,
                t_preds,
                paired.BOOTSTRAP_DEFAULT_METRICS,
                args.bootstrap,
                args.bootstrap_size,
                args.bootstrap_seed,
                ENSEMBLE_SOURCE,
                "tianji",
            ),
            bootstrap_delta_ci_named(
                y,
                e_preds,
                i_preds,
                paired.BOOTSTRAP_DEFAULT_METRICS,
                args.bootstrap,
                args.bootstrap_size,
                args.bootstrap_seed + 1,
                ENSEMBLE_SOURCE,
                "ifs",
            ),
        ]
        pd.concat(boot_frames, ignore_index=True).to_csv(out_dir / "paired_bootstrap_delta_ci.csv", index=False)

    if not args.no_per_sample_csv:
        sample_df = meta_test.reset_index(drop=True).copy()
        sample_df["y_true"] = y
        sample_df["vis_raw_m"] = raw_vis
        for source, probs, preds in (
            ("tianji", t_probs, t_preds),
            ("ifs", i_probs, i_preds),
            (ENSEMBLE_SOURCE, e_probs, e_preds),
        ):
            sample_df[f"{source}_pred"] = preds
            sample_df[f"{source}_p_fog"] = probs[:, 0]
            sample_df[f"{source}_p_mist"] = probs[:, 1]
            sample_df[f"{source}_p_clear"] = probs[:, 2]
            sample_df[f"{source}_correct"] = preds == y
        sample_df["ensemble_wins_vs_tianji"] = sample_df[f"{ENSEMBLE_SOURCE}_correct"] & ~sample_df["tianji_correct"]
        sample_df["ensemble_wins_vs_ifs"] = sample_df[f"{ENSEMBLE_SOURCE}_correct"] & ~sample_df["ifs_correct"]
        sample_df["ensemble_loses_vs_tianji"] = ~sample_df[f"{ENSEMBLE_SOURCE}_correct"] & sample_df["tianji_correct"]
        sample_df["ensemble_loses_vs_ifs"] = ~sample_df[f"{ENSEMBLE_SOURCE}_correct"] & sample_df["ifs_correct"]
        sample_df.to_csv(out_dir / "per_sample_softmax_ensemble_eval.csv", index=False)

    run_config = {
        "args": vars(args),
        "specs": {k: vars(v) for k, v in specs.items()},
        "model_arch": args.model_arch,
        "build_configs": build_configs,
        "paired_validation_rows": int(len(y_val)),
        "paired_test_rows": int(len(y)),
        "ensemble": {
            "method": "equal-weight mean of post-softmax class probabilities",
            "weights": {"tianji": 0.5, "ifs": 0.5},
            "threshold_mode": args.ensemble_threshold_mode,
            "threshold_source": e_threshold_source,
            "fog_threshold": e_thresholds.get("fog"),
            "mist_threshold": e_thresholds.get("mist"),
        },
        "source_thresholds": {
            source: {
                "threshold_source": out.threshold_source,
                "temperature": out.temperature,
                "fog_threshold": out.thresholds.get("fog"),
                "mist_threshold": out.thresholds.get("mist"),
                "checkpoint": out.spec.ckpt_path,
            }
            for source, out in outputs.items()
        },
        "evaluation_scope": (
            "validation split calibrates/selects ensemble thresholds; "
            "test split is held out for final metrics"
        ),
        "class_definition": {
            "0": "0 <= visibility < 500 m",
            "1": "500 <= visibility < 1000 m",
            "2": "visibility >= 1000 m",
        },
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    write_report(
        out_dir / "softmax_ensemble_report.txt",
        args,
        specs,
        build_configs,
        overall_df,
        delta_e_t,
        delta_e_i,
        len(y_val),
        len(y),
        e_thresholds,
        e_threshold_source,
    )

    if not args.no_figures:
        plot_ensemble_key_metrics_figure(overall_df, out_dir)

    print("\n[OK] wrote mean-softmax ensemble outputs to:", out_dir, flush=True)
    print(
        delta_e_t[delta_e_t["metric"].isin(paired.BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False),
        flush=True,
    )
    print(
        "\n[ensemble minus IFS]",
        flush=True,
    )
    print(
        delta_e_i[delta_e_i["metric"].isin(paired.BOOTSTRAP_DEFAULT_METRICS)].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
