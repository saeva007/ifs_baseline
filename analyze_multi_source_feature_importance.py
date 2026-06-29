#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired multi-source grouped permutation importance for Static-RNN models.

The same (valid time, station) rows and donor permutations are used for every
source. Dynamic variables are permuted as complete 12 h sequences. Marginal
model reliance is the primary analysis; a meteorology-stratified permutation
is reported as a dependence-aware sensitivity check for correlated inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EVAL_DIR_CANDIDATES = [ROOT / "paper_eval", ROOT / "vis_eval"]
FEATURE_CATALOG_CANDIDATES = [path / "feature_catalog_pm10_pm25.py" for path in EVAL_DIR_CANDIDATES]
for path in (SCRIPT_DIR, *EVAL_DIR_CANDIDATES, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from feature_catalog_pm10_pm25 import permutation_groups

    FEATURE_CATALOG_PATH = str(Path(sys.modules["feature_catalog_pm10_pm25"].__file__).resolve())
    HAS_FEATURE_CATALOG = True
except ModuleNotFoundError:
    FEATURE_CATALOG_PATH = ""
    HAS_FEATURE_CATALOG = False

    def permutation_groups(
        window_size: int,
        dyn_vars_count: int,
        extra_feat_dim: int,
        dynamic_feature_order: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """Dynamic-only fallback when the paper-eval feature catalog is absent."""
        del extra_feat_dim
        if not dynamic_feature_order or len(dynamic_feature_order) != int(dyn_vars_count):
            checked = ", ".join(str(path) for path in FEATURE_CATALOG_CANDIDATES)
            raise FileNotFoundError(
                "feature_catalog_pm10_pm25.py was not found and dataset_build_config.json "
                f"does not provide a valid dynamic_feature_order. Checked: {checked}"
            )
        groups: List[Dict[str, object]] = []
        for feature_idx, feature in enumerate(dynamic_feature_order):
            columns = [t * int(dyn_vars_count) + feature_idx for t in range(int(window_size))]
            groups.append(
                {
                    "feature": str(feature),
                    "block": "dynamic_12h",
                    "columns": columns,
                    "n_columns": len(columns),
                }
            )
        return groups


PACKAGE_DEFINITIONS = {
    "native_near_surface_moisture": ["RH2M", "D2M", "DPD", "Q_1000", "DP_1000"],
    "native_low_level_moisture_structure": ["RH_925", "Q_1000", "Q_925", "DP_1000", "DP_925"],
    "native_thermodynamics": ["T2M", "T_925", "INVERSION", "MSLP"],
    "native_wind_ventilation": [
        "U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925", "W_925", "W_1000",
    ],
    "native_cloud_precip_radiation": ["PRECIP", "SW_RAD", "CAPE", "LCC"],
    "native_aerosol": ["PM10_ugm3", "PM25_ugm3"],
}

SHARED_MOISTURE_CANDIDATES = ["RH2M", "DPD", "RH_925", "Q_1000", "Q_925", "DP_1000", "DP_925"]
METRICS = ("low_vis_csi", "low_vis_recall", "low_vis_precision", "low_vis_brier")
SOURCE_COLORS = {
    "tianji": "#176B87",
    "ifs": "#B65C19",
    "era5": "#555B66",
    "pangu": "#684A9B",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Paired grouped permutation/model-reliance analysis for source-trained Static-RNN models."
    )
    ap.add_argument(
        "--sources",
        required=True,
        help="Semicolon-separated tag=data_dir|checkpoint|scaler-or-AUTO|label specs.",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reference_source", default="tianji")
    ap.add_argument("--static_rnn_train_dir", default=os.environ.get("STATIC_RNN_TRAIN_DIR", str(ROOT / "train")))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--sample_size", type=int, default=100000, help="Uniform common-row sample; 0 uses all rows.")
    ap.add_argument("--limit_rows", type=int, default=0, help="Debug limit applied before cross-source alignment.")
    ap.add_argument("--min_low_vis", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--modes", default="marginal,stratified", help="marginal and/or stratified.")
    ap.add_argument(
        "--group_scope",
        choices=["dynamic", "all"],
        default="dynamic",
        help="dynamic keeps source-comparable 12 h variables; all also includes static/engineered groups.",
    )
    ap.add_argument("--max_groups", type=int, default=0, help="Smoke-test limit after group construction.")
    ap.add_argument("--no_packages", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--allow_legacy_time_alignment", action="store_true")
    ap.add_argument("--allow_partial_load", action="store_true")
    ap.add_argument("--checkpoint_tag", default="S2_PhaseB_best_score")
    return ap.parse_args()


def load_source_evaluator():
    path = SCRIPT_DIR / "test_PMST_overlap_forecast_source_s2.py"
    spec = importlib.util.spec_from_file_location("source_eval_for_importance", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_sources(text: str, ev, args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, str]]:
    specs: Dict[str, object] = {}
    labels: Dict[str, str] = {}
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad source spec {item!r}; expected tag=data|checkpoint|scaler|label")
        tag, payload = item.split("=", 1)
        parts = [part.strip() for part in payload.split("|")]
        if len(parts) < 3:
            raise ValueError(f"Bad source spec {item!r}; expected tag=data|checkpoint|scaler|label")
        tag = tag.strip()
        labels[tag] = parts[3] if len(parts) > 3 and parts[3] else tag
        specs[tag] = ev.SourceSpec(name=tag, data_dir=parts[0], ckpt_path=parts[1], scaler_path=parts[2])
    if len(specs) < 2:
        raise ValueError("At least two sources are required for paired multi-source importance.")
    if args.reference_source not in specs:
        raise KeyError(f"reference_source={args.reference_source!r} is not in --sources")
    return specs, labels


def evaluator_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_arch="static_rnn",
        static_rnn_train_dir=cli.static_rnn_train_dir,
        static_rnn_encoder="gru",
        static_rnn_hidden_dim=256,
        static_rnn_static_hidden_dim=96,
        static_rnn_fe_hidden_dim=128,
        static_rnn_fusion_hidden_dim=256,
        static_rnn_veg_emb_dim=16,
        static_rnn_rnn_layers=1,
        static_rnn_dropout=0.2,
        static_rnn_bidirectional=False,
        static_rnn_pooling="mean",
        static_rnn_no_fe=False,
        static_rnn_no_pm=False,
        window=12,
        dyn_vars_count=27,
        expected_extra_dim=36,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        device=cli.device,
        threshold_mode="argmax",
        threshold_search_policy="response",
        fog_threshold=0.5,
        mist_threshold=0.5,
        no_temp_scaling=True,
        temp_lr=0.01,
        temp_max_iter=10,
        skip_validation_inference=True,
        limit_samples=cli.limit_rows,
        allow_partial_load=cli.allow_partial_load,
        checkpoint_tag=cli.checkpoint_tag,
        ckpt_dir=str(SCRIPT_DIR / "checkpoints"),
        allow_legacy_time_alignment=cli.allow_legacy_time_alignment,
    )


def load_source_for_importance(source: str, spec, train_mod, args: argparse.Namespace, device, ev):
    """Load model and alignment metadata without running full-test inference."""
    source_dyn_vars = ev.source_dyn_vars_count(spec.data_dir, args.dyn_vars_count)
    _, _, dynamic_order = ev.dataset_layout_from_config(spec.data_dir)
    feature_dim, extra_dim = ev.infer_feature_layout(
        spec.data_dir,
        "test",
        args.window,
        source_dyn_vars,
        args.expected_extra_dim,
    )
    scaler = ev.joblib.load(spec.scaler_path)
    model = ev.load_model(
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
    if device.type == "cuda" and ev.torch.cuda.device_count() > 1:
        model = ev.torch.nn.DataParallel(model)
        print(f"[{source}] DataParallel on {ev.torch.cuda.device_count()} devices", flush=True)
    y_raw, y_cls = ev.build_y_cls_raw(np.load(Path(spec.data_dir) / "y_test.npy"))
    if args.limit_samples and args.limit_samples > 0:
        n = min(int(args.limit_samples), len(y_cls))
        y_raw = y_raw[:n]
        y_cls = y_cls[:n]
        indices = np.arange(n, dtype=np.int64)
    else:
        indices = None
    meta = ev.load_meta(spec.data_dir, "test", indices=indices)
    if meta is None:
        raise FileNotFoundError(f"{source}: meta_test.csv is required for paired feature importance")
    return ev.SourceEval(
        source=source,
        spec=spec,
        feature_dim=feature_dim,
        extra_feat_dim=extra_dim,
        dyn_vars_count=source_dyn_vars,
        dynamic_feature_order=dynamic_order,
        temperature=1.0,
        thresholds={"fog": math.nan, "mist": math.nan},
        threshold_source="argmax",
        val_metrics={"n": 0},
        test_probs=np.zeros((0, 3), dtype=np.float32),
        test_preds=np.zeros(0, dtype=np.int64),
        test_targets=y_cls,
        test_raw_vis=y_raw,
        test_meta=meta,
        model=model,
        scaler=scaler,
    )


def uniform_common_sample(y: np.ndarray, sample_size: int, seed: int, min_low_vis: int) -> np.ndarray:
    n = len(y)
    if sample_size <= 0 or sample_size >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=int(sample_size), replace=False))
    low_count = int(np.sum(np.asarray(y)[idx] <= 1))
    if low_count < int(min_low_vis):
        raise RuntimeError(
            f"Uniform paired sample contains only {low_count} low-visibility cases; "
            f"increase --sample_size or lower --min_low_vis for a smoke test."
        )
    return idx


def canonical_order(order: Optional[Sequence[str]], dyn_vars: int) -> List[str]:
    if not order:
        raise ValueError("dataset_build_config.json must provide dynamic_feature_order for publishable importance analysis.")
    names = [str(name) for name in order]
    if len(names) != int(dyn_vars):
        raise ValueError(f"dynamic_feature_order length {len(names)} != dyn_vars {dyn_vars}")
    return names


def dynamic_columns(features: Iterable[str], order: Sequence[str], window: int) -> List[int]:
    lookup = {str(name).upper(): idx for idx, name in enumerate(order)}
    feature_indices = [lookup[name.upper()] for name in features if name.upper() in lookup]
    return [t * len(order) + idx for t in range(int(window)) for idx in feature_indices]


def add_physical_packages(
    groups: List[Dict[str, object]],
    order: Sequence[str],
    window: int,
    shared_dynamic: Sequence[str],
) -> List[Dict[str, object]]:
    out = list(groups)
    lookup = {name.upper(): name for name in order}
    packages = dict(PACKAGE_DEFINITIONS)
    packages["shared_low_level_moisture"] = [name for name in SHARED_MOISTURE_CANDIDATES if name.upper() in {x.upper() for x in shared_dynamic}]
    packages["shared_dynamic_all"] = list(shared_dynamic)
    for name, candidates in packages.items():
        members = [lookup[item.upper()] for item in candidates if item.upper() in lookup]
        if not members:
            continue
        cols = dynamic_columns(members, order, window)
        out.append(
            {
                "feature": name,
                "block": "physical_package",
                "columns": cols,
                "n_columns": len(cols),
                "members": members,
                "analysis_level": "shared_package" if name.startswith("shared_") else "source_native_package",
            }
        )
    return out


def group_members(group: Dict[str, object]) -> List[str]:
    members = group.get("members")
    if isinstance(members, (list, tuple)):
        return [str(v) for v in members]
    if group.get("block") == "dynamic_12h":
        return [str(group["feature"])]
    return []


def source_color(tag: str) -> str:
    low = tag.lower()
    for family, color in SOURCE_COLORS.items():
        if family in low:
            return color
    return "#555B66"


def marginal_donor(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    donor = rng.permutation(n)
    return donor, float(np.mean(donor != np.arange(n))), 0.0


def _quantile_codes(values: np.ndarray, bins: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.full(len(values), -1, dtype=np.int16)
    if finite.sum() < bins:
        return out
    edges = np.unique(np.nanquantile(values[finite], np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return out
    out[finite] = np.digitize(values[finite], edges[1:-1], right=True).astype(np.int16)
    return out


def stratified_donor(
    meta: pd.DataFrame,
    rows: np.ndarray,
    order: Sequence[str],
    window: int,
    excluded_features: Sequence[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    n = len(rows)
    time = pd.to_datetime(meta.get("time", pd.Series([pd.NaT] * n)), errors="coerce")
    month = time.dt.month.fillna(1).to_numpy(dtype=np.int16)
    season = ((month % 12) // 3).astype(np.int16)
    hour_bin = (time.dt.hour.fillna(0).to_numpy(dtype=np.int16) // 6).astype(np.int16)
    lat = pd.to_numeric(meta.get("lat", pd.Series(np.zeros(n))), errors="coerce").fillna(0).to_numpy()
    lon = pd.to_numeric(meta.get("lon", pd.Series(np.zeros(n))), errors="coerce").fillna(0).to_numpy()
    lat_bin = np.floor(lat / 10.0).astype(np.int16)
    lon_bin = np.floor(lon / 10.0).astype(np.int16)

    excluded = {name.upper() for name in excluded_features}
    lookup = {name.upper(): idx for idx, name in enumerate(order)}
    anchor_code = np.full(n, -1, dtype=np.int16)
    for anchor in ("Q_1000", "RH2M", "RH_925", "DP_1000", "T2M"):
        if anchor in lookup and anchor not in excluded:
            col = (int(window) - 1) * len(order) + lookup[anchor]
            anchor_code = _quantile_codes(rows[:, col])
            break

    full_key = pd.MultiIndex.from_arrays([season, hour_bin, lat_bin, lon_bin, anchor_code])
    coarse_key = pd.MultiIndex.from_arrays([season, hour_bin, anchor_code])
    donor = np.arange(n, dtype=np.int64)
    unresolved = np.ones(n, dtype=bool)
    for _, positions in pd.Series(np.arange(n)).groupby(full_key, sort=False):
        pos = positions.to_numpy(dtype=np.int64)
        if len(pos) >= 2:
            donor[pos] = rng.permutation(pos)
            unresolved[pos] = False
    fallback_count = int(unresolved.sum())
    if fallback_count:
        unresolved_idx = np.flatnonzero(unresolved)
        unresolved_keys = coarse_key[unresolved_idx]
        for _, positions in pd.Series(unresolved_idx).groupby(unresolved_keys, sort=False):
            pos = positions.to_numpy(dtype=np.int64)
            if len(pos) >= 2:
                donor[pos] = rng.permutation(pos)
    moved_fraction = float(np.mean(donor != np.arange(n)))
    return donor, moved_fraction, fallback_count / max(n, 1)


def daily_binary_statistics(
    y_true: np.ndarray,
    probs: np.ndarray,
    dates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_low = np.asarray(y_true) <= 1
    pred_low = np.argmax(probs, axis=1) <= 1
    low_prob = probs[:, 0] + probs[:, 1]
    codes, unique_dates = pd.factorize(pd.Series(dates).astype(str), sort=True)
    n_dates = len(unique_dates)

    def bc(values: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=values.astype(np.float64), minlength=n_dates)

    stats = np.column_stack(
        [
            bc(y_low & pred_low),
            bc(~y_low & pred_low),
            bc(y_low & ~pred_low),
            bc(~y_low & ~pred_low),
            bc((low_prob - y_low.astype(np.float64)) ** 2),
            bc(np.ones(len(y_low), dtype=np.float64)),
        ]
    )
    return stats, np.asarray(unique_dates)


def metrics_from_stats(stats: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, _tn, brier_sum, n = np.asarray(stats, dtype=np.float64).sum(axis=0)
    return {
        "low_vis_csi": float(tp / (tp + fp + fn)) if tp + fp + fn > 0 else math.nan,
        "low_vis_recall": float(tp / (tp + fn)) if tp + fn > 0 else math.nan,
        "low_vis_precision": float(tp / (tp + fp)) if tp + fp > 0 else math.nan,
        "low_vis_brier": float(brier_sum / n) if n > 0 else math.nan,
    }


def importance_delta(metric: str, baseline: float, permuted: float) -> float:
    return float(permuted - baseline) if metric == "low_vis_brier" else float(baseline - permuted)


def date_block_importance_ci(
    baseline_daily: np.ndarray,
    permuted_daily: Sequence[np.ndarray],
    date_draws: Optional[np.ndarray],
    repeat_draws: Optional[np.ndarray],
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, np.ndarray]]:
    if date_draws is None or repeat_draws is None or len(baseline_daily) < 5:
        empty_ci = {metric: (math.nan, math.nan, math.nan) for metric in METRICS}
        return empty_ci, {metric: np.asarray([], dtype=np.float64) for metric in METRICS}
    draws: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    for draw_i in range(len(date_draws)):
        date_idx = date_draws[draw_i]
        repeat_idx = int(repeat_draws[draw_i] % len(permuted_daily))
        base = metrics_from_stats(baseline_daily[date_idx])
        perm = metrics_from_stats(permuted_daily[repeat_idx][date_idx])
        for metric in METRICS:
            draws[metric].append(importance_delta(metric, base[metric], perm[metric]))
    result: Dict[str, Tuple[float, float, float]] = {}
    arrays: Dict[str, np.ndarray] = {}
    for metric, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        arrays[metric] = arr
        result[metric] = (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
            float(np.nanmean(arr > 0.0)),
        )
    return result, arrays


def evaluate_importance(
    source: str,
    label: str,
    source_eval,
    rows: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame,
    groups: Sequence[Dict[str, object]],
    shared_features: set,
    train_mod,
    ev,
    eval_args: argparse.Namespace,
    cli: argparse.Namespace,
    date_draws: Optional[np.ndarray],
    repeat_draws: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[Tuple[str, str, str, str], np.ndarray]]:
    dates = pd.to_datetime(meta.get("time", pd.Series(np.arange(len(meta)))), errors="coerce").dt.strftime("%Y-%m-%d")
    dates = dates.fillna(pd.Series(np.arange(len(meta))).astype(str)).to_numpy()
    baseline_probs = ev.predict_static_rows_for_swap(rows, source_eval, train_mod, eval_args, ev.resolve_device(cli.device))
    baseline_daily, unique_dates = daily_binary_statistics(y_true, baseline_probs, dates)
    baseline_metrics = metrics_from_stats(baseline_daily)
    records: List[Dict[str, object]] = []
    draw_cache: Dict[Tuple[str, str, str, str], np.ndarray] = {}
    modes = [mode.strip().lower() for mode in cli.modes.split(",") if mode.strip()]
    unknown_modes = sorted(set(modes) - {"marginal", "stratified"})
    if unknown_modes:
        raise ValueError(f"Unknown permutation mode(s): {unknown_modes}")

    for group_i, group in enumerate(groups, start=1):
        columns = np.asarray(group["columns"], dtype=np.int64)
        members = group_members(group)
        for mode_i, mode in enumerate(modes):
            permuted_metrics: List[Dict[str, float]] = []
            permuted_daily: List[np.ndarray] = []
            moved: List[float] = []
            fallback: List[float] = []
            print(
                f"[importance] {source} {group_i}/{len(groups)} {mode} {group['block']}::{group['feature']}",
                flush=True,
            )
            for repeat in range(int(cli.repeats)):
                stable_group = zlib.crc32(f"{group['block']}::{group['feature']}".encode("utf-8"))
                seed = int(cli.seed + stable_group + mode_i * 1009 + repeat * 97)
                rng = np.random.default_rng(seed)
                if mode == "marginal":
                    donor, moved_fraction, fallback_fraction = marginal_donor(len(rows), rng)
                else:
                    donor, moved_fraction, fallback_fraction = stratified_donor(
                        meta,
                        rows,
                        source_eval.dynamic_feature_order,
                        eval_args.window,
                        members,
                        rng,
                    )
                permuted = rows.copy()
                permuted[:, columns] = rows[donor][:, columns]
                probs = ev.predict_static_rows_for_swap(permuted, source_eval, train_mod, eval_args, ev.resolve_device(cli.device))
                daily, dates_check = daily_binary_statistics(y_true, probs, dates)
                if not np.array_equal(unique_dates, dates_check):
                    raise RuntimeError("Date aggregation changed during permutation.")
                permuted_daily.append(daily)
                permuted_metrics.append(metrics_from_stats(daily))
                moved.append(moved_fraction)
                fallback.append(fallback_fraction)
            ci, importance_draws = date_block_importance_ci(
                baseline_daily,
                permuted_daily,
                date_draws,
                repeat_draws,
            )
            row: Dict[str, object] = {
                "source": source,
                "source_label": label,
                "permutation_mode": mode,
                "feature": group["feature"],
                "block": group["block"],
                "analysis_level": group.get("analysis_level", "individual_group"),
                "members": ",".join(members),
                "n_columns": int(len(columns)),
                "n_samples": int(len(rows)),
                "n_dates": int(len(unique_dates)),
                "repeats": int(cli.repeats),
                "mean_moved_fraction": float(np.mean(moved)),
                "mean_fallback_fraction": float(np.mean(fallback)),
                "shared_across_sources": bool(
                    group.get("analysis_level") == "shared_package"
                    or (group.get("analysis_level", "individual_group") == "individual_group" and str(group["feature"]) in shared_features)
                ),
            }
            for metric in METRICS:
                perm_values = np.asarray([values[metric] for values in permuted_metrics], dtype=np.float64)
                point = importance_delta(metric, baseline_metrics[metric], float(np.nanmean(perm_values)))
                lo, hi, p_positive = ci[metric]
                row[f"baseline_{metric}"] = baseline_metrics[metric]
                row[f"permuted_{metric}_mean"] = float(np.nanmean(perm_values))
                row[f"importance_{metric}"] = point
                row[f"importance_{metric}_ci_low"] = lo
                row[f"importance_{metric}_ci_high"] = hi
                row[f"importance_{metric}_p_positive"] = p_positive
                row[f"importance_{metric}_ci_excludes_zero"] = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
                draw_cache[(mode, str(group["block"]), str(group["feature"]), metric)] = importance_draws[metric]
            records.append(row)
    return pd.DataFrame(records), baseline_metrics, draw_cache


def paired_source_difference_table(
    results: pd.DataFrame,
    draw_cache: Dict[str, Dict[Tuple[str, str, str, str], np.ndarray]],
    source_order: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for left_i, left in enumerate(source_order):
        for right in source_order[left_i + 1 :]:
            left_rows = results[results["source"] == left]
            right_rows = results[results["source"] == right]
            keys = ["permutation_mode", "block", "feature"]
            merged = left_rows.merge(right_rows, on=keys, suffixes=("_left", "_right"))
            for _, item in merged.iterrows():
                pair_comparable = (
                    item["analysis_level_left"] == "individual_group"
                    and item["analysis_level_right"] == "individual_group"
                ) or (
                    item["analysis_level_left"] == "shared_package"
                    and item["analysis_level_right"] == "shared_package"
                    and item["members_left"] == item["members_right"]
                )
                if not pair_comparable:
                    continue
                key_base = (str(item["permutation_mode"]), str(item["block"]), str(item["feature"]))
                for metric in METRICS:
                    left_draws = draw_cache[left].get((*key_base, metric), np.asarray([]))
                    right_draws = draw_cache[right].get((*key_base, metric), np.asarray([]))
                    delta_draws = left_draws - right_draws if len(left_draws) and len(left_draws) == len(right_draws) else np.asarray([])
                    lo = float(np.nanpercentile(delta_draws, 2.5)) if len(delta_draws) else math.nan
                    hi = float(np.nanpercentile(delta_draws, 97.5)) if len(delta_draws) else math.nan
                    rows.append(
                        {
                            "left_source": left,
                            "left_label": item["source_label_left"],
                            "right_source": right,
                            "right_label": item["source_label_right"],
                            "permutation_mode": key_base[0],
                            "block": key_base[1],
                            "feature": key_base[2],
                            "pairwise_scope": (
                                "global_shared"
                                if bool(item["shared_across_sources_left"]) and bool(item["shared_across_sources_right"])
                                else "pair_shared"
                            ),
                            "members": item["members_left"],
                            "metric": metric,
                            "left_minus_right_importance": float(item[f"importance_{metric}_left"] - item[f"importance_{metric}_right"]),
                            "ci_low": lo,
                            "ci_high": hi,
                            "p_left_greater": float(np.nanmean(delta_draws > 0.0)) if len(delta_draws) else math.nan,
                            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                        }
                    )
    return pd.DataFrame(rows)


def plot_shared_heatmap(results: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sub = results[
        (results["permutation_mode"] == "marginal")
        & (results["block"] == "dynamic_12h")
        & results["shared_across_sources"].astype(bool)
    ].copy()
    if sub.empty:
        return
    table = sub.pivot(index="feature", columns="source_label", values="importance_low_vis_csi")
    order = table.abs().mean(axis=1).sort_values(ascending=False).index
    table = table.loc[order]
    vmax = float(np.nanmax(np.abs(table.to_numpy())))
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(max(5.8, 1.15 * table.shape[1]), max(4.0, 0.32 * table.shape[0] + 1.2)))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_title("Shared-input model reliance for low-visibility CSI")
    ax.set_xlabel("Source-trained model")
    ax.set_ylabel("Complete 12 h input sequence")
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("CSI decrease after permutation")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig_multi_source_shared_feature_importance.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_shared_moisture(results: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = results[
        (results["feature"] == "shared_low_level_moisture")
        & (results["permutation_mode"].isin(["marginal", "stratified"]))
    ].copy()
    if sub.empty:
        return
    labels = sub["source_label"].drop_duplicates().tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.7, max(3.2, 0.55 * len(labels) + 1.4)))
    offsets = {"marginal": -0.10, "stratified": 0.10}
    markers = {"marginal": "o", "stratified": "s"}
    available_modes = [mode for mode in ("marginal", "stratified") if mode in set(sub["permutation_mode"])]
    for mode in available_modes:
        cur = sub[sub["permutation_mode"] == mode].set_index("source_label").reindex(labels)
        values = cur["importance_low_vis_csi"].to_numpy(dtype=float)
        lo = cur["importance_low_vis_csi_ci_low"].to_numpy(dtype=float)
        hi = cur["importance_low_vis_csi_ci_high"].to_numpy(dtype=float)
        for idx, label in enumerate(labels):
            source = str(cur.loc[label, "source"])
            ax.errorbar(
                values[idx],
                y[idx] + offsets[mode],
                xerr=np.array([[values[idx] - lo[idx]], [hi[idx] - values[idx]]]),
                fmt=markers[mode],
                color=source_color(source),
                markerfacecolor=source_color(source) if mode == "marginal" else "white",
                capsize=2.2,
                ms=5.0,
                lw=1.0,
            )
    ax.axvline(0.0, color="#202124", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Low-visibility CSI decrease after moisture-package permutation")
    ax.set_title("Reliance on the shared low-level moisture package")
    if "marginal" in available_modes:
        ax.plot([], [], "o", color="#555B66", label="Marginal model reliance")
    if "stratified" in available_modes:
        ax.plot([], [], "s", color="#555B66", markerfacecolor="white", label="Meteorology-stratified sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig_multi_source_shared_moisture_reliance.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_method_note(out_dir: Path, cli: argparse.Namespace, labels: Dict[str, str], shared_dynamic: Sequence[str]) -> None:
    lines = [
        "# Multi-source feature-importance method",
        "",
        "This analysis estimates model reliance, not causal importance. Each dynamic variable is permuted as its complete 12 h sequence on the same paired valid-time/station sample for every source-trained model.",
        "",
        "The primary result is marginal grouped permutation importance. The secondary result permutes within season, six-hour UTC bin, coarse geographic cell, and an available non-target moisture-state quartile. This stratified analysis is a dependence-aware sensitivity check, not an exact implementation of the random-forest conditional permutation algorithm.",
        "",
        "Low-visibility CSI is the primary endpoint. Recall and precision identify whether a decrease is caused by missed events or false alarms, and low-visibility Brier score checks the probabilistic response. Positive importance always means degraded performance after permutation.",
        "",
        f"The common-row sample is uniform and preserves event prevalence (sample_size={cli.sample_size}); it is not class-balanced. Uncertainty uses paired valid-date block bootstrap with {cli.bootstrap_iters} iterations.",
        "",
        f"Shared dynamic inputs: {', '.join(shared_dynamic)}.",
        "",
        "Cross-source claims must use rows marked shared_across_sources=true. Source-native packages answer within-model questions only because their membership differs when a source lacks variables such as RH2M.",
        "Direct source-to-source claims should use multi_source_pairwise_feature_importance_differences.csv. Its confidence intervals reuse the same valid-date bootstrap draws and the same marginal donor maps on both models. pair_shared rows permit controlled comparisons such as Tianji versus T2ND RH2M; global_shared rows are comparable across every source.",
        "",
        "References: Fisher, Rudin and Dominici (2019), JMLR 20:177; Strobl et al. (2008), BMC Bioinformatics 9:307; Hamill (1999), Weather and Forecasting 14:155-167. Directional response curves should use ALE following Apley and Zhu (2020), JRSS-B 82:1059-1086.",
        "",
        "Sources: " + "; ".join(f"{tag}={label}" for tag, label in labels.items()),
    ]
    (out_dir / "feature_importance_method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    if cli.group_scope == "all" and not HAS_FEATURE_CATALOG:
        checked = ", ".join(str(path) for path in FEATURE_CATALOG_CANDIDATES)
        raise FileNotFoundError(
            "--group_scope all requires feature_catalog_pm10_pm25.py for static and "
            f"engineered groups. Checked: {checked}. Use --group_scope dynamic or deploy the catalog."
        )
    if HAS_FEATURE_CATALOG:
        print(f"[catalog] using {FEATURE_CATALOG_PATH}", flush=True)
    else:
        print(
            "[catalog] feature_catalog_pm10_pm25.py not found; using the dynamic-only "
            "dataset_build_config fallback.",
            flush=True,
        )
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = load_source_evaluator()
    eval_args = evaluator_args(cli)
    eval_args = ev.ensure_static_rnn_dataset_args(eval_args)
    specs, labels = parse_sources(cli.sources, ev, cli)
    ev.fill_auto_scaler_paths(specs, eval_args)
    for tag, spec in specs.items():
        if not Path(spec.data_dir).is_dir():
            raise FileNotFoundError(f"{tag} data directory does not exist: {spec.data_dir}")
        ev.require_file(spec.ckpt_path, f"{tag} checkpoint")
        ev.require_file(spec.scaler_path, f"{tag} scaler")

    configs = {tag: ev.read_build_config(spec.data_dir) for tag, spec in specs.items()}
    ev.validate_build_time_alignment(configs, cli.allow_legacy_time_alignment)
    train_mod = ev.import_training_module(eval_args)
    device = ev.resolve_device(cli.device)
    source_order = [cli.reference_source] + [tag for tag in specs if tag != cli.reference_source]
    source_evals = {
        tag: load_source_for_importance(tag, specs[tag], train_mod, eval_args, device, ev)
        for tag in source_order
    }
    aligned, meta_common = ev.align_sources_to_reference(source_evals, source_order, strict_meta=False)
    y_common = source_evals[cli.reference_source].test_targets[aligned[cli.reference_source]]
    sample_pos = uniform_common_sample(y_common, cli.sample_size, cli.seed, cli.min_low_vis)
    y_sample = y_common[sample_pos]
    meta_sample = meta_common.iloc[sample_pos].reset_index(drop=True)
    sample_dates = pd.to_datetime(meta_sample.get("time", pd.Series(np.arange(len(meta_sample)))), errors="coerce")
    sample_dates = sample_dates.dt.strftime("%Y-%m-%d").fillna(pd.Series(np.arange(len(meta_sample))).astype(str))
    n_dates = int(sample_dates.nunique())
    if cli.bootstrap_iters > 0:
        bootstrap_rng = np.random.default_rng(cli.seed + 811)
        date_draws = bootstrap_rng.integers(0, n_dates, size=(int(cli.bootstrap_iters), n_dates))
        repeat_draws = bootstrap_rng.integers(0, max(int(cli.repeats), 1), size=int(cli.bootstrap_iters))
    else:
        date_draws = None
        repeat_draws = None

    orders = {
        tag: canonical_order(source_evals[tag].dynamic_feature_order, source_evals[tag].dyn_vars_count)
        for tag in source_order
    }
    shared_dynamic = [name for name in orders[source_order[0]] if all(name in orders[tag] for tag in source_order[1:])]
    shared_feature_names = set(shared_dynamic)
    all_results: List[pd.DataFrame] = []
    all_draws: Dict[str, Dict[Tuple[str, str, str, str], np.ndarray]] = {}
    baseline_records: List[Dict[str, object]] = []
    for tag in source_order:
        source_eval = source_evals[tag]
        x_test = np.load(Path(source_eval.spec.data_dir) / "X_test.npy", mmap_mode="r")
        row_idx = aligned[tag][sample_pos]
        rows = np.asarray(x_test[row_idx], dtype=np.float32)
        groups = permutation_groups(
            eval_args.window,
            source_eval.dyn_vars_count,
            source_eval.extra_feat_dim,
            orders[tag],
        )
        if cli.group_scope == "dynamic":
            groups = [group for group in groups if group.get("block") == "dynamic_12h"]
        for group in groups:
            group.setdefault("analysis_level", "individual_group")
        if not cli.no_packages:
            groups = add_physical_packages(groups, orders[tag], eval_args.window, shared_dynamic)
        if cli.max_groups > 0:
            groups = groups[: cli.max_groups]
        result, baseline, source_draws = evaluate_importance(
            tag,
            labels[tag],
            source_eval,
            rows,
            y_sample,
            meta_sample,
            groups,
            shared_feature_names,
            train_mod,
            ev,
            eval_args,
            cli,
            date_draws,
            repeat_draws,
        )
        all_results.append(result)
        all_draws[tag] = source_draws
        baseline_records.append({"source": tag, "source_label": labels[tag], "n_samples": len(rows), **baseline})

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(out_dir / "multi_source_grouped_permutation_importance.csv", index=False, float_format="%.8f")
    pd.DataFrame(baseline_records).to_csv(out_dir / "multi_source_importance_baseline_metrics.csv", index=False)
    results[results["shared_across_sources"].astype(bool)].to_csv(
        out_dir / "multi_source_shared_feature_importance.csv", index=False, float_format="%.8f"
    )
    paired_differences = paired_source_difference_table(results, all_draws, source_order)
    paired_differences.to_csv(
        out_dir / "multi_source_pairwise_feature_importance_differences.csv",
        index=False,
        float_format="%.8f",
    )
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sources": [
                    {
                        "tag": tag,
                        "label": labels[tag],
                        "data_dir": specs[tag].data_dir,
                        "checkpoint": specs[tag].ckpt_path,
                        "scaler": specs[tag].scaler_path,
                        "dynamic_feature_order": orders[tag],
                    }
                    for tag in source_order
                ],
                "reference_source": cli.reference_source,
                "common_rows_before_sampling": int(len(y_common)),
                "sample_rows": int(len(sample_pos)),
                "sample_low_vis_rows": int(np.sum(y_sample <= 1)),
                "sampling": "uniform without replacement on common valid-time/station rows",
                "shared_dynamic_features": shared_dynamic,
                "permutation_modes": cli.modes,
                "group_scope": cli.group_scope,
                "feature_catalog_path": FEATURE_CATALOG_PATH,
                "feature_catalog_fallback": bool(not HAS_FEATURE_CATALOG),
                "repeats": cli.repeats,
                "bootstrap_iters": cli.bootstrap_iters,
                "bootstrap_unit": "valid date",
                "interpretation": "model reliance/sensitivity, not causal importance",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    write_method_note(out_dir, cli, labels, shared_dynamic)
    if not cli.no_plot:
        plot_shared_heatmap(results, out_dir)
        plot_shared_moisture(results, out_dir)
    print(f"[OK] wrote multi-source feature importance to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
