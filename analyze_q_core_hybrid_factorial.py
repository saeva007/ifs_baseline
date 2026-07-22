#!/usr/bin/env python3
"""Analyze a three-seed Pangu/Tianji q-core hybrid factorial experiment.

Primary endpoints are threshold-free low-visibility average precision and
low-visibility CSI/recall at a validation-matched false-positive rate.  The
script computes exact package-level Shapley contributions, second-order
interactions, paired UTC-date block-bootstrap intervals, calibration tables,
and Pangu-hit/Tianji-hit event case-control exports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


GROUP_PROFILES: Dict[str, Dict[str, object]] = {
    "mtw": {
        "dataset_prefix": "mtw",
        "source_prefix": "qcore_hybrid_",
        "description": "Original three-package factorial: moisture, thermal/pressure, wind.",
        "order": ("M", "T", "W"),
        "labels": {"M": "Moisture", "T": "Thermal/pressure", "W": "Wind"},
    },
    "mt2pw": {
        "dataset_prefix": "mt2pw",
        "source_prefix": "qcore_hybrid_mt2pw_",
        "description": "T-package follow-up factorial: split T into T2M and MSLP.",
        "order": ("M", "T2", "P", "W"),
        "labels": {"M": "Moisture", "T2": "T2M", "P": "MSLP", "W": "Wind"},
    },
    "m925b": {
        "dataset_prefix": "m925b",
        "source_prefix": "qcore_joint_",
        "description": (
            "Joint-structure factorial: low-level moisture/thermodynamic state, explicit T925, "
            "and remaining q-core background."
        ),
        "order": ("M", "H", "B"),
        "labels": {
            "M": "Moisture/thermodynamic state",
            "H": "T925",
            "B": "Remaining q-core background",
        },
    },
    "mhtpw": {
        "dataset_prefix": "mhtpw",
        "source_prefix": "qcore_hybrid_mhtpw_",
        "description": (
            "Final five-package q-core+T925 factorial with a coupled 925-hPa state and a "
            "prespecified low-level wind/ventilation block."
        ),
        "order": ("M", "H", "T", "P", "W"),
        "labels": {
            "M": "Near-surface moisture",
            "H": "925-hPa thermo-moisture",
            "T": "T2M",
            "P": "MSLP",
            "W": "Low-level wind/ventilation",
        },
    },
}
GROUP_PROFILE = "mtw"
GROUP_ORDER = tuple(GROUP_PROFILES[GROUP_PROFILE]["order"])  # type: ignore[arg-type]
GROUP_LABELS = dict(GROUP_PROFILES[GROUP_PROFILE]["labels"])  # type: ignore[arg-type]
DATASET_PREFIX = str(GROUP_PROFILES[GROUP_PROFILE]["dataset_prefix"])
SOURCE_PREFIX = str(GROUP_PROFILES[GROUP_PROFILE]["source_prefix"])
PRIMARY_METRICS = ("low_vis_ap", "low_vis_csi_matched_fpr", "low_vis_recall_matched_fpr")
EVENT_FEATURES = (
    "T2M",
    "T_925",
    "MSLP",
    "WSPD10",
    "RH_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
)
OBS_VALID_RANGES = {
    "rhu": (0.0, 100.0),
    "tem": (-80.0, 60.0),
    "win_s_avg_10mi": (0.0, 80.0),
    "pre_1h": (0.0, 500.0),
    "prs_sea": (800.0, 1100.0),
}


def set_group_profile(profile: str, dataset_prefix: str | None = None, source_prefix: str | None = None) -> None:
    global GROUP_PROFILE, GROUP_ORDER, GROUP_LABELS, DATASET_PREFIX, SOURCE_PREFIX
    key = str(profile).strip().lower()
    if key not in GROUP_PROFILES:
        raise ValueError(f"Unknown group profile {profile!r}; choose from {sorted(GROUP_PROFILES)}")
    spec = GROUP_PROFILES[key]
    order = tuple(str(name) for name in tuple(spec["order"]))
    labels = {str(name): str(label) for name, label in dict(spec["labels"]).items()}
    missing = [name for name in order if name not in labels]
    if missing:
        raise ValueError(f"Group profile {profile!r} has missing labels: {missing}")
    prefix = str(dataset_prefix).strip() if dataset_prefix else str(spec["dataset_prefix"])
    src_prefix = str(source_prefix).strip() if source_prefix else str(spec["source_prefix"])
    if not prefix or any(ch.isspace() for ch in prefix) or "/" in prefix or "\\" in prefix:
        raise ValueError(f"Invalid dataset prefix: {prefix!r}")
    if not src_prefix or any(ch.isspace() for ch in src_prefix):
        raise ValueError(f"Invalid source prefix: {src_prefix!r}")
    GROUP_PROFILE = key
    GROUP_ORDER = order
    GROUP_LABELS = labels
    DATASET_PREFIX = prefix
    SOURCE_PREFIX = src_prefix


def all_masks() -> Tuple[str, ...]:
    width = len(GROUP_ORDER)
    return tuple(f"{value:0{width}b}" for value in range(1 << width))


def zero_mask() -> str:
    return "0" * len(GROUP_ORDER)


def one_mask() -> str:
    return "1" * len(GROUP_ORDER)


@dataclass
class SampleSet:
    frame: pd.DataFrame
    y: np.ndarray
    score: np.ndarray
    pred: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True, help="Contains seed_<seed>/ evaluator outputs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,2025,20260702")
    ap.add_argument("--masks", default="all")
    ap.add_argument(
        "--group-profile",
        default="mtw",
        choices=sorted(GROUP_PROFILES),
        help="Physical package profile used by the evaluator source tags and Shapley masks.",
    )
    ap.add_argument("--dataset-prefix", default="", help="Optional dataset prefix override recorded in the report.")
    ap.add_argument("--source-prefix", default="", help="Optional evaluator source-tag prefix override.")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260702)
    ap.add_argument("--bootstrap-max-rows", type=int, default=0, help="Optional deterministic paired-row cap; 0 uses all rows.")
    ap.add_argument("--ap-hist-initial-bins", type=int, default=4096)
    ap.add_argument("--ap-hist-max-bins", type=int, default=65536)
    ap.add_argument("--ap-hist-max-error", type=float, default=5.0e-4)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--event-features", default=",".join(EVENT_FEATURES))
    ap.add_argument("--obs-root", default="")
    ap.add_argument("--era5-data-dir", default="", help="Optional paired ERA5 q-core reference-analysis dataset.")
    ap.add_argument("--paper-eval-dir", default="/public/home/putianshu/vis_mlp/paper_eval")
    ap.add_argument("--require-ale", action="store_true", help="Fail unless every requested seed has a completed endpoint ALE table.")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def parse_csv(value: str) -> List[str]:
    normalized = str(value).replace(";", ",").replace(":", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def parse_masks(value: str) -> List[str]:
    expected = set(all_masks())
    masks = list(all_masks()) if str(value).strip().lower() == "all" else parse_csv(value)
    bad = [mask for mask in masks if mask not in expected]
    if bad:
        raise ValueError(f"Invalid {GROUP_PROFILE} masks: {bad}; expected {len(GROUP_ORDER)}-bit factorial masks")
    missing = sorted(expected - set(masks))
    if missing:
        raise ValueError(f"Exact {len(GROUP_ORDER)}-package Shapley requires all {len(expected)} masks; missing {missing}")
    return list(all_masks())


def load_ale_outputs(eval_root: Path, seeds: Sequence[int], required: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for seed in seeds:
        path = eval_root / "ale" / f"seed_{seed}" / "q_core_trajectory_ale.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    if required and missing:
        raise FileNotFoundError("Required ALE outputs are missing: " + "; ".join(missing))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    all_rows = pd.concat(frames, ignore_index=True)
    keys = [column for column in ("source", "source_label", "feature", "bin") if column in all_rows]
    mean_rows = all_rows.groupby(keys, as_index=False).mean(numeric_only=True)
    return all_rows, mean_rows


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def load_samples(path: Path) -> SampleSet:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"time", "station_id", "y_cls", "pred", "p_fog", "p_mist", "p_clear"}
    if not required.issubset(df.columns):
        raise KeyError(f"{path}: missing {sorted(required - set(df.columns))}")
    time = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError(f"{path}: invalid timestamps")
    df = df.copy()
    df["time_utc"] = time
    df["time_key"] = time.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["station_key"] = normalize_station(df["station_id"])
    if df[["time_key", "station_key"]].duplicated().any():
        raise ValueError(f"{path}: duplicate (time, station_id)")
    y = df["y_cls"].to_numpy(dtype=np.int64)
    pred = df["pred"].to_numpy(dtype=np.int64)
    score = df["p_fog"].to_numpy(dtype=np.float64) + df["p_mist"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise ValueError(f"{path}: non-finite probabilities")
    return SampleSet(df, y, score, pred)


def assert_aligned(reference: SampleSet, other: SampleSet, label: str) -> None:
    ref_keys = reference.frame[["time_key", "station_key"]]
    other_keys = other.frame[["time_key", "station_key"]]
    if not ref_keys.equals(other_keys):
        raise ValueError(f"{label}: sample keys/order differ")
    if not np.array_equal(reference.y, other.y):
        raise ValueError(f"{label}: labels differ")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_ranking_curve(y: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError("binary targets and scores must have the same shape")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    if not len(y_sorted):
        return np.asarray([0.0]), np.asarray([0.0]), np.asarray([math.inf])
    distinct = np.r_[np.flatnonzero(np.diff(score_sorted) != 0), len(score_sorted) - 1]
    tp = np.cumsum(y_sorted, dtype=np.float64)[distinct]
    fp = (distinct + 1).astype(np.float64) - tp
    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    tpr = tp / positives if positives else np.zeros_like(tp)
    fpr = fp / negatives if negatives else np.zeros_like(fp)
    thresholds = score_sorted[distinct]
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[math.inf, thresholds]


def average_precision_binary(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    _, recall, thresholds = binary_ranking_curve(y, score)
    fpr, _, _ = binary_ranking_curve(y, score)
    negatives = float(len(y) - positives)
    tp = recall * positives
    fp = fpr * negatives
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return float(np.sum(np.diff(recall) * precision[1:]))


def roc_auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return math.nan
    fpr, tpr, _ = binary_ranking_curve(y, score)
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def ece_binary(y: np.ndarray, score: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(score, edges[1:-1], right=True), 0, bins - 1)
    total = max(len(y), 1)
    value = 0.0
    for idx in range(bins):
        mask = bucket == idx
        if mask.any():
            value += float(mask.sum()) / total * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return float(value)


def reliability_rows(y: np.ndarray, score: np.ndarray, bins: int) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(score, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for idx in range(bins):
        mask = bucket == idx
        rows.append(
            {
                "bin": idx,
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "n": int(mask.sum()),
                "mean_probability": float(score[mask].mean()) if mask.any() else math.nan,
                "observed_frequency": float(y[mask].mean()) if mask.any() else math.nan,
            }
        )
    return rows


def threshold_metrics(y_cls: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = y_cls <= 1
    pred = score >= float(threshold)
    tp = float(np.sum(y & pred))
    fp = float(np.sum(~y & pred))
    fn = float(np.sum(y & ~pred))
    tn = float(np.sum(~y & ~pred))
    return {
        "low_vis_threshold": float(threshold),
        "low_vis_precision_matched_fpr": safe_div(tp, tp + fp),
        "low_vis_recall_matched_fpr": safe_div(tp, tp + fn),
        "low_vis_csi_matched_fpr": safe_div(tp, tp + fp + fn),
        "low_vis_fpr_matched_fpr": safe_div(fp, fp + tn),
        "low_vis_tp": tp,
        "low_vis_fp": fp,
        "low_vis_fn": fn,
        "low_vis_tn": tn,
    }


def argmax_metrics(y_cls: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, name in enumerate(("ultra_low", "moderate_low", "clear")):
        tp = float(np.sum((y_cls == cid) & (pred == cid)))
        fp = float(np.sum((y_cls != cid) & (pred == cid)))
        fn = float(np.sum((y_cls == cid) & (pred != cid)))
        out[f"{name}_precision_argmax"] = safe_div(tp, tp + fp)
        out[f"{name}_recall_argmax"] = safe_div(tp, tp + fn)
        out[f"{name}_csi_argmax"] = safe_div(tp, tp + fp + fn)
    low_y = y_cls <= 1
    low_p = pred <= 1
    tp = float(np.sum(low_y & low_p))
    fp = float(np.sum(~low_y & low_p))
    fn = float(np.sum(low_y & ~low_p))
    tn = float(np.sum(~low_y & ~low_p))
    out.update(
        {
            "low_vis_precision_argmax": safe_div(tp, tp + fp),
            "low_vis_recall_argmax": safe_div(tp, tp + fn),
            "low_vis_csi_argmax": safe_div(tp, tp + fp + fn),
            "low_vis_fpr_argmax": safe_div(fp, fp + tn),
        }
    )
    return out


def probability_metrics(y_cls: np.ndarray, score: np.ndarray, bins: int) -> Dict[str, float]:
    y = (y_cls <= 1).astype(np.int64)
    return {
        "low_vis_ap": average_precision_binary(y, score),
        "low_vis_roc_auc": roc_auc_binary(y, score),
        "low_vis_brier": float(np.mean((score - y) ** 2)),
        "low_vis_ece": ece_binary(y, score, bins),
        "low_vis_base_rate": float(y.mean()),
    }


def match_fpr_threshold(y_cls: np.ndarray, score: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    y = (y_cls <= 1).astype(np.int64)
    fpr, tpr, thresholds = binary_ranking_curve(y, score)
    finite = np.isfinite(thresholds)
    candidates = np.flatnonzero(finite)
    if not len(candidates):
        raise RuntimeError("No finite threshold candidates")
    distance = np.abs(fpr[candidates] - float(target_fpr))
    best_distance = float(distance.min())
    tied = candidates[np.isclose(distance, best_distance, rtol=0.0, atol=1.0e-12)]
    best = int(tied[np.argmax(tpr[tied])])
    return float(thresholds[best]), float(fpr[best])


def mask_to_set(mask: str) -> frozenset[str]:
    return frozenset(group for group, bit in zip(GROUP_ORDER, mask) if bit == "1")


def set_to_mask(groups: Iterable[str]) -> str:
    selected = set(groups)
    return "".join("1" if group in selected else "0" for group in GROUP_ORDER)


def shapley_values(values: Mapping[str, float]) -> Dict[str, float]:
    n = len(GROUP_ORDER)
    denom = math.factorial(n)
    out: Dict[str, float] = {}
    for group in GROUP_ORDER:
        others = [name for name in GROUP_ORDER if name != group]
        phi = 0.0
        for bits in range(1 << len(others)):
            subset = {others[i] for i in range(len(others)) if bits & (1 << i)}
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / denom
            phi += weight * (values[set_to_mask(subset | {group})] - values[set_to_mask(subset)])
        out[group] = float(phi)
    return out


def pair_interactions(values: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(GROUP_ORDER)
    for i, left in enumerate(GROUP_ORDER):
        for right in GROUP_ORDER[i + 1 :]:
            others = [name for name in GROUP_ORDER if name not in {left, right}]
            interaction = 0.0
            for bits in range(1 << len(others)):
                subset = {others[j] for j in range(len(others)) if bits & (1 << j)}
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(n - len(subset) - 2)
                    / math.factorial(n - 1)
                )
                delta = (
                    values[set_to_mask(subset | {left, right})]
                    - values[set_to_mask(subset | {left})]
                    - values[set_to_mask(subset | {right})]
                    + values[set_to_mask(subset)]
                )
                interaction += weight * delta
            out[f"{left}:{right}"] = float(interaction)
    return out


def load_all(
    eval_root: Path,
    seeds: Sequence[int],
    masks: Sequence[str],
    prefix: str,
) -> Tuple[Dict[Tuple[int, str], SampleSet], Dict[Tuple[int, str], SampleSet]]:
    val: Dict[Tuple[int, str], SampleSet] = {}
    test: Dict[Tuple[int, str], SampleSet] = {}
    ref_val: SampleSet | None = None
    ref_test: SampleSet | None = None
    for seed in seeds:
        seed_dir = eval_root / f"seed_{seed}"
        for mask in masks:
            tag = f"{prefix}{mask}"
            val_set = load_samples(seed_dir / f"per_sample_val_{tag}.csv")
            test_set = load_samples(seed_dir / f"per_sample_{tag}.csv")
            if ref_val is None:
                ref_val, ref_test = val_set, test_set
            else:
                assert_aligned(ref_val, val_set, f"validation seed={seed} mask={mask}")
                assert_aligned(ref_test, test_set, f"test seed={seed} mask={mask}")
            val[(seed, mask)] = val_set
            test[(seed, mask)] = test_set
    return val, test


def load_common_core(
    eval_root: Path,
    seeds: Sequence[int],
    reference_val: Mapping[Tuple[int, str], SampleSet],
    reference_test: Mapping[Tuple[int, str], SampleSet],
) -> Tuple[Dict[int, SampleSet], Dict[int, SampleSet]]:
    val: Dict[int, SampleSet] = {}
    test: Dict[int, SampleSet] = {}
    for seed in seeds:
        seed_dir = eval_root / f"seed_{seed}"
        val_path = seed_dir / "per_sample_val_tianji_common_core.csv"
        test_path = seed_dir / "per_sample_tianji_common_core.csv"
        if not val_path.is_file() or not test_path.is_file():
            return {}, {}
        val[seed] = load_samples(val_path)
        test[seed] = load_samples(test_path)
        assert_aligned(reference_val[(seed, one_mask())], val[seed], f"common-core validation seed={seed}")
        assert_aligned(reference_test[(seed, one_mask())], test[seed], f"common-core test seed={seed}")
    return val, test


def point_analysis(
    val: Mapping[Tuple[int, str], SampleSet],
    test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    masks: Sequence[str],
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[Tuple[int, str], float]]:
    baseline_fprs = []
    for seed in seeds:
        sample = val[(seed, zero_mask())]
        baseline_fprs.append(argmax_metrics(sample.y, sample.pred)["low_vis_fpr_argmax"])
    target_fpr = float(np.median(baseline_fprs))
    thresholds: Dict[Tuple[int, str], float] = {}
    rows: List[Dict[str, object]] = []
    reliability: List[Dict[str, object]] = []
    for seed in seeds:
        for mask in masks:
            val_set = val[(seed, mask)]
            test_set = test[(seed, mask)]
            threshold, achieved_val_fpr = match_fpr_threshold(val_set.y, val_set.score, target_fpr)
            thresholds[(seed, mask)] = threshold
            metrics: Dict[str, float] = {}
            metrics.update(probability_metrics(test_set.y, test_set.score, bins))
            metrics.update(threshold_metrics(test_set.y, test_set.score, threshold))
            metrics.update(argmax_metrics(test_set.y, test_set.pred))
            rows.append(
                {
                    "seed": seed,
                    "mask": mask,
                    "groups_from_tianji": ",".join(mask_to_set(mask)),
                    "target_validation_fpr": target_fpr,
                    "achieved_validation_fpr": achieved_val_fpr,
                    **metrics,
                }
            )
            y_binary = (test_set.y <= 1).astype(np.int64)
            for rel in reliability_rows(y_binary, test_set.score, bins):
                reliability.append({"seed": seed, "mask": mask, **rel})
    return pd.DataFrame(rows), pd.DataFrame(reliability), target_fpr, thresholds


def aggregate_effects(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shapley_rows: List[Dict[str, object]] = []
    interaction_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        per_seed: Dict[int, Dict[str, float]] = {}
        for seed, part in metrics.groupby("seed"):
            values = dict(zip(part["mask"].astype(str), part[metric].astype(float)))
            per_seed[int(seed)] = shapley_values(values)
            interactions = pair_interactions(values)
            for pair, value in interactions.items():
                interaction_rows.append({"seed": int(seed), "metric": metric, "pair": pair, "interaction": value})
            total = values[one_mask()] - values[zero_mask()]
            phi_sum = sum(per_seed[int(seed)].values())
            efficiency_rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "endpoint_delta_111_minus_000": total,
                    "endpoint_delta_all1_minus_all0": total,
                    "all0_mask": zero_mask(),
                    "all1_mask": one_mask(),
                    "shapley_sum": phi_sum,
                    "absolute_error": abs(total - phi_sum),
                }
            )
        for group in GROUP_ORDER:
            values = np.asarray([per_seed[seed][group] for seed in sorted(per_seed)], dtype=np.float64)
            shapley_rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "shapley_mean": float(values.mean()),
                    "shapley_seed_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "shapley_seed_min": float(values.min()),
                    "shapley_seed_max": float(values.max()),
                    "seed_sign_consistent_positive": bool(np.all(values > 0.0)),
                }
            )
    efficiency = pd.DataFrame(efficiency_rows)
    if not np.all(efficiency["absolute_error"].to_numpy(dtype=float) <= 1.0e-10):
        raise RuntimeError("Shapley efficiency identity failed")
    return pd.DataFrame(shapley_rows), pd.DataFrame(interaction_rows), efficiency


def common_core_availability_analysis(
    common_val: Mapping[int, SampleSet],
    common_test: Mapping[int, SampleSet],
    qcore_val: Mapping[Tuple[int, str], SampleSet],
    qcore_test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    target_fpr: float,
    iterations: int,
    rng_seed: int,
    max_rows: int,
    ap_hist_initial_bins: int,
    ap_hist_max_bins: int,
    ap_hist_max_error: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not common_val:
        return pd.DataFrame(), pd.DataFrame()
    point_rows: List[Dict[str, object]] = []
    thresholds: Dict[Tuple[int, str], float] = {}
    for seed in seeds:
        for label, val_set, test_set in (
            ("q_core", qcore_val[(seed, one_mask())], qcore_test[(seed, one_mask())]),
            ("common_core_plus_rh2m_dpd", common_val[seed], common_test[seed]),
        ):
            threshold, achieved = match_fpr_threshold(val_set.y, val_set.score, target_fpr)
            thresholds[(seed, label)] = threshold
            prob = probability_metrics(test_set.y, test_set.score, bins=15)
            fixed = threshold_metrics(test_set.y, test_set.score, threshold)
            point_rows.append(
                {
                    "seed": seed,
                    "feature_set": label,
                    "target_validation_fpr": target_fpr,
                    "achieved_validation_fpr": achieved,
                    **prob,
                    **fixed,
                }
            )
    point = pd.DataFrame(point_rows)

    ref = qcore_test[(seeds[0], one_mask())]
    rng = np.random.default_rng(rng_seed + 811)
    working_q = {seed: qcore_test[(seed, one_mask())] for seed in seeds}
    working_c = dict(common_test)
    if max_rows > 0 and len(ref.y) > max_rows:
        keep = np.sort(rng.choice(len(ref.y), size=max_rows, replace=False).astype(np.int64))
        ref = subset_sample(ref, keep)
        working_q = {seed: subset_sample(sample, keep) for seed, sample in working_q.items()}
        working_c = {seed: subset_sample(sample, keep) for seed, sample in working_c.items()}
    dates = ref.frame["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    unique_dates, date_codes = np.unique(dates, return_inverse=True)
    n_dates = len(unique_dates)
    histogram_samples: Dict[Tuple[int, str], SampleSet] = {}
    for seed in seeds:
        histogram_samples[(seed, "q_core")] = working_q[seed]
        histogram_samples[(seed, "common_core_plus_rh2m_dpd")] = working_c[seed]
    precomputed, _ap_bins, _max_ap_error = adaptive_ap_histograms(
        histogram_samples,
        thresholds,
        date_codes,
        n_dates,
        ap_hist_initial_bins,
        ap_hist_max_bins,
        ap_hist_max_error,
    )
    draws: List[Dict[str, object]] = []
    for iteration in range(iterations):
        date_weights = np.bincount(
            rng.integers(0, n_dates, size=n_dates), minlength=n_dates
        ).astype(np.int64)
        for metric in PRIMARY_METRICS:
            deltas = []
            for seed in seeds:
                values = {}
                for label in ("q_core", "common_core_plus_rh2m_dpd"):
                    positive, negative, confusion = precomputed[(seed, label)]
                    if metric == "low_vis_ap":
                        values[label] = ap_from_histogram(date_weights @ positive, date_weights @ negative)
                    else:
                        tp, fp, fn, _tn = date_weights @ confusion
                        values[label] = (
                            safe_div(float(tp), float(tp + fp + fn))
                            if metric == "low_vis_csi_matched_fpr"
                            else safe_div(float(tp), float(tp + fn))
                        )
                q_value = values["q_core"]
                c_value = values["common_core_plus_rh2m_dpd"]
                deltas.append(c_value - q_value)
            draws.append({"iteration": iteration, "metric": metric, "delta_common_core_minus_q_core": float(np.mean(deltas))})
    draw_df = pd.DataFrame(draws)
    summary_rows = []
    for metric, part in draw_df.groupby("metric", sort=False):
        values = part["delta_common_core_minus_q_core"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "delta_common_core_minus_q_core": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_iterations": iterations,
                "bootstrap_unit": "UTC_valid_date",
                "interpretation": "incremental RH2M+DPD information package; not a general causal effect",
            }
        )
    return point, pd.DataFrame(summary_rows)


def subset_sample(sample: SampleSet, idx: np.ndarray) -> SampleSet:
    return SampleSet(sample.frame.iloc[idx].reset_index(drop=True), sample.y[idx], sample.score[idx], sample.pred[idx])


def ap_from_histogram(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)[::-1]
    negative = np.asarray(negative, dtype=np.float64)[::-1]
    total_positive = float(positive.sum())
    if total_positive <= 0:
        return math.nan
    tp = np.cumsum(positive)
    fp = np.cumsum(negative)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return float(np.sum((positive / total_positive) * precision))


def adaptive_ap_histograms(
    samples: Mapping[object, SampleSet],
    thresholds: Mapping[object, float],
    date_codes: np.ndarray,
    n_dates: int,
    initial_bins: int,
    max_bins: int,
    max_error: float,
) -> Tuple[Dict[object, Tuple[np.ndarray, np.ndarray, np.ndarray]], int, float]:
    if initial_bins < 2 or max_bins < initial_bins:
        raise ValueError(f"Invalid AP histogram bin range: {initial_bins}..{max_bins}")
    if max_error <= 0:
        raise ValueError("AP histogram maximum error must be positive")
    bins = int(initial_bins)
    while True:
        precomputed: Dict[object, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        largest_error = 0.0
        for key, sample in samples.items():
            y_low = sample.y <= 1
            score_bin = np.clip((sample.score * (bins - 1)).astype(np.int32), 0, bins - 1)
            positive = np.zeros((n_dates, bins), dtype=np.int32)
            negative = np.zeros((n_dates, bins), dtype=np.int32)
            np.add.at(positive, (date_codes[y_low], score_bin[y_low]), 1)
            np.add.at(negative, (date_codes[~y_low], score_bin[~y_low]), 1)
            pred = sample.score >= thresholds[key]
            confusion = np.zeros((n_dates, 4), dtype=np.int64)
            np.add.at(confusion[:, 0], date_codes[y_low & pred], 1)
            np.add.at(confusion[:, 1], date_codes[~y_low & pred], 1)
            np.add.at(confusion[:, 2], date_codes[y_low & ~pred], 1)
            np.add.at(confusion[:, 3], date_codes[~y_low & ~pred], 1)
            exact_ap = average_precision_binary(y_low.astype(np.int64), sample.score)
            approximate_ap = ap_from_histogram(positive.sum(axis=0), negative.sum(axis=0))
            largest_error = max(largest_error, abs(exact_ap - approximate_ap))
            precomputed[key] = positive, negative, confusion
        if largest_error <= max_error:
            return precomputed, bins, largest_error
        if bins >= max_bins:
            raise RuntimeError(
                f"Bootstrap AP histogram approximation error {largest_error:.6g} exceeds "
                f"{max_error:.6g} at maximum {bins} bins"
            )
        bins = min(bins * 2, max_bins)


def bootstrap_effects(
    test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    masks: Sequence[str],
    thresholds: Mapping[Tuple[int, str], float],
    iterations: int,
    rng_seed: int,
    max_rows: int,
    ap_hist_initial_bins: int,
    ap_hist_max_bins: int,
    ap_hist_max_error: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference = test[(seeds[0], zero_mask())]
    n = len(reference.y)
    rng = np.random.default_rng(rng_seed)
    if max_rows > 0 and n > max_rows:
        keep = np.sort(rng.choice(n, size=max_rows, replace=False).astype(np.int64))
        working = {key: subset_sample(sample, keep) for key, sample in test.items()}
        reference = working[(seeds[0], zero_mask())]
    else:
        working = dict(test)
    dates = reference.frame["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    unique_dates, date_codes = np.unique(dates, return_inverse=True)
    n_dates = len(unique_dates)
    precomputed, ap_bins, max_ap_approximation_error = adaptive_ap_histograms(
        working,
        thresholds,
        date_codes,
        n_dates,
        ap_hist_initial_bins,
        ap_hist_max_bins,
        ap_hist_max_error,
    )

    draws: List[Dict[str, object]] = []
    gap_draws: List[Dict[str, object]] = []
    for iteration in range(iterations):
        chosen = rng.integers(0, n_dates, size=n_dates)
        date_weights = np.bincount(chosen, minlength=n_dates).astype(np.int64)
        metric_values: Dict[str, Dict[str, float]] = {metric: {} for metric in PRIMARY_METRICS}
        for mask in masks:
            seed_metrics: Dict[str, List[float]] = {metric: [] for metric in PRIMARY_METRICS}
            for seed in seeds:
                positive, negative, confusion = precomputed[(seed, mask)]
                positive_draw = date_weights @ positive
                negative_draw = date_weights @ negative
                tp, fp, fn, _tn = date_weights @ confusion
                seed_metrics["low_vis_ap"].append(ap_from_histogram(positive_draw, negative_draw))
                seed_metrics["low_vis_csi_matched_fpr"].append(safe_div(float(tp), float(tp + fp + fn)))
                seed_metrics["low_vis_recall_matched_fpr"].append(safe_div(float(tp), float(tp + fn)))
            for metric in PRIMARY_METRICS:
                metric_values[metric][mask] = float(np.mean(seed_metrics[metric]))
        for metric in PRIMARY_METRICS:
            phi = shapley_values(metric_values[metric])
            interactions = pair_interactions(metric_values[metric])
            for group, value in phi.items():
                draws.append(
                    {
                        "iteration": iteration,
                        "metric": metric,
                        "effect": "shapley",
                        "term": group,
                        "value": value,
                    }
                )
            for pair, value in interactions.items():
                draws.append(
                    {
                        "iteration": iteration,
                        "metric": metric,
                        "effect": "interaction",
                        "term": pair,
                        "value": value,
                    }
                )
            gap_draws.append(
                {
                    "iteration": iteration,
                    "metric": metric,
                    "delta_111_minus_000": metric_values[metric][one_mask()] - metric_values[metric][zero_mask()],
                    "delta_all1_minus_all0": metric_values[metric][one_mask()] - metric_values[metric][zero_mask()],
                    "all0_mask": zero_mask(),
                    "all1_mask": one_mask(),
                }
            )
    draw_df = pd.DataFrame(draws)
    summary_rows: List[Dict[str, object]] = []
    for (metric, effect, term), part in draw_df.groupby(["metric", "effect", "term"], sort=False):
        values = part["value"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "effect": effect,
                "term": term,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_iterations": int(iterations),
                "bootstrap_rows": int(len(reference.y)),
                "bootstrap_unit": "UTC_valid_date",
                "ap_histogram_bins": ap_bins,
                "ap_point_approximation_max_abs_error": max_ap_approximation_error,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(gap_draws)


def read_eval_data_dir(seed_dir: Path, tag: str) -> Path:
    path = seed_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    specs = cfg.get("specs", {})
    if tag not in specs or not specs[tag].get("data_dir"):
        raise KeyError(f"{path}: missing data_dir for {tag}")
    return Path(specs[tag]["data_dir"]).expanduser()


def attach_source_features(
    events: pd.DataFrame,
    test_reference: SampleSet,
    pangu_dir: Path,
    tianji_dir: Path,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    with (pangu_dir / "dataset_build_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = [str(value) for value in cfg["dynamic_feature_order"]]
    dyn = int(cfg["dyn_vars"])
    window = int(cfg["window"])
    p_x = np.load(pangu_dir / "X_test.npy", mmap_mode="r")
    t_x = np.load(tianji_dir / "X_test.npy", mmap_mode="r")
    if len(p_x) != len(test_reference.y) or len(t_x) != len(test_reference.y):
        raise ValueError("Event feature matrices do not match evaluator row count")
    row_idx = events["_row_index"].to_numpy(dtype=np.int64)
    current_offset = (window - 1) * dyn
    out = events.copy()
    for feature in feature_names:
        if feature not in order:
            continue
        col = current_offset + order.index(feature)
        p_values = np.asarray(p_x[row_idx, col], dtype=np.float64)
        t_values = np.asarray(t_x[row_idx, col], dtype=np.float64)
        out[f"{feature}_pangu"] = p_values
        out[f"{feature}_tianji"] = t_values
        out[f"{feature}_tianji_minus_pangu"] = t_values - p_values
    static_start = window * dyn
    out["orography_center_m"] = np.asarray(p_x[row_idx, static_start + 2], dtype=np.float64)
    return out


def current_feature_values(data_dir: Path, n: int, feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    with (data_dir / "dataset_build_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = [str(value) for value in cfg["dynamic_feature_order"]]
    dyn, window = int(cfg["dyn_vars"]), int(cfg["window"])
    x = np.load(data_dir / "X_test.npy", mmap_mode="r")
    if len(x) != n:
        raise ValueError(f"{data_dir}: X_test rows {len(x)} != paired evaluator rows {n}")
    current = (window - 1) * dyn
    return {
        feature: np.asarray(x[:, current + order.index(feature)], dtype=np.float64)
        for feature in feature_names
        if feature in order
    }


def clean_observation_values(column: str, values) -> np.ndarray:
    # pandas 2.2+ may expose a read-only zero-copy view here; the QC operations
    # below intentionally mutate the array, so request an owned buffer.
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1.0e5] = np.nan
    if column == "prs_sea":
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmedian(np.abs(finite))) > 2000.0:
            arr = arr / 100.0
    bounds = OBS_VALID_RANGES.get(column)
    if bounds is not None:
        lo, hi = bounds
        arr[(arr < lo) | (arr > hi)] = np.nan
    return arr


def continuous_error_metrics(forecast: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(forecast) & np.isfinite(reference)
    if not valid.any():
        return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan, "correlation": math.nan}
    f, r = forecast[valid], reference[valid]
    corr = float(np.corrcoef(f, r)[0, 1]) if len(f) > 1 and np.std(f) > 0 and np.std(r) > 0 else math.nan
    return {
        "n": int(len(f)),
        "bias": float(np.mean(f - r)),
        "mae": float(np.mean(np.abs(f - r))),
        "rmse": float(np.sqrt(np.mean((f - r) ** 2))),
        "correlation": corr,
    }


def observation_anchored_quality(
    reference: SampleSet,
    pangu_dir: Path,
    tianji_dir: Path,
    obs_root: str,
    paper_eval_dir: str,
    common_core_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frame = reference.frame.copy()
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame = attach_source_features(frame, reference, pangu_dir, tianji_dir, ("T2M", "WSPD10", "MSLP"))
    frame, info = attach_observations(frame, Path(obs_root), Path(paper_eval_dir))
    if common_core_dir is not None:
        common_values = current_feature_values(common_core_dir, len(frame), ("RH2M",))
        if "RH2M" in common_values:
            frame["RH2M_tianji_common_core"] = common_values["RH2M"]
    mapping = {"T2M": "tem", "WSPD10": "win_s_avg_10mi", "MSLP": "prs_sea"}
    rows: List[Dict[str, object]] = []
    low = reference.y <= 1
    for feature, obs_column in mapping.items():
        if obs_column not in frame:
            continue
        obs = clean_observation_values(obs_column, frame[obs_column])
        for source in ("pangu", "tianji"):
            column = f"{feature}_{source}"
            if column not in frame:
                continue
            forecast = frame[column].to_numpy(dtype=np.float64)
            if feature == "T2M" and np.nanmedian(forecast) > 150.0:
                forecast = forecast - 273.15
            if feature == "MSLP" and np.nanmedian(forecast) > 2000.0:
                forecast = forecast / 100.0
            for scope, mask in (("all_paired_test", np.ones(len(frame), dtype=bool)), ("true_low_visibility", low)):
                metrics = continuous_error_metrics(forecast[mask], obs[mask])
                rows.append(
                    {
                        "feature": feature,
                        "observation_column": obs_column,
                        "source": source,
                        "scope": scope,
                        **metrics,
                    }
                )
    if "rhu" in frame and "RH2M_tianji_common_core" in frame:
        obs = clean_observation_values("rhu", frame["rhu"])
        forecast = frame["RH2M_tianji_common_core"].to_numpy(dtype=np.float64)
        if np.nanmedian(np.abs(forecast)) <= 1.5:
            forecast = forecast * 100.0
        for scope, mask in (("all_paired_test", np.ones(len(frame), dtype=bool)), ("true_low_visibility", low)):
            rows.append(
                {
                    "feature": "RH2M",
                    "observation_column": "rhu",
                    "source": "tianji_common_core",
                    "scope": scope,
                    **continuous_error_metrics(forecast[mask], obs[mask]),
                }
            )
    return pd.DataFrame(rows), info


def binary_event_metrics(observed: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(observed & forecast))
    fp = float(np.sum(~observed & forecast))
    fn = float(np.sum(observed & ~forecast))
    tn = float(np.sum(~observed & ~forecast))
    return {
        "pod": safe_div(tp, tp + fn),
        "far": safe_div(fp, tp + fp),
        "pofd": safe_div(fp, fp + tn),
        "csi": safe_div(tp, tp + fp + fn),
        "frequency_bias": safe_div(tp + fp, tp + fn),
    }


def q_reference_quality(
    reference: SampleSet,
    pangu_dir: Path,
    tianji_dir: Path,
    era5_dir: Path,
) -> pd.DataFrame:
    source_dirs = {"pangu": pangu_dir, "tianji": tianji_dir, "era5_reference_analysis": era5_dir}
    ref_index = pd.MultiIndex.from_frame(reference.frame[["time_key", "station_key"]])
    source_positions: Dict[str, np.ndarray] = {}
    source_values: Dict[str, Dict[str, np.ndarray]] = {}
    common = np.ones(len(ref_index), dtype=bool)
    for source, data_dir in source_dirs.items():
        meta = pd.read_csv(data_dir / "meta_test.csv", usecols=["time", "station_id"])
        time = pd.to_datetime(meta["time"], errors="coerce", utc=True)
        source_index = pd.MultiIndex.from_arrays(
            [time.dt.strftime("%Y-%m-%d %H:%M:%S"), normalize_station(meta["station_id"])],
            names=["time_key", "station_key"],
        )
        if source_index.has_duplicates:
            raise ValueError(f"{data_dir}: duplicate test metadata keys")
        positions = source_index.get_indexer(ref_index)
        source_positions[source] = positions
        common &= positions >= 0
        source_values[source] = current_feature_values(data_dir, len(meta), ("Q_1000", "Q_925"))
    if not common.any():
        raise RuntimeError("No common Pangu/Tianji/ERA5 q-core test samples for Q reference analysis")
    values = {
        source: {
            feature: feature_values[source_positions[source][common]]
            for feature, feature_values in source_values[source].items()
        }
        for source in source_dirs
    }
    rows: List[Dict[str, object]] = []
    for feature in ("Q_1000", "Q_925"):
        ref = values["era5_reference_analysis"][feature].copy()
        if np.nanmedian(np.abs(ref)) < 0.1:
            ref *= 1000.0
        for source in ("pangu", "tianji"):
            forecast = values[source][feature].copy()
            if np.nanmedian(np.abs(forecast)) < 0.1:
                forecast *= 1000.0
            valid = np.isfinite(ref) & np.isfinite(forecast) & (ref >= 0.0) & (forecast >= 0.0)
            error = continuous_error_metrics(forecast[valid], ref[valid])
            rows.append(
                {
                    "feature": feature,
                    "source": source,
                    "analysis": "continuous_error_vs_reference_analysis",
                    "quantile": math.nan,
                    "reference_threshold_g_kg": math.nan,
                    "source_threshold_g_kg": math.nan,
                    **error,
                }
            )
            for quantile in (0.90, 0.95, 0.99):
                ref_threshold = float(np.quantile(ref[valid], quantile))
                source_threshold = float(np.quantile(forecast[valid], quantile))
                observed = ref[valid] >= ref_threshold
                for analysis, predicted, threshold in (
                    ("exact_reference_threshold", forecast[valid] >= ref_threshold, ref_threshold),
                    ("quantile_matched", forecast[valid] >= source_threshold, source_threshold),
                ):
                    event = binary_event_metrics(observed, predicted)
                    rows.append(
                        {
                            "feature": feature,
                            "source": source,
                            "analysis": analysis,
                            "quantile": quantile,
                            "reference_threshold_g_kg": ref_threshold,
                            "source_threshold_g_kg": threshold,
                            "n": int(valid.sum()),
                            "bias": math.nan,
                            "mae": math.nan,
                            "rmse": math.nan,
                            "correlation": math.nan,
                            **event,
                        }
                    )
    return pd.DataFrame(rows)


def qc_elevation_sensitivity(
    test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    thresholds: Mapping[Tuple[int, str], float],
    eval_root: Path,
    prefix: str,
) -> pd.DataFrame:
    seed_dir = eval_root / f"seed_{seeds[0]}"
    data_dirs = {
        "pangu": read_eval_data_dir(seed_dir, f"{prefix}{zero_mask()}"),
        "tianji": read_eval_data_dir(seed_dir, f"{prefix}{one_mask()}"),
    }
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for source, data_dir in data_dirs.items():
        with (data_dir / "dataset_build_config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        order = [str(value) for value in cfg["dynamic_feature_order"]]
        dyn, window = int(cfg["dyn_vars"]), int(cfg["window"])
        x = np.load(data_dir / "X_test.npy", mmap_mode="r")
        current = (window - 1) * dyn
        arrays[source] = {
            "q1000": np.asarray(x[:, current + order.index("Q_1000")], dtype=np.float64),
            "q925": np.asarray(x[:, current + order.index("Q_925")], dtype=np.float64),
            "orography": np.asarray(x[:, window * dyn + 2], dtype=np.float64),
        }
    n = len(test[(seeds[0], zero_mask())].y)
    if any(len(values["q1000"]) != n for values in arrays.values()):
        raise ValueError("QC/elevation arrays do not match evaluator test rows")
    p_valid = np.isfinite(arrays["pangu"]["q1000"]) & np.isfinite(arrays["pangu"]["q925"])
    p_valid &= (arrays["pangu"]["q1000"] >= 0.0) & (arrays["pangu"]["q925"] >= 0.0)
    orography = arrays["pangu"]["orography"]
    scenarios = {
        "all": np.ones(n, dtype=bool),
        "pangu_nonnegative_q": p_valid,
        "elevation_le_100m": np.isfinite(orography) & (orography <= 100.0),
        "elevation_100_500m": np.isfinite(orography) & (orography > 100.0) & (orography <= 500.0),
        "elevation_gt_500m": np.isfinite(orography) & (orography > 500.0),
    }
    rows: List[Dict[str, object]] = []
    for source, mask in (("pangu", zero_mask()), ("tianji", one_mask())):
        q1000 = arrays[source]["q1000"]
        q925 = arrays[source]["q925"]
        nonphysical = (~np.isfinite(q1000)) | (~np.isfinite(q925)) | (q1000 < 0.0) | (q925 < 0.0)
        for scenario, sample_mask in scenarios.items():
            if not sample_mask.any():
                continue
            for seed in seeds:
                sample = test[(seed, mask)]
                prob = probability_metrics(sample.y[sample_mask], sample.score[sample_mask], bins=10)
                fixed = threshold_metrics(
                    sample.y[sample_mask], sample.score[sample_mask], thresholds[(seed, mask)]
                )
                rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "scenario": scenario,
                        "n": int(sample_mask.sum()),
                        "source_nonphysical_q_count": int(np.sum(nonphysical & sample_mask)),
                        "source_nonphysical_q_fraction": float(np.mean(nonphysical[sample_mask])),
                        "orography_median_m": float(np.nanmedian(orography[sample_mask])),
                        "low_vis_ap": prob["low_vis_ap"],
                        "low_vis_csi_matched_fpr": fixed["low_vis_csi_matched_fpr"],
                        "low_vis_recall_matched_fpr": fixed["low_vis_recall_matched_fpr"],
                        "low_vis_fpr_matched_fpr": fixed["low_vis_fpr_matched_fpr"],
                    }
                )
    return pd.DataFrame(rows)


def attach_observations(events: pd.DataFrame, obs_root: Path, paper_eval_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    module_path = paper_eval_dir / "analyze_key_variable_quality.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("qcore_key_variable_quality", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    meta = events[["time_utc", "station_key"]].rename(columns={"time_utc": "time"}).copy()
    # analyze_key_variable_quality.choose_obs_time_shift predates the q-core
    # analysis and uses timezone-naive UTC timestamps for its probe merge.
    # Keep event tables UTC-aware elsewhere, but pass a naive UTC view into
    # the observation-alignment helper to avoid pandas aware-vs-naive merge
    # failures.
    meta["time"] = pd.to_datetime(meta["time"], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    shift, obs, diag = mod.choose_obs_time_shift(obs_root, meta, 96)
    keep = [column for column in ("rhu", "tem", "win_s_avg_10mi", "pre_1h", "prs_sea") if column in obs.columns]
    obs = obs[["time", "station_key", *keep]].copy()
    obs["time"] = pd.to_datetime(obs["time"], errors="coerce", utc=True)
    for column in keep:
        obs[column] = clean_observation_values(column, obs[column])
    merged = events.merge(obs, left_on=["time_utc", "station_key"], right_on=["time", "station_key"], how="left")
    if "time" in merged:
        merged = merged.drop(columns=["time"])
    return merged, {
        "obs_root": str(obs_root),
        "obs_time_shift_to_utc_hours": float(shift),
        "matched_event_rows": int(merged[keep].notna().any(axis=1).sum()) if keep else 0,
        "available_observation_columns": keep,
        "timezone_probe": diag.to_dict(orient="records"),
    }


def event_case_control(
    val: Mapping[Tuple[int, str], SampleSet],
    test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    target_fpr: float,
    eval_root: Path,
    prefix: str,
    feature_names: Sequence[str],
    obs_root: str,
    paper_eval_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    all0 = zero_mask()
    all1 = one_mask()
    ref = test[(seeds[0], all0)]
    y_low = ref.y <= 1
    mean_val: Dict[str, np.ndarray] = {}
    mean_test: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}
    for mask in (all0, all1):
        mean_val[mask] = np.mean(np.stack([val[(seed, mask)].score for seed in seeds], axis=0), axis=0)
        mean_test[mask] = np.mean(np.stack([test[(seed, mask)].score for seed in seeds], axis=0), axis=0)
        thresholds[mask], _ = match_fpr_threshold(val[(seeds[0], mask)].y, mean_val[mask], target_fpr)
    p_hit = mean_test[all0] >= thresholds[all0]
    t_hit = mean_test[all1] >= thresholds[all1]
    category = np.full(len(ref.y), "not_low_visibility", dtype=object)
    category[y_low & p_hit & t_hit] = "both_hit"
    category[y_low & ~p_hit & ~t_hit] = "both_miss"
    category[y_low & ~p_hit & t_hit] = "tianji_hit_pangu_miss"
    category[y_low & p_hit & ~t_hit] = "pangu_hit_tianji_miss"
    event_idx = np.flatnonzero(y_low)
    events = ref.frame.iloc[event_idx].reset_index(drop=True).copy()
    events["_row_index"] = event_idx
    events["y_cls"] = ref.y[event_idx]
    if "vis_raw_m" in ref.frame:
        events["vis_raw_m"] = ref.frame["vis_raw_m"].to_numpy()[event_idx]
    events["case_category"] = category[event_idx]
    events["pangu_seed_mean_low_vis_probability"] = mean_test[all0][event_idx]
    events["tianji_seed_mean_low_vis_probability"] = mean_test[all1][event_idx]
    events["pangu_matched_fpr_threshold"] = thresholds[all0]
    events["tianji_matched_fpr_threshold"] = thresholds[all1]

    seed_dir = eval_root / f"seed_{seeds[0]}"
    pangu_dir = read_eval_data_dir(seed_dir, f"{prefix}{all0}")
    tianji_dir = read_eval_data_dir(seed_dir, f"{prefix}{all1}")
    events = attach_source_features(events, ref, pangu_dir, tianji_dir, feature_names)
    obs_info: Dict[str, object] = {"enabled": False}
    if obs_root:
        events, detail = attach_observations(events, Path(obs_root), Path(paper_eval_dir))
        obs_info = {"enabled": True, **detail}

    numeric = [
        column
        for column in events.select_dtypes(include=[np.number]).columns
        if column not in {"_row_index", "y_cls"}
    ]
    summary_rows: List[Dict[str, object]] = []
    for category_name, part in events.groupby("case_category", sort=False):
        row: Dict[str, object] = {"case_category": category_name, "n": int(len(part))}
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_median"] = float(values.median())
        summary_rows.append(row)
    return events, pd.DataFrame(summary_rows), {"thresholds": thresholds, "observations": obs_info}


def make_figures(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    shapley: pd.DataFrame,
    boot: pd.DataFrame,
    events: pd.DataFrame,
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] figures skipped: {exc}")
        return
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    aggregate = metrics.groupby("mask", sort=True)[list(PRIMARY_METRICS)].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(aggregate))
    ax.plot(x, aggregate["low_vis_ap"], marker="o", label="Low-vis AP")
    ax.plot(x, aggregate["low_vis_csi_matched_fpr"], marker="s", label="CSI at matched FPR")
    ax.set_xticks(x, aggregate["mask"], rotation=45)
    ax.set_xlabel(f"{GROUP_PROFILE} source mask (1 = Tianji; order={','.join(GROUP_ORDER)})")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    plot = shapley[shapley["metric"] == "low_vis_ap"].copy()
    ci = boot[(boot["metric"] == "low_vis_ap") & (boot["effect"] == "shapley")].set_index("term")
    y = np.arange(len(plot))
    means = plot["shapley_mean"].to_numpy(dtype=float)
    ci_low = np.asarray([float(ci.loc[group, "ci_low"]) for group in plot["group"]])
    ci_high = np.asarray([float(ci.loc[group, "ci_high"]) for group in plot["group"]])
    ax.barh(y, means, color="#2878B5", alpha=0.82, label="Point estimate")
    for y_pos, low, high in zip(y, ci_low, ci_high):
        if np.isfinite(low) and np.isfinite(high):
            left, right = sorted((low, high))
            ax.hlines(y_pos, left, right, color="#17202A", linewidth=1.4, zorder=4)
            ax.plot([left, right], [y_pos, y_pos], linestyle="none", marker="|", markersize=6, color="#17202A", zorder=5)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y, plot["group_label"])
    ax.set_xlabel("Exact Shapley contribution to Low-vis AP")
    ax.plot([], [], color="#17202A", marker="|", label="95% UTC-date bootstrap CI")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    all0 = zero_mask()
    all1 = one_mask()
    endpoint = aggregate[aggregate["mask"].isin([all0, all1])].set_index("mask")
    width = 0.34
    values = [endpoint.loc[all0, "low_vis_ap"], endpoint.loc[all1, "low_vis_ap"]]
    ax.bar(np.arange(2) - width / 2, values, width, label="AP", color="#8C564B")
    values_csi = [endpoint.loc[all0, "low_vis_csi_matched_fpr"], endpoint.loc[all1, "low_vis_csi_matched_fpr"]]
    ax.bar(np.arange(2) + width / 2, values_csi, width, label="CSI at matched FPR", color="#2CA02C")
    ax.set_xticks(np.arange(2), ["Pangu q-core", "Tianji q-core"])
    ax.set_ylabel("Score")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    counts = events["case_category"].value_counts()
    ax.barh(np.arange(len(counts)), counts.to_numpy(), color="#6F4E7C")
    ax.set_yticks(np.arange(len(counts)), counts.index.str.replace("_", " "))
    ax.set_xlabel("True Low-vis event samples")
    ax.set_title("Matched-FPR event case control")
    for panel_label, ax in zip("abcd", axes.ravel()):
        ax.text(-0.12, 1.04, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig_q_core_hybrid_factorial_mechanism.{ext}", dpi=300)
    plt.close(fig)

    endpoint = reliability[reliability["mask"].isin([zero_mask(), one_mask()])].copy()
    if not endpoint.empty:
        endpoint = (
            endpoint.groupby(["mask", "bin"], as_index=False)
            .agg(mean_probability=("mean_probability", "mean"), observed_frequency=("observed_frequency", "mean"), n=("n", "sum"))
        )
        fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#777777", label="Perfect calibration")
        for mask, label, color, marker in (
            (zero_mask(), "Pangu q-core", "#684A9B", "o"),
            (one_mask(), "Tianji q-core", "#176B87", "s"),
        ):
            part = endpoint[(endpoint["mask"] == mask) & (endpoint["n"] > 0)].sort_values("mean_probability")
            ax.plot(part["mean_probability"], part["observed_frequency"], marker=marker, color=color, label=label)
        ax.set(xlabel="Predicted Low-vis probability", ylabel="Observed Low-vis frequency", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir / f"fig_q_core_endpoint_reliability.{ext}", dpi=300)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_group_profile(args.group_profile, args.dataset_prefix or None, args.source_prefix or None)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    masks = parse_masks(args.masks)
    if len(seeds) not in {1, 3}:
        raise ValueError("Use one seed for an end-to-end smoke test or exactly three seeds for formal analysis")
    val, test = load_all(eval_root, seeds, masks, SOURCE_PREFIX)
    common_val, common_test = load_common_core(eval_root, seeds, val, test)
    metrics, reliability, target_fpr, thresholds = point_analysis(val, test, seeds, masks, args.ece_bins)
    shapley, interactions, efficiency = aggregate_effects(metrics)
    bootstrap, gap_draws = bootstrap_effects(
        test,
        seeds,
        masks,
        thresholds,
        args.bootstrap_iters,
        args.bootstrap_seed,
        args.bootstrap_max_rows,
        args.ap_hist_initial_bins,
        args.ap_hist_max_bins,
        args.ap_hist_max_error,
    )
    availability_point, availability_ci = common_core_availability_analysis(
        common_val,
        common_test,
        val,
        test,
        seeds,
        target_fpr,
        args.bootstrap_iters,
        args.bootstrap_seed,
        args.bootstrap_max_rows,
        args.ap_hist_initial_bins,
        args.ap_hist_max_bins,
        args.ap_hist_max_error,
    )
    shapley_ci = bootstrap[(bootstrap["effect"] == "shapley")][["metric", "term", "ci_low", "ci_high"]].rename(
        columns={"term": "group"}
    )
    shapley = shapley.merge(shapley_ci, on=["metric", "group"], how="left", validate="one_to_one")
    events, event_summary, event_info = event_case_control(
        val,
        test,
        seeds,
        target_fpr,
        eval_root,
        SOURCE_PREFIX,
        parse_csv(args.event_features),
        args.obs_root,
        args.paper_eval_dir,
    )
    qc_sensitivity = qc_elevation_sensitivity(test, seeds, thresholds, eval_root, SOURCE_PREFIX)
    seed_dir = eval_root / f"seed_{seeds[0]}"
    pangu_data_dir = read_eval_data_dir(seed_dir, f"{SOURCE_PREFIX}{zero_mask()}")
    tianji_data_dir = read_eval_data_dir(seed_dir, f"{SOURCE_PREFIX}{one_mask()}")
    observation_quality = pd.DataFrame()
    observation_quality_info: Dict[str, object] = {"enabled": False}
    if args.obs_root:
        common_core_dir = (
            read_eval_data_dir(seed_dir, "tianji_common_core") if common_val else None
        )
        observation_quality, observation_quality_info = observation_anchored_quality(
            test[(seeds[0], zero_mask())],
            pangu_data_dir,
            tianji_data_dir,
            args.obs_root,
            args.paper_eval_dir,
            common_core_dir,
        )
        observation_quality_info = {"enabled": True, **observation_quality_info}
    q_quality = pd.DataFrame()
    if args.era5_data_dir:
        q_quality = q_reference_quality(
            test[(seeds[0], zero_mask())], pangu_data_dir, tianji_data_dir, Path(args.era5_data_dir)
        )
    ale_all, ale_mean = load_ale_outputs(eval_root, seeds, args.require_ale)

    metrics.to_csv(out_dir / "hybrid_factorial_metrics_by_seed.csv", index=False)
    metrics.groupby("mask", sort=True).mean(numeric_only=True).reset_index().to_csv(
        out_dir / "hybrid_factorial_metrics_seed_mean.csv", index=False
    )
    reliability.to_csv(out_dir / "hybrid_reliability_bins.csv", index=False)
    shapley.to_csv(out_dir / "hybrid_exact_shapley_effects.csv", index=False)
    interactions.to_csv(out_dir / "hybrid_second_order_interactions_by_seed.csv", index=False)
    efficiency.to_csv(out_dir / "hybrid_shapley_efficiency_check.csv", index=False)
    bootstrap.to_csv(out_dir / "hybrid_date_block_bootstrap_ci.csv", index=False)
    gap_draws.to_csv(out_dir / "hybrid_total_gap_bootstrap_draws.csv.gz", index=False, compression="gzip")
    events.to_csv(out_dir / "event_case_control_samples.csv.gz", index=False, compression="gzip")
    event_summary.to_csv(out_dir / "event_case_control_environment_summary.csv", index=False)
    qc_sensitivity.to_csv(out_dir / "q_core_qc_elevation_sensitivity.csv", index=False)
    if not observation_quality.empty:
        observation_quality.to_csv(out_dir / "observation_anchored_source_quality.csv", index=False)
    if not q_quality.empty:
        q_quality.to_csv(out_dir / "q_reference_analysis_quality_and_extreme_placement.csv", index=False)
    if not ale_all.empty:
        ale_all.to_csv(out_dir / "q_core_trajectory_ale_all_seeds.csv", index=False)
        ale_mean.to_csv(out_dir / "q_core_trajectory_ale_seed_mean.csv", index=False)
    if not availability_point.empty:
        availability_point.to_csv(out_dir / "rh2m_dpd_information_package_metrics_by_seed.csv", index=False)
        availability_ci.to_csv(out_dir / "rh2m_dpd_information_package_date_bootstrap_ci.csv", index=False)

    ap_rows = shapley[shapley["metric"] == "low_vis_ap"].sort_values("shapley_mean", ascending=False)
    moisture = ap_rows[ap_rows["group"] == "M"].iloc[0]
    formal_three_seed = len(seeds) == 3
    if GROUP_PROFILE == "mtw":
        gate_passed = bool(
            formal_three_seed
            and str(ap_rows.iloc[0]["group"]) == "M"
            and float(moisture["ci_low"]) > 0.0
            and bool(moisture["seed_sign_consistent_positive"])
        )
        gate = {
            "status": "passed" if gate_passed else "not_passed",
            "criterion": "M is the largest Low-vis AP Shapley contribution, date-block CI excludes zero, and all seed effects are positive",
            "largest_group": str(ap_rows.iloc[0]["group"]),
            "moisture_shapley_mean": float(moisture["shapley_mean"]),
            "moisture_ci": [float(moisture["ci_low"]), float(moisture["ci_high"])],
            "moisture_seed_sign_consistent_positive": bool(moisture["seed_sign_consistent_positive"]),
            "formal_three_seed_analysis": formal_three_seed,
            "next_step": "build M1000-only and M925-only hybrids" if gate_passed else "do not claim a moisture mechanism; investigate the winning package",
        }
    elif GROUP_PROFILE == "m925b":
        interaction_rows = interactions[
            (interactions["metric"] == "low_vis_ap") & (interactions["pair"] == "M:H")
        ].copy()
        interaction_ci = bootstrap[
            (bootstrap["metric"] == "low_vis_ap")
            & (bootstrap["effect"] == "interaction")
            & (bootstrap["term"] == "M:H")
        ]
        if interaction_rows.empty or interaction_ci.empty:
            raise RuntimeError("m925b analysis is missing the Low-vis AP M:H interaction")
        interaction_values = interaction_rows["interaction"].to_numpy(dtype=float)
        ci_row = interaction_ci.iloc[0]
        positive = bool(
            formal_three_seed
            and np.all(interaction_values > 0.0)
            and float(ci_row["ci_low"]) > 0.0
        )
        gate = {
            "status": "performance_interaction_supported" if positive else "performance_interaction_not_supported",
            "criterion": (
                "The Low-vis AP Shapley M:T925 interaction is positive in all three seeds and its "
                "UTC-date block-bootstrap interval excludes zero."
            ),
            "interaction_pair": "M:H",
            "interaction_mean": float(np.mean(interaction_values)),
            "interaction_seed_values": [float(value) for value in interaction_values],
            "interaction_ci": [float(ci_row["ci_low"]), float(ci_row["ci_high"])],
            "formal_three_seed_analysis": formal_three_seed,
            "next_step": (
                "combine this predictive-complementarity result with ERA5-reference joint-quality and hit/miss evidence"
            ),
            "claim_limit": (
                "A positive interaction supports complementarity of the M and T925 source blocks under retraining; "
                "it does not alone prove physical-law consistency."
            ),
        }
    elif GROUP_PROFILE == "mhtpw":
        gate = {
            "status": "primary_factorial_complete" if formal_three_seed else "smoke_only",
            "criterion": (
                "All 32 prespecified M-H-T-P-W coalitions are evaluated for three independent training seeds."
            ),
            "largest_group": str(ap_rows.iloc[0]["group"]),
            "moisture_shapley_mean": float(moisture["shapley_mean"]),
            "moisture_ci": [float(moisture["ci_low"]), float(moisture["ci_high"])],
            "moisture_seed_sign_consistent_positive": bool(moisture["seed_sign_consistent_positive"]),
            "formal_three_seed_analysis": formal_three_seed,
            "wind_scope": "10 m plus 925 hPa low-level wind/ventilation package",
            "next_step": (
                "interpret package Shapley effects with layer-wise grouped model reliance and source-quality evidence"
            ),
            "claim_limit": (
                "Package effects quantify predictive source-block attribution under retraining, not isolated-variable "
                "causality or proof of governing-equation consistency."
            ),
        }
    else:
        gate = {
            "status": "not_applicable",
            "criterion": "The MTW moisture follow-up gate applies only to the original mtw profile.",
            "largest_group": str(ap_rows.iloc[0]["group"]),
            "moisture_shapley_mean": float(moisture["shapley_mean"]),
            "moisture_ci": [float(moisture["ci_low"]), float(moisture["ci_high"])],
            "moisture_seed_sign_consistent_positive": bool(moisture["seed_sign_consistent_positive"]),
            "formal_three_seed_analysis": formal_three_seed,
            "next_step": "interpret the mt2pw T-package split; compare T2M and MSLP Shapley effects and interactions",
        }
    report = {
        "status": "passed",
        "seeds": seeds,
        "masks": masks,
        "group_profile": GROUP_PROFILE,
        "group_order": list(GROUP_ORDER),
        "dataset_prefix": DATASET_PREFIX,
        "source_prefix": SOURCE_PREFIX,
        "all0_mask": zero_mask(),
        "all1_mask": one_mask(),
        "target_validation_fpr": target_fpr,
        "primary_endpoints": list(PRIMARY_METRICS),
        "bootstrap": {
            "iterations": args.bootstrap_iters,
            "seed": args.bootstrap_seed,
            "unit": "UTC_valid_date",
            "max_rows": args.bootstrap_max_rows,
            "low_vis_ap_method": "adaptive score histogram per UTC date; exact point AP is reported separately",
            "ap_histogram_initial_bins": args.ap_hist_initial_bins,
            "ap_histogram_max_bins": args.ap_hist_max_bins,
            "ap_histogram_selected_bins": int(bootstrap["ap_histogram_bins"].max()),
            "ap_point_approximation_max_abs_error": float(bootstrap["ap_point_approximation_max_abs_error"].max()),
            "acceptance": f"analysis stops if histogram AP differs from exact point AP by more than {args.ap_hist_max_error:g}",
        },
        "event_case_control": event_info,
        "observation_anchored_source_quality": observation_quality_info,
        "q_reference_analysis_quality": {
            "enabled": bool(not q_quality.empty),
            "era5_role": "reference analysis, not truth",
        },
        "ale_response_analysis": {
            "enabled": bool(not ale_all.empty),
            "required": bool(args.require_ale),
            "method": "first-order ALE of 12 h trajectory mean with UTC-date bootstrap",
        },
        "rh2m_dpd_information_package_analysis": {
            "enabled": bool(not availability_point.empty),
            "role": "Tianji common-core versus Tianji q-core paired comparison",
        },
        "mechanism_gate": gate,
        "moisture_followup_gate": gate,
        "interpretation": "controlled source-block retraining attribution; not a single-variable causal effect",
        "era5_role": "reference analysis only",
    }
    gate_filename = "joint_structure_performance_gate.json" if GROUP_PROFILE == "m925b" else "moisture_followup_gate.json"
    with (out_dir / gate_filename).open("w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)
    with (out_dir / "hybrid_factorial_analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not args.no_figures:
        make_figures(metrics, reliability, shapley, bootstrap, events, out_dir)
    print(f"[OK] q-core hybrid factorial analysis written to {out_dir}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
