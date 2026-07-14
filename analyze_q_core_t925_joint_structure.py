#!/usr/bin/env python3
"""Paper-grade T925--moisture joint-structure diagnosis for Pangu versus Tianji.

The default diagnostic-only mode combines two new evidence layers with the
already-completed MTW/mt2pw performance experiment:

1. paired source quality against ERA5 reference analysis;
2. joint-state errors in existing Tianji-hit/Pangu-miss cases.

An explicitly optional factorial-confirmatory mode can additionally read a
retrained M x T925 interaction. It is not required for the default diagnosis.

The analysis separates marginal accuracy from dependence using an empirical-
copula MMD.  The Gaussian-kernel MMD is approximated with fixed random Fourier
features so UTC-date block bootstrap remains tractable on station-scale data.
The 925-hPa saturation deficit is diagnosed with the same Bolton-form saturation
vapour-pressure relation used by the shared dataset builder.

Method anchors
--------------
Gretton et al. (2012), JMLR 13, 723--773, MMD two-sample testing.
Rahimi and Recht (2007), NeurIPS 20, random Fourier kernel features.
Hamill (1999), Weather and Forecasting 14, 155--167, case-day resampling.
Bolton (1980), Monthly Weather Review 108, 1046--1053, saturation formula.
Wilks (2015), QJRMS 141, 945--952, empirical-copula dependence templates.

ERA5 is always a reference analysis, never labelled as truth.  A positive
factorial interaction is predictive complementarity under retraining, not by
itself proof that a source obeys the governing equations more closely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SOURCES = ("pangu", "tianji", "era5_reference_analysis")
FORECAST_SOURCES = ("pangu", "tianji")
FEATURES = ("T_925", "Q_925", "RH_925")
ALLOWED_FEATURE_SETS = {"source_full", "q_core_t925_no_rh2m"}
EXPECTED_UNIT_POLICY = "pmst_canonical_units_v2_20260630"
EXPECTED_UNITS = {"T_925": "K", "Q_925": "kg kg-1", "RH_925": "%"}
SOURCE_LABELS = {
    "pangu": "Pangu",
    "tianji": "Tianji",
    "era5_reference_analysis": "ERA5 reference analysis",
}
COLORS = {"pangu": "#5B2C83", "tianji": "#2FBF9F", "era5_reference_analysis": "#777777"}


@dataclass
class DatasetLayout:
    path: Path
    order: Tuple[str, ...]
    window: int
    dyn_vars: int
    fe_dim: int
    feature_set: str
    config: Dict[str, object]


@dataclass
class AlignedSplit:
    keys: pd.DataFrame
    positions: Dict[str, np.ndarray]
    values: Dict[str, Dict[str, np.ndarray]]
    visibility_m: np.ndarray
    orography_m: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-dir", required=True)
    ap.add_argument("--tianji-dir", required=True)
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument(
        "--analysis-mode",
        choices=("diagnostic_only", "factorial_confirmatory"),
        default="diagnostic_only",
        help="diagnostic_only never requires or interprets newly trained M-H-B models.",
    )
    ap.add_argument("--event-analysis-dir", default="", help="Existing MTW/mt2pw analysis containing event samples.")
    ap.add_argument("--factorial-analysis-dir", default="", help="Required only for factorial_confirmatory mode.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fit-max-rows", type=int, default=200000)
    ap.add_argument("--test-max-rows", type=int, default=200000)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260714)
    ap.add_argument("--rff-dim", type=int, default=512)
    ap.add_argument("--rff-bandwidth-sample", type=int, default=2000)
    ap.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    ap.add_argument("--near-saturation-quantile", type=float, default=0.10)
    ap.add_argument("--min-event-coverage", type=float, default=0.80)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def load_layout(path: Path, source: str) -> DatasetLayout:
    cfg_path = path / "dataset_build_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = tuple(str(value) for value in cfg.get("dynamic_feature_order", []))
    missing = [name for name in FEATURES if name not in order]
    if missing:
        raise ValueError(f"{path}: q-core+T925 layout is missing {missing}; order={order}")
    feature_set = str(cfg.get("feature_set", "")).strip().lower().replace("-", "_")
    if feature_set not in ALLOWED_FEATURE_SETS:
        raise ValueError(f"{path}: feature_set={feature_set!r}; expected one of {sorted(ALLOWED_FEATURE_SETS)}")
    if str(cfg.get("canonical_unit_policy", "")) != EXPECTED_UNIT_POLICY:
        raise ValueError(
            f"{path}: canonical_unit_policy predates {EXPECTED_UNIT_POLICY}; rebuild data before diagnosis"
        )
    declared_units = cfg.get("canonical_dynamic_units")
    if not isinstance(declared_units, Mapping):
        raise ValueError(f"{path}: canonical_dynamic_units metadata is missing")
    mismatch = {
        feature: (declared_units.get(feature), expected)
        for feature, expected in EXPECTED_UNITS.items()
        if declared_units.get(feature) != expected
    }
    if mismatch:
        raise ValueError(f"{path}: canonical unit mismatch: {mismatch}")
    if str(cfg.get("time_coordinate", "")).upper() != "UTC":
        raise ValueError(f"{path}: time_coordinate must be explicitly recorded as UTC")
    native = {str(value) for value in cfg.get("native_source_features", [])}
    derived = {str(value) for value in cfg.get("derived_source_features", [])}
    if "T_925" not in native:
        raise ValueError(f"{path}: T_925 must be documented as a native source field")
    if source in {"pangu", "tianji"} and "Q_925" not in native:
        raise ValueError(f"{path}: {source} Q_925 must be documented as native")
    if source == "era5_reference_analysis" and "Q_925" not in native:
        if "Q_925" not in derived or not {"T_925", "RH_925"}.issubset(native):
            raise ValueError(f"{path}: ERA5 Q_925 needs native Q or documented derivation from native T925/RH925")
    if source == "pangu":
        lead = cfg.get("source_forecast_lead")
        if not isinstance(lead, Mapping) or not bool(lead.get("available")):
            raise ValueError(f"{path}: Pangu forecast-lead provenance is missing")
        lead_min = float(lead.get("min_hours", math.nan))
        lead_max = float(lead.get("max_hours", math.nan))
        if not (math.isclose(lead_min, 12.0, abs_tol=1.0e-6) and math.isclose(lead_max, 23.0, abs_tol=1.0e-6)):
            raise ValueError(f"{path}: expected canonical Pangu 12--23 h lead, got {lead}")
    window = int(cfg["window"])
    dyn_vars = int(cfg["dyn_vars"])
    if window != 12 or dyn_vars != len(order):
        raise ValueError(f"{path}: invalid window/dyn_vars metadata")
    return DatasetLayout(
        path=path,
        order=order,
        window=window,
        dyn_vars=dyn_vars,
        fe_dim=int(cfg["fe_dim"]),
        feature_set=feature_set,
        config=dict(cfg),
    )


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def metadata(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path / f"meta_{split}.csv")
    required = {"time", "station_id"}
    if not required.issubset(frame):
        raise KeyError(f"{path}/meta_{split}.csv lacks {sorted(required - set(frame))}")
    time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError(f"{path}/meta_{split}.csv has invalid timestamps")
    out = pd.DataFrame(
        {
            "time_utc": time,
            "time_key": time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "station_key": normalize_station(frame["station_id"]),
            "source_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    if out[["time_key", "station_key"]].duplicated().any():
        raise ValueError(f"{path}/meta_{split}.csv has duplicate paired keys")
    return out


def key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["time_key", "station_key"]])


def balanced_day_subset(keys: pd.DataFrame, max_rows: int, seed: int) -> np.ndarray:
    n = len(keys)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    dates = keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    if max_rows < len(unique):
        chosen_days = unique[np.sort(rng.choice(len(unique), size=max_rows, replace=False))]
        return np.asarray([np.flatnonzero(dates == day)[0] for day in chosen_days], dtype=np.int64)
    quota = max(1, max_rows // len(unique))
    selected = []
    for day in unique:
        rows = np.flatnonzero(dates == day)
        take = min(len(rows), quota)
        selected.extend(np.sort(rng.choice(rows, size=take, replace=False)).tolist())
    remaining = max_rows - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(selected, dtype=np.int64), assume_unique=False)
        if len(pool):
            selected.extend(np.sort(rng.choice(pool, size=min(remaining, len(pool)), replace=False)).tolist())
    return np.sort(np.asarray(selected[:max_rows], dtype=np.int64))


def extract_sequences(layout: DatasetLayout, split: str, rows: np.ndarray, chunk: int = 20000) -> Dict[str, np.ndarray]:
    x = np.load(layout.path / f"X_{split}.npy", mmap_mode="r")
    if int(x.shape[1]) - layout.window * layout.dyn_vars - layout.fe_dim != 6:
        raise ValueError(f"{layout.path}/{split}: expected six static features")
    result = {
        name: np.empty((len(rows), layout.window), dtype=np.float32)
        for name in FEATURES
    }
    flat_columns = {
        name: [step * layout.dyn_vars + layout.order.index(name) for step in range(layout.window)]
        for name in FEATURES
    }
    for start in range(0, len(rows), chunk):
        end = min(start + chunk, len(rows))
        block = np.asarray(x[rows[start:end]], dtype=np.float32)
        for name, columns in flat_columns.items():
            result[name][start:end] = block[:, columns]
    return result


def align_split(
    layouts: Mapping[str, DatasetLayout],
    split: str,
    max_rows: int,
    seed: int,
) -> AlignedSplit:
    frames = {source: metadata(layout.path, split) for source, layout in layouts.items()}
    reference = frames["pangu"]
    common = key_index(reference)
    for source in SOURCES[1:]:
        common = common[common.isin(key_index(frames[source]))]
    if len(common) == 0:
        raise RuntimeError(f"{split}: no common Pangu/Tianji/ERA5 samples")
    keys = reference.set_index(["time_key", "station_key"]).loc[common].reset_index()
    keep = balanced_day_subset(keys, max_rows, seed)
    keys = keys.iloc[keep].reset_index(drop=True)
    wanted = key_index(keys)
    positions: Dict[str, np.ndarray] = {}
    for source, frame in frames.items():
        lookup = pd.Series(frame["source_row"].to_numpy(dtype=np.int64), index=key_index(frame))
        pos = lookup.reindex(wanted)
        if pos.isna().any():
            raise RuntimeError(f"{split}/{source}: aligned rows disappeared")
        positions[source] = pos.to_numpy(dtype=np.int64)
    values = {
        source: extract_sequences(layout, split, positions[source])
        for source, layout in layouts.items()
    }
    labels = {
        source: np.asarray(
            np.load(layout.path / f"y_{split}.npy", mmap_mode="r")[positions[source]],
            dtype=np.float64,
        )
        for source, layout in layouts.items()
    }
    reference_labels = labels["pangu"]
    for source in SOURCES[1:]:
        same = np.isclose(reference_labels, labels[source], rtol=0.0, atol=1.0e-3, equal_nan=False)
        if not np.all(same):
            raise ValueError(f"{split}/{source}: {int((~same).sum())} paired visibility labels differ from Pangu")
    p_x = np.load(layouts["pangu"].path / f"X_{split}.npy", mmap_mode="r")
    p_static_start = layouts["pangu"].window * layouts["pangu"].dyn_vars
    p_rows = positions["pangu"]
    visibility = reference_labels
    orography = np.asarray(p_x[p_rows, p_static_start + 2], dtype=np.float64)
    return AlignedSplit(keys, positions, values, visibility, orography)


def qsat_kgkg(t_k: np.ndarray, pressure_hpa: float = 925.0) -> np.ndarray:
    t_c = np.asarray(t_k, dtype=np.float64) - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    eps = 0.622
    return eps * es / np.maximum(float(pressure_hpa) - (1.0 - eps) * es, 1.0e-8)


def rh_from_tq(t_k: np.ndarray, q_kgkg: np.ndarray, pressure_hpa: float = 925.0) -> np.ndarray:
    q = np.asarray(q_kgkg, dtype=np.float64)
    e = q * float(pressure_hpa) / np.maximum(0.622 + (1.0 - 0.622) * q, 1.0e-8)
    t_c = np.asarray(t_k, dtype=np.float64) - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    return 100.0 * e / np.maximum(es, 1.0e-8)


def saturation_deficit_gkg(t_k: np.ndarray, q_kgkg: np.ndarray) -> np.ndarray:
    return (qsat_kgkg(t_k) - np.asarray(q_kgkg, dtype=np.float64)) * 1000.0


def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Cannot estimate robust scale from no finite values")
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1.0e-8:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1.0e-8:
        raise ValueError("Reference-analysis feature has zero robust scale")
    return scale


def continuous_metrics(forecast: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(forecast) & np.isfinite(reference)
    if not valid.any():
        return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan, "correlation": math.nan}
    f = np.asarray(forecast[valid], dtype=np.float64)
    r = np.asarray(reference[valid], dtype=np.float64)
    corr = float(np.corrcoef(f, r)[0, 1]) if len(f) > 1 and np.std(f) > 0 and np.std(r) > 0 else math.nan
    return {
        "n": int(len(f)),
        "bias": float(np.mean(f - r)),
        "mae": float(np.mean(np.abs(f - r))),
        "rmse": float(np.sqrt(np.mean((f - r) ** 2))),
        "correlation": corr,
    }


def binary_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(observed, dtype=bool)
    pred = np.asarray(predicted, dtype=bool)
    tp = float(np.sum(obs & pred))
    fp = float(np.sum(~obs & pred))
    fn = float(np.sum(obs & ~pred))
    tn = float(np.sum(~obs & ~pred))
    div = lambda a, b: float(a / b) if b else math.nan
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pod": div(tp, tp + fn),
        "far": div(fp, tp + fp),
        "pofd": div(fp, fp + tn),
        "csi": div(tp, tp + fp + fn),
        "frequency_bias": div(tp + fp, tp + fn),
    }


def rank_cdf(sorted_fit: np.ndarray, values: np.ndarray) -> np.ndarray:
    fit = np.asarray(sorted_fit, dtype=np.float64)
    fit = fit[np.isfinite(fit)]
    fit.sort()
    if len(fit) < 2:
        raise ValueError("Empirical CDF needs at least two finite fitting values")
    ranks = np.searchsorted(fit, np.asarray(values, dtype=np.float64), side="right")
    return np.clip((ranks + 0.5) / (len(fit) + 1.0), 1.0e-6, 1.0 - 1.0e-6)


def rff_parameters(reference: np.ndarray, dim: int, sample_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    valid = reference[np.all(np.isfinite(reference), axis=1)]
    if len(valid) < 10:
        raise ValueError("Too few reference copula rows for RFF bandwidth")
    sample = valid[rng.choice(len(valid), size=min(sample_n, len(valid)), replace=False)]
    diff = sample[:, None, :] - sample[None, :, :]
    distance = np.sqrt(np.sum(diff * diff, axis=-1))
    upper = distance[np.triu_indices(len(sample), k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0)]
    bandwidth = float(np.median(upper)) if len(upper) else 0.25
    bandwidth = max(bandwidth, 1.0e-3)
    omega = rng.normal(size=(reference.shape[1], dim)) / bandwidth
    phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    return omega.astype(np.float64), phase.astype(np.float64), bandwidth


def rff(values: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return (math.sqrt(2.0 / omega.shape[1]) * np.cos(values @ omega + phase)).astype(np.float64)


def rff_sum(values: np.ndarray, omega: np.ndarray, phase: np.ndarray, chunk_rows: int = 8192) -> np.ndarray:
    """Sum random Fourier features without materializing a station-day matrix."""
    total = np.zeros(omega.shape[1], dtype=np.float64)
    for start in range(0, len(values), chunk_rows):
        total += np.sum(rff(values[start : start + chunk_rows], omega, phase), axis=0)
    return total


def rbf_kernel_mean(left: np.ndarray, right: np.ndarray, bandwidth: float, chunk_rows: int = 256) -> float:
    """Blockwise mean RBF kernel value for an exact MMD subset audit."""
    total = 0.0
    count = 0
    denom = 2.0 * bandwidth * bandwidth
    for left_start in range(0, len(left), chunk_rows):
        left_block = left[left_start : left_start + chunk_rows]
        for right_start in range(0, len(right), chunk_rows):
            right_block = right[right_start : right_start + chunk_rows]
            distance2 = np.sum((left_block[:, None, :] - right_block[None, :, :]) ** 2, axis=2)
            total += float(np.exp(-distance2 / denom).sum())
            count += int(distance2.size)
    return total / max(count, 1)


def exact_biased_mmd2(source: np.ndarray, reference: np.ndarray, bandwidth: float) -> float:
    return float(
        rbf_kernel_mean(source, source, bandwidth)
        + rbf_kernel_mean(reference, reference, bandwidth)
        - 2.0 * rbf_kernel_mean(source, reference, bandwidth)
    )


def daily_rff_sums(
    source: np.ndarray,
    reference: np.ndarray,
    dates: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, codes = np.unique(dates, return_inverse=True)
    sums = np.zeros((len(unique), omega.shape[1]), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for day in range(len(unique)):
        rows = np.flatnonzero(codes == day)
        valid = np.all(np.isfinite(source[rows]), axis=1) & np.all(np.isfinite(reference[rows]), axis=1)
        rows = rows[valid]
        if not len(rows):
            continue
        sums[day] = rff_sum(source[rows], omega, phase) - rff_sum(reference[rows], omega, phase)
        counts[day] = len(rows)
    return unique, sums, counts


def mmd_from_daily(sums: np.ndarray, counts: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones(len(counts), dtype=np.int64)
    n = float(weights @ counts)
    if n <= 0:
        return math.nan
    mean = (weights @ sums) / n
    return float(mean @ mean)


def aggregate_daily(values: np.ndarray, dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, codes = np.unique(dates, return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for day in range(len(unique)):
        part = np.asarray(values[codes == day], dtype=np.float64)
        part = part[np.isfinite(part)]
        if len(part):
            sums[day] = float(np.sum(part))
            counts[day] = len(part)
    return unique, sums, counts


def bootstrap_pair_loss(
    pangu_loss: np.ndarray,
    tianji_loss: np.ndarray,
    dates: np.ndarray,
    iterations: int,
    seed: int,
    square_root: bool,
) -> Dict[str, float]:
    unique, p_sum, p_n = aggregate_daily(pangu_loss, dates)
    unique_t, t_sum, t_n = aggregate_daily(tianji_loss, dates)
    if not np.array_equal(unique, unique_t):
        raise RuntimeError("Daily loss aggregation differs across paired sources")
    rng = np.random.default_rng(seed)
    p_draw = np.empty(iterations, dtype=np.float64)
    t_draw = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(chosen, minlength=len(unique))
        p_value = float(weights @ p_sum) / max(float(weights @ p_n), 1.0)
        t_value = float(weights @ t_sum) / max(float(weights @ t_n), 1.0)
        p_draw[i] = math.sqrt(max(p_value, 0.0)) if square_root else p_value
        t_draw[i] = math.sqrt(max(t_value, 0.0)) if square_root else t_value
    delta = p_draw - t_draw
    p_point = float(np.nansum(pangu_loss) / max(np.isfinite(pangu_loss).sum(), 1))
    t_point = float(np.nansum(tianji_loss) / max(np.isfinite(tianji_loss).sum(), 1))
    if square_root:
        p_point, t_point = math.sqrt(max(p_point, 0.0)), math.sqrt(max(t_point, 0.0))
    return {
        "pangu": p_point,
        "tianji": t_point,
        "delta_pangu_minus_tianji": p_point - t_point,
        "delta_ci_low": float(np.quantile(delta, 0.025)),
        "delta_ci_high": float(np.quantile(delta, 0.975)),
        "pangu_ci_low": float(np.quantile(p_draw, 0.025)),
        "pangu_ci_high": float(np.quantile(p_draw, 0.975)),
        "tianji_ci_low": float(np.quantile(t_draw, 0.025)),
        "tianji_ci_high": float(np.quantile(t_draw, 0.975)),
    }


def bootstrap_mmd_pair(
    p_sums: np.ndarray,
    p_counts: np.ndarray,
    t_sums: np.ndarray,
    t_counts: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    if p_sums.shape != t_sums.shape or not np.array_equal(p_counts, t_counts):
        raise ValueError("Paired RFF day aggregates must have matching shape and counts")
    rng = np.random.default_rng(seed)
    p_draw = np.empty(iterations, dtype=np.float64)
    t_draw = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        chosen = rng.integers(0, len(p_counts), size=len(p_counts))
        weights = np.bincount(chosen, minlength=len(p_counts))
        p_draw[i] = mmd_from_daily(p_sums, p_counts, weights)
        t_draw[i] = mmd_from_daily(t_sums, t_counts, weights)
    delta = p_draw - t_draw
    p_point = mmd_from_daily(p_sums, p_counts)
    t_point = mmd_from_daily(t_sums, t_counts)
    return {
        "pangu": p_point,
        "tianji": t_point,
        "delta_pangu_minus_tianji": p_point - t_point,
        "delta_ci_low": float(np.quantile(delta, 0.025)),
        "delta_ci_high": float(np.quantile(delta, 0.975)),
        "pangu_ci_low": float(np.quantile(p_draw, 0.025)),
        "pangu_ci_high": float(np.quantile(p_draw, 0.975)),
        "tianji_ci_low": float(np.quantile(t_draw, 0.025)),
        "tianji_ci_high": float(np.quantile(t_draw, 0.975)),
    }


def valid_masks(split: AlignedSplit, low_vis_threshold_m: float = 1000.0) -> Dict[str, np.ndarray]:
    finite = np.ones(len(split.keys), dtype=bool)
    physical = np.ones(len(split.keys), dtype=bool)
    for source in SOURCES:
        t = split.values[source]["T_925"][:, -1]
        q = split.values[source]["Q_925"][:, -1]
        finite &= np.isfinite(t) & np.isfinite(q)
        physical &= np.isfinite(t) & (t >= 180.0) & (t <= 340.0)
        physical &= np.isfinite(q) & (q >= 0.0) & (q <= 0.08)
    return {
        "all_finite": finite,
        "union_physical": finite & physical,
        "true_low_visibility": finite & np.isfinite(split.visibility_m) & (split.visibility_m <= low_vis_threshold_m),
        "elevation_le_500m": finite & np.isfinite(split.orography_m) & (split.orography_m <= 500.0),
    }


def source_quality(
    fit: AlignedSplit,
    test: AlignedSplit,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    masks = valid_masks(test, args.low_vis_threshold_m)
    marginal_rows = []
    closure_rows = []
    temporal_rows = []
    tail_rows = []
    qc_rows = []
    fit_finite = valid_masks(fit, args.low_vis_threshold_m)["all_finite"]
    ref_fit_t = fit.values["era5_reference_analysis"]["T_925"][:, -1][fit_finite]
    ref_fit_q = fit.values["era5_reference_analysis"]["Q_925"][:, -1][fit_finite] * 1000.0
    scale_t, scale_q = robust_scale(ref_fit_t), robust_scale(ref_fit_q)
    fit_deficit = {
        source: saturation_deficit_gkg(
            fit.values[source]["T_925"][:, -1], fit.values[source]["Q_925"][:, -1]
        )
        for source in SOURCES
    }
    quantile = float(args.near_saturation_quantile)
    ref_threshold = float(np.nanquantile(fit_deficit["era5_reference_analysis"][fit_finite], quantile))
    source_thresholds = {
        source: float(np.nanquantile(fit_deficit[source][fit_finite], quantile))
        for source in FORECAST_SOURCES
    }

    for source in SOURCES:
        t = test.values[source]["T_925"][:, -1]
        q = test.values[source]["Q_925"][:, -1]
        rh = test.values[source]["RH_925"][:, -1]
        invalid = (~np.isfinite(t)) | (~np.isfinite(q)) | (t < 180.0) | (t > 340.0) | (q < 0.0) | (q > 0.08)
        qc_rows.append(
            {
                "source": source,
                "rows": int(len(t)),
                "invalid_or_nonphysical_tq_rows": int(invalid.sum()),
                "invalid_or_nonphysical_tq_fraction": float(np.mean(invalid)),
                "negative_q925_rows": int(np.sum(np.isfinite(q) & (q < 0.0))),
                "q925_min_gkg": float(np.nanmin(q) * 1000.0),
                "q925_max_gkg": float(np.nanmax(q) * 1000.0),
            }
        )
        closure = rh - rh_from_tq(t, q)
        closure_rows.append(
            {
                "source": source,
                "n": int(np.isfinite(closure).sum()),
                "closure_bias_rh_pct": float(np.nanmean(closure)),
                "closure_mae_rh_pct": float(np.nanmean(np.abs(closure))),
                "closure_rmse_rh_pct": float(np.sqrt(np.nanmean(closure * closure))),
                "fraction_abs_closure_gt_2pct": float(np.nanmean(np.abs(closure) > 2.0)),
                "role": "thermodynamic closure QC, not an independent skill metric",
            }
        )

    ref_current = {
        "T_925": test.values["era5_reference_analysis"]["T_925"][:, -1],
        "Q_925": test.values["era5_reference_analysis"]["Q_925"][:, -1] * 1000.0,
        "RH_925": test.values["era5_reference_analysis"]["RH_925"][:, -1],
    }
    deficit = {
        source: saturation_deficit_gkg(
            test.values[source]["T_925"][:, -1], test.values[source]["Q_925"][:, -1]
        )
        for source in SOURCES
    }
    for scope, mask in masks.items():
        for source in FORECAST_SOURCES:
            current = {
                "T_925": test.values[source]["T_925"][:, -1],
                "Q_925": test.values[source]["Q_925"][:, -1] * 1000.0,
                "RH_925": test.values[source]["RH_925"][:, -1],
            }
            for feature in FEATURES:
                marginal_rows.append(
                    {
                        "scope": scope,
                        "feature": feature,
                        "source": source,
                        "reference": "ERA5 reference analysis",
                        **continuous_metrics(current[feature][mask], ref_current[feature][mask]),
                    }
                )
            marginal_rows.append(
                {
                    "scope": scope,
                    "feature": "SATURATION_DEFICIT_925",
                    "source": source,
                    "reference": "ERA5 reference analysis",
                    **continuous_metrics(deficit[source][mask], deficit["era5_reference_analysis"][mask]),
                }
            )
            err_t = (current["T_925"] - ref_current["T_925"]) / scale_t
            err_q = (current["Q_925"] - ref_current["Q_925"]) / scale_q
            joint_loss = err_t * err_t + err_q * err_q
            marginal_rows.append(
                {
                    "scope": scope,
                    "feature": "T925_Q925_STANDARDIZED_VECTOR",
                    "source": source,
                    "reference": "ERA5 reference analysis",
                    "n": int(np.isfinite(joint_loss[mask]).sum()),
                    "bias": math.nan,
                    "mae": float(np.nanmean(np.sqrt(joint_loss[mask]))),
                    "rmse": float(np.sqrt(np.nanmean(joint_loss[mask]))),
                    "correlation": math.nan,
                }
            )

    ref_obs = deficit["era5_reference_analysis"] <= ref_threshold
    for source in FORECAST_SOURCES:
        for method, predicted, threshold in (
            ("exact_reference_threshold", deficit[source] <= ref_threshold, ref_threshold),
            ("quantile_matched", deficit[source] <= source_thresholds[source], source_thresholds[source]),
        ):
            valid = masks["all_finite"]
            tail_rows.append(
                {
                    "source": source,
                    "method": method,
                    "quantile": quantile,
                    "reference_threshold_gkg": ref_threshold,
                    "source_threshold_gkg": threshold,
                    **binary_metrics(ref_obs[valid], predicted[valid]),
                }
            )

    for source in FORECAST_SOURCES:
        t_seq = test.values[source]["T_925"]
        q_seq = test.values[source]["Q_925"] * 1000.0
        rt = test.values["era5_reference_analysis"]["T_925"]
        rq = test.values["era5_reference_analysis"]["Q_925"] * 1000.0
        dt, dq = t_seq[:, -1] - t_seq[:, -4], q_seq[:, -1] - q_seq[:, -4]
        rdt, rdq = rt[:, -1] - rt[:, -4], rq[:, -1] - rq[:, -4]
        loss = ((dt - rdt) / scale_t) ** 2 + ((dq - rdq) / scale_q) ** 2
        source_rank_corr = float(pd.Series(dt).rank().corr(pd.Series(dq).rank()))
        ref_rank_corr = float(pd.Series(rdt).rank().corr(pd.Series(rdq).rank()))
        deficit_seq = saturation_deficit_gkg(t_seq, test.values[source]["Q_925"])
        ref_deficit_seq = saturation_deficit_gkg(rt, test.values["era5_reference_analysis"]["Q_925"])
        temporal_rows.append(
            {
                "source": source,
                "n": int(np.isfinite(loss).sum()),
                "three_hour_tendency_vector_rmse": float(np.sqrt(np.nanmean(loss))),
                "tendency_rank_correlation": source_rank_corr,
                "reference_tendency_rank_correlation": ref_rank_corr,
                "absolute_rank_correlation_error": abs(source_rank_corr - ref_rank_corr),
                "saturation_deficit_12h_rmse_gkg": float(np.sqrt(np.nanmean((deficit_seq - ref_deficit_seq) ** 2))),
            }
        )

    details = {
        "reference_scales": {"T925_K": scale_t, "Q925_gkg": scale_q},
        "near_saturation_quantile": quantile,
        "reference_threshold_gkg": ref_threshold,
        "source_quantile_thresholds_gkg": source_thresholds,
        "scope_rows": {scope: int(mask.sum()) for scope, mask in masks.items()},
    }
    return (
        pd.DataFrame(marginal_rows),
        pd.DataFrame(closure_rows),
        pd.DataFrame(temporal_rows),
        pd.DataFrame(tail_rows),
        pd.DataFrame(qc_rows),
        details,
    )


def dependence_and_bootstrap(
    fit: AlignedSplit,
    test: AlignedSplit,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    fit_mask = valid_masks(fit, args.low_vis_threshold_m)["all_finite"]
    test_mask = valid_masks(test, args.low_vis_threshold_m)["all_finite"]
    dates = test.keys.loc[test_mask, "time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    fit_cdfs: Dict[str, Dict[str, np.ndarray]] = {}
    test_copula: Dict[str, np.ndarray] = {}
    for source in SOURCES:
        fit_t = fit.values[source]["T_925"][:, -1][fit_mask]
        fit_q = fit.values[source]["Q_925"][:, -1][fit_mask]
        fit_cdfs[source] = {"T_925": np.sort(fit_t[np.isfinite(fit_t)]), "Q_925": np.sort(fit_q[np.isfinite(fit_q)])}
        test_t = test.values[source]["T_925"][:, -1][test_mask]
        test_q = test.values[source]["Q_925"][:, -1][test_mask]
        test_copula[source] = np.column_stack(
            [rank_cdf(fit_cdfs[source]["T_925"], test_t), rank_cdf(fit_cdfs[source]["Q_925"], test_q)]
        )
    ref_fit_copula = np.column_stack(
        [
            rank_cdf(fit_cdfs["era5_reference_analysis"]["T_925"], fit.values["era5_reference_analysis"]["T_925"][:, -1][fit_mask]),
            rank_cdf(fit_cdfs["era5_reference_analysis"]["Q_925"], fit.values["era5_reference_analysis"]["Q_925"][:, -1][fit_mask]),
        ]
    )
    omega, phase, bandwidth = rff_parameters(
        ref_fit_copula, args.rff_dim, args.rff_bandwidth_sample, args.bootstrap_seed
    )
    ref = test_copula["era5_reference_analysis"]
    audit_rng = np.random.default_rng(args.bootstrap_seed + 17)
    audit_n = min(int(args.rff_bandwidth_sample), len(ref))
    audit_rows = np.sort(audit_rng.choice(len(ref), size=audit_n, replace=False))
    ref_audit = ref[audit_rows]
    daily = {}
    dependence_rows = []
    ref_corr = float(np.corrcoef(ref[:, 0], ref[:, 1])[0, 1])
    ref_cov = np.cov(ref.T)
    for source in FORECAST_SOURCES:
        unique, sums, counts = daily_rff_sums(test_copula[source], ref, dates, omega, phase)
        daily[source] = (unique, sums, counts)
        corr = float(np.corrcoef(test_copula[source][:, 0], test_copula[source][:, 1])[0, 1])
        cov_error = float(np.linalg.norm(np.cov(test_copula[source].T) - ref_cov, ord="fro"))
        source_audit = test_copula[source][audit_rows]
        exact_subset = exact_biased_mmd2(source_audit, ref_audit, bandwidth)
        rff_subset = float(
            np.sum((np.mean(rff(source_audit, omega, phase), axis=0) - np.mean(rff(ref_audit, omega, phase), axis=0)) ** 2)
        )
        dependence_rows.append(
            {
                "source": source,
                "n": int(len(ref)),
                "copula_mmd2_rff": mmd_from_daily(sums, counts),
                "copula_rank_correlation": corr,
                "reference_rank_correlation": ref_corr,
                "absolute_rank_correlation_error": abs(corr - ref_corr),
                "copula_covariance_frobenius_error": cov_error,
                "rff_dim": args.rff_dim,
                "rbf_bandwidth": bandwidth,
                "exact_subset_mmd2": exact_subset,
                "rff_subset_mmd2": rff_subset,
                "rff_subset_absolute_error": abs(rff_subset - exact_subset),
                "exact_subset_n": audit_n,
                "exact_subset_seed": args.bootstrap_seed + 17,
                "fit_policy": "source-specific train empirical marginals; paired test copula comparison",
            }
        )
    if not np.array_equal(daily["pangu"][0], daily["tianji"][0]):
        raise RuntimeError("Pangu and Tianji copula day aggregates differ")
    mmd_boot = bootstrap_mmd_pair(
        daily["pangu"][1], daily["pangu"][2], daily["tianji"][1], daily["tianji"][2],
        args.bootstrap_iters, args.bootstrap_seed,
    )
    bootstrap_rows = [{"metric": "copula_mmd2_rff", **mmd_boot, "bootstrap_unit": "UTC_valid_date"}]

    ref_t = test.values["era5_reference_analysis"]["T_925"][:, -1][test_mask]
    ref_q = test.values["era5_reference_analysis"]["Q_925"][:, -1][test_mask] * 1000.0
    scale_t, scale_q = robust_scale(ref_t), robust_scale(ref_q)
    losses: Dict[str, Dict[str, np.ndarray]] = {source: {} for source in FORECAST_SOURCES}
    ref_deficit = saturation_deficit_gkg(
        test.values["era5_reference_analysis"]["T_925"][:, -1][test_mask],
        test.values["era5_reference_analysis"]["Q_925"][:, -1][test_mask],
    )
    for source in FORECAST_SOURCES:
        t = test.values[source]["T_925"][:, -1][test_mask]
        q = test.values[source]["Q_925"][:, -1][test_mask] * 1000.0
        source_deficit = saturation_deficit_gkg(
            test.values[source]["T_925"][:, -1][test_mask], test.values[source]["Q_925"][:, -1][test_mask]
        )
        losses[source]["saturation_deficit_rmse_gkg"] = (source_deficit - ref_deficit) ** 2
        losses[source]["standardized_joint_vector_rmse"] = ((t - ref_t) / scale_t) ** 2 + ((q - ref_q) / scale_q) ** 2
        losses[source]["t925_rmse_k"] = (t - ref_t) ** 2
        losses[source]["q925_rmse_gkg"] = (q - ref_q) ** 2
    for index, metric in enumerate(losses["pangu"]):
        result = bootstrap_pair_loss(
            losses["pangu"][metric], losses["tianji"][metric], dates,
            args.bootstrap_iters, args.bootstrap_seed + index + 1, square_root=True,
        )
        bootstrap_rows.append({"metric": metric, **result, "bootstrap_unit": "UTC_valid_date"})
    return pd.DataFrame(dependence_rows), pd.DataFrame(bootstrap_rows), losses


def performance_evidence(analysis_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    metrics = pd.read_csv(analysis_dir / "hybrid_factorial_metrics_by_seed.csv")
    interactions = pd.read_csv(analysis_dir / "hybrid_second_order_interactions_by_seed.csv")
    ci = pd.read_csv(analysis_dir / "hybrid_date_block_bootstrap_ci.csv")
    gap_draws = pd.read_csv(analysis_dir / "hybrid_total_gap_bootstrap_draws.csv.gz")
    rows = []
    for seed, part in metrics.groupby("seed"):
        values = dict(zip(part["mask"].astype(str).str.zfill(3), part["low_vis_ap"].astype(float)))
        for background_bit, background_label in (("0", "Pangu background"), ("1", "Tianji background")):
            v00 = values[f"00{background_bit}"]
            v01 = values[f"01{background_bit}"]
            v10 = values[f"10{background_bit}"]
            v11 = values[f"11{background_bit}"]
            interaction = v11 - v10 - v01 + v00
            rows.append(
                {
                    "seed": int(seed),
                    "background": background_label,
                    "background_bit": background_bit,
                    "metric": "low_vis_ap",
                    "m_t925_interaction": interaction,
                    "coherent_minus_cross_source_mean": 0.5 * interaction,
                    "formula": "0.5 * [(M_T,H_T)+(M_P,H_P)-(M_T,H_P)-(M_P,H_T)]",
                }
            )
    pair = interactions[(interactions["metric"] == "low_vis_ap") & (interactions["pair"] == "M:H")]
    pair_ci = ci[(ci["metric"] == "low_vis_ap") & (ci["effect"] == "interaction") & (ci["term"] == "M:H")]
    if pair.empty or pair_ci.empty:
        raise RuntimeError("Factorial analysis lacks the primary Low-vis AP M:H interaction")
    gap = gap_draws[gap_draws["metric"] == "low_vis_ap"]["delta_all1_minus_all0"].to_numpy(dtype=float)
    values = pair["interaction"].to_numpy(dtype=float)
    evidence = {
        "interaction_mean": float(values.mean()),
        "interaction_seed_values": [float(value) for value in values],
        "interaction_ci": [float(pair_ci.iloc[0]["ci_low"]), float(pair_ci.iloc[0]["ci_high"])],
        "interaction_all_seeds_positive": bool(np.all(values > 0.0)),
        "endpoint_gap_ci": [float(np.quantile(gap, 0.025)), float(np.quantile(gap, 0.975))],
        "endpoint_gap_mean": float(np.mean(gap)),
    }
    return pd.DataFrame(rows), evidence


def event_linkage(
    analysis_dir: Path,
    test: AlignedSplit,
    losses: Mapping[str, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    event_path = analysis_dir / "event_case_control_samples.csv.gz"
    if not event_path.is_file():
        return pd.DataFrame(), {"enabled": False, "reason": "event case-control file missing"}
    events = pd.read_csv(event_path)
    if "category" not in events and "case_category" in events:
        events = events.rename(columns={"case_category": "category"})
    required = {"time_utc", "station_key", "category"}
    if not required.issubset(events):
        return pd.DataFrame(), {"enabled": False, "reason": f"event table lacks {sorted(required - set(events))}"}
    sample = test.keys.copy()
    sample["time_key"] = sample["time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sample["station_key"] = normalize_station(sample["station_key"])
    finite = valid_masks(test, args.low_vis_threshold_m)["all_finite"]
    sample = sample.loc[finite, ["time_key", "station_key", "time_utc"]].reset_index(drop=True)
    sample["pangu_joint_error"] = np.sqrt(losses["pangu"]["standardized_joint_vector_rmse"])
    sample["tianji_joint_error"] = np.sqrt(losses["tianji"]["standardized_joint_vector_rmse"])
    sample["pangu_saturation_abs_error_gkg"] = np.sqrt(losses["pangu"]["saturation_deficit_rmse_gkg"])
    sample["tianji_saturation_abs_error_gkg"] = np.sqrt(losses["tianji"]["saturation_deficit_rmse_gkg"])
    events["time_key"] = pd.to_datetime(events["time_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["station_key"] = normalize_station(events["station_key"])
    merged = events[["time_key", "station_key", "category"]].merge(
        sample, on=["time_key", "station_key"], how="inner", validate="one_to_one"
    )
    input_target_rows = int((events["category"] == "tianji_hit_pangu_miss").sum())
    rows = []
    for category, part in merged.groupby("category"):
        rows.append(
            {
                "category": category,
                "n": int(len(part)),
                "pangu_joint_error_mean": float(part["pangu_joint_error"].mean()),
                "tianji_joint_error_mean": float(part["tianji_joint_error"].mean()),
                "joint_error_delta_pangu_minus_tianji": float((part["pangu_joint_error"] - part["tianji_joint_error"]).mean()),
                "pangu_saturation_abs_error_gkg_mean": float(part["pangu_saturation_abs_error_gkg"].mean()),
                "tianji_saturation_abs_error_gkg_mean": float(part["tianji_saturation_abs_error_gkg"].mean()),
                "saturation_error_delta_pangu_minus_tianji": float(
                    (part["pangu_saturation_abs_error_gkg"] - part["tianji_saturation_abs_error_gkg"]).mean()
                ),
            }
        )
    target = merged[merged["category"] == "tianji_hit_pangu_miss"].copy()
    target_coverage = float(len(target) / max(input_target_rows, 1))
    if target.empty:
        return pd.DataFrame(rows), {
            "enabled": True,
            "input_event_rows": int(len(events)),
            "matched_rows": int(len(merged)),
            "input_target_category_rows": input_target_rows,
            "target_category_rows": 0,
            "target_category_coverage": target_coverage,
        }
    dates = pd.to_datetime(target["time_utc"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    delta = target["pangu_joint_error"].to_numpy() - target["tianji_joint_error"].to_numpy()
    unique, sums, counts = aggregate_daily(delta, dates)
    rng = np.random.default_rng(args.bootstrap_seed + 99)
    draws = np.empty(args.bootstrap_iters, dtype=np.float64)
    for i in range(args.bootstrap_iters):
        chosen = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(chosen, minlength=len(unique))
        draws[i] = float(weights @ sums) / max(float(weights @ counts), 1.0)
    info = {
        "enabled": True,
        "input_event_rows": int(len(events)),
        "matched_rows": int(len(merged)),
        "input_target_category_rows": input_target_rows,
        "target_category_rows": int(len(target)),
        "target_category_coverage": target_coverage,
        "target_joint_error_delta": float(np.mean(delta)),
        "target_joint_error_delta_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_unit": "UTC_valid_date",
    }
    return pd.DataFrame(rows), info


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_key_sha256(keys: pd.DataFrame) -> str:
    payload = "\n".join(
        keys["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ") + "|" + normalize_station(keys["station_key"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_figures(
    bootstrap: pd.DataFrame,
    performance: Mapping[str, object] | None,
    event_info: Mapping[str, object],
    out_dir: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )

    def bar_metric(metric: str, ylabel: str, stem: str) -> None:
        row = bootstrap[bootstrap["metric"] == metric].iloc[0]
        means = [float(row["pangu"]), float(row["tianji"])]
        lows = [float(row["pangu_ci_low"]), float(row["tianji_ci_low"])]
        highs = [float(row["pangu_ci_high"]), float(row["tianji_ci_high"])]
        err = np.vstack([np.maximum(0.0, np.asarray(means) - lows), np.maximum(0.0, highs - np.asarray(means))])
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.bar([0, 1], means, yerr=err, capsize=6, color=[COLORS["pangu"], COLORS["tianji"]], width=0.62)
        ax.set_xticks([0, 1], ["Pangu", "Tianji"])
        ax.set_ylabel(ylabel)
        ax.set_title("Error relative to ERA5 reference analysis")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
        for index, value in enumerate(means):
            ax.annotate(
                f"{value:.3g}",
                (index, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
        save_figure(fig, out_dir, stem)
        plt.close(fig)

    bar_metric("saturation_deficit_rmse_gkg", "925-hPa saturation-deficit RMSE (g kg$^{-1}$)", "fig_joint_saturation_deficit_quality")
    bar_metric("copula_mmd2_rff", "Empirical-copula MMD$^2$", "fig_t925_q925_copula_quality")

    if performance is not None:
        ci = performance["interaction_ci"]
        mean = float(performance["interaction_mean"])
        seed_values = np.asarray(performance["interaction_seed_values"], dtype=float)
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.axhline(0.0, color="#555555", linewidth=1.2)
        ax.errorbar([0], [mean], yerr=[[max(0.0, mean - float(ci[0]))], [max(0.0, float(ci[1]) - mean)]], fmt="o", color="#D95F02", markersize=11, capsize=7, linewidth=2.2)
        offsets = np.linspace(-0.10, 0.10, len(seed_values))
        ax.scatter(offsets, seed_values, color="#7F2704", s=55, zorder=3, label="Training seeds")
        ax.set_xlim(-0.45, 0.45)
        ax.set_xticks([0], ["M × T925"])
        ax.set_ylabel("Low-visibility AP interaction")
        ax.set_title("Retrained source-block complementarity")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
        save_figure(fig, out_dir, "fig_m_t925_performance_interaction")
        plt.close(fig)

    if event_info.get("enabled") and int(event_info.get("target_category_rows", 0)) > 0:
        mean = float(event_info["target_joint_error_delta"])
        ci = [float(value) for value in event_info["target_joint_error_delta_ci"]]
        err = [[max(0.0, mean - ci[0])], [max(0.0, ci[1] - mean)]]
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.axhline(0.0, color="#555555", linewidth=1.2)
        ax.errorbar([0], [mean], yerr=err, fmt="o", color=COLORS["pangu"], markersize=12, capsize=7, linewidth=2.3)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0], ["Tianji hit / Pangu miss"])
        ax.set_ylabel("Pangu − Tianji standardized joint error")
        ax.set_title("Joint-state error in missed low-visibility cases")
        ax.annotate(
            f"n = {int(event_info['target_category_rows']):,}",
            (0, mean),
            xytext=(18, 8),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
        save_figure(fig, out_dir, "fig_event_joint_state_error")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters < 1:
        raise ValueError("--bootstrap-iters must be positive")
    if args.rff_dim < 8:
        raise ValueError("--rff-dim must be at least 8")
    if args.rff_bandwidth_sample < 10:
        raise ValueError("--rff-bandwidth-sample must be at least 10")
    if not 0.0 < args.near_saturation_quantile < 1.0:
        raise ValueError("--near-saturation-quantile must lie strictly between 0 and 1")
    if not 0.0 < args.min_event_coverage <= 1.0:
        raise ValueError("--min-event-coverage must lie in (0, 1]")
    if args.fit_max_rows < 0 or args.test_max_rows < 0:
        raise ValueError("row caps must be non-negative; use 0 for no cap")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layouts = {
        "pangu": load_layout(Path(args.pangu_dir).expanduser().resolve(), "pangu"),
        "tianji": load_layout(Path(args.tianji_dir).expanduser().resolve(), "tianji"),
        "era5_reference_analysis": load_layout(
            Path(args.era5_dir).expanduser().resolve(), "era5_reference_analysis"
        ),
    }
    fit = align_split(layouts, "train", args.fit_max_rows, args.bootstrap_seed)
    test = align_split(layouts, "test", args.test_max_rows, args.bootstrap_seed + 1)
    marginal, closure, temporal, tails, qc, quality_info = source_quality(fit, test, args)
    dependence, bootstrap, losses = dependence_and_bootstrap(fit, test, args)
    performance: Dict[str, object] | None = None
    coherence = pd.DataFrame()
    if args.analysis_mode == "factorial_confirmatory":
        if not args.factorial_analysis_dir:
            raise ValueError("--factorial-analysis-dir is required for factorial_confirmatory mode")
        factorial_dir = Path(args.factorial_analysis_dir).expanduser().resolve()
        coherence, performance = performance_evidence(factorial_dir)
    event_dir_text = args.event_analysis_dir or args.factorial_analysis_dir
    if not event_dir_text:
        raise ValueError("--event-analysis-dir is required so existing MTW/mt2pw hit/miss cases can be reused")
    event_dir = Path(event_dir_text).expanduser().resolve()
    events, event_info = event_linkage(event_dir, test, losses, args)

    marginal.to_csv(out_dir / "joint_structure_marginal_and_vector_quality.csv", index=False)
    closure.to_csv(out_dir / "joint_structure_thermodynamic_closure_qc.csv", index=False)
    temporal.to_csv(out_dir / "joint_structure_temporal_quality.csv", index=False)
    tails.to_csv(out_dir / "joint_saturation_extreme_placement.csv", index=False)
    qc.to_csv(out_dir / "joint_structure_source_qc.csv", index=False)
    dependence.to_csv(out_dir / "t925_q925_empirical_copula_quality.csv", index=False)
    bootstrap.to_csv(out_dir / "joint_structure_utc_date_bootstrap_ci.csv", index=False)
    if not coherence.empty:
        coherence.to_csv(out_dir / "m_t925_coherence_contrasts_by_seed.csv", index=False)
    if not events.empty:
        events.to_csv(out_dir / "joint_structure_event_case_summary.csv", index=False)

    sat = bootstrap[bootstrap["metric"] == "saturation_deficit_rmse_gkg"].iloc[0]
    copula = bootstrap[bootstrap["metric"] == "copula_mmd2_rff"].iloc[0]
    dependence_by_source = dependence.set_index("source")
    exact_subset_rank_stable = bool(
        float(dependence_by_source.loc["pangu", "exact_subset_mmd2"])
        > float(dependence_by_source.loc["tianji", "exact_subset_mmd2"])
    )
    quality_supported = bool(
        float(sat["delta_ci_low"]) > 0.0
        and float(copula["delta_ci_low"]) > 0.0
        and exact_subset_rank_stable
    )
    performance_supported = None
    if performance is not None:
        performance_supported = bool(
            float(performance["interaction_ci"][0]) > 0.0
            and bool(performance["interaction_all_seeds_positive"])
            and float(performance["endpoint_gap_ci"][0]) > 0.0
        )
    event_supported = bool(
        event_info.get("enabled")
        and int(event_info.get("target_category_rows", 0)) > 0
        and float(event_info.get("target_category_coverage", 0.0)) >= args.min_event_coverage
        and float(event_info.get("target_joint_error_delta_ci", [math.nan, math.nan])[0]) > 0.0
    )
    diagnostic_supported = quality_supported and event_supported
    all_supported = diagnostic_supported and bool(performance_supported)
    if args.analysis_mode == "diagnostic_only":
        gate_status = "joint_structure_diagnostic_supported" if diagnostic_supported else "joint_structure_diagnostic_not_supported"
    else:
        gate_status = "joint_structure_supported" if all_supported else "joint_structure_not_fully_supported"
    gate = {
        "status": gate_status,
        "analysis_mode": args.analysis_mode,
        "new_training_models_used": 0 if args.analysis_mode == "diagnostic_only" else 27,
        "quality_against_reference_analysis_supported": quality_supported,
        "copula_rff_ranking_confirmed_by_exact_subset": exact_subset_rank_stable,
        "predictive_m_t925_interaction_supported": performance_supported,
        "tianji_hit_pangu_miss_linkage_supported": event_supported,
        "performance": performance,
        "event_linkage": event_info,
        "minimum_event_coverage": args.min_event_coverage,
        "primary_quality_deltas": {
            "saturation_deficit_rmse_pangu_minus_tianji": float(sat["delta_pangu_minus_tianji"]),
            "saturation_deficit_rmse_delta_ci": [float(sat["delta_ci_low"]), float(sat["delta_ci_high"])],
            "copula_mmd2_pangu_minus_tianji": float(copula["delta_pangu_minus_tianji"]),
            "copula_mmd2_delta_ci": [float(copula["delta_ci_low"]), float(copula["delta_ci_high"])],
        },
        "claim_if_supported": (
            "Within Pangu-2025, 2025, 12--23 h lead and this low-visibility task, Pangu has a larger "
            "low-level T925--moisture joint-structure discrepancy than Tianji, and that discrepancy is "
            "associated with existing Tianji-hit/Pangu-miss cases."
            if args.analysis_mode == "diagnostic_only"
            else "Within the same scope, the combined quality, event and retrained-interaction evidence supports "
            "a source-dependent low-level T925--moisture joint-structure contribution beyond marginal errors."
        ),
        "claim_limit": (
            "Diagnostic-only results are associative mechanism evidence, not a direct T925 intervention. The experiment "
            "does not prove violation of atmospheric governing equations and must not be generalized to all AI models."
            if args.analysis_mode == "diagnostic_only"
            else "The experiment does not prove violation of atmospheric governing equations and must not be generalized "
            "to all AI weather models. Factorial interactions can also reflect representation and distribution shift."
        ),
    }
    report = {
        "status": "completed",
        "analysis_mode": args.analysis_mode,
        "new_training_models_used": 0 if args.analysis_mode == "diagnostic_only" else 27,
        "era5_role": "reference analysis, not truth",
        "source_feature_sets": {source: layout.feature_set for source, layout in layouts.items()},
        "event_analysis_dir": str(event_dir),
        "factorial_profile": "m925b" if performance is not None else None,
        "optional_factorial_groups": {
            "M": ["Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"],
            "H": ["T_925"],
            "B": ["T2M", "MSLP", "U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925"],
        },
        "fit_rows": int(len(fit.keys)),
        "test_rows": int(len(test.keys)),
        "bootstrap": {"iterations": args.bootstrap_iters, "seed": args.bootstrap_seed, "unit": "UTC_valid_date"},
        "mmd": {
            "method": "Gaussian-kernel MMD on source-specific train empirical-copula transforms",
            "approximation": "fixed random Fourier features",
            "rff_dim": args.rff_dim,
            "exact_subset_audit_rows": int(dependence["exact_subset_n"].iloc[0]),
            "exact_subset_ranking_confirmed": exact_subset_rank_stable,
        },
        "provenance": {
            source: {
                "dataset_dir": str(layout.path),
                "dataset_config_sha256": sha256_file(layout.path / "dataset_build_config.json"),
            }
            for source, layout in layouts.items()
        },
        "existing_event_artifact": {
            "path": str(event_dir / "event_case_control_samples.csv.gz"),
            "sha256": (
                sha256_file(event_dir / "event_case_control_samples.csv.gz")
                if (event_dir / "event_case_control_samples.csv.gz").is_file()
                else None
            ),
            "role": "reused completed MTW/mt2pw hit-miss classification; no new model inference",
        },
        "paired_alignment": {
            "fit_key_sha256": paired_key_sha256(fit.keys),
            "test_key_sha256": paired_key_sha256(test.keys),
        },
        "quality": quality_info,
        "figure_contract": {
            "backend": "Python/matplotlib only",
            "one_claim_per_figure": True,
            "formats": ["PNG", "SVG", "PDF", "TIFF"],
            "core_conclusions": [
                "joint saturation-state accuracy versus ERA5 reference analysis",
                "T925-Q925 empirical-copula discrepancy",
                "joint-state error in asymmetric low-visibility outcomes",
            ] + (["retrained M by T925 predictive interaction"] if performance is not None else []),
        },
        "method_references": [
            "Gretton et al. 2012 JMLR 13:723-773",
            "Rahimi and Recht 2007 NeurIPS 20",
            "Hamill 1999 Weather and Forecasting 14:155-167",
            "Bolton 1980 Monthly Weather Review 108:1046-1053",
            "Wilks 2015 QJRMS 141:945-952",
        ],
    }
    with (out_dir / "joint_structure_evidence_gate.json").open("w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)
    with (out_dir / "joint_structure_analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not args.no_figures:
        make_figures(bootstrap, performance, event_info, out_dir)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
