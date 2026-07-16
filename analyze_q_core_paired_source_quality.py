#!/usr/bin/env python3
"""Paired pressure-level and surface source-quality diagnosis with no training.

The analysis uses one common ``(valid_time, station_id)`` intersection for
Pangu, Tianji and ERA5.  Pressure-level variables are verified point by point
against ERA5 reference analysis.  Surface T2M, WSPD10 and MSLP are verified
against automatic-station observations, with ERA5 included as an analysis
benchmark rather than as a third forecast.

All uncertainty intervals resample UTC valid dates jointly, preserving the
spatial dependence among stations on the same weather day.  The main analysis
keeps every finite paired value.  Broad physical-range violations are reported
separately and are never silently removed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import analyze_q_core_t925_joint_structure as joint


SOURCES = ("pangu", "tianji", "era5_reference_analysis")
FORECAST_SOURCES = ("pangu", "tianji")

# These are the pressure-level fields available in the completed
# q_core_t925_no_rh2m diagnostic datasets.  T_925 is diagnosis-only; the other
# fields are present in the original q_core_no_rh2m input layout.
PRESSURE_LEVEL_FEATURES = (
    "T_925",
    "RH_925",
    "U_925",
    "V_925",
    "WSPD925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
)
SURFACE_OBS_FEATURES = ("T2M", "WSPD10", "MSLP")
REQUIRED_FEATURES = PRESSURE_LEVEL_FEATURES + SURFACE_OBS_FEATURES

ORIGINAL_Q_CORE_INPUTS = {
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
}

FEATURE_INFO: Dict[str, Dict[str, object]] = {
    "T_925": {
        "unit": "K",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (180.0, 340.0),
        "family": "thermal",
        "lineage": "native source field; diagnosis-only, absent from original q-core input",
        "independent_evidence": True,
    },
    "RH_925": {
        "unit": "%",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (0.0, 110.0),
        "family": "moisture",
        "lineage": "asymmetric: derived from T925/Q925 for Pangu; source lineage differs across products",
        "independent_evidence": False,
    },
    "U_925": {
        "unit": "m s-1",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (-150.0, 150.0),
        "family": "horizontal_wind",
        "lineage": "native horizontal wind component",
        "independent_evidence": True,
    },
    "V_925": {
        "unit": "m s-1",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (-150.0, 150.0),
        "family": "horizontal_wind",
        "lineage": "native horizontal wind component",
        "independent_evidence": True,
    },
    "WSPD925": {
        "unit": "m s-1",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (0.0, 150.0),
        "family": "horizontal_wind",
        "lineage": "derived from U925/V925; not independent of the component errors",
        "independent_evidence": False,
    },
    "DP_1000": {
        "unit": "K",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (180.0, 340.0),
        "family": "moisture",
        "lineage": "derived from Q1000 at fixed pressure; not independent of Q1000",
        "independent_evidence": False,
    },
    "DP_925": {
        "unit": "K",
        "scale": 1.0,
        "offset": 0.0,
        "broad_range": (180.0, 340.0),
        "family": "moisture",
        "lineage": "derived from Q925 at fixed pressure; not independent of Q925",
        "independent_evidence": False,
    },
    "Q_1000": {
        "unit": "g kg-1",
        "scale": 1000.0,
        "offset": 0.0,
        "broad_range": (0.0, 80.0),
        "family": "moisture",
        "lineage": "native forecast-source specific humidity; ERA5 derivation is recorded in dataset provenance",
        "independent_evidence": True,
    },
    "Q_925": {
        "unit": "g kg-1",
        "scale": 1000.0,
        "offset": 0.0,
        "broad_range": (0.0, 80.0),
        "family": "moisture",
        "lineage": "native Pangu/Tianji specific humidity; ERA5 may be derived from T925/RH925",
        "independent_evidence": True,
    },
}

SURFACE_INFO = {
    "T2M": {"observation_column": "tem", "unit": "degC", "scale": 1.0, "offset": -273.15},
    "WSPD10": {"observation_column": "win_s_avg_10mi", "unit": "m s-1", "scale": 1.0, "offset": 0.0},
    "MSLP": {"observation_column": "prs_sea", "unit": "hPa", "scale": 0.01, "offset": 0.0},
}

SCOPES = ("all_paired_test", "true_low_visibility", "elevation_le_500m")


@dataclass
class QualitySplit:
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
    ap.add_argument("--obs-root", required=True)
    ap.add_argument("--paper-eval-dir", default="/public/home/putianshu/vis_mlp/paper_eval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-max-rows", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260715)
    ap.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    return ap.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_feature(feature: str, values: np.ndarray) -> np.ndarray:
    info = FEATURE_INFO[feature]
    return np.asarray(values, dtype=np.float64) * float(info["scale"]) + float(info["offset"])


def convert_surface(feature: str, values: np.ndarray) -> np.ndarray:
    info = SURFACE_INFO[feature]
    arr = np.asarray(values, dtype=np.float64)
    if feature == "T2M":
        finite = arr[np.isfinite(arr)]
        offset = float(info["offset"]) if finite.size and float(np.nanmedian(finite)) > 150.0 else 0.0
        return arr + offset
    if feature == "MSLP":
        finite = arr[np.isfinite(arr)]
        scale = float(info["scale"]) if finite.size and float(np.nanmedian(np.abs(finite))) > 2000.0 else 1.0
        return arr * scale
    return arr


def extract_current(
    layout: joint.DatasetLayout,
    split: str,
    rows: np.ndarray,
    features: Sequence[str],
    chunk_rows: int = 50000,
) -> Dict[str, np.ndarray]:
    missing = [feature for feature in features if feature not in layout.order]
    if missing:
        raise ValueError(f"{layout.path}: required quality features are missing: {missing}")
    x = np.load(layout.path / f"X_{split}.npy", mmap_mode="r")
    expected = layout.window * layout.dyn_vars + layout.fe_dim + 6
    if int(x.shape[1]) != expected:
        raise ValueError(f"{layout.path}/{split}: X columns={x.shape[1]} != expected {expected}")
    result = {feature: np.empty(len(rows), dtype=np.float64) for feature in features}
    current = (layout.window - 1) * layout.dyn_vars
    columns = {feature: current + layout.order.index(feature) for feature in features}
    for start in range(0, len(rows), chunk_rows):
        end = min(start + chunk_rows, len(rows))
        block = np.asarray(x[rows[start:end]], dtype=np.float64)
        for feature, column in columns.items():
            result[feature][start:end] = block[:, column]
    return result


def align_test(
    layouts: Mapping[str, joint.DatasetLayout],
    max_rows: int,
    seed: int,
) -> QualitySplit:
    frames = {source: joint.metadata(layout.path, "test") for source, layout in layouts.items()}
    reference = frames["pangu"]
    common = joint.key_index(reference)
    for source in SOURCES[1:]:
        common = common[common.isin(joint.key_index(frames[source]))]
    if len(common) == 0:
        raise RuntimeError("test: no common Pangu/Tianji/ERA5 rows")
    keys = reference.set_index(["time_key", "station_key"]).loc[common].reset_index()
    keep = joint.balanced_day_subset(keys, max_rows, seed)
    keys = keys.iloc[keep].reset_index(drop=True)
    wanted = joint.key_index(keys)
    positions: Dict[str, np.ndarray] = {}
    for source, frame in frames.items():
        lookup = pd.Series(frame["source_row"].to_numpy(dtype=np.int64), index=joint.key_index(frame))
        position = lookup.reindex(wanted)
        if position.isna().any():
            raise RuntimeError(f"test/{source}: rows disappeared after common-key alignment")
        positions[source] = position.to_numpy(dtype=np.int64)

    values = {
        source: extract_current(layout, "test", positions[source], REQUIRED_FEATURES)
        for source, layout in layouts.items()
    }
    labels = {
        source: np.asarray(
            np.load(layout.path / "y_test.npy", mmap_mode="r")[positions[source]], dtype=np.float64
        )
        for source, layout in layouts.items()
    }
    reference_labels = labels["pangu"]
    for source in SOURCES[1:]:
        equal = np.isclose(reference_labels, labels[source], rtol=0.0, atol=1.0e-3, equal_nan=False)
        if not np.all(equal):
            raise ValueError(f"test/{source}: {int((~equal).sum())} paired visibility labels differ")

    pangu_layout = layouts["pangu"]
    pangu_x = np.load(pangu_layout.path / "X_test.npy", mmap_mode="r")
    static_start = pangu_layout.window * pangu_layout.dyn_vars
    orography = np.asarray(pangu_x[positions["pangu"], static_start + 2], dtype=np.float64)
    return QualitySplit(keys, positions, values, reference_labels, orography)


def scope_masks(split: QualitySplit, low_vis_threshold_m: float) -> Dict[str, np.ndarray]:
    n = len(split.keys)
    return {
        "all_paired_test": np.ones(n, dtype=bool),
        "true_low_visibility": np.isfinite(split.visibility_m)
        & (split.visibility_m < low_vis_threshold_m),
        "elevation_le_500m": np.isfinite(split.orography_m) & (split.orography_m <= 500.0),
    }


def source_metrics(forecast: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(forecast, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    corr = (
        float(np.corrcoef(forecast, reference)[0, 1])
        if len(diff) > 1 and np.std(forecast) > 0.0 and np.std(reference) > 0.0
        else math.nan
    )
    return {
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "correlation": corr,
    }


def pressure_level_quality(
    split: QualitySplit,
    iterations: int,
    seed: int,
    low_vis_threshold_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = split.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    masks = scope_masks(split, low_vis_threshold_m)
    rows = []
    qc_rows = []
    offset = 0
    for feature in PRESSURE_LEVEL_FEATURES:
        converted = {
            source: convert_feature(feature, split.values[source][feature]) for source in SOURCES
        }
        lo, hi = FEATURE_INFO[feature]["broad_range"]
        for source in SOURCES:
            value = converted[source]
            finite = np.isfinite(value)
            out_of_range = finite & ((value < float(lo)) | (value > float(hi)))
            qc_rows.append(
                {
                    "feature": feature,
                    "source": source,
                    "unit": FEATURE_INFO[feature]["unit"],
                    "rows": int(len(value)),
                    "finite_rows": int(finite.sum()),
                    "nonfinite_rows": int((~finite).sum()),
                    "broad_range_low": float(lo),
                    "broad_range_high": float(hi),
                    "outside_broad_range_rows": int(out_of_range.sum()),
                    "outside_broad_range_fraction": float(out_of_range.sum() / max(finite.sum(), 1)),
                    "minimum": float(np.nanmin(value)) if finite.any() else math.nan,
                    "maximum": float(np.nanmax(value)) if finite.any() else math.nan,
                    "excluded_from_primary_rmse": False,
                }
            )
        complete = np.logical_and.reduce([np.isfinite(converted[source]) for source in SOURCES])
        for scope in SCOPES:
            valid = masks[scope] & complete
            n = int(valid.sum())
            if n == 0:
                raise RuntimeError(f"No complete {feature} rows for scope={scope}")
            reference = converted["era5_reference_analysis"][valid]
            pangu = converted["pangu"][valid]
            tianji = converted["tianji"][valid]
            inferred = joint.bootstrap_pair_loss(
                (pangu - reference) ** 2,
                (tianji - reference) ** 2,
                dates[valid],
                iterations,
                seed + offset,
                square_root=True,
            )
            p_metrics = source_metrics(pangu, reference)
            t_metrics = source_metrics(tianji, reference)
            rows.append(
                {
                    "feature": feature,
                    "family": FEATURE_INFO[feature]["family"],
                    "unit": FEATURE_INFO[feature]["unit"],
                    "scope": scope,
                    "metric": "pointwise_rmse",
                    "sequence_position": "last_input_step_at_valid_time",
                    "reference": "ERA5 reference analysis",
                    "n": n,
                    "represented_utc_dates": int(np.unique(dates[valid]).size),
                    **inferred,
                    "pangu_bias": p_metrics["bias"],
                    "pangu_mae": p_metrics["mae"],
                    "pangu_correlation": p_metrics["correlation"],
                    "tianji_bias": t_metrics["bias"],
                    "tianji_mae": t_metrics["mae"],
                    "tianji_correlation": t_metrics["correlation"],
                    "original_q_core_input": feature in ORIGINAL_Q_CORE_INPUTS,
                    "independent_evidence": bool(FEATURE_INFO[feature]["independent_evidence"]),
                    "lineage_caveat": FEATURE_INFO[feature]["lineage"],
                    "primary_filter_policy": "common finite rows; broad-range failures retained and reported separately",
                    "bootstrap_iterations": int(iterations),
                    "bootstrap_seed": int(seed + offset),
                    "bootstrap_unit": "UTC_valid_date",
                    "delta_direction": "positive means Pangu has larger RMSE than Tianji",
                }
            )
            offset += 1

    # A vector-wind RMSE is included as the preferred combined summary of the
    # U/V component pair. WSPD925 remains in the per-variable table but is
    # marked as algebraically dependent; none should be double-counted.
    wind = {
        source: {
            component: convert_feature(component, split.values[source][component])
            for component in ("U_925", "V_925")
        }
        for source in SOURCES
    }
    complete_wind = np.logical_and.reduce(
        [np.isfinite(wind[source][component]) for source in SOURCES for component in ("U_925", "V_925")]
    )
    for scope in SCOPES:
        valid = masks[scope] & complete_wind
        n = int(valid.sum())
        p_loss = sum(
            (wind["pangu"][component][valid] - wind["era5_reference_analysis"][component][valid]) ** 2
            for component in ("U_925", "V_925")
        )
        t_loss = sum(
            (wind["tianji"][component][valid] - wind["era5_reference_analysis"][component][valid]) ** 2
            for component in ("U_925", "V_925")
        )
        inferred = joint.bootstrap_pair_loss(
            p_loss,
            t_loss,
            dates[valid],
            iterations,
            seed + offset,
            square_root=True,
        )
        rows.append(
            {
                "feature": "UV_925_VECTOR",
                "family": "horizontal_wind",
                "unit": "m s-1",
                "scope": scope,
                "metric": "vector_rmse",
                "sequence_position": "last_input_step_at_valid_time",
                "reference": "ERA5 reference analysis",
                "n": n,
                "represented_utc_dates": int(np.unique(dates[valid]).size),
                **inferred,
                "original_q_core_input": True,
                "independent_evidence": False,
                "lineage_caveat": (
                    "derived verification score from native U925/V925 components; "
                    "use as the preferred wind-family summary, not additional independent evidence"
                ),
                "primary_filter_policy": "common finite rows; broad-range failures retained and reported separately",
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(seed + offset),
                "bootstrap_unit": "UTC_valid_date",
                "delta_direction": "positive means Pangu has larger vector RMSE than Tianji",
            }
        )
        offset += 1
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def aggregate_daily_losses(losses: Mapping[str, np.ndarray], dates: np.ndarray):
    unique, codes = np.unique(dates, return_inverse=True)
    sums = {source: np.zeros(len(unique), dtype=np.float64) for source in losses}
    counts = np.zeros(len(unique), dtype=np.int64)
    for day in range(len(unique)):
        selected = codes == day
        counts[day] = int(selected.sum())
        for source, loss in losses.items():
            sums[source][day] = float(np.sum(np.asarray(loss)[selected]))
    return unique, sums, counts


def bootstrap_multi_source_rmse(
    losses: Mapping[str, np.ndarray],
    dates: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[str, float]]]:
    unique, sums, counts = aggregate_daily_losses(losses, dates)
    if len(unique) == 0:
        raise RuntimeError("No represented UTC dates for RMSE bootstrap")
    rng = np.random.default_rng(seed)
    draws = {source: np.empty(iterations, dtype=np.float64) for source in losses}
    for i in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(chosen, minlength=len(unique))
        denominator = max(float(weights @ counts), 1.0)
        for source in losses:
            draws[source][i] = math.sqrt(max(float(weights @ sums[source]) / denominator, 0.0))
    source_results: Dict[str, Dict[str, float]] = {}
    for source, loss in losses.items():
        point = math.sqrt(float(np.mean(loss)))
        source_results[source] = {
            "rmse": point,
            "ci_low": float(np.quantile(draws[source], 0.025)),
            "ci_high": float(np.quantile(draws[source], 0.975)),
        }
    pair_results: Dict[Tuple[str, str], Dict[str, float]] = {}
    for left, right in itertools.combinations(losses, 2):
        delta = draws[left] - draws[right]
        pair_results[(left, right)] = {
            "delta_left_minus_right": source_results[left]["rmse"] - source_results[right]["rmse"],
            "delta_ci_low": float(np.quantile(delta, 0.025)),
            "delta_ci_high": float(np.quantile(delta, 0.975)),
        }
    return source_results, pair_results


def surface_observation_quality(
    split: QualitySplit,
    obs_root: Path,
    paper_eval_dir: Path,
    iterations: int,
    seed: int,
    low_vis_threshold_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    frame = split.keys[["time_utc", "station_key"]].copy()
    frame["_aligned_row"] = np.arange(len(frame), dtype=np.int64)
    for source in SOURCES:
        for feature in SURFACE_OBS_FEATURES:
            frame[f"{feature}_{source}"] = split.values[source][feature]
    frame, obs_info = joint.hybrid_analysis.attach_observations(frame, obs_root, paper_eval_dir)
    if len(frame) != len(split.keys) or frame["_aligned_row"].duplicated().any():
        raise RuntimeError("Observation attachment changed common test-row cardinality")
    frame = frame.sort_values("_aligned_row", kind="stable").reset_index(drop=True)
    if not np.array_equal(frame["_aligned_row"].to_numpy(dtype=np.int64), np.arange(len(frame))):
        raise RuntimeError("Observation attachment changed common test-row order")

    dates = split.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    masks = scope_masks(split, low_vis_threshold_m)
    source_rows = []
    pair_rows = []
    counts: Dict[str, Dict[str, int]] = {}
    offset = 0
    for feature in SURFACE_OBS_FEATURES:
        info = SURFACE_INFO[feature]
        obs_column = str(info["observation_column"])
        if obs_column not in frame:
            raise KeyError(f"Observation attachment lacks required column {obs_column}")
        observation = joint.hybrid_analysis.clean_observation_values(obs_column, frame[obs_column])
        forecasts = {
            source: convert_surface(feature, frame[f"{feature}_{source}"].to_numpy(dtype=np.float64))
            for source in SOURCES
        }
        counts[feature] = {}
        complete = np.isfinite(observation) & np.logical_and.reduce(
            [np.isfinite(forecasts[source]) for source in SOURCES]
        )
        for scope in SCOPES:
            valid = masks[scope] & complete
            n = int(valid.sum())
            if n == 0:
                raise RuntimeError(f"No common observation rows for feature={feature}, scope={scope}")
            counts[feature][scope] = n
            losses = {
                source: (forecasts[source][valid] - observation[valid]) ** 2 for source in SOURCES
            }
            source_result, pair_result = bootstrap_multi_source_rmse(
                losses, dates[valid], iterations, seed + offset
            )
            for source in SOURCES:
                metrics = source_metrics(forecasts[source][valid], observation[valid])
                source_rows.append(
                    {
                        "feature": feature,
                        "source": source,
                        "source_role": "forecast" if source in FORECAST_SOURCES else "reference-analysis benchmark",
                        "reference": "automatic station observations",
                        "observation_column": obs_column,
                        "unit": info["unit"],
                        "scope": scope,
                        "sequence_position": "last_input_step_at_valid_time",
                        "n": n,
                        "represented_utc_dates": int(np.unique(dates[valid]).size),
                        **source_result[source],
                        "bias": metrics["bias"],
                        "mae": metrics["mae"],
                        "correlation": metrics["correlation"],
                        "bootstrap_iterations": int(iterations),
                        "bootstrap_seed": int(seed + offset),
                        "bootstrap_unit": "UTC_valid_date",
                    }
                )
            for (left, right), inferred in pair_result.items():
                pair_rows.append(
                    {
                        "feature": feature,
                        "left_source": left,
                        "right_source": right,
                        "reference": "automatic station observations",
                        "unit": info["unit"],
                        "scope": scope,
                        "sequence_position": "last_input_step_at_valid_time",
                        "n": n,
                        "represented_utc_dates": int(np.unique(dates[valid]).size),
                        **inferred,
                        "delta_direction": "positive means left source has larger observation RMSE",
                        "bootstrap_iterations": int(iterations),
                        "bootstrap_seed": int(seed + offset),
                        "bootstrap_unit": "UTC_valid_date",
                    }
                )
            offset += 1
    return pd.DataFrame(source_rows), pd.DataFrame(pair_rows), {
        **obs_info,
        "common_rows_by_feature_and_scope": counts,
        "era5_role": "reference-analysis benchmark against observations; not a third forecast and not independent truth",
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters < 100:
        raise ValueError("--bootstrap-iters must be at least 100 for inference output")
    paths = {
        "pangu": Path(args.pangu_dir).expanduser().resolve(),
        "tianji": Path(args.tianji_dir).expanduser().resolve(),
        "era5_reference_analysis": Path(args.era5_dir).expanduser().resolve(),
    }
    layouts = {source: joint.load_layout(path, source) for source, path in paths.items()}
    for source, layout in layouts.items():
        missing = [feature for feature in REQUIRED_FEATURES if feature not in layout.order]
        if missing:
            raise ValueError(f"{source}: diagnostic dataset lacks {missing}; rebuild q_core_t925_no_rh2m data")

    split = align_test(layouts, args.test_max_rows, args.bootstrap_seed)
    pressure, qc = pressure_level_quality(
        split, args.bootstrap_iters, args.bootstrap_seed, args.low_vis_threshold_m
    )
    surface, surface_pairs, obs_info = surface_observation_quality(
        split,
        Path(args.obs_root).expanduser().resolve(),
        Path(args.paper_eval_dir).expanduser().resolve(),
        args.bootstrap_iters,
        args.bootstrap_seed + 1000,
        args.low_vis_threshold_m,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pressure.to_csv(out_dir / "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv", index=False)
    qc.to_csv(out_dir / "pressure_level_source_qc.csv", index=False)
    surface.to_csv(out_dir / "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv", index=False)
    surface_pairs.to_csv(
        out_dir / "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv", index=False
    )

    report = {
        "status": "completed",
        "analysis_type": "diagnostic_only_no_training",
        "new_training_models_used": 0,
        "test_rows": int(len(split.keys)),
        "represented_utc_dates": int(split.keys["time_utc"].dt.strftime("%Y-%m-%d").nunique()),
        "low_visibility_definition": {
            "operator": "<",
            "threshold_m": float(args.low_vis_threshold_m),
            "boundary_value_is_clear": True,
            "visibility_equal_threshold_rows": int(
                (
                    np.isfinite(split.visibility_m)
                    & np.isclose(
                        split.visibility_m,
                        float(args.low_vis_threshold_m),
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                ).sum()
            ),
        },
        "paired_key": ["valid_time_utc", "station_id"],
        "sequence_position": (
            "last input step at valid time, matching the observation-anchored pointwise comparison; "
            "the overlapping 12 h trajectory is not duplicated in this diagnosis"
        ),
        "bootstrap": {
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
            "unit": "UTC_valid_date",
            "spatial_dependence": "all stations on a sampled date are resampled jointly",
        },
        "pressure_level_features": list(PRESSURE_LEVEL_FEATURES),
        "original_q_core_pressure_level_inputs": [
            feature for feature in PRESSURE_LEVEL_FEATURES if feature in ORIGINAL_Q_CORE_INPUTS
        ],
        "diagnosis_only_pressure_level_features": [
            feature for feature in PRESSURE_LEVEL_FEATURES if feature not in ORIGINAL_Q_CORE_INPUTS
        ],
        "upper_wind_input_audit": {
            "present": ["U_925", "V_925", "WSPD925"],
            "WSPD925_lineage": (
                "derived from U925/V925; use UV_925_VECTOR as the preferred wind-family summary "
                "without double-counting component and speed metrics"
            ),
            "absent": ["W_925", "WDIR925"],
        },
        "input_datasets": {
            source: {
                "path": str(layout.path),
                "feature_set": layout.feature_set,
                "dynamic_feature_order": list(layout.order),
                "config_sha256": sha256(layout.path / "dataset_build_config.json"),
            }
            for source, layout in layouts.items()
        },
        "surface_observation_alignment": obs_info,
        "reference_roles": {
            "pressure_level": "ERA5 reference analysis, not truth",
            "surface": "automatic-station observations are the primary reference",
            "era5_surface": "analysis benchmark against observations; not a third forecast and not independent",
        },
        "interpretation_constraints": [
            "RMSE comparisons use exact common finite station-time rows for Pangu, Tianji and ERA5.",
            "Broad physical-range failures are reported in pressure_level_source_qc.csv and retained in primary RMSE.",
            "Derived DP, RH and WSPD fields are not independent evidence from their parent variables.",
            "T925 is diagnosis-only and was not used by the original q-core training experiment.",
            "A closer match to ERA5 does not by itself imply a closer match to observations.",
        ],
        "outputs": [
            "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv",
            "pressure_level_source_qc.csv",
            "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv",
            "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv",
        ],
    }
    (out_dir / "paired_source_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
