#!/usr/bin/env python3
"""Validation-frozen task-relevant tail-fidelity analysis for Tianji and Pangu.

This no-training diagnostic reuses the paired q-core+T925 station datasets used
by ``analyze_q_core_paired_source_quality.py``.  Every dynamic value is the
last step of the existing 12-h IDW-to-station input window.  Surface variables
are verified against automatic-station observations; pressure-level variables
are verified against ERA5 reference analysis (reference, not truth).

The validation split fixes:

* Q05/Q95 reference and source-specific thresholds;
* which tail is associated with observed low visibility for each variable;
* monthly reference centres/scales and the joint-tail rule.

The test split is evaluation-only.  Outputs quantify distributional tail
amplitude, placement of reference extremes, and joint entry into the
low-visibility-associated tail.  More extreme values are never equated with
greater accuracy: a source is favoured only when it better reproduces the
reference tail event or tail amplitude.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import analyze_q_core_paired_source_quality as quality


FORECAST_SOURCES = ("pangu", "tianji")
REFERENCE_SOURCE = "era5_reference_analysis"
DEFAULT_FEATURES = (
    "T2M",
    "WSPD10",
    "MSLP",
    "T_925",
    "Q_1000",
    "Q_925",
    "UV_925_VECTOR",
)
DEFAULT_JOINT_FEATURES = (
    "T2M",
    "WSPD10",
    "T_925",
    "Q_1000",
    "Q_925",
    "UV_925_VECTOR",
)
SCOPES = ("all_paired", "true_low_visibility")


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-dir", required=True)
    ap.add_argument("--tianji-dir", required=True)
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--obs-root", required=True)
    ap.add_argument(
        "--paper-eval-dir",
        default="/public/home/putianshu/vis_mlp/paper_eval",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--joint-features", default=",".join(DEFAULT_JOINT_FEATURES))
    ap.add_argument("--joint-min-count", type=int, default=2)
    ap.add_argument("--low-quantile", type=float, default=0.05)
    ap.add_argument("--high-quantile", type=float, default=0.95)
    ap.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    ap.add_argument("--fit-max-rows", type=int, default=0)
    ap.add_argument("--test-max-rows", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260811)
    return ap.parse_args()


def safe_div(num: float | np.ndarray, den: float | np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(num, dtype=np.float64) / np.asarray(den, dtype=np.float64)
    if out.ndim == 0:
        return float(out) if np.isfinite(out) else math.nan
    out[~np.isfinite(out)] = np.nan
    return out


def event_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = float(np.sum(observed & predicted))
    fp = float(np.sum(~observed & predicted))
    fn = float(np.sum(observed & ~predicted))
    tn = float(np.sum(~observed & ~predicted))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": safe_div(tp, tp + fn),
        "precision": safe_div(tp, tp + fp),
        "csi": safe_div(tp, tp + fp + fn),
        "fpr": safe_div(fp, fp + tn),
        "frequency_bias": safe_div(tp + fp, tp + fn),
    }


def align_split(
    layouts: Mapping[str, quality.joint.DatasetLayout],
    split: str,
    max_rows: int,
    seed: int,
) -> quality.QualitySplit:
    frames = {
        source: quality.joint.metadata(layout.path, split)
        for source, layout in layouts.items()
    }
    reference = frames["pangu"]
    common = quality.joint.key_index(reference)
    for source in quality.SOURCES[1:]:
        common = common[common.isin(quality.joint.key_index(frames[source]))]
    if len(common) == 0:
        raise RuntimeError(f"{split}: no common Pangu/Tianji/ERA5 rows")
    keys = reference.set_index(["time_key", "station_key"]).loc[common].reset_index()
    keep = quality.joint.balanced_day_subset(keys, max_rows, seed)
    keys = keys.iloc[keep].reset_index(drop=True)
    wanted = quality.joint.key_index(keys)
    positions: Dict[str, np.ndarray] = {}
    for source, frame in frames.items():
        lookup = pd.Series(
            frame["source_row"].to_numpy(dtype=np.int64),
            index=quality.joint.key_index(frame),
        )
        position = lookup.reindex(wanted)
        if position.isna().any():
            raise RuntimeError(f"{split}/{source}: rows disappeared after alignment")
        positions[source] = position.to_numpy(dtype=np.int64)

    values = {
        source: quality.extract_current(
            layout,
            split,
            positions[source],
            quality.REQUIRED_FEATURES,
        )
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
    for source in quality.SOURCES[1:]:
        equal = np.isclose(
            reference_labels,
            labels[source],
            rtol=0.0,
            atol=1.0e-3,
            equal_nan=False,
        )
        if not np.all(equal):
            raise ValueError(
                f"{split}/{source}: {int((~equal).sum())} paired visibility labels differ"
            )

    pangu_layout = layouts["pangu"]
    pangu_x = np.load(pangu_layout.path / f"X_{split}.npy", mmap_mode="r")
    static_start = pangu_layout.window * pangu_layout.dyn_vars
    orography = np.asarray(
        pangu_x[positions["pangu"], static_start + 2], dtype=np.float64
    )
    return quality.QualitySplit(keys, positions, values, reference_labels, orography)


def attach_surface_observations(
    split: quality.QualitySplit,
    obs_root: Path,
    paper_eval_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frame = split.keys[["time_utc", "station_key"]].copy()
    frame["_aligned_row"] = np.arange(len(frame), dtype=np.int64)
    for source in quality.SOURCES:
        for feature in quality.SURFACE_OBS_FEATURES:
            frame[f"{feature}_{source}"] = split.values[source][feature]
    frame, info = quality.joint.hybrid_analysis.attach_observations(
        frame, obs_root, paper_eval_dir
    )
    if len(frame) != len(split.keys) or frame["_aligned_row"].duplicated().any():
        raise RuntimeError("Observation attachment changed paired-row cardinality")
    frame = frame.sort_values("_aligned_row", kind="stable").reset_index(drop=True)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["_aligned_row"].to_numpy(dtype=np.int64), expected):
        raise RuntimeError("Observation attachment changed paired-row order")
    return frame, info


def feature_bundle(
    split: quality.QualitySplit,
    obs_frame: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    bundle: Dict[str, Dict[str, object]] = {}
    for feature in ("T2M", "WSPD10", "MSLP"):
        info = quality.SURFACE_INFO[feature]
        obs_column = str(info["observation_column"])
        reference = quality.joint.hybrid_analysis.clean_observation_values(
            obs_column, obs_frame[obs_column]
        )
        bundle[feature] = {
            "unit": str(info["unit"]),
            "reference_name": "automatic station observations",
            "reference": np.asarray(reference, dtype=np.float64),
            **{
                source: quality.convert_surface(
                    feature,
                    obs_frame[f"{feature}_{source}"].to_numpy(dtype=np.float64),
                )
                for source in FORECAST_SOURCES
            },
        }

    for feature in ("T_925", "Q_1000", "Q_925"):
        bundle[feature] = {
            "unit": str(quality.FEATURE_INFO[feature]["unit"]),
            "reference_name": "ERA5 reference analysis",
            "reference": quality.convert_feature(
                feature, split.values[REFERENCE_SOURCE][feature]
            ),
            **{
                source: quality.convert_feature(feature, split.values[source][feature])
                for source in FORECAST_SOURCES
            },
        }

    vector_values = {}
    for source in (*FORECAST_SOURCES, REFERENCE_SOURCE):
        u = quality.convert_feature("U_925", split.values[source]["U_925"])
        v = quality.convert_feature("V_925", split.values[source]["V_925"])
        vector_values[source] = np.hypot(u, v)
    bundle["UV_925_VECTOR"] = {
        "unit": "m s-1",
        "reference_name": "ERA5 reference analysis",
        "reference": vector_values[REFERENCE_SOURCE],
        "pangu": vector_values["pangu"],
        "tianji": vector_values["tianji"],
    }
    return bundle


def scope_mask(
    visibility_m: np.ndarray,
    scope: str,
    low_vis_threshold_m: float,
) -> np.ndarray:
    finite = np.isfinite(visibility_m)
    if scope == "all_paired":
        return finite
    if scope == "true_low_visibility":
        return finite & (visibility_m < low_vis_threshold_m)
    raise KeyError(scope)


def quantile_summary(
    test_bundle: Mapping[str, Mapping[str, object]],
    visibility_m: np.ndarray,
    features: Sequence[str],
    low_q: float,
    high_q: float,
    low_vis_threshold_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    shift_rows = []
    lowvis = np.isfinite(visibility_m) & (visibility_m < low_vis_threshold_m)
    clear = np.isfinite(visibility_m) & (visibility_m >= low_vis_threshold_m)
    for feature in features:
        data = test_bundle[feature]
        reference = np.asarray(data["reference"], dtype=np.float64)
        sources = {
            source: np.asarray(data[source], dtype=np.float64)
            for source in FORECAST_SOURCES
        }
        complete = np.isfinite(reference) & np.logical_and.reduce(
            [np.isfinite(value) for value in sources.values()]
        )
        for scope in SCOPES:
            valid = complete & scope_mask(visibility_m, scope, low_vis_threshold_m)
            if not valid.any():
                continue
            ref_q = np.quantile(reference[valid], [low_q, 0.5, high_q])
            ref_width = float(ref_q[2] - ref_q[0])
            ref_lower_amplitude = float(ref_q[1] - ref_q[0])
            ref_upper_amplitude = float(ref_q[2] - ref_q[1])
            for source, values in sources.items():
                src_q = np.quantile(values[valid], [low_q, 0.5, high_q])
                src_width = float(src_q[2] - src_q[0])
                src_lower_amplitude = float(src_q[1] - src_q[0])
                src_upper_amplitude = float(src_q[2] - src_q[1])
                outer_abs_error = 0.5 * (
                    abs(float(src_q[0] - ref_q[0]))
                    + abs(float(src_q[2] - ref_q[2]))
                )
                rows.append(
                    {
                        "feature": feature,
                        "unit": data["unit"],
                        "reference": data["reference_name"],
                        "scope": scope,
                        "source": source,
                        "n": int(valid.sum()),
                        "low_quantile": low_q,
                        "high_quantile": high_q,
                        "reference_q05": float(ref_q[0]),
                        "source_q05": float(src_q[0]),
                        "q05_error_source_minus_reference": float(src_q[0] - ref_q[0]),
                        "reference_q50": float(ref_q[1]),
                        "source_q50": float(src_q[1]),
                        "q50_error_source_minus_reference": float(src_q[1] - ref_q[1]),
                        "reference_q95": float(ref_q[2]),
                        "source_q95": float(src_q[2]),
                        "q95_error_source_minus_reference": float(src_q[2] - ref_q[2]),
                        "reference_central90_width": ref_width,
                        "source_central90_width": src_width,
                        "central90_width_ratio": safe_div(src_width, ref_width),
                        "reference_lower_tail_amplitude_q50_minus_q05": ref_lower_amplitude,
                        "source_lower_tail_amplitude_q50_minus_q05": src_lower_amplitude,
                        "lower_tail_amplitude_retention_ratio": safe_div(
                            src_lower_amplitude, ref_lower_amplitude
                        ),
                        "reference_upper_tail_amplitude_q95_minus_q50": ref_upper_amplitude,
                        "source_upper_tail_amplitude_q95_minus_q50": src_upper_amplitude,
                        "upper_tail_amplitude_retention_ratio": safe_div(
                            src_upper_amplitude, ref_upper_amplitude
                        ),
                        "outer_quantile_abs_error": outer_abs_error,
                        "outer_quantile_abs_error_normalized_by_reference_central90_width": safe_div(
                            outer_abs_error, ref_width
                        ),
                    }
                )
        fit = complete & lowvis
        background = complete & clear
        if fit.any() and background.any():
            ref_shift = float(np.median(reference[fit]) - np.median(reference[background]))
            for source, values in sources.items():
                source_shift = float(np.median(values[fit]) - np.median(values[background]))
                shift_rows.append(
                    {
                        "feature": feature,
                        "unit": data["unit"],
                        "reference": data["reference_name"],
                        "source": source,
                        "n_low_visibility": int(fit.sum()),
                        "n_clear": int(background.sum()),
                        "reference_lowvis_minus_clear_median": ref_shift,
                        "source_lowvis_minus_clear_median": source_shift,
                        "shift_error_source_minus_reference": source_shift - ref_shift,
                        "shift_retention_ratio": safe_div(source_shift, ref_shift),
                        "task_tail_direction": "upper" if ref_shift >= 0.0 else "lower",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(shift_rows)


def _confusion_by_date(
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "date": dates,
            "tp": observed & predicted,
            "fp": ~observed & predicted,
            "fn": observed & ~predicted,
            "tn": ~observed & ~predicted,
        }
    )
    return (
        frame.groupby("date", sort=True)[["tp", "fp", "fn", "tn"]]
        .sum()
        .to_numpy(dtype=np.float64)
    )


def _metric_from_counts(counts: np.ndarray, metric: str) -> np.ndarray:
    tp, fp, fn, tn = (counts[..., index] for index in range(4))
    if metric == "pod":
        return safe_div(tp, tp + fn)
    if metric == "precision":
        return safe_div(tp, tp + fp)
    if metric == "csi":
        return safe_div(tp, tp + fp + fn)
    if metric == "fpr":
        return safe_div(fp, fp + tn)
    raise KeyError(metric)


def bootstrap_event_pair(
    observed: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    dates: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    daily = {
        source: _confusion_by_date(observed, predictions[source], dates)
        for source in FORECAST_SOURCES
    }
    n_dates = daily["pangu"].shape[0]
    if n_dates == 0:
        return {}
    rng = np.random.default_rng(seed)
    draw_index = rng.integers(0, n_dates, size=(iterations, n_dates))
    draw_counts = {
        source: daily[source][draw_index].sum(axis=1)
        for source in FORECAST_SOURCES
    }
    result: Dict[str, Dict[str, float]] = {}
    for metric in ("pod", "precision", "csi", "fpr"):
        draws = {
            source: _metric_from_counts(draw_counts[source], metric)
            for source in FORECAST_SOURCES
        }
        delta = draws["tianji"] - draws["pangu"]
        result[metric] = {
            "pangu_ci_low": float(np.nanquantile(draws["pangu"], 0.025)),
            "pangu_ci_high": float(np.nanquantile(draws["pangu"], 0.975)),
            "tianji_ci_low": float(np.nanquantile(draws["tianji"], 0.025)),
            "tianji_ci_high": float(np.nanquantile(draws["tianji"], 0.975)),
            "delta_tianji_minus_pangu_ci_low": float(np.nanquantile(delta, 0.025)),
            "delta_tianji_minus_pangu_ci_high": float(np.nanquantile(delta, 0.975)),
        }
    return result


def tail_placement(
    fit_bundle: Mapping[str, Mapping[str, object]],
    test_bundle: Mapping[str, Mapping[str, object]],
    fit_visibility: np.ndarray,
    test_visibility: np.ndarray,
    test_dates: np.ndarray,
    features: Sequence[str],
    low_q: float,
    high_q: float,
    low_vis_threshold_m: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    offset = 0
    for feature in features:
        fit_data = fit_bundle[feature]
        test_data = test_bundle[feature]
        fit_ref = np.asarray(fit_data["reference"], dtype=np.float64)
        test_ref = np.asarray(test_data["reference"], dtype=np.float64)
        fit_sources = {
            source: np.asarray(fit_data[source], dtype=np.float64)
            for source in FORECAST_SOURCES
        }
        test_sources = {
            source: np.asarray(test_data[source], dtype=np.float64)
            for source in FORECAST_SOURCES
        }
        fit_complete = np.isfinite(fit_ref) & np.logical_and.reduce(
            [np.isfinite(value) for value in fit_sources.values()]
        )
        test_complete = np.isfinite(test_ref) & np.logical_and.reduce(
            [np.isfinite(value) for value in test_sources.values()]
        )
        direction_fit = fit_complete & np.isfinite(fit_visibility)
        lowvis_fit = direction_fit & (fit_visibility < low_vis_threshold_m)
        clear_fit = direction_fit & (fit_visibility >= low_vis_threshold_m)
        if not lowvis_fit.any() or not clear_fit.any():
            continue
        reference_shift = float(
            np.median(fit_ref[lowvis_fit]) - np.median(fit_ref[clear_fit])
        )
        task_tail = "upper" if reference_shift >= 0.0 else "lower"
        for scope in SCOPES:
            fit_scope = fit_complete & scope_mask(
                fit_visibility, scope, low_vis_threshold_m
            )
            test_scope = test_complete & scope_mask(
                test_visibility, scope, low_vis_threshold_m
            )
            if not fit_scope.any() or not test_scope.any():
                continue
            ref_thresholds = {
                "lower": float(np.quantile(fit_ref[fit_scope], low_q)),
                "upper": float(np.quantile(fit_ref[fit_scope], high_q)),
            }
            source_thresholds = {
                source: {
                    "lower": float(np.quantile(values[fit_scope], low_q)),
                    "upper": float(np.quantile(values[fit_scope], high_q)),
                }
                for source, values in fit_sources.items()
            }
            ref_eval = test_ref[test_scope]
            dates_eval = test_dates[test_scope]
            for tail in ("lower", "upper"):
                observed = (
                    ref_eval <= ref_thresholds[tail]
                    if tail == "lower"
                    else ref_eval >= ref_thresholds[tail]
                )
                for threshold_mode in ("exact_reference_threshold", "quantile_matched"):
                    predictions = {}
                    for source, values in test_sources.items():
                        threshold = (
                            ref_thresholds[tail]
                            if threshold_mode == "exact_reference_threshold"
                            else source_thresholds[source][tail]
                        )
                        evaluated = values[test_scope]
                        predictions[source] = (
                            evaluated <= threshold
                            if tail == "lower"
                            else evaluated >= threshold
                        )
                        rows.append(
                            {
                                "feature": feature,
                                "unit": fit_data["unit"],
                                "reference": fit_data["reference_name"],
                                "scope": scope,
                                "tail": tail,
                                "task_relevant_tail": bool(tail == task_tail),
                                "validation_reference_lowvis_minus_clear_median": reference_shift,
                                "threshold_mode": threshold_mode,
                                "source": source,
                                "validation_reference_threshold": ref_thresholds[tail],
                                "validation_source_threshold": source_thresholds[source][tail],
                                "test_n": int(test_scope.sum()),
                                "test_reference_extreme_n": int(observed.sum()),
                                **event_metrics(observed, predictions[source]),
                            }
                        )
                    inferred = bootstrap_event_pair(
                        observed,
                        predictions,
                        dates_eval,
                        bootstrap_iters,
                        bootstrap_seed + offset,
                    )
                    point = {
                        source: event_metrics(observed, predictions[source])
                        for source in FORECAST_SOURCES
                    }
                    for metric, ci in inferred.items():
                        bootstrap_rows.append(
                            {
                                "feature": feature,
                                "scope": scope,
                                "tail": tail,
                                "task_relevant_tail": bool(tail == task_tail),
                                "threshold_mode": threshold_mode,
                                "metric": metric,
                                "pangu": point["pangu"][metric],
                                "tianji": point["tianji"][metric],
                                "delta_tianji_minus_pangu": point["tianji"][metric]
                                - point["pangu"][metric],
                                **ci,
                                "bootstrap_unit": "UTC_valid_date",
                                "bootstrap_iterations": bootstrap_iters,
                            }
                        )
                    offset += 1
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def fit_monthly_reference_transform(
    fit_bundle: Mapping[str, Mapping[str, object]],
    fit_months: np.ndarray,
    complete: np.ndarray,
    features: Sequence[str],
) -> Dict[str, Dict[str, Dict[int | str, float]]]:
    transform: Dict[str, Dict[str, Dict[int | str, float]]] = {}
    for feature in features:
        reference = np.asarray(fit_bundle[feature]["reference"], dtype=np.float64)
        centres: Dict[int | str, float] = {}
        scales: Dict[int | str, float] = {}
        valid_all = complete & np.isfinite(reference)
        global_median = float(np.median(reference[valid_all]))
        q25, q75 = np.quantile(reference[valid_all], [0.25, 0.75])
        global_scale = float(q75 - q25)
        if not np.isfinite(global_scale) or global_scale <= 1.0e-12:
            global_scale = float(np.std(reference[valid_all]))
        global_scale = max(global_scale, 1.0e-12)
        centres["global"] = global_median
        scales["global"] = global_scale
        for month in range(1, 13):
            valid = valid_all & (fit_months == month)
            if int(valid.sum()) < 50:
                centres[month] = global_median
                scales[month] = global_scale
                continue
            median = float(np.median(reference[valid]))
            q25, q75 = np.quantile(reference[valid], [0.25, 0.75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 1.0e-12:
                scale = global_scale
            centres[month] = median
            scales[month] = max(scale, 1.0e-12)
        transform[feature] = {"centre": centres, "scale": scales}
    return transform


def apply_monthly_transform(
    values: np.ndarray,
    months: np.ndarray,
    transform: Mapping[str, Mapping[int | str, float]],
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float64)
    for month in range(1, 13):
        mask = months == month
        centre = float(transform["centre"].get(month, transform["centre"]["global"]))
        scale = float(transform["scale"].get(month, transform["scale"]["global"]))
        output[mask] = (values[mask] - centre) / scale
    return output


def joint_tail_analysis(
    fit_bundle: Mapping[str, Mapping[str, object]],
    test_bundle: Mapping[str, Mapping[str, object]],
    fit_visibility: np.ndarray,
    test_visibility: np.ndarray,
    fit_months: np.ndarray,
    test_months: np.ndarray,
    test_dates: np.ndarray,
    features: Sequence[str],
    min_count: int,
    low_vis_threshold_m: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_complete = np.isfinite(fit_visibility)
    test_complete = np.isfinite(test_visibility)
    for feature in features:
        for key in ("reference", *FORECAST_SOURCES):
            fit_complete &= np.isfinite(np.asarray(fit_bundle[feature][key]))
            test_complete &= np.isfinite(np.asarray(test_bundle[feature][key]))
    if not fit_complete.any() or not test_complete.any():
        raise RuntimeError("No complete rows for the requested joint-tail features")
    if min_count < 1 or min_count > len(features):
        raise ValueError("joint-min-count must be between 1 and the feature count")

    transforms = fit_monthly_reference_transform(
        fit_bundle, fit_months, fit_complete, features
    )
    definitions = []
    fit_exceed: Dict[str, Dict[str, np.ndarray]] = {
        key: {} for key in ("reference", *FORECAST_SOURCES)
    }
    test_exceed: Dict[str, Dict[str, np.ndarray]] = {
        key: {} for key in ("reference", *FORECAST_SOURCES)
    }
    fit_low = fit_complete & (fit_visibility < low_vis_threshold_m)
    fit_clear = fit_complete & (fit_visibility >= low_vis_threshold_m)
    for feature in features:
        transformed_fit = {
            key: apply_monthly_transform(
                np.asarray(fit_bundle[feature][key], dtype=np.float64),
                fit_months,
                transforms[feature],
            )
            for key in ("reference", *FORECAST_SOURCES)
        }
        transformed_test = {
            key: apply_monthly_transform(
                np.asarray(test_bundle[feature][key], dtype=np.float64),
                test_months,
                transforms[feature],
            )
            for key in ("reference", *FORECAST_SOURCES)
        }
        ref_shift = float(
            np.median(transformed_fit["reference"][fit_low])
            - np.median(transformed_fit["reference"][fit_clear])
        )
        sign = 1.0 if ref_shift >= 0.0 else -1.0
        adverse_fit_ref = sign * transformed_fit["reference"]
        threshold = float(np.quantile(adverse_fit_ref[fit_complete], 0.95))
        for key in ("reference", *FORECAST_SOURCES):
            fit_exceed[key][feature] = sign * transformed_fit[key] >= threshold
            test_exceed[key][feature] = sign * transformed_test[key] >= threshold
        definitions.append(
            {
                "feature": feature,
                "unit": fit_bundle[feature]["unit"],
                "reference": fit_bundle[feature]["reference_name"],
                "monthly_standardization": "reference monthly median and IQR; validation split only",
                "validation_reference_lowvis_minus_clear_robust_z": ref_shift,
                "task_tail_direction": "upper" if sign > 0 else "lower",
                "validation_adverse_tail_threshold_robust_z": threshold,
            }
        )

    fit_counts = {
        key: np.sum(
            np.column_stack([fit_exceed[key][feature] for feature in features]), axis=1
        )
        for key in ("reference", *FORECAST_SOURCES)
    }
    test_counts = {
        key: np.sum(
            np.column_stack([test_exceed[key][feature] for feature in features]), axis=1
        )
        for key in ("reference", *FORECAST_SOURCES)
    }
    rows = []
    bootstrap_rows = []
    for scope in SCOPES:
        valid = test_complete & scope_mask(
            test_visibility, scope, low_vis_threshold_m
        )
        observed = test_counts["reference"][valid] >= min_count
        predictions = {
            source: test_counts[source][valid] >= min_count
            for source in FORECAST_SOURCES
        }
        lowvis = test_visibility[valid] < low_vis_threshold_m
        baseline_lowvis_rate = float(np.mean(lowvis)) if len(lowvis) else math.nan
        for source in FORECAST_SOURCES:
            predicted = predictions[source]
            predicted_lowvis_rate = (
                float(np.mean(lowvis[predicted])) if predicted.any() else math.nan
            )
            rows.append(
                {
                    "scope": scope,
                    "source": source,
                    "features": ",".join(features),
                    "joint_min_count": min_count,
                    "test_n": int(valid.sum()),
                    "reference_joint_tail_n": int(observed.sum()),
                    "source_joint_tail_n": int(predicted.sum()),
                    "test_lowvis_rate": baseline_lowvis_rate,
                    "lowvis_rate_within_source_joint_tail": predicted_lowvis_rate,
                    "lowvis_enrichment_ratio": safe_div(
                        predicted_lowvis_rate, baseline_lowvis_rate
                    ),
                    **event_metrics(observed, predicted),
                }
            )
        inferred = bootstrap_event_pair(
            observed,
            predictions,
            test_dates[valid],
            bootstrap_iters,
            bootstrap_seed + len(bootstrap_rows),
        )
        point = {
            source: event_metrics(observed, predictions[source])
            for source in FORECAST_SOURCES
        }
        for metric, ci in inferred.items():
            bootstrap_rows.append(
                {
                    "scope": scope,
                    "features": ",".join(features),
                    "joint_min_count": min_count,
                    "metric": metric,
                    "pangu": point["pangu"][metric],
                    "tianji": point["tianji"][metric],
                    "delta_tianji_minus_pangu": point["tianji"][metric]
                    - point["pangu"][metric],
                    **ci,
                    "bootstrap_unit": "UTC_valid_date",
                    "bootstrap_iterations": bootstrap_iters,
                }
            )
    return pd.DataFrame(definitions), pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def write_summary(
    out_dir: Path,
    quantiles: pd.DataFrame,
    tail_metrics: pd.DataFrame,
    joint_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Task-relevant tail-fidelity summary",
        "",
        "All thresholds and task-tail directions were fixed on validation data; test data were evaluation-only.",
        "Dynamic variables are the last step of the existing 12-h IDW-to-station input sequence.",
        "ERA5 is reference analysis for pressure-level variables, not independent truth.",
        "Tail-amplitude retention is source (Q50-Q05 or Q95-Q50) divided by the corresponding reference amplitude; the fidelity target is 1, not the largest value.",
        "",
        "## Marginal tails in observed low-visibility samples",
        "",
    ]
    subset = quantiles[quantiles["scope"] == "true_low_visibility"]
    for feature in subset["feature"].drop_duplicates():
        part = subset[subset["feature"] == feature].set_index("source")
        if not set(FORECAST_SOURCES).issubset(part.index):
            continue
        winner = min(
            FORECAST_SOURCES,
            key=lambda source: float(part.loc[source, "outer_quantile_abs_error"]),
        )
        lines.append(
            f"- {feature}: smaller Q05/Q95 absolute error = {winner}; "
            f"Pangu {float(part.loc['pangu', 'outer_quantile_abs_error']):.4g}, "
            f"Tianji {float(part.loc['tianji', 'outer_quantile_abs_error']):.4g}."
        )
    lines.extend(["", "## Validation-defined task-tail placement", ""])
    subset = tail_metrics[
        (tail_metrics["scope"] == "all_paired")
        & tail_metrics["task_relevant_tail"]
        & (tail_metrics["threshold_mode"] == "exact_reference_threshold")
    ]
    for feature in subset["feature"].drop_duplicates():
        part = subset[subset["feature"] == feature].set_index("source")
        if not set(FORECAST_SOURCES).issubset(part.index):
            continue
        winner = max(
            FORECAST_SOURCES,
            key=lambda source: float(part.loc[source, "csi"]),
        )
        lines.append(
            f"- {feature}: higher reference-tail CSI = {winner}; "
            f"Pangu {float(part.loc['pangu', 'csi']):.3f}, "
            f"Tianji {float(part.loc['tianji', 'csi']):.3f}."
        )
    lines.extend(["", "## Joint task tail", ""])
    subset = joint_metrics[joint_metrics["scope"] == "all_paired"].set_index("source")
    if set(FORECAST_SOURCES).issubset(subset.index):
        winner = max(
            FORECAST_SOURCES,
            key=lambda source: float(subset.loc[source, "csi"]),
        )
        lines.append(
            f"- Higher joint-tail CSI = {winner}; "
            f"Pangu {float(subset.loc['pangu', 'csi']):.3f}, "
            f"Tianji {float(subset.loc['tianji', 'csi']):.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A wider or stronger tail is not automatically better. The paper-facing claim should rely on lower quantile error and better placement of reference tail events. Joint-tail directions are observationally defined on validation data and remain task-specific.",
            "",
        ]
    )
    (out_dir / "task_tail_fidelity_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.low_quantile < 0.5:
        raise ValueError("--low-quantile must be in (0, 0.5)")
    if not 0.5 < args.high_quantile < 1.0:
        raise ValueError("--high-quantile must be in (0.5, 1)")
    if args.low_quantile >= args.high_quantile:
        raise ValueError("low quantile must be smaller than high quantile")
    if args.bootstrap_iters < 100:
        raise ValueError("--bootstrap-iters must be at least 100")

    paths = {
        "pangu": Path(args.pangu_dir).expanduser().resolve(),
        "tianji": Path(args.tianji_dir).expanduser().resolve(),
        REFERENCE_SOURCE: Path(args.era5_dir).expanduser().resolve(),
    }
    layouts = {
        source: quality.joint.load_layout(path, source)
        for source, path in paths.items()
    }
    for source, layout in layouts.items():
        missing = [
            feature
            for feature in quality.REQUIRED_FEATURES
            if feature not in layout.order
        ]
        if missing:
            raise ValueError(f"{source}: diagnostic dataset lacks {missing}")

    fit = align_split(layouts, "val", args.fit_max_rows, args.bootstrap_seed)
    test = align_split(layouts, "test", args.test_max_rows, args.bootstrap_seed + 1)
    obs_root = Path(args.obs_root).expanduser().resolve()
    paper_eval_dir = Path(args.paper_eval_dir).expanduser().resolve()
    fit_obs, fit_obs_info = attach_surface_observations(fit, obs_root, paper_eval_dir)
    test_obs, test_obs_info = attach_surface_observations(test, obs_root, paper_eval_dir)
    fit_bundle = feature_bundle(fit, fit_obs)
    test_bundle = feature_bundle(test, test_obs)

    features = parse_csv(args.features)
    joint_features = parse_csv(args.joint_features)
    unknown = [feature for feature in (*features, *joint_features) if feature not in fit_bundle]
    if unknown:
        raise KeyError(f"Unsupported features: {sorted(set(unknown))}")

    quantiles, shifts = quantile_summary(
        test_bundle,
        test.visibility_m,
        features,
        args.low_quantile,
        args.high_quantile,
        args.low_vis_threshold_m,
    )
    tail_metrics, tail_bootstrap = tail_placement(
        fit_bundle,
        test_bundle,
        fit.visibility_m,
        test.visibility_m,
        test.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy(),
        features,
        args.low_quantile,
        args.high_quantile,
        args.low_vis_threshold_m,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    joint_definitions, joint_metrics, joint_bootstrap = joint_tail_analysis(
        fit_bundle,
        test_bundle,
        fit.visibility_m,
        test.visibility_m,
        fit.keys["time_utc"].dt.month.to_numpy(dtype=np.int64),
        test.keys["time_utc"].dt.month.to_numpy(dtype=np.int64),
        test.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy(),
        joint_features,
        args.joint_min_count,
        args.low_vis_threshold_m,
        args.bootstrap_iters,
        args.bootstrap_seed + 100000,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    quantiles.to_csv(out_dir / "marginal_q05_q95_tail_fidelity.csv", index=False)
    shifts.to_csv(out_dir / "event_conditioned_distribution_shift.csv", index=False)
    tail_metrics.to_csv(out_dir / "reference_tail_placement_metrics.csv", index=False)
    tail_bootstrap.to_csv(
        out_dir / "reference_tail_placement_utc_date_bootstrap_ci.csv", index=False
    )
    joint_definitions.to_csv(out_dir / "joint_tail_feature_definitions.csv", index=False)
    joint_metrics.to_csv(out_dir / "joint_tail_placement_metrics.csv", index=False)
    joint_bootstrap.to_csv(
        out_dir / "joint_tail_placement_utc_date_bootstrap_ci.csv", index=False
    )
    write_summary(out_dir, quantiles, tail_metrics, joint_metrics)
    report = {
        "status": "passed",
        "analysis": "validation-frozen task-relevant marginal and joint tail fidelity",
        "fit_split": "validation",
        "evaluation_split": "test",
        "input_sequence_position": "last step of existing 12-h IDW-to-station sequence",
        "features": list(features),
        "joint_features": list(joint_features),
        "joint_min_count": int(args.joint_min_count),
        "quantiles": [float(args.low_quantile), float(args.high_quantile)],
        "low_visibility_threshold_m": float(args.low_vis_threshold_m),
        "bootstrap": {
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
            "unit": "UTC_valid_date",
        },
        "fit_rows": int(len(fit.keys)),
        "test_rows": int(len(test.keys)),
        "fit_observation_attachment": fit_obs_info,
        "test_observation_attachment": test_obs_info,
        "reference_roles": {
            "surface": "automatic station observations",
            "pressure_level": "ERA5 reference analysis, not independent truth",
        },
        "metric_definitions": {
            "lower_tail_amplitude_retention_ratio": "(source Q50 - source Q05) / (reference Q50 - reference Q05); target is 1",
            "upper_tail_amplitude_retention_ratio": "(source Q95 - source Q50) / (reference Q95 - reference Q50); target is 1",
            "outer_quantile_abs_error": "mean absolute source-reference error at Q05 and Q95",
            "reference_tail_placement": "source detection of reference tail events using validation-frozen thresholds",
            "joint_tail": "at least joint_min_count variables simultaneously enter their validation-defined low-visibility-associated reference tails",
        },
        "claim_limit": (
            "More extreme is not assumed better. Support requires lower Q05/Q95 error, "
            "better placement of reference tail events, or both."
        ),
    }
    (out_dir / "task_tail_fidelity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] task-tail fidelity analysis written to {out_dir}")
    print((out_dir / "task_tail_fidelity_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
