#!/usr/bin/env python3
"""Validation-frozen task-tail CSI by forecast lead for T2M and Q1000.

This is a no-training diagnostic.  Validation data determine each feature's
low-visibility-associated tail and the physical reference threshold.  Test
data are evaluation-only and are grouped into the existing 12--23 h stitched
forecast leads.  Pangu and Tianji are evaluated on identical paired rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import analyze_q_core_paired_source_quality as quality
import analyze_q_core_task_tail_fidelity as tail_fidelity


FORECAST_SOURCES = tail_fidelity.FORECAST_SOURCES
REFERENCE_SOURCE = tail_fidelity.REFERENCE_SOURCE
DEFAULT_FEATURES = ("T2M", "Q_1000")
EXPECTED_LEADS = tuple(range(12, 24))
SCOPE = "all_paired"
THRESHOLD_MODE = "exact_reference_threshold"


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pangu-dir", required=True)
    parser.add_argument("--tianji-dir", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--obs-root", required=True)
    parser.add_argument(
        "--paper-eval-dir",
        default="/public/home/putianshu/vis_mlp/paper_eval",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--low-quantile", type=float, default=0.05)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    parser.add_argument("--fit-max-rows", type=int, default=0)
    parser.add_argument("--test-max-rows", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    return parser.parse_args()


def derive_lead_hours(keys: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Map valid UTC hours to the stitched 12--23 h forecast lead."""

    times = pd.to_datetime(keys["time_utc"], errors="coerce")
    if times.isna().any():
        raise ValueError("time_utc contains invalid timestamps")
    hours = times.dt.hour.to_numpy(dtype=np.int64)
    leads = np.where(hours < 12, hours + 12, hours).astype(np.int64)
    initializations = pd.DatetimeIndex(times - pd.to_timedelta(leads, unit="h"))
    if np.any((leads < EXPECTED_LEADS[0]) | (leads > EXPECTED_LEADS[-1])):
        raise ValueError(
            f"derived leads outside {EXPECTED_LEADS[0]}--{EXPECTED_LEADS[-1]} h"
        )
    if not set(initializations.hour.tolist()).issubset({0, 12}):
        raise ValueError("lead derivation produced initialization hours other than 00/12Z")
    return leads, initializations


def _complete_rows(data: Mapping[str, object]) -> np.ndarray:
    reference = np.asarray(data["reference"], dtype=np.float64)
    return np.isfinite(reference) & np.logical_and.reduce(
        [
            np.isfinite(np.asarray(data[source], dtype=np.float64))
            for source in FORECAST_SOURCES
        ]
    )


def task_tail_csi_by_lead(
    fit_bundle: Mapping[str, Mapping[str, object]],
    test_bundle: Mapping[str, Mapping[str, object]],
    fit_visibility_m: np.ndarray,
    test_visibility_m: np.ndarray,
    test_leads: np.ndarray,
    test_dates: np.ndarray,
    features: Sequence[str],
    low_quantile: float,
    high_quantile: float,
    low_vis_threshold_m: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Evaluate validation-defined task-tail placement for each test lead."""

    fit_visibility_m = np.asarray(fit_visibility_m, dtype=np.float64)
    test_visibility_m = np.asarray(test_visibility_m, dtype=np.float64)
    test_leads = np.asarray(test_leads, dtype=np.int64)
    test_dates = np.asarray(test_dates)
    if len(test_leads) != len(test_visibility_m) or len(test_dates) != len(test_leads):
        raise ValueError("test leads, dates and visibility arrays must have equal length")

    rows = []
    threshold_records: Dict[str, object] = {}
    aggregate_max_abs_error = 0.0
    available_leads = tuple(sorted(int(value) for value in np.unique(test_leads)))

    for feature in features:
        fit_data = fit_bundle[feature]
        test_data = test_bundle[feature]
        fit_reference = np.asarray(fit_data["reference"], dtype=np.float64)
        test_reference = np.asarray(test_data["reference"], dtype=np.float64)
        fit_sources = {
            source: np.asarray(fit_data[source], dtype=np.float64)
            for source in FORECAST_SOURCES
        }
        test_sources = {
            source: np.asarray(test_data[source], dtype=np.float64)
            for source in FORECAST_SOURCES
        }
        fit_complete = _complete_rows(fit_data) & np.isfinite(fit_visibility_m)
        test_complete = _complete_rows(test_data) & np.isfinite(test_visibility_m)
        low_visibility = fit_complete & (fit_visibility_m < low_vis_threshold_m)
        clear = fit_complete & (fit_visibility_m >= low_vis_threshold_m)
        if not low_visibility.any() or not clear.any():
            raise ValueError(f"{feature}: validation lacks low-visibility or clear rows")

        reference_shift = float(
            np.median(fit_reference[low_visibility])
            - np.median(fit_reference[clear])
        )
        task_tail = "upper" if reference_shift >= 0.0 else "lower"
        quantile = high_quantile if task_tail == "upper" else low_quantile
        reference_threshold = float(np.quantile(fit_reference[fit_complete], quantile))
        source_thresholds = {
            source: float(np.quantile(values[fit_complete], quantile))
            for source, values in fit_sources.items()
        }
        threshold_records[feature] = {
            "unit": str(fit_data["unit"]),
            "reference": str(fit_data["reference_name"]),
            "task_tail": task_tail,
            "validation_reference_lowvis_minus_clear_median": reference_shift,
            "quantile": float(quantile),
            "validation_reference_threshold": reference_threshold,
            "validation_source_thresholds": source_thresholds,
            "validation_complete_rows": int(fit_complete.sum()),
        }

        if task_tail == "upper":
            observed_all = test_reference[test_complete] >= reference_threshold
        else:
            observed_all = test_reference[test_complete] <= reference_threshold

        direct_metrics = {}
        for source, values in test_sources.items():
            evaluated = values[test_complete]
            predicted_all = (
                evaluated >= reference_threshold
                if task_tail == "upper"
                else evaluated <= reference_threshold
            )
            direct_metrics[source] = tail_fidelity.event_metrics(
                observed_all, predicted_all
            )

        feature_start = len(rows)
        for lead_hour in available_leads:
            selected = test_complete & (test_leads == lead_hour)
            if not selected.any():
                raise ValueError(f"{feature}: no complete rows for lead {lead_hour}")
            reference_values = test_reference[selected]
            observed = (
                reference_values >= reference_threshold
                if task_tail == "upper"
                else reference_values <= reference_threshold
            )
            dates = test_dates[selected]
            for source, values in test_sources.items():
                evaluated = values[selected]
                predicted = (
                    evaluated >= reference_threshold
                    if task_tail == "upper"
                    else evaluated <= reference_threshold
                )
                metrics = tail_fidelity.event_metrics(observed, predicted)
                rows.append(
                    {
                        "feature": feature,
                        "unit": fit_data["unit"],
                        "reference": fit_data["reference_name"],
                        "scope": SCOPE,
                        "tail": task_tail,
                        "task_relevant_tail": True,
                        "threshold_mode": THRESHOLD_MODE,
                        "source": source,
                        "lead_hour": int(lead_hour),
                        "validation_reference_lowvis_minus_clear_median": reference_shift,
                        "validation_reference_threshold": reference_threshold,
                        "validation_source_threshold": source_thresholds[source],
                        "test_n": int(selected.sum()),
                        "test_dates": int(pd.Series(dates).nunique()),
                        "test_reference_extreme_n": int(observed.sum()),
                        "tp": int(metrics["tp"]),
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                        "tn": int(metrics["tn"]),
                        "pod": metrics["pod"],
                        "precision": metrics["precision"],
                        "csi": metrics["csi"],
                        "fpr": metrics["fpr"],
                        "frequency_bias": metrics["frequency_bias"],
                    }
                )

        feature_rows = pd.DataFrame(rows[feature_start:])
        for source in FORECAST_SOURCES:
            part = feature_rows[feature_rows["source"] == source]
            counts = part[["tp", "fp", "fn", "tn"]].sum().to_numpy(dtype=float)
            direct = np.asarray(
                [direct_metrics[source][name] for name in ("tp", "fp", "fn", "tn")],
                dtype=float,
            )
            aggregate_max_abs_error = max(
                aggregate_max_abs_error, float(np.max(np.abs(counts - direct)))
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("task-tail by-lead analysis produced no rows")
    if result.duplicated(["feature", "source", "lead_hour"]).any():
        raise ValueError("duplicate feature/source/lead rows")
    if aggregate_max_abs_error != 0.0:
        raise ValueError(
            "lead-resolved confusion counts do not reproduce all-test counts: "
            f"max error={aggregate_max_abs_error}"
        )
    expected_rows = len(features) * len(FORECAST_SOURCES) * len(available_leads)
    if len(result) != expected_rows:
        raise ValueError(f"expected {expected_rows} output rows, found {len(result)}")

    validation = {
        "status": "passed",
        "available_leads": list(available_leads),
        "features": list(features),
        "sources": list(FORECAST_SOURCES),
        "rows": int(len(result)),
        "aggregate_confusion_count_max_abs_error": aggregate_max_abs_error,
        "thresholds": threshold_records,
    }
    return result, validation


def main() -> None:
    args = parse_args()
    if not 0.0 < args.low_quantile < 0.5:
        raise ValueError("--low-quantile must be in (0, 0.5)")
    if not 0.5 < args.high_quantile < 1.0:
        raise ValueError("--high-quantile must be in (0.5, 1)")
    if args.fit_max_rows < 0 or args.test_max_rows < 0:
        raise ValueError("row limits must be non-negative")

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
            feature for feature in quality.REQUIRED_FEATURES if feature not in layout.order
        ]
        if missing:
            raise ValueError(f"{source}: diagnostic dataset lacks {missing}")

    fit = tail_fidelity.align_split(
        layouts, "val", args.fit_max_rows, args.selection_seed
    )
    test = tail_fidelity.align_split(
        layouts, "test", args.test_max_rows, args.selection_seed + 1
    )
    obs_root = Path(args.obs_root).expanduser().resolve()
    paper_eval_dir = Path(args.paper_eval_dir).expanduser().resolve()
    fit_obs, fit_obs_info = tail_fidelity.attach_surface_observations(
        fit, obs_root, paper_eval_dir
    )
    test_obs, test_obs_info = tail_fidelity.attach_surface_observations(
        test, obs_root, paper_eval_dir
    )
    fit_bundle = tail_fidelity.feature_bundle(fit, fit_obs)
    test_bundle = tail_fidelity.feature_bundle(test, test_obs)

    features = parse_csv(args.features)
    unknown = [feature for feature in features if feature not in fit_bundle]
    if unknown:
        raise KeyError(f"unsupported features: {sorted(set(unknown))}")
    leads, initializations = derive_lead_hours(test.keys)
    dates = test.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    result, validation = task_tail_csi_by_lead(
        fit_bundle,
        test_bundle,
        fit.visibility_m,
        test.visibility_m,
        leads,
        dates,
        features,
        args.low_quantile,
        args.high_quantile,
        args.low_vis_threshold_m,
    )
    if tuple(validation["available_leads"]) != EXPECTED_LEADS:
        raise ValueError(
            f"expected stitched leads {EXPECTED_LEADS}, found "
            f"{tuple(validation['available_leads'])}"
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    output_name = "reference_tail_placement_by_lead_hour.csv"
    result.to_csv(out_dir / output_name, index=False, float_format="%.10g")
    report = {
        "status": "passed",
        "analysis": "validation-frozen task-tail placement by forecast lead",
        "analysis_type": "diagnostic_only_no_training",
        "fit_split": "validation",
        "evaluation_split": "test",
        "scope": SCOPE,
        "threshold_mode": THRESHOLD_MODE,
        "features": list(features),
        "sources": list(FORECAST_SOURCES),
        "quantiles": [float(args.low_quantile), float(args.high_quantile)],
        "low_visibility_threshold_m": float(args.low_vis_threshold_m),
        "stitching_rule": "00/12Z initializations, each contributing leads 12-23",
        "lead_derivation": "lead = valid_hour + 12 if valid_hour < 12 else valid_hour",
        "initialization_hour_counts": {
            str(hour): int(np.sum(initializations.hour == hour))
            for hour in sorted(set(initializations.hour.tolist()))
        },
        "fit_rows": int(len(fit.keys)),
        "test_rows": int(len(test.keys)),
        "fit_observation_attachment": fit_obs_info,
        "test_observation_attachment": test_obs_info,
        "input_datasets": {source: str(path) for source, path in paths.items()},
        "validation": validation,
        "outputs": [output_name],
    }
    (out_dir / "task_tail_by_leadtime_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] task-tail by-lead analysis written to {out_dir}", flush=True)
    print(out_dir / output_name, flush=True)


if __name__ == "__main__":
    main()
