#!/usr/bin/env python3
"""Three-seed paired Pangu--Tianji fair analysis for q-core plus T925.

The script consumes the per-sample validation and test outputs produced by
``sub_static_rnn_q_core_fair_eval.slurm``.  It keeps the formal q-core protocol:
threshold-free Low-vis AP, validation-only matched-FPR test metrics, argmax
metrics, and a joint UTC-valid-date block bootstrap averaged across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from analyze_q_core_hybrid_factorial import (
    SampleSet,
    adaptive_ap_histograms,
    ap_from_histogram,
    argmax_metrics,
    assert_aligned,
    load_samples,
    match_fpr_threshold,
    probability_metrics,
    reliability_rows,
    safe_div,
    subset_sample,
    threshold_metrics,
)
from plot_pangu_qcore_mechanism_ppt import (
    plot_endpoint,
    plot_qcore_argmax_overview,
    qcore_argmax_source,
)


SOURCE_ORDER = ("pangu", "tianji")
SOURCE_MASK = {"pangu": "0000", "tianji": "1111"}
PRIMARY_METRICS = (
    "low_vis_ap",
    "low_vis_csi_matched_fpr",
    "low_vis_recall_matched_fpr",
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).replace(":", ",").split(",") if part.strip()]


def parse_formats(value: str) -> List[str]:
    allowed = {"svg", "pdf", "png", "tiff"}
    formats: List[str] = []
    for item in parse_csv(value):
        ext = item.lower().lstrip(".")
        if ext == "tif":
            ext = "tiff"
        if ext not in allowed:
            raise ValueError(f"Unsupported figure format {item!r}; choose from {sorted(allowed)}")
        if ext not in formats:
            formats.append(ext)
    if not formats:
        raise ValueError("At least one figure format is required")
    return formats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42:2025:20260702")
    ap.add_argument("--tianji-tag", default="tianji")
    ap.add_argument("--pangu-tag", default="pangu2025_q_core_t925_no_rh2m")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260721)
    ap.add_argument("--bootstrap-max-rows", type=int, default=0)
    ap.add_argument("--ap-hist-initial-bins", type=int, default=4096)
    ap.add_argument("--ap-hist-max-bins", type=int, default=65536)
    ap.add_argument("--ap-hist-max-error", type=float, default=5.0e-4)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--formats", default="svg,pdf,png,tiff")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def align_pair(left: SampleSet, right: SampleSet, label: str) -> Tuple[SampleSet, SampleSet]:
    def keyed(sample: SampleSet) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(sample.frame[["time_key", "station_key"]])

    left_key = keyed(left)
    right_key = keyed(right)
    common = left_key.intersection(right_key, sort=False).sort_values()
    if len(common) == 0:
        raise RuntimeError(f"{label}: no common (valid_time, station_id) rows")
    left_idx = left_key.get_indexer(common)
    right_idx = right_key.get_indexer(common)
    if (left_idx < 0).any() or (right_idx < 0).any():
        raise RuntimeError(f"{label}: failed to index the paired row intersection")
    aligned_left = subset_sample(left, left_idx.astype(np.int64))
    aligned_right = subset_sample(right, right_idx.astype(np.int64))
    assert_aligned(aligned_left, aligned_right, label)
    return aligned_left, aligned_right


def load_all(
    eval_root: Path,
    seeds: Sequence[int],
    tianji_tag: str,
    pangu_tag: str,
) -> Tuple[Dict[Tuple[int, str], SampleSet], Dict[Tuple[int, str], SampleSet]]:
    val: Dict[Tuple[int, str], SampleSet] = {}
    test: Dict[Tuple[int, str], SampleSet] = {}
    for seed in seeds:
        seed_dir = eval_root / f"seed_{seed}"
        pangu_val = load_samples(seed_dir / f"per_sample_val_{pangu_tag}.csv")
        tianji_val = load_samples(seed_dir / f"per_sample_val_{tianji_tag}.csv")
        pangu_test = load_samples(seed_dir / f"per_sample_{pangu_tag}.csv")
        tianji_test = load_samples(seed_dir / f"per_sample_{tianji_tag}.csv")
        pangu_val, tianji_val = align_pair(pangu_val, tianji_val, f"seed {seed} validation")
        pangu_test, tianji_test = align_pair(pangu_test, tianji_test, f"seed {seed} test")
        val[(seed, "pangu")], val[(seed, "tianji")] = pangu_val, tianji_val
        test[(seed, "pangu")], test[(seed, "tianji")] = pangu_test, tianji_test

    val_ref = val[(seeds[0], "pangu")]
    test_ref = test[(seeds[0], "pangu")]
    for key, sample in val.items():
        assert_aligned(val_ref, sample, f"validation cross-seed/source {key}")
    for key, sample in test.items():
        assert_aligned(test_ref, sample, f"test cross-seed/source {key}")
    return val, test


def point_metrics(
    val: Mapping[Tuple[int, str], SampleSet],
    test: Mapping[Tuple[int, str], SampleSet],
    seeds: Sequence[int],
    ece_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[int, str], float], float]:
    pangu_argmax_fpr = {
        seed: argmax_metrics(val[(seed, "pangu")].y, val[(seed, "pangu")].pred)[
            "low_vis_fpr_argmax"
        ]
        for seed in seeds
    }
    target_fpr = float(np.median(np.asarray(list(pangu_argmax_fpr.values()), dtype=np.float64)))
    rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    reliability: List[Dict[str, object]] = []
    thresholds: Dict[Tuple[int, str], float] = {}

    for seed in seeds:
        for source in SOURCE_ORDER:
            val_set = val[(seed, source)]
            test_set = test[(seed, source)]
            threshold, achieved = match_fpr_threshold(val_set.y, val_set.score, target_fpr)
            thresholds[(seed, source)] = threshold
            row: Dict[str, object] = {
                "seed": int(seed),
                "source": source,
                "mask": SOURCE_MASK[source],
                "n_val": int(len(val_set.y)),
                "n_test": int(len(test_set.y)),
                "target_validation_fpr": target_fpr,
                "achieved_validation_fpr": achieved,
                "low_vis_threshold": threshold,
            }
            row.update(probability_metrics(test_set.y, test_set.score, bins=ece_bins))
            row.update(threshold_metrics(test_set.y, test_set.score, threshold))
            row.update(argmax_metrics(test_set.y, test_set.pred))
            rows.append(row)
            threshold_rows.append(
                {
                    "seed": int(seed),
                    "source": source,
                    "pangu_validation_argmax_fpr": pangu_argmax_fpr[seed],
                    "target_validation_fpr": target_fpr,
                    "selected_probability_threshold": threshold,
                    "achieved_validation_fpr": achieved,
                }
            )
            for record in reliability_rows((test_set.y <= 1).astype(np.int64), test_set.score, ece_bins):
                reliability.append({"seed": int(seed), "source": source, **record})
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows), pd.DataFrame(reliability), thresholds, target_fpr


def paired_bootstrap(
    test: Mapping[Tuple[int, str], SampleSet],
    thresholds: Mapping[Tuple[int, str], float],
    seeds: Sequence[int],
    iterations: int,
    rng_seed: int,
    max_rows: int,
    initial_bins: int,
    max_bins: int,
    max_error: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if iterations < 20:
        raise ValueError("bootstrap-iters must be at least 20")
    reference = test[(seeds[0], "pangu")]
    working = dict(test)
    rng = np.random.default_rng(rng_seed)
    if max_rows > 0 and len(reference.y) > max_rows:
        keep = np.sort(rng.choice(len(reference.y), size=max_rows, replace=False).astype(np.int64))
        working = {key: subset_sample(sample, keep) for key, sample in working.items()}
        reference = working[(seeds[0], "pangu")]

    dates = reference.frame["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    unique_dates, date_codes = np.unique(dates, return_inverse=True)
    precomputed, selected_bins, approximation_error = adaptive_ap_histograms(
        working,
        thresholds,
        date_codes,
        len(unique_dates),
        initial_bins,
        max_bins,
        max_error,
    )

    draws: List[Dict[str, object]] = []
    for iteration in range(iterations):
        chosen = rng.integers(0, len(unique_dates), size=len(unique_dates))
        date_weights = np.bincount(chosen, minlength=len(unique_dates)).astype(np.int64)
        seed_delta: Dict[str, List[float]] = {metric: [] for metric in PRIMARY_METRICS}
        for seed in seeds:
            values: Dict[str, Dict[str, float]] = {}
            for source in SOURCE_ORDER:
                positive, negative, confusion = precomputed[(seed, source)]
                tp, fp, fn, _tn = date_weights @ confusion
                values[source] = {
                    "low_vis_ap": ap_from_histogram(date_weights @ positive, date_weights @ negative),
                    "low_vis_csi_matched_fpr": safe_div(float(tp), float(tp + fp + fn)),
                    "low_vis_recall_matched_fpr": safe_div(float(tp), float(tp + fn)),
                }
            for metric in PRIMARY_METRICS:
                seed_delta[metric].append(values["tianji"][metric] - values["pangu"][metric])
        for metric in PRIMARY_METRICS:
            draws.append(
                {
                    "iteration": int(iteration),
                    "metric": metric,
                    "delta_all1_minus_all0": float(np.mean(seed_delta[metric])),
                    "delta_tianji_minus_pangu": float(np.mean(seed_delta[metric])),
                }
            )

    draw_df = pd.DataFrame(draws)
    summary_rows: List[Dict[str, object]] = []
    for metric, part in draw_df.groupby("metric", sort=False):
        values = part["delta_tianji_minus_pangu"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric,
                "delta_tianji_minus_pangu": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "direction_consistent_across_bootstrap": float(np.mean(values > 0.0)),
                "bootstrap_iterations": int(iterations),
                "bootstrap_unit": "UTC_valid_date",
            }
        )
    info = {
        "bootstrap_rows": int(len(reference.y)),
        "n_valid_dates": int(len(unique_dates)),
        "ap_histogram_bins": int(selected_bins),
        "ap_point_approximation_max_abs_error": float(approximation_error),
    }
    return draw_df, pd.DataFrame(summary_rows), info


def endpoint_source(metrics: pd.DataFrame, gap_draws: pd.DataFrame, metric: str) -> pd.DataFrame:
    part = metrics[["seed", "mask", "source", metric]].rename(columns={metric: "value"}).copy()
    draws = pd.to_numeric(
        gap_draws.loc[gap_draws["metric"] == metric, "delta_tianji_minus_pangu"],
        errors="coerce",
    ).dropna()
    if draws.empty:
        raise ValueError(f"{metric}: no finite bootstrap gap draws")
    means = part.groupby("source", sort=False)["value"].mean()
    delta = float(means["tianji"] - means["pangu"])
    summary = pd.DataFrame(
        [
            {
                "seed": "mean",
                "mask": SOURCE_MASK[source],
                "source": source,
                "value": float(means[source]),
                "delta_tianji_minus_pangu": delta,
                "delta_ci_low": float(draws.quantile(0.025)),
                "delta_ci_high": float(draws.quantile(0.975)),
            }
            for source in SOURCE_ORDER
        ]
    )
    part["delta_tianji_minus_pangu"] = np.nan
    part["delta_ci_low"] = np.nan
    part["delta_ci_high"] = np.nan
    return pd.concat([part, summary], ignore_index=True)


def make_figures(
    metrics: pd.DataFrame,
    gap_draws: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    ap_source = endpoint_source(metrics, gap_draws, "low_vis_ap")
    recall_source = endpoint_source(metrics, gap_draws, "low_vis_recall_matched_fpr")
    argmax_source = qcore_argmax_source(metrics)
    ap_source.to_csv(out_dir / "source_data_01_qcore_t925_lowvis_ap.csv", index=False)
    recall_source.to_csv(out_dir / "source_data_02_qcore_t925_matched_fpr_recall.csv", index=False)
    argmax_source.to_csv(out_dir / "source_data_03_qcore_t925_argmax.csv", index=False)
    plot_endpoint(
        ap_source,
        "Low-vis average precision",
        "Fair Q-Core + T925 Low-vis AP",
        "01_qcore_t925_lowvis_ap",
        out_dir,
        formats,
        dpi,
        "three seeds; paired test samples",
    )
    plot_endpoint(
        recall_source,
        "Low-vis recall at matched FPR",
        "Fair Q-Core + T925 Matched-FPR Recall",
        "02_qcore_t925_matched_fpr_recall",
        out_dir,
        formats,
        dpi,
        "validation-only threshold selection",
    )
    plot_qcore_argmax_overview(
        argmax_source,
        out_dir,
        formats,
        dpi,
        title="Fair Q-Core + T925 Argmax Performance",
        output_name="03_qcore_t925_argmax_lowvis_overview",
    )


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integer seeds")
    if args.dpi < 300:
        raise ValueError("Use --dpi >= 300; 600 is recommended for formal export")
    formats = parse_formats(args.formats)
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val, test = load_all(eval_root, seeds, args.tianji_tag, args.pangu_tag)
    metrics, threshold_df, reliability, thresholds, target_fpr = point_metrics(
        val, test, seeds, args.ece_bins
    )
    gap_draws, bootstrap_summary, bootstrap_info = paired_bootstrap(
        test,
        thresholds,
        seeds,
        args.bootstrap_iters,
        args.bootstrap_seed,
        args.bootstrap_max_rows,
        args.ap_hist_initial_bins,
        args.ap_hist_max_bins,
        args.ap_hist_max_error,
    )

    metrics.to_csv(out_dir / "qcore_t925_metrics_by_seed.csv", index=False)
    metrics.groupby(["source", "mask"], as_index=False).mean(numeric_only=True).to_csv(
        out_dir / "qcore_t925_metrics_seed_mean.csv", index=False
    )
    threshold_df.to_csv(out_dir / "qcore_t925_validation_thresholds.csv", index=False)
    reliability.to_csv(out_dir / "qcore_t925_reliability_by_seed.csv", index=False)
    gap_draws.to_csv(out_dir / "qcore_t925_bootstrap_gap_draws.csv", index=False)
    bootstrap_summary.to_csv(out_dir / "qcore_t925_bootstrap_summary.csv", index=False)
    if not args.no_figures:
        make_figures(metrics, gap_draws, out_dir, formats, args.dpi)

    report = {
        "status": "passed",
        "feature_set": "q_core_t925_no_rh2m",
        "dynamic_variables": 18,
        "sources": {"tianji": args.tianji_tag, "pangu": args.pangu_tag},
        "seeds": seeds,
        "paired_validation_rows": int(len(val[(seeds[0], "pangu")].y)),
        "paired_test_rows": int(len(test[(seeds[0], "pangu")].y)),
        "target_validation_fpr": target_fpr,
        "threshold_policy": "per-seed/source probability threshold selected on validation only to match the median Pangu validation argmax FPR",
        "primary_metrics": list(PRIMARY_METRICS),
        "bootstrap": {
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
            "unit": "UTC_valid_date",
            **bootstrap_info,
        },
        "claim_scope": "Pangu-2025 versus Tianji, 2025, canonical 12--23 h Pangu leads, shared q-core+T925 variables",
    }
    with (out_dir / "qcore_t925_fair_analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(metrics.groupby("source")[["low_vis_ap", "low_vis_recall_matched_fpr", "low_vis_csi_matched_fpr"]].mean())
    print(f"[OK] q-core+T925 fair analysis written to {out_dir}")


if __name__ == "__main__":
    main()
