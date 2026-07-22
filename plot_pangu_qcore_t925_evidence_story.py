#!/usr/bin/env python3
"""Redraw the q-core evidence story around the completed q-core+T925 endpoints.

Performance and every endpoint-conditioned event panel are rebuilt from the
completed three-seed q-core+T925 fair run.  Source-quality and QC panels reuse
their model-independent paired diagnoses.  The old no-T925 Shapley panel is
intentionally omitted; it must be replaced only by the completed five-package
MHTPW factorial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_pangu_qcore_mechanism_ppt as story
from analyze_q_core_t925_upper_air_disagreement import plot_summary as plot_upper_air


PRESSURE_FIGURE_SPECS: Tuple[Mapping[str, str], ...] = (
    {
        "feature": "T_925",
        "label": "925-hPa temperature",
        "title": "925-hPa Temperature RMSE",
        "unit": "K",
        "output": "07a_t925_quality",
    },
    {
        "feature": "Q_1000",
        "label": "1000-hPa specific humidity",
        "title": "1000-hPa Specific-humidity RMSE",
        "unit": r"g kg$^{-1}$",
        "output": "07b_q1000_quality",
    },
    {
        "feature": "Q_925",
        "label": "925-hPa specific humidity",
        "title": "925-hPa Specific-humidity RMSE",
        "unit": r"g kg$^{-1}$",
        "output": "07c_q925_quality",
    },
    {
        "feature": "UV_925_VECTOR",
        "label": "925-hPa vector wind",
        "title": "925-hPa Vector-wind RMSE",
        "unit": r"m s$^{-1}$",
        "output": "07d_uv925_quality",
    },
)
PRESSURE_SCOPES = ("all_paired_test", "true_low_visibility")
PRESSURE_SCOPE_LABELS = {
    "all_paired_test": "All test samples",
    "true_low_visibility": "Observed visibility <1 km",
}
PRESSURE_SCOPE_MARKERS = {
    "all_paired_test": "o",
    "true_low_visibility": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fair-eval-root", required=True)
    parser.add_argument("--paired-quality-dir", required=True)
    parser.add_argument("--event-analysis-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--formats", default="svg:pdf:png:tiff")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--surface-view", choices=("both", "all", "low"), default="both")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}")


def load_t925_performance(eval_root: Path) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    analysis = eval_root / "analysis"
    report = json.loads(
        (analysis / "qcore_t925_fair_analysis_report.json").read_text(encoding="utf-8")
    )
    if report.get("status") != "passed":
        raise ValueError(f"q-core+T925 fair analysis is not completed: {report.get('status')}")
    if report.get("seeds") != [42, 2025, 20260702]:
        raise ValueError(f"unexpected q-core+T925 seeds: {report.get('seeds')}")
    metrics = pd.read_csv(analysis / "qcore_t925_metrics_by_seed.csv")
    require_columns(
        metrics,
        [
            "seed",
            "source",
            "low_vis_ap",
            "low_vis_recall_matched_fpr",
            "low_vis_precision_argmax",
            "low_vis_recall_argmax",
            "low_vis_csi_argmax",
            "low_vis_fpr_argmax",
        ],
        "qcore_t925_metrics_by_seed.csv",
    )
    if set(metrics["source"].astype(str)) != {"pangu", "tianji"}:
        raise ValueError("q-core+T925 metrics must contain exactly Pangu and Tianji")
    expected = {(seed, source) for seed in (42, 2025, 20260702) for source in ("pangu", "tianji")}
    actual = set(zip(metrics["seed"].astype(int), metrics["source"].astype(str)))
    if actual != expected:
        raise ValueError("q-core+T925 metrics are not the exact 3-seed x 2-source matrix")
    metrics = metrics.copy()
    metrics["mask"] = metrics["source"].map({"pangu": "0000", "tianji": "1111"})
    draws = pd.read_csv(analysis / "qcore_t925_bootstrap_gap_draws.csv")
    require_columns(draws, ["iteration", "metric", "delta_all1_minus_all0"], "gap draws")
    if draws["iteration"].nunique() != 1000:
        raise ValueError("q-core+T925 gap table must contain 1000 bootstrap iterations")
    return report, metrics, draws


def plot_t925_flow(
    out_dir: Path, formats: Sequence[str], dpi: int
) -> Tuple[pd.DataFrame, List[str]]:
    fig, ax = plt.subplots(figsize=story.FIGURE_SIZES["flow"])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.055)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.94,
        "Controlled Q-Core + T925 Evidence Chain",
        fontsize=10.8,
        fontweight="bold",
        color=story.INK,
        ha="center",
        va="top",
    )
    xs = [0.07, 0.28, 0.49, 0.70, 0.91]
    ax.plot([xs[0], xs[-1]], [0.68, 0.68], color=story.LIGHT_GREY, linewidth=1.1)
    color = "#4F5963"
    story.flow_step(ax, xs[0], "1", "Fair design", ["same samples and labels", "shared q-core + T925", "Pangu 12–23 h"], color)
    story.flow_step(ax, xs[1], "2", "Performance", ["Low-vis AP", "matched-FPR recall", "3 training seeds"], color)
    story.flow_step(ax, xs[2], "3", "Attribution", ["32 source combinations", "M / H / T / P / W", "exact group Shapley"], color)
    story.flow_step(ax, xs[3], "4", "Quality checks", ["surface → observations", "upper air → ERA5", "paired UTC-date CIs"], color)
    story.flow_step(ax, xs[4], "5", "Bounded inference", ["performance + attribution", "+ quality + hit / miss", "task-specific conclusion"], color)
    ax.plot([0.03, 0.97], [0.17, 0.17], color=story.INK, linewidth=1.2)
    ax.text(0.03, 0.135, "Supported: task-relevant source-information differences", ha="left", va="top", fontsize=6.9, fontweight="bold", color=story.INK)
    ax.text(0.56, 0.135, "Not supported: a universal physical-inconsistency claim", ha="left", va="top", fontsize=6.7, color=story.INK)
    source = pd.DataFrame(
        [
            {"stage": 1, "evidence": "q-core+T925 fair endpoints"},
            {"stage": 2, "evidence": "three-seed performance and matched-FPR events"},
            {"stage": 3, "evidence": "five-package MHTPW exact Shapley"},
            {"stage": 4, "evidence": "station observations and ERA5 reference analysis"},
            {"stage": 5, "evidence": "bounded event-level mechanism interpretation"},
        ]
    )
    return source, story.save_figure(
        fig, out_dir / "00_qcore_full_experiment_flow", formats, dpi
    )


def pressure_rmse_source(pressure: pd.DataFrame) -> pd.DataFrame:
    required = [
        "feature",
        "scope",
        "pangu",
        "tianji",
        "pangu_ci_low",
        "pangu_ci_high",
        "tianji_ci_low",
        "tianji_ci_high",
        "delta_pangu_minus_tianji",
        "delta_ci_low",
        "delta_ci_high",
        "n",
        "represented_utc_dates",
    ]
    require_columns(pressure, required, "pressure-level paired quality")
    rows: List[Dict[str, object]] = []
    for spec in PRESSURE_FIGURE_SPECS:
        for scope in PRESSURE_SCOPES:
            part = pressure[
                (pressure["feature"] == spec["feature"])
                & (pressure["scope"] == scope)
            ]
            if len(part) != 1:
                raise ValueError(
                    f"Expected one pressure-quality row for {spec['feature']}/{scope}"
                )
            row = part.iloc[0]
            for source_key in ("pangu", "tianji"):
                rows.append(
                    {
                        **spec,
                        "scope": scope,
                        "scope_label": PRESSURE_SCOPE_LABELS[scope],
                        "source": source_key,
                        "rmse": float(row[source_key]),
                        "ci_low": float(row[f"{source_key}_ci_low"]),
                        "ci_high": float(row[f"{source_key}_ci_high"]),
                        "pangu_minus_tianji": float(
                            row["delta_pangu_minus_tianji"]
                        ),
                        "delta_ci_low": float(row["delta_ci_low"]),
                        "delta_ci_high": float(row["delta_ci_high"]),
                        "n": int(row["n"]),
                        "represented_utc_dates": int(
                            row["represented_utc_dates"]
                        ),
                        "reference": "ERA5 reference analysis",
                    }
                )
    return pd.DataFrame(rows)


def pressure_ratio_source(pressure: pd.DataFrame) -> pd.DataFrame:
    ratio_columns = [
        "tianji_to_pangu_ratio",
        "tianji_to_pangu_ratio_ci_low",
        "tianji_to_pangu_ratio_ci_high",
        "valid_relative_bootstrap_draws",
    ]
    require_columns(
        pressure,
        ratio_columns,
        "pressure-level quality; rerun the paired-quality analysis for exact paired RMSE-ratio CIs",
    )
    rows: List[Dict[str, object]] = []
    for spec in PRESSURE_FIGURE_SPECS:
        for scope in PRESSURE_SCOPES:
            part = pressure[
                (pressure["feature"] == spec["feature"])
                & (pressure["scope"] == scope)
            ]
            if len(part) != 1:
                raise ValueError(
                    f"Expected one pressure-ratio row for {spec['feature']}/{scope}"
                )
            row = part.iloc[0]
            ratio = float(row["tianji_to_pangu_ratio"])
            expected = float(row["tianji"]) / float(row["pangu"])
            if not np.isclose(ratio, expected, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(f"RMSE ratio identity failed for {spec['feature']}/{scope}")
            rows.append(
                {
                    **spec,
                    "scope": scope,
                    "scope_label": PRESSURE_SCOPE_LABELS[scope],
                    "tianji_to_pangu_rmse_ratio": ratio,
                    "ratio_ci_low": float(row["tianji_to_pangu_ratio_ci_low"]),
                    "ratio_ci_high": float(row["tianji_to_pangu_ratio_ci_high"]),
                    "pangu_rmse": float(row["pangu"]),
                    "tianji_rmse": float(row["tianji"]),
                    "n": int(row["n"]),
                    "represented_utc_dates": int(row["represented_utc_dates"]),
                    "valid_relative_bootstrap_draws": int(
                        row["valid_relative_bootstrap_draws"]
                    ),
                    "reference": "ERA5 reference analysis",
                }
            )
    return pd.DataFrame(rows)


def plot_pressure_ratio_overview(
    source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int
) -> List[str]:
    fig, ax = plt.subplots(figsize=story.FIGURE_SIZES["pressure"])
    fig.subplots_adjust(left=0.31, right=0.97, top=0.76, bottom=0.24)
    y_base = {
        str(spec["feature"]): float(len(PRESSURE_FIGURE_SPECS) - 1 - index)
        for index, spec in enumerate(PRESSURE_FIGURE_SPECS)
    }
    offsets = {"all_paired_test": 0.11, "true_low_visibility": -0.11}
    for row in source.itertuples(index=False):
        ratio = float(row.tianji_to_pangu_rmse_ratio)
        low = min(float(row.ratio_ci_low), ratio)
        high = max(float(row.ratio_ci_high), ratio)
        if low <= 1.0 <= high:
            color = "#697077"
        elif ratio < 1.0:
            color = story.TIANJI
        else:
            color = story.PANGU_DARK
        y = y_base[str(row.feature)] + offsets[str(row.scope)]
        ax.plot([low, high], [y, y], color=color, linewidth=1.8, solid_capstyle="round")
        ax.scatter(
            ratio,
            y,
            s=44,
            marker=PRESSURE_SCOPE_MARKERS[str(row.scope)],
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
    ax.axvline(1.0, color=story.INK, linewidth=0.9)
    ax.set_xscale("log", base=2)
    lows = source["ratio_ci_low"].to_numpy(dtype=float)
    highs = source["ratio_ci_high"].to_numpy(dtype=float)
    if np.any(lows <= 0.0):
        raise ValueError("RMSE-ratio confidence intervals must be positive")
    lo_log = float(np.log2(lows.min()))
    hi_log = float(np.log2(highs.max()))
    span = max(hi_log - lo_log, 0.6)
    ax.set_xlim(2.0 ** (lo_log - 0.10 * span), 2.0 ** (hi_log + 0.10 * span))
    ticks = np.asarray([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    visible = ticks[(ticks >= ax.get_xlim()[0]) & (ticks <= ax.get_xlim()[1])]
    ax.set_xticks(visible, [f"{value:g}" for value in visible])
    ax.set_yticks(
        [y_base[str(spec["feature"])] for spec in PRESSURE_FIGURE_SPECS],
        [str(spec["label"]) for spec in PRESSURE_FIGURE_SPECS],
    )
    ax.set_ylim(-0.48, len(PRESSURE_FIGURE_SPECS) - 0.50)
    ax.set_xlabel("Tianji RMSE / Pangu RMSE (log scale)")
    story.add_mainline_figure_title(fig, "Pressure-Level RMSE Ratios", y=0.95)
    ax.text(
        0.01,
        1.025,
        "← Tianji closer",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=story.TIANJI,
        ha="left",
    )
    ax.text(
        0.99,
        1.025,
        "Pangu closer →",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=story.PANGU_DARK,
        ha="right",
    )
    scope_handles = [
        Line2D(
            [0],
            [0],
            marker=PRESSURE_SCOPE_MARKERS[scope],
            linestyle="none",
            markersize=5.2,
            markerfacecolor=story.INK,
            markeredgecolor="white",
            label=PRESSURE_SCOPE_LABELS[scope],
        )
        for scope in PRESSURE_SCOPES
    ]
    ax.legend(
        handles=scope_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    story.style_axis(ax, horizontal_grid=False, vertical_grid=True)
    fig.text(
        0.5,
        0.055,
        "Pointwise RMSE against ERA5 reference analysis; 95% CIs use paired UTC-date bootstrap.",
        ha="center",
        fontsize=6.7,
        color="#4A5056",
    )
    return story.save_figure(fig, out_dir / "07_pressure_level_quality", formats, dpi)


def plot_pressure_feature_rmse(
    source: pd.DataFrame,
    spec: Mapping[str, str],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    feature = str(spec["feature"])
    part = source[source["feature"] == feature]
    if len(part) != 2 * len(PRESSURE_SCOPES):
        raise ValueError(f"Incomplete absolute RMSE source table for {feature}")
    y_base = {
        scope: float(len(PRESSURE_SCOPES) - 1 - index)
        for index, scope in enumerate(PRESSURE_SCOPES)
    }
    offsets = {"pangu": 0.10, "tianji": -0.10}
    fig, ax = plt.subplots(figsize=(story.SINGLE_COLUMN_WIDTH, 2.75))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.76, bottom=0.24)
    for source_key in ("pangu", "tianji"):
        for row in part[part["source"] == source_key].itertuples(index=False):
            estimate = float(row.rmse)
            low = min(float(row.ci_low), estimate)
            high = max(float(row.ci_high), estimate)
            ax.errorbar(
                estimate,
                y_base[str(row.scope)] + offsets[source_key],
                xerr=np.asarray([[estimate - low], [high - estimate]]),
                fmt=story.SOURCE_MARKERS[source_key],
                markersize=5.5,
                color=story.SOURCE_COLORS[source_key],
                ecolor=story.SOURCE_COLORS[source_key],
                elinewidth=1.2,
                capsize=2.5,
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=story.SOURCE_LABELS[source_key]
                if str(row.scope) == PRESSURE_SCOPES[0]
                else None,
                zorder=3,
            )
    ax.set_yticks(
        [y_base[scope] for scope in PRESSURE_SCOPES],
        [PRESSURE_SCOPE_LABELS[scope] for scope in PRESSURE_SCOPES],
    )
    ax.set_xlim(0.0, max(float(part["ci_high"].max()) * 1.10, 1.0e-6))
    ax.set_ylim(-0.42, 1.42)
    ax.set_xlabel(f"RMSE vs ERA5 analysis ({spec['unit']})")
    story.add_mainline_figure_title(fig, str(spec["title"]), y=0.95)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    story.style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return story.save_figure(fig, out_dir / str(spec["output"]), formats, dpi)


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("dpi must be at least 300")
    fair_root = Path(args.fair_eval_root).expanduser().resolve()
    quality_dir = story.resolve_quality_dir(Path(args.paired_quality_dir))
    event_dir = Path(args.event_analysis_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    formats = story.ordered_formats(args.formats)
    scopes = {
        "both": ["all_paired_test", "true_low_visibility"],
        "all": ["all_paired_test"],
        "low": ["true_low_visibility"],
    }[args.surface_view]

    fair_report, metrics, gap_draws = load_t925_performance(fair_root)
    quality_report, quality = story.load_quality(
        quality_dir, "true_low_visibility" in scopes
    )
    event_report = json.loads(
        (event_dir / "upper_air_disagreement_analysis_report.json").read_text(encoding="utf-8")
    )
    if event_report.get("status") != "completed" or event_report.get("new_training_models_used") != 0:
        raise ValueError("upper-air disagreement analysis is incomplete or used new training")
    events = pd.read_csv(event_dir / "event_case_control_samples.csv.gz")
    event_counts = pd.read_csv(event_dir / "upper_air_disagreement_event_counts.csv")
    upper = pd.read_csv(event_dir / "upper_air_disagreement_bias_source_data.csv")
    require_columns(events, story.EVENT_OBSERVATION_COLUMNS, "T925 event samples")
    require_columns(event_counts, ["case_category", "n"], "T925 event counts")

    event_quality = story.event_observation_advantage_source(events)
    event_state = story.event_forecast_state_contrast_source(events)
    event_bias = story.event_observation_bias_source(events)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, List[str]] = {}

    flow, generated["00_qcore_full_experiment_flow"] = plot_t925_flow(
        out_dir, formats, args.dpi
    )
    flow.to_csv(source_dir / "00_qcore_full_experiment_flow.csv", index=False)

    ap = story.endpoint_source(metrics, gap_draws, "low_vis_ap")
    ap.to_csv(source_dir / "01_qcore_lowvis_ap.csv", index=False)
    generated["01_qcore_lowvis_ap"] = story.plot_endpoint(
        ap,
        "Low-vis average precision (AP)",
        "Fair Q-Core + T925 Low-vis Average Precision",
        "01_qcore_lowvis_ap",
        out_dir,
        formats,
        args.dpi,
        "AP uses no decision threshold.",
    )
    recall = story.endpoint_source(metrics, gap_draws, "low_vis_recall_matched_fpr")
    recall.to_csv(source_dir / "02_qcore_matched_fpr_recall.csv", index=False)
    generated["02_qcore_matched_fpr_recall"] = story.plot_endpoint(
        recall,
        "Low-vis recall",
        "Fair Q-Core + T925 Recall at Matched FPR",
        "02_qcore_matched_fpr_recall",
        out_dir,
        formats,
        args.dpi,
        "Thresholds were selected on validation at one target FPR and frozen for test.",
    )
    argmax = story.qcore_argmax_source(metrics)
    argmax.to_csv(source_dir / "02a_qcore_argmax_lowvis_overview.csv", index=False)
    generated["02a_qcore_argmax_lowvis_overview"] = story.plot_qcore_argmax_overview(
        argmax,
        out_dir,
        formats,
        args.dpi,
        title="Fair Q-Core + T925 Argmax Performance",
        output_name="02a_qcore_argmax_lowvis_overview",
    )

    surface_specs = [
        ("T2M", "2-m temperature", "°C", "04_surface_t2m_quality"),
        ("WSPD10", "10-m wind speed", r"m s$^{-1}$", "05_surface_wspd10_quality"),
        ("MSLP", "Mean sea-level pressure", "hPa", "06_surface_mslp_quality"),
    ]
    for feature, label, unit, name in surface_specs:
        source = story.surface_feature_source(
            quality["surface"], quality["surface_pairs"], feature, scopes
        )
        source.to_csv(source_dir / f"{name}.csv", index=False)
        generated[name] = story.plot_surface_feature(
            source, label, unit, scopes, name, out_dir, formats, args.dpi
        )

    pressure_ratio = pressure_ratio_source(quality["pressure"])
    pressure_ratio.to_csv(source_dir / "07_pressure_level_quality.csv", index=False)
    generated["07_pressure_level_quality"] = plot_pressure_ratio_overview(
        pressure_ratio, out_dir, formats, args.dpi
    )
    pressure_rmse = pressure_rmse_source(quality["pressure"])
    for spec in PRESSURE_FIGURE_SPECS:
        feature_source = pressure_rmse[
            pressure_rmse["feature"] == str(spec["feature"])
        ].copy()
        output_name = str(spec["output"])
        feature_source.to_csv(source_dir / f"{output_name}.csv", index=False)
        generated[output_name] = plot_pressure_feature_rmse(
            feature_source, spec, out_dir, formats, args.dpi
        )
    unique = story.event_source(event_counts)
    unique.to_csv(source_dir / "08_qcore_t925_unique_event_hits.csv", index=False)
    generated["08_qcore_t925_unique_event_hits"] = story.plot_events(
        unique, out_dir, formats, args.dpi
    )

    event_quality.to_csv(source_dir / "09_qcore_t925_observation_anchored_advantage.csv", index=False)
    generated["09_qcore_t925_observation_anchored_advantage"] = story.plot_event_observation_advantage(
        event_quality, out_dir, formats, args.dpi
    )
    event_state.to_csv(source_dir / "09b_qcore_t925_disagreement_state.csv", index=False)
    generated["09b_qcore_t925_disagreement_state"] = story.plot_event_forecast_state_contrast(
        event_state, out_dir, formats, args.dpi
    )
    event_bias.to_csv(source_dir / "09c_qcore_t925_surface_observation_bias.csv", index=False)
    generated["09c_qcore_t925_surface_observation_bias"] = story.plot_event_observation_bias(
        event_bias, out_dir, formats, args.dpi
    )
    upper.to_csv(source_dir / "09d_qcore_t925_upper_air_reference_bias.csv", index=False)
    generated["09d_qcore_t925_upper_air_reference_bias"] = plot_upper_air(
        upper, out_dir, formats, args.dpi
    )

    qc = story.qc_source(quality["qc"])
    qc.to_csv(source_dir / "10_pressure_qc.csv", index=False)
    generated["10_pressure_qc"] = story.plot_qc(qc, out_dir, formats, args.dpi)

    report = {
        "status": "completed",
        "story_version": "q_core_t925_pre_factorial_attribution",
        "fair_performance_report": str(fair_root / "analysis" / "qcore_t925_fair_analysis_report.json"),
        "fair_performance_status": fair_report.get("status"),
        "paired_quality_status": quality_report.get("status"),
        "event_analysis": str(event_dir),
        "generated": generated,
        "shapley": {
            "status": "intentionally_omitted",
            "reason": "the previous Shapley matrix excluded T925; wait for completed MHTPW 32-mask factorial",
        },
        "recomputation_policy": {
            "performance_and_event_panels": "recomputed from q-core+T925 three-seed endpoints",
            "source_quality_and_qc_panels": (
                "re-exported from model-independent paired diagnoses; pressure-level "
                "RMSE ratios and their CIs are computed jointly in the UTC-date bootstrap"
            ),
        },
        "pressure_level_figure_roles": {
            "07_pressure_level_quality": (
                "cross-unit overview using exact paired Tianji/Pangu RMSE ratios for "
                "all-test and observed-Low-vis scopes"
            ),
            "07a_to_07d": (
                "native-unit pointwise RMSE and source-specific 95% CIs for the same "
                "four non-redundant pressure-level quantities and both scopes"
            ),
            "09d": (
                "endpoint-conditioned signed bias and paired MAE for T925, Q1000, "
                "Q925 and 925-hPa wind speed"
            ),
        },
        "claim_limits": [
            "ERA5 is a reference analysis, not truth or an independent forecast source.",
            "Pangu RH925 is derived from T925/Q925 and is not independent evidence.",
            "Event panels are endpoint-conditioned descriptive associations, not causal interventions.",
        ],
    }
    (out_dir / "qcore_t925_evidence_story_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] q-core+T925 evidence-story figures written to {out_dir}")


if __name__ == "__main__":
    main()
