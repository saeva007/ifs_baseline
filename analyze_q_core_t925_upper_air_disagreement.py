#!/usr/bin/env python3
"""Upper-air source quality in q-core+T925 model-disagreement Low-vis cases.

This diagnostic is deliberately independent of the 32-mask factorial.  Event
categories are defined from three-seed Pangu/Tianji endpoint probabilities at
validation-matched false-positive rates.  Forecast T925, Q925 and 925-hPa wind
speed are then compared point by point with ERA5 reference analysis at the
last input step (valid time).  ERA5 is never described as truth.

The primary estimates are signed bias and paired MAE.  Confidence intervals
use a joint UTC-valid-date block bootstrap so all stations on a sampled weather
day are retained together.  The result is endpoint-conditioned descriptive
evidence, not a causal source intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_q_core_hybrid_factorial import (
    argmax_metrics,
    attach_observations,
    match_fpr_threshold,
)
from analyze_q_core_t925_fair import load_all
from analyze_q_core_t925_joint_structure import DatasetLayout, load_layout, metadata
from plot_pangu_qcore_mechanism_ppt import (
    FIGURE_SIZES,
    GRID_GREY,
    INK,
    LIGHT_GREY,
    PANGU,
    TIANJI,
    ordered_formats,
    save_figure,
    style_axis,
)


FEATURE_SPECS: Tuple[Mapping[str, object], ...] = (
    {
        "feature": "T_925",
        "label": "925-hPa temperature",
        "unit": "K",
        "members": ("T_925",),
    },
    {
        "feature": "Q_925",
        "label": "925-hPa specific humidity",
        "unit": r"g kg$^{-1}$",
        "members": ("Q_925",),
    },
    {
        "feature": "WSPD925",
        "label": "925-hPa wind speed",
        "unit": r"m s$^{-1}$",
        "members": ("U_925", "V_925"),
    },
)
CATEGORIES = (
    ("tianji_hit_pangu_miss", "Physics-only\nhit"),
    ("pangu_hit_tianji_miss", "AI-only\nhit"),
)


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).replace(":", ",").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="Three-seed q-core+T925 evaluator root.")
    parser.add_argument("--pangu-data-dir", required=True)
    parser.add_argument("--tianji-data-dir", required=True)
    parser.add_argument("--era5-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="42:2025:20260702")
    parser.add_argument("--pangu-tag", default="pangu2025_q_core_t925_no_rh2m")
    parser.add_argument("--tianji-tag", default="tianji")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    parser.add_argument("--min-event-coverage", type=float, default=0.98)
    parser.add_argument("--obs-root", default="")
    parser.add_argument("--paper-eval-dir", default="/public/home/putianshu/vis_mlp/paper_eval")
    parser.add_argument("--formats", default="svg:pdf:png:tiff")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_events(
    val: Mapping[Tuple[int, str], object],
    test: Mapping[Tuple[int, str], object],
    seeds: Sequence[int],
    low_vis_threshold_m: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Define endpoint disagreements from seed-mean Low-vis probabilities."""

    pangu_argmax_fpr = [
        argmax_metrics(val[(seed, "pangu")].y, val[(seed, "pangu")].pred)[
            "low_vis_fpr_argmax"
        ]
        for seed in seeds
    ]
    target_fpr = float(np.median(np.asarray(pangu_argmax_fpr, dtype=float)))
    mean_val = {
        source: np.mean(
            np.stack([val[(seed, source)].score for seed in seeds], axis=0), axis=0
        )
        for source in ("pangu", "tianji")
    }
    mean_test = {
        source: np.mean(
            np.stack([test[(seed, source)].score for seed in seeds], axis=0), axis=0
        )
        for source in ("pangu", "tianji")
    }
    reference_val = val[(seeds[0], "pangu")]
    reference_test = test[(seeds[0], "pangu")]
    thresholds: Dict[str, float] = {}
    achieved: Dict[str, float] = {}
    for source in ("pangu", "tianji"):
        thresholds[source], achieved[source] = match_fpr_threshold(
            reference_val.y, mean_val[source], target_fpr
        )

    vis = pd.to_numeric(reference_test.frame["vis_raw_m"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(vis)):
        raise ValueError("test per-sample visibility contains non-finite values")
    y_low = reference_test.y <= 1
    strict_low = vis < float(low_vis_threshold_m)
    if not np.array_equal(y_low, strict_low):
        mismatch = int(np.count_nonzero(y_low != strict_low))
        raise ValueError(
            f"class-label Low-vis membership differs from visibility < {low_vis_threshold_m:g} m "
            f"for {mismatch} rows"
        )
    pangu_hit = mean_test["pangu"] >= thresholds["pangu"]
    tianji_hit = mean_test["tianji"] >= thresholds["tianji"]
    category = np.full(len(vis), "not_low_visibility", dtype=object)
    category[y_low & pangu_hit & tianji_hit] = "both_hit"
    category[y_low & ~pangu_hit & ~tianji_hit] = "both_miss"
    category[y_low & ~pangu_hit & tianji_hit] = "tianji_hit_pangu_miss"
    category[y_low & pangu_hit & ~tianji_hit] = "pangu_hit_tianji_miss"

    keep = y_low & np.isin(category, [item[0] for item in CATEGORIES])
    events = reference_test.frame.loc[
        keep, ["time_utc", "time_key", "station_key", "vis_raw_m"]
    ].copy()
    events["case_category"] = category[keep]
    events["pangu_seed_mean_low_vis_probability"] = mean_test["pangu"][keep]
    events["tianji_seed_mean_low_vis_probability"] = mean_test["tianji"][keep]
    if events[["time_key", "station_key"]].duplicated().any():
        raise ValueError("event prediction rows contain duplicate station-time keys")
    counts = events["case_category"].value_counts().to_dict()
    for category, _ in CATEGORIES:
        if int(counts.get(category, 0)) < 100:
            raise ValueError(f"{category}: fewer than 100 event rows")
    return events.reset_index(drop=True), {
        "target_validation_fpr": target_fpr,
        "pangu_seed_argmax_validation_fpr": pangu_argmax_fpr,
        "seed_mean_probability_thresholds": thresholds,
        "achieved_seed_mean_validation_fpr": achieved,
        "event_counts": {str(key): int(value) for key, value in counts.items()},
    }


def event_positions(layout: DatasetLayout, events: pd.DataFrame) -> np.ndarray:
    frame = metadata(layout.path, "test")
    lookup = pd.Series(
        frame["source_row"].to_numpy(dtype=np.int64),
        index=pd.MultiIndex.from_frame(frame[["time_key", "station_key"]]),
    )
    wanted = pd.MultiIndex.from_frame(events[["time_key", "station_key"]])
    positions = lookup.reindex(wanted)
    if positions.isna().any():
        raise ValueError(
            f"{layout.path}: {int(positions.isna().sum())} event station-time rows are missing"
        )
    return positions.to_numpy(dtype=np.int64)


def restrict_events_to_common_layouts(
    events: pd.DataFrame,
    layouts: Mapping[str, DatasetLayout],
    minimum_coverage: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("min-event-coverage must be in (0, 1]")
    wanted = pd.MultiIndex.from_frame(events[["time_key", "station_key"]])
    common = np.ones(len(events), dtype=bool)
    missing_by_source: Dict[str, int] = {}
    for source, layout in layouts.items():
        frame = metadata(layout.path, "test")
        available = pd.MultiIndex.from_frame(frame[["time_key", "station_key"]])
        present = wanted.isin(available)
        missing_by_source[source] = int(np.count_nonzero(~present))
        common &= present
    coverage = float(np.mean(common)) if len(common) else 0.0
    if coverage < minimum_coverage:
        raise ValueError(
            f"common Pangu/Tianji/ERA5 event coverage {coverage:.6f} is below "
            f"{minimum_coverage:.6f}; missing={missing_by_source}"
        )
    aligned = events.loc[common].reset_index(drop=True)
    counts_before = events["case_category"].value_counts().to_dict()
    counts_after = aligned["case_category"].value_counts().to_dict()
    for category, _ in CATEGORIES:
        if int(counts_after.get(category, 0)) < 100:
            raise ValueError(f"{category}: fewer than 100 common-reference event rows")
    return aligned, {
        "rows_before_common_reference": int(len(events)),
        "rows_after_common_reference": int(len(aligned)),
        "coverage": coverage,
        "minimum_required_coverage": float(minimum_coverage),
        "missing_by_source": missing_by_source,
        "category_counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "category_counts_after": {str(k): int(v) for k, v in counts_after.items()},
    }


def last_step_columns(layout: DatasetLayout, names: Sequence[str]) -> Dict[str, int]:
    missing = [name for name in names if name not in layout.order]
    if missing:
        raise KeyError(f"{layout.path}: dynamic feature order lacks {missing}")
    return {
        name: (layout.window - 1) * layout.dyn_vars + layout.order.index(name)
        for name in names
    }


def extract_source_state(layout: DatasetLayout, positions: np.ndarray) -> Dict[str, np.ndarray]:
    names = (
        "T2M",
        "MSLP",
        "WSPD10",
        "T_925",
        "RH_925",
        "Q_925",
        "U_925",
        "V_925",
    )
    columns = last_step_columns(layout, names)
    x = np.load(layout.path / "X_test.npy", mmap_mode="r")
    expected = layout.window * layout.dyn_vars + layout.fe_dim + 6
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"{layout.path}/X_test.npy: expected width {expected}, got {x.shape}")
    result = {
        name: np.asarray(x[positions, column], dtype=np.float64)
        for name, column in columns.items()
    }
    units = dict(layout.config.get("canonical_dynamic_units", {}))
    if units.get("T_925") != "K" or units.get("Q_925") != "kg kg-1":
        raise ValueError(
            f"{layout.path}: unexpected T925/Q925 units "
            f"{units.get('T_925')!r}/{units.get('Q_925')!r}"
        )
    result["Q_925"] *= 1000.0
    result["WSPD925"] = np.hypot(result["U_925"], result["V_925"])
    return result


def attach_source_state(
    events: pd.DataFrame, layouts: Mapping[str, DatasetLayout]
) -> pd.DataFrame:
    output = events.copy()
    for dataset_key, layout in layouts.items():
        positions = event_positions(layout, events)
        values = extract_source_state(layout, positions)
        for feature in ("T_925", "Q_925", "WSPD925"):
            output[f"{feature}_{dataset_key}"] = values[feature]
        if dataset_key in {"pangu", "tianji"}:
            for feature in ("T2M", "WSPD10", "RH_925", "MSLP"):
                output[f"{feature}_{dataset_key}"] = values[feature]
    return output


def block_bootstrap_summary(
    events: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    if iterations < 200:
        raise ValueError("Use at least 200 UTC-date bootstrap iterations")
    work = events.copy()
    work["utc_valid_date"] = pd.to_datetime(
        work["time_utc"], errors="raise", utc=True
    ).dt.floor("D")
    required_values = [
        f"{feature}_{dataset}"
        for feature in ("T_925", "Q_925", "WSPD925")
        for dataset in ("tianji", "pangu", "era5")
    ]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for category_key, category_label in CATEGORIES:
        category = work[work["case_category"] == category_key].copy()
        finite = np.ones(len(category), dtype=bool)
        for column in required_values:
            values = pd.to_numeric(category[column], errors="coerce").to_numpy(dtype=float)
            finite &= np.isfinite(values)
        category = category.loc[finite].reset_index(drop=True)
        if len(category) < 100:
            raise ValueError(f"{category_key}: fewer than 100 complete upper-air rows")
        dates = pd.Index(sorted(category["utc_valid_date"].unique()))
        if len(dates) < 10:
            raise ValueError(f"{category_key}: fewer than 10 represented UTC dates")
        date_draws = rng.integers(0, len(dates), size=(iterations, len(dates)))

        for spec in FEATURE_SPECS:
            feature = str(spec["feature"])
            reference = category[f"{feature}_era5"].to_numpy(dtype=float)
            errors = {
                "physics": category[f"{feature}_tianji"].to_numpy(dtype=float) - reference,
                "ai": category[f"{feature}_pangu"].to_numpy(dtype=float) - reference,
            }
            daily_rows: Dict[str, np.ndarray] = {}
            daily_n = category.groupby("utc_valid_date", sort=True).size().reindex(dates).to_numpy(dtype=float)
            draw_n = daily_n[date_draws].sum(axis=1)
            if bool((draw_n <= 0).any()):
                raise RuntimeError(f"{feature}/{category_key}: empty bootstrap draw")
            for role in ("physics", "ai"):
                temp = pd.DataFrame(
                    {
                        "date": category["utc_valid_date"].to_numpy(),
                        "error": errors[role],
                        "absolute_error": np.abs(errors[role]),
                    }
                )
                daily = temp.groupby("date", sort=True).agg(
                    error=("error", "sum"),
                    absolute_error=("absolute_error", "sum"),
                ).reindex(dates)
                daily_rows[f"{role}_error"] = daily["error"].to_numpy(dtype=float)
                daily_rows[f"{role}_absolute_error"] = daily["absolute_error"].to_numpy(dtype=float)

            mae_draws = {
                role: daily_rows[f"{role}_absolute_error"][date_draws].sum(axis=1) / draw_n
                for role in ("physics", "ai")
            }
            mae_delta_draws = mae_draws["physics"] - mae_draws["ai"]
            mae_delta = float(np.mean(np.abs(errors["physics"])) - np.mean(np.abs(errors["ai"])))
            delta_low, delta_high = np.quantile(mae_delta_draws, [0.025, 0.975])
            for role, source_label, source_key in (
                ("physics", "Physics forecast", "tianji"),
                ("ai", "AI forecast", "pangu"),
            ):
                bias_draws = daily_rows[f"{role}_error"][date_draws].sum(axis=1) / draw_n
                bias_low, bias_high = np.quantile(bias_draws, [0.025, 0.975])
                mae_low, mae_high = np.quantile(mae_draws[role], [0.025, 0.975])
                rows.append(
                    {
                        "feature": feature,
                        "label": spec["label"],
                        "unit": spec["unit"],
                        "case_category": category_key,
                        "case_label": category_label.replace("\n", " "),
                        "source_role": role,
                        "source_label": source_label,
                        "source_dataset": source_key,
                        "bias_forecast_minus_reference": float(np.mean(errors[role])),
                        "bias_ci_low": float(bias_low),
                        "bias_ci_high": float(bias_high),
                        "mae": float(np.mean(np.abs(errors[role]))),
                        "mae_ci_low": float(mae_low),
                        "mae_ci_high": float(mae_high),
                        "paired_mae_difference_physics_minus_ai": mae_delta,
                        "paired_mae_difference_ci_low": float(delta_low),
                        "paired_mae_difference_ci_high": float(delta_high),
                        "n_complete_paired": int(len(category)),
                        "represented_utc_dates": int(len(dates)),
                        "reference": "ERA5 reference analysis",
                        "sequence_position": "last input step at valid time",
                        "bootstrap_unit": "UTC_valid_date",
                        "bootstrap_iterations": int(iterations),
                        "selection_note": (
                            "three-seed endpoint disagreement at validation-matched FPR; "
                            "descriptive association, not a causal source intervention"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_summary(
    source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int
) -> List[str]:
    roles = (
        ("physics", "Physics forecast", TIANJI, "o"),
        ("ai", "AI forecast", PANGU, "D"),
    )
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["event_bias"])
    fig.subplots_adjust(left=0.095, right=0.985, top=0.73, bottom=0.27, wspace=0.46)
    for panel_index, (ax, spec) in enumerate(zip(axes, FEATURE_SPECS)):
        feature = str(spec["feature"])
        part = source[source["feature"] == feature]
        x = np.arange(len(CATEGORIES), dtype=float)
        for role_index, (role, role_label, color, marker) in enumerate(roles):
            selected = []
            for category_key, _ in CATEGORIES:
                row = part[
                    (part["case_category"] == category_key)
                    & (part["source_role"] == role)
                ]
                if len(row) != 1:
                    raise ValueError(f"missing summary row for {feature}/{category_key}/{role}")
                selected.append(row.iloc[0])
            estimates = np.asarray(
                [float(row["bias_forecast_minus_reference"]) for row in selected]
            )
            lows = np.asarray([float(row["bias_ci_low"]) for row in selected])
            highs = np.asarray([float(row["bias_ci_high"]) for row in selected])
            visual_lows = np.minimum(lows, estimates)
            visual_highs = np.maximum(highs, estimates)
            xpos = x + (-0.11 if role_index == 0 else 0.11)
            ax.errorbar(
                xpos,
                estimates,
                yerr=np.vstack(
                    [estimates - visual_lows, visual_highs - estimates]
                ),
                fmt=marker,
                markersize=5.2,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.65,
                ecolor=color,
                elinewidth=1.35,
                capsize=3.0,
                label=role_label if panel_index == 0 else None,
                zorder=3,
            )
        ax.axhline(0.0, color=INK, linewidth=0.85, zorder=1)
        tick_labels = []
        for category_key, category_label in CATEGORIES:
            row = part[
                (part["case_category"] == category_key)
                & (part["source_role"] == "physics")
            ].iloc[0]
            tick_labels.append(
                f"{category_label}\nn={int(row['n_complete_paired']):,}"
            )
        ax.set_xticks(x, tick_labels)
        ax.set_title(str(spec["label"]), loc="left", fontsize=9.0, fontweight="bold", pad=5)
        ax.set_ylabel(f"Forecast − ERA5 analysis ({spec['unit']})")
        ax.text(
            -0.17,
            1.05,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
        )
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.65)
        ax.grid(axis="x", visible=False)
        style_axis(ax, horizontal_grid=True, vertical_grid=False)

        data_ymin, data_ymax = ax.get_ylim()
        data_span = max(data_ymax - data_ymin, 1e-6)
        ax.set_ylim(data_ymin, data_ymax + 0.30 * data_span)
        for category_index, (category_key, _) in enumerate(CATEGORIES):
            row = part[
                (part["case_category"] == category_key)
                & (part["source_role"] == "physics")
            ].iloc[0]
            delta = float(row["paired_mae_difference_physics_minus_ai"])
            low = float(row["paired_mae_difference_ci_low"])
            high = float(row["paired_mae_difference_ci_high"])
            ax.text(
                0.25 + 0.50 * category_index,
                0.98,
                f"ΔMAE {delta:+.2f}\n[{low:+.2f}, {high:+.2f}]",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.2,
                color="#40464D",
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": LIGHT_GREY,
                    "linewidth": 0.55,
                },
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Upper-air Forecast Biases in Model-disagreement Low-vis Cases",
        x=0.5,
        y=0.985,
        fontsize=10.8,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.055,
        "Signed bias (forecast − ERA5 reference analysis); 95% CIs use UTC-date block bootstrap. "
        "ΔMAE = Physics − AI (negative favours Physics).\n"
        "ERA5 is a reference analysis, not truth; endpoint-conditioned diagnostic, not a causal intervention.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#4A5056",
    )
    return save_figure(
        fig,
        out_dir / "09d_disagreement_case_upper_air_reference_bias",
        formats,
        dpi,
    )


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("formal upper-air diagnosis requires three unique seeds")
    if args.bootstrap_iters < 200:
        raise ValueError("bootstrap-iters must be at least 200")
    if args.dpi < 300:
        raise ValueError("dpi must be at least 300")
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    layouts = {
        "pangu": load_layout(Path(args.pangu_data_dir).expanduser().resolve(), "pangu"),
        "tianji": load_layout(Path(args.tianji_data_dir).expanduser().resolve(), "tianji"),
        "era5": load_layout(Path(args.era5_data_dir).expanduser().resolve(), "era5_reference_analysis"),
    }
    for key, layout in layouts.items():
        if layout.feature_set != "q_core_t925_no_rh2m":
            raise ValueError(f"{key}: expected q_core_t925_no_rh2m, got {layout.feature_set}")

    val, test = load_all(eval_root, seeds, args.tianji_tag, args.pangu_tag)
    events, event_info = classify_events(
        val, test, seeds, float(args.low_vis_threshold_m)
    )
    events, alignment_info = restrict_events_to_common_layouts(
        events, layouts, float(args.min_event_coverage)
    )
    events = attach_source_state(events, layouts)
    observation_info: Dict[str, object] = {"enabled": False}
    if args.obs_root:
        events, detail = attach_observations(
            events,
            Path(args.obs_root).expanduser().resolve(),
            Path(args.paper_eval_dir).expanduser().resolve(),
        )
        observation_info = {"enabled": True, **detail}
    summary = block_bootstrap_summary(
        events, int(args.bootstrap_iters), int(args.bootstrap_seed)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "upper_air_disagreement_bias_source_data.csv", index=False)
    counts = (
        events.groupby("case_category", as_index=False)
        .agg(n=("case_category", "size"), represented_utc_dates=("time_utc", lambda x: pd.to_datetime(x, utc=True).dt.floor("D").nunique()))
    )
    counts.to_csv(out_dir / "upper_air_disagreement_event_counts.csv", index=False)
    events.to_csv(out_dir / "event_case_control_samples.csv.gz", index=False, compression="gzip")
    generated: List[str] = []
    if not args.no_figure:
        generated = plot_summary(
            summary, out_dir, ordered_formats(args.formats), int(args.dpi)
        )

    report = {
        "status": "completed",
        "analysis_type": "diagnostic_only_no_training",
        "new_training_models_used": 0,
        "event_definition": (
            "three-seed mean Low-vis probabilities from q-core+T925 endpoints; "
            "source-specific thresholds selected on validation at the median Pangu argmax FPR"
        ),
        "eval_root": str(eval_root),
        "seeds": seeds,
        "source_tags": {"physics_tianji": args.tianji_tag, "ai_pangu": args.pangu_tag},
        "event_thresholds": event_info,
        "event_reference_alignment": alignment_info,
        "surface_observation_alignment": observation_info,
        "upper_air_features": [str(spec["feature"]) for spec in FEATURE_SPECS],
        "reference": "ERA5 reference analysis, not truth",
        "sequence_position": "last input step at valid time",
        "bootstrap": {
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
            "unit": "UTC_valid_date",
            "spatial_dependence": "all stations on a sampled date are resampled jointly",
        },
        "dataset_provenance": {
            key: {
                "path": str(layout.path),
                "config_sha256": sha256_file(layout.path / "dataset_build_config.json"),
            }
            for key, layout in layouts.items()
        },
        "outputs": {
            "source_data": "upper_air_disagreement_bias_source_data.csv",
            "event_counts": "upper_air_disagreement_event_counts.csv",
            "event_samples": "event_case_control_samples.csv.gz",
            "figures": generated,
        },
        "interpretation_constraints": [
            "ERA5 is a reference analysis rather than truth or an independent forecast source.",
            "T925 and Q925 are native source variables; WSPD925 is derived from U925 and V925.",
            "The estimates are conditioned on endpoint disagreement and are descriptive, not causal.",
            "This analysis does not require or use the unfinished 32-mask factorial.",
        ],
    }
    (out_dir / "upper_air_disagreement_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] upper-air disagreement diagnosis written to {out_dir}")


if __name__ == "__main__":
    main()
