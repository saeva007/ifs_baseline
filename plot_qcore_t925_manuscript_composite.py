#!/usr/bin/env python3
"""Redraw the Tianji--Pangu endpoint and variable-quality manuscript figure.

This script never pastes pre-rendered panels.  Every axis is drawn from the
endpoint-comparison and paired variable-quality tables on one shared canvas so
panel geometry, typography, and margins remain consistent.  Two systematic-
bias layouts are exported for direct visual comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_source_palette import SOURCE_COLORS, SOURCE_DARK_COLORS
from paper_figure_geometry import (
    ENDPOINT_BAR_WIDTH,
    HORIZONTAL_BAR_HEIGHT,
    INTERVAL_CAPSIZE,
    INTERVAL_CAPTHICK,
    INTERVAL_LINEWIDTH,
    INTERVAL_MARKERSIZE,
)
import plot_common_variable_error_regimes as quality_plot
import plot_q_core_task_tail_fidelity_preview as tail_plot
import plot_viscast_controlled_attribution_composite as controlled_plot


FIGURE_WIDTH = 7.60
TIANJI = SOURCE_COLORS["tianji"]
PANGU = SOURCE_COLORS["pangu"]
TIANJI_DARK = SOURCE_DARK_COLORS["tianji"]
PANGU_DARK = SOURCE_DARK_COLORS["pangu"]
INK = "#17191B"
GRID = "#E8EAEB"
PACKAGE_COLORS = {
    "M": "#4D8C57",
    "H": "#4E79A7",
    "T": "#C66A3D",
    "T2": "#C66A3D",
    "P": "#70757A",
    "W": "#C59A32",
    "B": "#7A6F9B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(
            "/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/"
            "q_core_t925_factorial/qcore_t925_mhtpw_formal_v2_20260728/analysis"
        ),
        help="Directory containing hybrid factorial analysis CSV files.",
    )
    p.add_argument(
        "--endpoint-dir",
        type=Path,
        default=Path(
            "/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/"
            "q_core_t925_fair/qcore_t925_fair_formal_v1_20260721/analysis"
        ),
        help="Completed fair-comparison analysis directory used for AP and matched-FPR recall.",
    )
    p.add_argument(
        "--upper-air-dir",
        type=Path,
        default=None,
        help="Directory containing upper_air_disagreement_bias_source_data.csv; auto-resolved when omitted.",
    )
    p.add_argument(
        "--paired-quality-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the pressure-level and surface paired-quality CSV files "
            "used by the four-panel variable-quality analysis."
        ),
    )
    p.add_argument(
        "--tail-analysis-dir",
        type=Path,
        default=None,
        help="Directory containing joint-tail placement metrics and UTC-date bootstrap intervals.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--figure-stem", default="fig_tianji_pangu_main_composite")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_columns(df: pd.DataFrame, names: Iterable[str], source: Path) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def normalize_mask(values: pd.Series, width: int) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


def resolve_upper_air_dir(analysis_dir: Path, requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser().resolve())
    candidates.extend([analysis_dir, analysis_dir.parent])
    for candidate in candidates:
        if (candidate / "upper_air_disagreement_bias_source_data.csv").is_file():
            return candidate
    for root in (analysis_dir.parent, analysis_dir.parent.parent, analysis_dir.parents[2]):
        matches = sorted(
            root.glob("**/upper_air_disagreement_bias_source_data.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0].parent
    raise FileNotFoundError(
        "Could not locate upper_air_disagreement_bias_source_data.csv; pass --upper-air-dir."
    )


def resolve_paired_quality_dir(
    analysis_dir: Path,
    endpoint_dir: Path,
    requested: Path | None,
) -> Path:
    required = (
        "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv",
        "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv",
        "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv",
    )
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser().resolve())
    candidates.extend(
        [
            analysis_dir,
            analysis_dir.parent,
            endpoint_dir,
            endpoint_dir.parent,
        ]
    )
    for candidate in candidates:
        for resolved in (candidate, candidate / "analysis"):
            if all((resolved / name).is_file() for name in required):
                return resolved
    raise FileNotFoundError(
        "Could not locate the paired variable-quality CSV files; pass --paired-quality-dir."
    )


def load_inputs(endpoint_dir: Path) -> Dict[str, pd.DataFrame]:
    paths = {
        "metrics": endpoint_dir / "qcore_t925_metrics_by_seed.csv",
        "gap": endpoint_dir / "qcore_t925_bootstrap_gap_draws.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    tables = {
        "metrics": pd.read_csv(paths["metrics"], dtype={"mask": str}),
        "gap": pd.read_csv(paths["gap"]),
    }
    require_columns(tables["metrics"], ["mask", "seed", "low_vis_ap", "low_vis_recall_matched_fpr"], paths["metrics"])
    require_columns(tables["gap"], ["metric", "delta_all1_minus_all0"], paths["gap"])
    return tables


def load_mechanism_inputs(
    analysis_dir: Path,
    upper_air_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shapley_path = analysis_dir / "hybrid_exact_shapley_effects.csv"
    bias_path = upper_air_dir / "upper_air_disagreement_bias_source_data.csv"
    for path in (shapley_path, bias_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shapley = pd.read_csv(shapley_path)
    bias = pd.read_csv(bias_path)
    require_columns(
        shapley,
        ["metric", "group", "group_label", "shapley_mean", "ci_low", "ci_high"],
        shapley_path,
    )
    require_columns(
        bias,
        ["case_category", "source_role", "n_complete_paired"],
        bias_path,
    )
    return shapley, bias


def load_tail_inputs(tail_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ci_path = tail_dir / "joint_tail_placement_utc_date_bootstrap_ci.csv"
    metrics_path = tail_dir / "joint_tail_placement_metrics.csv"
    for path in (ci_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    ci = pd.read_csv(ci_path)
    metrics = pd.read_csv(metrics_path)
    require_columns(
        ci,
        [
            "scope",
            "metric",
            "delta_tianji_minus_pangu",
            "delta_tianji_minus_pangu_ci_low",
            "delta_tianji_minus_pangu_ci_high",
        ],
        ci_path,
    )
    require_columns(metrics, ["scope", "reference_joint_tail_n"], metrics_path)
    return ci, metrics


def panel_label(ax, letter: str) -> None:
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)


def style_axis(ax, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#31363B")
        spine.set_linewidth(0.75)


def endpoint_rows(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    widths = metrics["mask"].astype(str).str.replace(r"\.0$", "", regex=True).str.len()
    width = int(widths.max())
    out = metrics.copy()
    out["mask"] = normalize_mask(out["mask"], width)
    keep = out[out["mask"].isin(["0" * width, "1" * width])].copy()
    if keep["mask"].nunique() != 2:
        raise ValueError("Factorial metrics do not contain both all-Pangu and all-Tianji endpoint masks")
    return keep, width


def draw_endpoint(
    ax,
    metrics: pd.DataFrame,
    gap: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    show_seed_range: bool = True,
) -> pd.DataFrame:
    endpoints, width = endpoint_rows(metrics)
    pangu_mask, tianji_mask = "0" * width, "1" * width
    seeds = sorted(set(endpoints["seed"].astype(int)))
    source_rows = []
    pangu_values = []
    tianji_values = []
    for seed in seeds:
        part = endpoints[endpoints["seed"].astype(int) == seed].set_index("mask")
        if pangu_mask not in part.index or tianji_mask not in part.index:
            raise ValueError(f"Missing endpoint for seed {seed}")
        y0 = float(part.loc[pangu_mask, metric])
        y1 = float(part.loc[tianji_mask, metric])
        pangu_values.append(y0)
        tianji_values.append(y1)
        source_rows.extend(
            [
                {"panel_metric": metric, "seed": seed, "source": "Pangu", "value": y0},
                {"panel_metric": metric, "seed": seed, "source": "Tianji", "value": y1},
            ]
        )
    distributions = [np.asarray(pangu_values, dtype=float), np.asarray(tianji_values, dtype=float)]
    pangu_mean = float(np.mean(distributions[0]))
    tianji_mean = float(np.mean(distributions[1]))
    means = np.asarray([pangu_mean, tianji_mean], dtype=float)
    lows = np.asarray([np.min(values) for values in distributions], dtype=float)
    highs = np.asarray([np.max(values) for values in distributions], dtype=float)
    positions = np.asarray([0, 1], dtype=float)
    bars = ax.bar(
        positions,
        means,
        width=ENDPOINT_BAR_WIDTH,
        color=[PANGU, TIANJI],
        edgecolor=[PANGU_DARK, TIANJI_DARK],
        linewidth=0.9,
        alpha=0.92,
        zorder=2,
    )
    if show_seed_range:
        for position, mean, low, high, color in zip(
            positions, means, lows, highs, (PANGU_DARK, TIANJI_DARK)
        ):
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="none",
                ecolor=color,
                elinewidth=INTERVAL_LINEWIDTH,
                capsize=INTERVAL_CAPSIZE,
                capthick=INTERVAL_CAPTHICK,
                zorder=3,
            )
    draws = pd.to_numeric(gap.loc[gap["metric"].astype(str) == metric, "delta_all1_minus_all0"], errors="coerce").dropna()
    delta = tianji_mean - pangu_mean
    if not draws.empty:
        lo, hi = np.percentile(draws.to_numpy(dtype=float), [2.5, 97.5])
        ax.text(0.5, 1.01, f"Δ {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", transform=ax.transAxes, ha="center", va="bottom", color=TIANJI_DARK, fontweight="bold", fontsize=7.5)
    ax.set_xticks([0, 1], ["Pangu", "Tianji"])
    ax.set_xlim(-0.55, 1.55)
    upper = min(1.0, max(0.10, float(highs.max()) * 1.13))
    ax.set_ylim(0.0, upper)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    style_axis(ax)
    return pd.DataFrame(source_rows)


def draw_shapley(ax, shapley: pd.DataFrame) -> pd.DataFrame:
    source = shapley[shapley["metric"].astype(str) == "low_vis_ap"].copy()
    source = source.sort_values("shapley_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(source))
    for yi, row in enumerate(source.itertuples(index=False)):
        color = PACKAGE_COLORS.get(str(row.group), "#6B7280")
        ax.errorbar(
            row.shapley_mean,
            yi,
            xerr=[[row.shapley_mean - row.ci_low], [row.ci_high - row.shapley_mean]],
            fmt="o",
            markersize=INTERVAL_MARKERSIZE,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=INTERVAL_LINEWIDTH,
            capsize=INTERVAL_CAPSIZE,
            capthick=INTERVAL_CAPTHICK,
            zorder=3,
        )
    ax.axvline(0.0, color=INK, linewidth=0.75)
    wrapped_labels = {
        "Near-surface moisture": "Near-surface\nmoisture",
        "925-hPa thermo-moisture": "925-hPa\nthermo-moisture",
        "Low-level wind/ventilation": "Low-level wind/\nventilation",
    }
    labels = [wrapped_labels.get(value, value) for value in source["group_label"].astype(str)]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Shapley contribution\nto Low-vis AP")
    ax.set_title("Exact source-package contributions", loc="left", fontweight="bold", pad=6)
    style_axis(ax, xgrid=True, ygrid=False)
    return source


def draw_hits(ax, bias: pd.DataFrame) -> pd.DataFrame:
    mapping = [
        ("tianji_hit_pangu_miss", "Tianji", TIANJI),
        ("pangu_hit_tianji_miss", "Pangu", PANGU),
    ]
    values = []
    for key, _, _ in mapping:
        counts = pd.to_numeric(
            bias.loc[
                (bias["case_category"].astype(str) == key)
                & (bias["source_role"].astype(str) == "physics"),
                "n_complete_paired",
            ],
            errors="coerce",
        ).dropna().unique()
        if len(counts) != 1:
            raise ValueError(f"Expected one complete-paired count for {key}, found {counts}")
        values.append(float(counts[0]))
    if not np.isfinite(values).all():
        raise ValueError("Event summary is missing source-exclusive hit counts")
    y = np.arange(2)[::-1]
    bars = ax.barh(
        y,
        values,
        color=[item[2] for item in mapping],
        height=HORIZONTAL_BAR_HEIGHT,
    )
    for bar, value, (_, _, color) in zip(bars, values, mapping):
        ax.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", ha="left", color=color, fontweight="bold")
    ax.set_yticks(y, [item[1] for item in mapping])
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Exclusive hits")
    ax.set_title("Source-exclusive Low-vis hits", loc="left", fontweight="bold", pad=6)
    style_axis(ax, xgrid=True, ygrid=False)
    return pd.DataFrame({"source": [item[1] for item in mapping], "exclusive_hits": values})


def select_joint_rows(joint_ci: pd.DataFrame, scope: str) -> pd.DataFrame:
    order = ["pod", "csi", "precision", "fpr"]
    source = joint_ci[
        (joint_ci["scope"].astype(str) == scope)
        & joint_ci["metric"].astype(str).isin(order)
    ].copy()
    source["metric"] = pd.Categorical(source["metric"], order, ordered=True)
    source = source.sort_values("metric").reset_index(drop=True)
    if len(source) != len(order):
        raise ValueError(f"{scope}: expected {len(order)} joint-tail rows, found {len(source)}")
    return source


def draw_joint_scope(
    ax,
    joint_ci: pd.DataFrame,
    joint_metrics: pd.DataFrame,
    scope: str,
    title: str,
    show_y: bool,
) -> pd.DataFrame:
    source = select_joint_rows(joint_ci, scope)
    counts = pd.to_numeric(
        joint_metrics.loc[
            joint_metrics["scope"].astype(str) == scope,
            "reference_joint_tail_n",
        ],
        errors="coerce",
    ).dropna().unique()
    if len(counts) != 1:
        raise ValueError(f"{scope}: expected one joint-tail count, found {counts}")
    labels = {
        "pod": "Tail-event recall",
        "csi": "CSI",
        "precision": "Precision",
        "fpr": "False-positive rate",
    }
    y = np.arange(len(source), dtype=float)
    ax.axvline(0.0, color=INK, linewidth=0.80, zorder=0)
    for yi, row in source.iterrows():
        value = float(row["delta_tianji_minus_pangu"])
        lo = float(row["delta_tianji_minus_pangu_ci_low"])
        hi = float(row["delta_tianji_minus_pangu_ci_high"])
        significant = not (lo <= 0.0 <= hi)
        color = TIANJI_DARK if significant and value > 0 else PANGU_DARK if significant else "#8B8F92"
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            markersize=INTERVAL_MARKERSIZE,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            ecolor=color,
            elinewidth=INTERVAL_LINEWIDTH,
            capsize=INTERVAL_CAPSIZE,
            capthick=INTERVAL_CAPTHICK,
            zorder=3,
        )
    ax.set_yticks(y, [labels[str(value)] for value in source["metric"].astype(str)] if show_y else [])
    if not show_y:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 0.11)
    ax.set_xlabel("Difference (Tianji − Pangu)")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    style_axis(ax)
    return source.assign(reference_joint_tail_n=int(counts[0]))


def draw_bias(ax, bias: pd.DataFrame, feature: str, title: str, unit: str, add_legend: bool = False) -> pd.DataFrame:
    categories = [("tianji_hit_pangu_miss", "Tianji-only hit"), ("pangu_hit_tianji_miss", "Pangu-only hit")]
    roles = [("physics", "Tianji", TIANJI, "o"), ("ai", "Pangu", PANGU, "D")]
    part = bias[bias["feature"].astype(str) == feature].copy()
    x = np.arange(len(categories), dtype=float)
    for ri, (role, label, color, marker) in enumerate(roles):
        rows = []
        for category, _ in categories:
            match = part[(part["case_category"].astype(str) == category) & (part["source_role"].astype(str) == role)]
            if len(match) != 1:
                raise ValueError(f"Missing bias row for {feature}/{category}/{role}")
            rows.append(match.iloc[0])
        estimate = np.asarray([float(row["bias_forecast_minus_reference"]) for row in rows])
        lo = np.asarray([float(row["bias_ci_low"]) for row in rows])
        hi = np.asarray([float(row["bias_ci_high"]) for row in rows])
        xpos = x + (-0.10 if ri == 0 else 0.10)
        ax.errorbar(
            xpos,
            estimate,
            yerr=np.vstack([estimate - np.minimum(lo, estimate), np.maximum(hi, estimate) - estimate]),
            fmt=marker,
            markersize=INTERVAL_MARKERSIZE,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            ecolor=color,
            elinewidth=INTERVAL_LINEWIDTH,
            capsize=INTERVAL_CAPSIZE,
            capthick=INTERVAL_CAPTHICK,
            label=label if add_legend else None,
            zorder=3,
        )
    tick_labels = []
    for category, label in categories:
        nrow = part[(part["case_category"].astype(str) == category) & (part["source_role"].astype(str) == "physics")]
        n = int(nrow.iloc[0]["n_complete_paired"])
        tick_labels.append(f"{label}\nn={n:,}")
    ax.axhline(0.0, color=INK, linewidth=0.75)
    ax.set_xticks(x, tick_labels)
    ax.set_ylabel(f"Forecast − ERA5 ({unit})")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    style_axis(ax)
    if add_legend:
        ax.legend(loc="upper right", ncol=2, handletextpad=0.35, columnspacing=0.8)
    return part


def export(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"dpi": dpi} if ext in {"png", "tiff"} else {}
        fig.savefig(out_dir / f"{stem}.{ext}", facecolor="white", **kwargs)


def main() -> None:
    args = parse_args()
    setup_style()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    endpoint_dir = args.endpoint_dir.expanduser().resolve()
    paired_quality_dir = resolve_paired_quality_dir(
        analysis_dir,
        endpoint_dir,
        args.paired_quality_dir,
    )
    tail_analysis_dir = (
        args.tail_analysis_dir.expanduser().resolve()
        if args.tail_analysis_dir is not None
        else analysis_dir
    )
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else analysis_dir / "manuscript_figures"
    endpoint_tables = load_inputs(endpoint_dir)
    joint_ci, joint_metrics = load_tail_inputs(tail_analysis_dir)
    quality_source = quality_plot.prepare_source(paired_quality_dir)
    placement_metrics = tail_plot.load_table(
        tail_analysis_dir,
        "reference_tail_placement_metrics.csv",
        ["feature", "tail", "task_relevant_tail"],
    )
    definitions = tail_plot.load_table(
        tail_analysis_dir,
        "joint_tail_feature_definitions.csv",
        ["feature", "task_tail_direction"],
    )
    placement_ci = tail_plot.load_table(
        tail_analysis_dir,
        "reference_tail_placement_utc_date_bootstrap_ci.csv",
        [
            "feature",
            "scope",
            "tail",
            "threshold_mode",
            "metric",
            "delta_tianji_minus_pangu",
            "delta_tianji_minus_pangu_ci_low",
            "delta_tianji_minus_pangu_ci_high",
        ],
    )
    relevant = placement_metrics[
        placement_metrics["task_relevant_tail"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    ].copy()
    directions = {
        str(row.feature): str(row.task_tail_direction)
        for row in definitions.itertuples(index=False)
    }
    for feature in tail_plot.FEATURE_ORDER:
        if feature in directions:
            continue
        tails = sorted(
            set(
                relevant.loc[
                    relevant["feature"].astype(str) == feature,
                    "tail",
                ].astype(str)
            )
        )
        if len(tails) != 1:
            raise ValueError(
                f"{feature}: expected one task-relevant tail in reference_tail_placement_metrics.csv, found {tails}"
            )
        directions[feature] = tails[0]
    placement = tail_plot.select_placement_rows(placement_ci, directions)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 9.45))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.86, 1.16, 1.18],
        left=0.185,
        right=0.985,
        top=0.955,
        bottom=0.065,
        hspace=0.59,
    )
    skill_grid = outer[0].subgridspec(1, 3, wspace=0.48)
    rmse_grid = outer[1].subgridspec(1, 2, wspace=0.19)
    tail_grid = outer[2].subgridspec(1, 3, width_ratios=[1.18, 1.0, 1.0], wspace=0.45)
    skill_axes = [fig.add_subplot(skill_grid[0, index]) for index in range(3)]
    rmse_axes = [fig.add_subplot(rmse_grid[0, index]) for index in range(2)]
    tail_axes = [fig.add_subplot(tail_grid[0, index]) for index in range(3)]

    metrics = endpoint_tables["metrics"]
    gap = endpoint_tables["gap"]
    source_frames = [
        controlled_plot.draw_endpoint_panel(
            skill_axes[0], metrics, "low_vis_ap", "Low-vis average precision", "Average precision"
        ),
        controlled_plot.draw_endpoint_panel(
            skill_axes[1], metrics, "low_vis_recall_matched_fpr", "Recall at matched FPR", "Low-vis recall"
        ),
        controlled_plot.draw_delta_panel(skill_axes[2], metrics, gap),
    ]
    for letter, axis in zip("abc", skill_axes):
        controlled_plot.panel_label(axis, letter, x=-0.24)

    for axis, scope, letter, show_y in (
        (rmse_axes[0], quality_plot.SCOPES[0], "d", True),
        (rmse_axes[1], quality_plot.SCOPES[1], "e", False),
    ):
        quality_plot.rmse_ratio_panel(
            axis,
            quality_source,
            scope,
            letter,
            show_y,
            show_reference_labels=False,
        )
        source_frames.append(
            quality_source[quality_source["scope"] == scope].assign(
                panel_metric=f"rmse_{scope}"
            )
        )
    rmse_xlim = (
        min(axis.get_xlim()[0] for axis in rmse_axes),
        max(axis.get_xlim()[1] for axis in rmse_axes),
    )
    rmse_tick_candidates = np.asarray([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0])
    rmse_ticks = rmse_tick_candidates[
        (rmse_tick_candidates >= rmse_xlim[0])
        & (rmse_tick_candidates <= rmse_xlim[1])
    ]
    for axis in rmse_axes:
        axis.set_xlim(*rmse_xlim)
        axis.set_xticks(rmse_ticks, [f"{value:g}" for value in rmse_ticks])

    source_frames.append(
        tail_plot.draw_tail_placement_panel(tail_axes[0], placement).assign(
            panel_metric="task_tail_placement"
        )
    )
    panel_label(tail_axes[0], "f")
    source_frames.extend(
        [
            draw_joint_scope(
                tail_axes[1],
                joint_ci,
                joint_metrics,
                "all_paired",
                "All paired samples",
                True,
            ).assign(panel_metric="joint_tail_all_paired"),
            draw_joint_scope(
                tail_axes[2],
                joint_ci,
                joint_metrics,
                "true_low_visibility",
                "Observed Low-vis (<1 km)",
                False,
            ).assign(panel_metric="joint_tail_low_visibility"),
        ]
    )
    for letter, axis in zip("gh", tail_axes[1:]):
        panel_label(axis, letter)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color=PANGU,
                markerfacecolor=PANGU,
                linestyle="none",
                label="Pangu (AI)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=TIANJI,
                markerfacecolor=TIANJI,
                linestyle="none",
                label="Tianji (physics)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.60, 0.995),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    export(fig, out_dir, args.figure_stem, args.dpi)
    plt.close(fig)
    pd.concat(source_frames, ignore_index=True, sort=False).to_csv(
        out_dir / f"{args.figure_stem}_source_data.csv",
        index=False,
        float_format="%.8f",
    )
    rendered = [str(out_dir / f"{args.figure_stem}.pdf")]

    manifest = {
        "analysis_dir": str(analysis_dir),
        "endpoint_dir": str(endpoint_dir),
        "paired_quality_dir": str(paired_quality_dir),
        "tail_analysis_dir": str(tail_analysis_dir),
        "figures": rendered,
        "panels": {
            "a-c": "forecast-source endpoint skill",
            "d-e": "paired RMSE ratios for all and observed Low-vis samples",
            "f": "task-tail placement",
            "g-h": "joint-tail recovery for all and observed Low-vis samples",
        },
        "task_tail_directions": directions,
        "rendering": "all panels redrawn from source tables on one canvas",
    }
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for figure in rendered:
        print(figure)


if __name__ == "__main__":
    main()
