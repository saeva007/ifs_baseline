#!/usr/bin/env python3
"""Draw the Nature-style common-variable main figure.

The figure is intentionally built around the *primary* T925-inclusive,
18-channel common-variable experiment.  It contains four tightly connected
views:

1. the controlled design (only forecast source changes);
2. paired three-seed endpoint skill for AP, recall and CSI;
3. observation-anchored near-surface state quality;
4. the imbalance in exclusive Low-vis hits.

No SHAP or source-block attribution result is drawn here.  That analysis can
be added later as an independent panel or supplementary figure without
changing the main comparison.

Style is inherited from the current manuscript plotting chain:
Arial/Helvetica-first typography, Tianji dark blue, Pangu violet, editable
SVG text, PDF Type-42 fonts, and 600-dpi raster output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge
import numpy as np
import pandas as pd

from paper_source_palette import (
    SOURCE_COLORS,
    SOURCE_DARK_COLORS,
    SOURCE_LIGHT_COLORS,
    SOURCE_MARKERS,
    SOURCE_PALE_COLORS,
)


# Nature Communications double-column width (183 mm).
FIGURE_SIZE = (7.205, 5.35)

PHYSICS = SOURCE_COLORS["tianji"]
PHYSICS_DARK = SOURCE_DARK_COLORS["tianji"]
PHYSICS_LIGHT = SOURCE_LIGHT_COLORS["tianji"]
PHYSICS_PALE = SOURCE_PALE_COLORS["tianji"]
AI = SOURCE_COLORS["pangu"]
AI_DARK = SOURCE_DARK_COLORS["pangu"]
AI_LIGHT = SOURCE_LIGHT_COLORS["pangu"]
AI_PALE = SOURCE_PALE_COLORS["pangu"]

INK = "#17191B"
MID = "#697075"
LIGHT = "#D9DCDE"
GRID = "#E8EAEB"
WHITE = "#FFFFFF"
PALE = "#F7F8F8"

METRICS: Tuple[Tuple[str, str, Tuple[float, float]], ...] = (
    ("low_vis_ap", "Low-vis AP", (0.29, 0.425)),
    ("low_vis_recall_matched_fpr", "Recall\nmatched FPR", (0.65, 0.785)),
    ("low_vis_csi_matched_fpr", "CSI\nmatched FPR", (0.155, 0.208)),
)

FEATURE_ORDER = ("T2M", "WSPD10", "MSLP")
FEATURE_LABELS = {
    "T2M": "2-m temperature",
    "WSPD10": "10-m wind speed",
    "MSLP": "Mean sea-level pressure",
}
SCOPE_ORDER = ("all_paired_test", "true_low_visibility")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-analysis-dir",
        required=True,
        help="Directory containing qcore_t925_metrics_by_seed.csv and bootstrap outputs.",
    )
    parser.add_argument(
        "--observation-quality-csv",
        required=True,
        help="observation_anchored_source_quality.csv for T2M, WSPD10 and MSLP.",
    )
    parser.add_argument(
        "--physics-only-hits",
        type=int,
        default=None,
        help="Override the physics-only Low-vis hit count.",
    )
    parser.add_argument(
        "--ai-only-hits",
        type=int,
        default=None,
        help="Override the AI-only Low-vis hit count.",
    )
    parser.add_argument(
        "--event-samples-csv",
        default=None,
        help=(
            "Event sample table containing case_category/category. Defaults to "
            "<primary-analysis-dir>/event_case_control_samples.csv.gz when manual "
            "hit counts are omitted."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stem",
        default="fig_common_variable_main",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png,tiff",
        help="Comma-separated formats; SVG is always emitted first.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--no-panel-exports",
        action="store_true",
        help="Write only the composite; by default panels a-d are also exported separately.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def resolve_exclusive_hits(
    args: argparse.Namespace,
    primary_dir: Path,
) -> Tuple[int, int, str]:
    manual = (args.physics_only_hits, args.ai_only_hits)
    if all(value is not None for value in manual):
        physics_hits, ai_hits = (int(value) for value in manual)
        source = "manual_cli"
    elif any(value is not None for value in manual):
        raise ValueError(
            "--physics-only-hits and --ai-only-hits must be supplied together"
        )
    else:
        event_path = (
            Path(args.event_samples_csv).expanduser().resolve()
            if args.event_samples_csv
            else primary_dir / "event_case_control_samples.csv.gz"
        )
        if not event_path.is_file():
            raise FileNotFoundError(
                "Exclusive-hit counts were not supplied and the event sample table "
                f"does not exist: {event_path}"
            )
        events = pd.read_csv(event_path)
        category_column = (
            "case_category"
            if "case_category" in events.columns
            else "category"
            if "category" in events.columns
            else None
        )
        if category_column is None:
            raise ValueError(
                f"{event_path} must contain case_category or category"
            )
        counts = events[category_column].astype(str).value_counts()
        physics_hits = int(counts.get("tianji_hit_pangu_miss", 0))
        ai_hits = int(counts.get("pangu_hit_tianji_miss", 0))
        source = str(event_path)
    if physics_hits <= 0 or ai_hits <= 0:
        raise ValueError(
            "Both exclusive-hit counts must be positive; got "
            f"physics={physics_hits}, ai={ai_hits}"
        )
    return physics_hits, ai_hits, source


def read_primary(primary_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = primary_dir / "qcore_t925_metrics_by_seed.csv"
    bootstrap_path = primary_dir / "qcore_t925_bootstrap_summary.csv"
    if not metrics_path.is_file() or not bootstrap_path.is_file():
        raise FileNotFoundError(
            f"Expected {metrics_path.name} and {bootstrap_path.name} under {primary_dir}"
        )
    metrics = pd.read_csv(metrics_path, dtype={"mask": str})
    bootstrap = pd.read_csv(bootstrap_path)
    metrics["mask"] = (
        metrics["mask"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    require_columns(
        metrics,
        ["seed", "source", "mask", *(name for name, _, _ in METRICS)],
        metrics_path.name,
    )
    require_columns(
        bootstrap,
        ["metric", "delta_tianji_minus_pangu", "ci_low", "ci_high"],
        bootstrap_path.name,
    )
    bootstrap = bootstrap.rename(
        columns={"delta_tianji_minus_pangu": "point_delta"}
    )
    endpoint = metrics[metrics["mask"].isin(["0000", "1111"])].copy()
    expected = {
        (seed, source)
        for seed in (42, 2025, 20260702)
        for source in ("pangu", "tianji")
    }
    actual = set(zip(endpoint["seed"].astype(int), endpoint["source"].astype(str)))
    if actual != expected:
        raise ValueError(
            "Primary endpoint table must contain three seeds for both Pangu and Tianji."
        )
    return endpoint, bootstrap


def prepare_endpoint_source(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    boot = bootstrap.set_index("metric")
    for metric, label, _ in METRICS:
        if metric not in boot.index:
            raise ValueError(f"Missing bootstrap summary for {metric}")
        part = metrics[["seed", "source", metric]].rename(columns={metric: "value"})
        means = part.groupby("source")["value"].mean()
        for row in part.itertuples(index=False):
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "seed": int(row.seed),
                    "source": str(row.source),
                    "value": float(row.value),
                    "mean_ai": float(means["pangu"]),
                    "mean_physics": float(means["tianji"]),
                    "delta_physics_minus_ai": float(boot.loc[metric, "point_delta"]),
                    "delta_ci_low": float(boot.loc[metric, "ci_low"]),
                    "delta_ci_high": float(boot.loc[metric, "ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def prepare_observation_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, ["feature", "source", "scope", "rmse"], path.name)
    part = frame[
        frame["feature"].isin(FEATURE_ORDER)
        & frame["scope"].isin(SCOPE_ORDER)
        & frame["source"].isin(["pangu", "tianji"])
    ].copy()
    indexed = part.set_index(["feature", "scope", "source"])
    rows: List[Dict[str, object]] = []
    for feature in FEATURE_ORDER:
        for scope in SCOPE_ORDER:
            try:
                ai_rmse = float(indexed.loc[(feature, scope, "pangu"), "rmse"])
                physics_rmse = float(indexed.loc[(feature, scope, "tianji"), "rmse"])
            except KeyError as exc:
                raise ValueError(f"Missing observation row for {feature}/{scope}") from exc
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "scope": scope,
                    "scope_label": "All test" if scope == "all_paired_test" else "Observed Low-vis",
                    "ai_rmse": ai_rmse,
                    "physics_rmse": physics_rmse,
                    "physics_to_ai_rmse_ratio": physics_rmse / ai_rmse,
                }
            )
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def title(ax: plt.Axes, text: str, y: float = 1.04, x: float = 0.08) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def style_axis(ax: plt.Axes, xgrid: bool = False) -> None:
    ax.tick_params(length=2.8, width=0.75, color=INK, pad=2.2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)
    if xgrid:
        ax.xaxis.grid(
            True,
            color=GRID,
            linewidth=0.55,
            linestyle=(0, (2.0, 2.0)),
        )
    ax.set_axisbelow(True)


def rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    wh: Tuple[float, float],
    text: str,
    face: str,
    edge: str,
    *,
    fontsize: float = 7.2,
    weight: str = "bold",
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=edge,
        linespacing=1.15,
    )


def arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str,
    width: float = 1.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def draw_design(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", x=-0.02, y=1.04)
    title(ax, "Common-variable control", y=1.04)

    rounded_box(
        ax,
        (0.08, 0.72),
        (0.84, 0.17),
        "18 shared dynamic channels\nSame samples · context · evaluation",
        PALE,
        INK,
        fontsize=6.2,
    )
    ax.plot([0.50, 0.50], [0.72, 0.64], color=INK, linewidth=1.0)
    ax.plot([0.24, 0.76], [0.64, 0.64], color=INK, linewidth=1.0)
    ax.plot([0.24, 0.24], [0.64, 0.57], color=AI, linewidth=1.35)
    ax.plot([0.76, 0.76], [0.64, 0.57], color=PHYSICS, linewidth=1.35)

    rounded_box(ax, (0.04, 0.41), (0.40, 0.16), "AI forecast\nPangu", AI_PALE, AI_DARK)
    rounded_box(
        ax,
        (0.56, 0.41),
        (0.40, 0.16),
        "Physics forecast\nTianji",
        PHYSICS_PALE,
        PHYSICS_DARK,
    )
    arrow(ax, (0.24, 0.41), (0.24, 0.31), AI, 1.35)
    arrow(ax, (0.76, 0.41), (0.76, 0.31), PHYSICS, 1.35)
    rounded_box(ax, (0.04, 0.13), (0.40, 0.18), "VisCast\npaired training", WHITE, AI_DARK, fontsize=6.3)
    rounded_box(
        ax,
        (0.56, 0.13),
        (0.40, 0.18),
        "VisCast\npaired training",
        WHITE,
        PHYSICS_DARK,
        fontsize=6.3,
    )


def metric_axis(
    ax: plt.Axes,
    endpoint: pd.DataFrame,
    metric: str,
    label: str,
    ylim: Tuple[float, float],
    show_ylabel: bool,
) -> None:
    part = endpoint[endpoint["metric"] == metric]
    ai = part[part["source"] == "pangu"].set_index("seed")
    physics = part[part["source"] == "tianji"].set_index("seed")
    seeds = sorted(set(ai.index).intersection(physics.index))

    for seed in seeds:
        y0 = float(ai.loc[seed, "value"])
        y1 = float(physics.loc[seed, "value"])
        ax.plot([0, 1], [y0, y1], color=LIGHT, linewidth=1.15, zorder=1)
        ax.scatter(
            0,
            y0,
            s=26,
            marker=SOURCE_MARKERS["pangu"],
            facecolor=AI_LIGHT,
            edgecolor=AI_DARK,
            linewidth=0.7,
            zorder=2,
        )
        ax.scatter(
            1,
            y1,
            s=26,
            marker=SOURCE_MARKERS["tianji"],
            facecolor=PHYSICS_LIGHT,
            edgecolor=PHYSICS_DARK,
            linewidth=0.7,
            zorder=2,
        )

    mean_ai = float(part["mean_ai"].iloc[0])
    mean_physics = float(part["mean_physics"].iloc[0])
    ax.plot(
        [0, 1],
        [mean_ai, mean_physics],
        color=INK,
        linewidth=1.25,
        zorder=3,
    )
    ax.scatter(
        0,
        mean_ai,
        s=72,
        marker=SOURCE_MARKERS["pangu"],
        facecolor=AI,
        edgecolor=WHITE,
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        1,
        mean_physics,
        s=72,
        marker=SOURCE_MARKERS["tianji"],
        facecolor=PHYSICS,
        edgecolor=WHITE,
        linewidth=0.8,
        zorder=4,
    )

    delta = float(part["delta_physics_minus_ai"].iloc[0])
    lo = float(part["delta_ci_low"].iloc[0])
    hi = float(part["delta_ci_high"].iloc[0])
    ax.text(
        0.5,
        1.21,
        f"Δ {delta:+.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=PHYSICS_DARK,
    )
    ax.text(
        0.5,
        1.14,
        f"[{lo:+.3f}, {hi:+.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=INK,
    )
    ax.text(
        -0.08,
        mean_ai,
        f"{mean_ai:.3f}",
        ha="right",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=AI_DARK,
        clip_on=False,
    )
    ax.text(
        1.08,
        mean_physics,
        f"{mean_physics:.3f}",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=PHYSICS_DARK,
        clip_on=False,
    )
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1], ["AI", "Physics"])
    ax.get_xticklabels()[0].set_color(AI_DARK)
    ax.get_xticklabels()[1].set_color(PHYSICS_DARK)
    ax.text(
        0.5,
        0.985,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        linespacing=1.0,
    )
    if not show_ylabel:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    style_axis(ax)


def draw_endpoints(
    parent_spec,
    fig: plt.Figure,
    endpoint: pd.DataFrame,
) -> None:
    inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=parent_spec, wspace=0.48)
    axes = [fig.add_subplot(inner[0, index]) for index in range(3)]
    for index, (metric, label, ylim) in enumerate(METRICS):
        metric_axis(
            axes[index],
            endpoint,
            metric,
            label,
            ylim,
            show_ylabel=index == 0,
        )
    panel_label(axes[0], "b", x=-0.26, y=1.36)
    axes[0].text(
        -0.10,
        1.36,
        "Common variables: physics retains higher Low-vis skill",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def draw_observation_quality(ax: plt.Axes, source: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.055, y=1.08)
    title(ax, "Observation-anchored near-surface error fingerprint", y=1.08, x=0.02)
    base = {
        feature: float(len(FEATURE_ORDER) - 1 - index)
        for index, feature in enumerate(FEATURE_ORDER)
    }
    offsets = {"all_paired_test": 0.13, "true_low_visibility": -0.13}
    markers = {"all_paired_test": "o", "true_low_visibility": "s"}
    faces = {"all_paired_test": WHITE, "true_low_visibility": PHYSICS}

    for row in source.itertuples(index=False):
        y = base[str(row.feature)] + offsets[str(row.scope)]
        ratio = float(row.physics_to_ai_rmse_ratio)
        ax.plot([ratio, 1.0], [y, y], color=LIGHT, linewidth=1.25, zorder=1)
        ax.scatter(
            ratio,
            y,
            s=34,
            marker=markers[str(row.scope)],
            facecolor=faces[str(row.scope)],
            edgecolor=PHYSICS_DARK,
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            ratio - 0.018,
            y,
            f"{ratio:.2f}",
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=PHYSICS_DARK,
        )
    ax.axvline(1.0, color=INK, linewidth=0.8)
    ax.set_yticks(
        [base[feature] for feature in FEATURE_ORDER],
        ["T2M", "10-m wind", "MSLP"],
    )
    ax.set_xlim(0.44, 1.07)
    ax.set_ylim(-0.50, len(FEATURE_ORDER) - 0.45)
    ax.set_xlabel("Physics / AI RMSE")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=WHITE,
                markeredgecolor=PHYSICS_DARK,
                markeredgewidth=0.8,
                markersize=4.8,
                label="All test",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=PHYSICS,
                markeredgecolor=PHYSICS_DARK,
                markeredgewidth=0.8,
                markersize=4.8,
                label="Observed Low-vis",
            ),
        ],
        loc="upper right",
        ncol=1,
        borderaxespad=0.2,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    style_axis(ax, xgrid=True)


def draw_exclusive_hits(ax: plt.Axes, physics_hits: int, ai_hits: int) -> pd.DataFrame:
    if physics_hits <= 0 or ai_hits <= 0:
        raise ValueError("Exclusive-hit counts must be positive")
    panel_label(ax, "d", x=-0.08, y=1.08)
    title(ax, "Exclusive Low-vis hits", y=1.08, x=0.03)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.32, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    total = physics_hits + ai_hits
    physics_fraction = physics_hits / total
    start = 90.0
    physics_end = start + 360.0 * physics_fraction
    ring_width = 0.23
    ax.add_patch(
        Wedge(
            (0, 0.08),
            0.80,
            start,
            physics_end,
            width=ring_width,
            facecolor=PHYSICS,
            edgecolor=WHITE,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Wedge(
            (0, 0.08),
            0.80,
            physics_end,
            start + 360.0,
            width=ring_width,
            facecolor=AI,
            edgecolor=WHITE,
            linewidth=1.0,
        )
    )
    ratio = physics_hits / ai_hits
    ax.text(
        0,
        0.16,
        f"{ratio:.2f}×",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=PHYSICS_DARK,
    )
    ax.text(
        0,
        -0.10,
        "Physics / AI",
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=INK,
    )

    ax.add_patch(Circle((-0.78, -0.90), 0.050, facecolor=PHYSICS, edgecolor=WHITE, linewidth=0.6))
    ax.text(
        -0.68,
        -0.90,
        f"Physics-only  {physics_hits:,}",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=PHYSICS_DARK,
    )
    ax.add_patch(Circle((-0.78, -1.14), 0.050, facecolor=AI, edgecolor=WHITE, linewidth=0.6))
    ax.text(
        -0.68,
        -1.14,
        f"AI-only  {ai_hits:,}",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=AI_DARK,
    )
    return pd.DataFrame(
        [
            {
                "category": "physics_only_hit",
                "n": physics_hits,
                "share_of_exclusive_hits": physics_fraction,
                "physics_to_ai_ratio": ratio,
            },
            {
                "category": "ai_only_hit",
                "n": ai_hits,
                "share_of_exclusive_hits": ai_hits / total,
                "physics_to_ai_ratio": ratio,
            },
        ]
    )


def make_figure(
    endpoints: pd.DataFrame,
    observations: pd.DataFrame,
    physics_hits: int,
    ai_hits: int,
) -> Tuple[plt.Figure, pd.DataFrame]:
    fig = plt.figure(figsize=FIGURE_SIZE)
    outer = GridSpec(
        2,
        3,
        figure=fig,
        left=0.080,
        right=0.975,
        bottom=0.075,
        top=0.820,
        width_ratios=[1.00, 1.25, 0.88],
        height_ratios=[1.03, 0.88],
        wspace=0.34,
        hspace=0.45,
    )
    design_ax = fig.add_subplot(outer[0, 0])
    draw_design(design_ax)
    draw_endpoints(outer[0, 1:3], fig, endpoints)
    obs_ax = fig.add_subplot(outer[1, 0:2])
    draw_observation_quality(obs_ax, observations)
    hit_ax = fig.add_subplot(outer[1, 2])
    event_source = draw_exclusive_hits(hit_ax, physics_hits, ai_hits)
    return fig, event_source


def make_panel_figures(
    endpoints: pd.DataFrame,
    observations: pd.DataFrame,
    physics_hits: int,
    ai_hits: int,
) -> Dict[str, plt.Figure]:
    figures: Dict[str, plt.Figure] = {}

    fig_a, ax_a = plt.subplots(figsize=(3.25, 2.75))
    fig_a.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.87)
    draw_design(ax_a)
    figures["a_common_variable_control"] = fig_a

    fig_b = plt.figure(figsize=(6.6, 2.75))
    grid_b = GridSpec(
        1,
        1,
        figure=fig_b,
        left=0.10,
        right=0.96,
        bottom=0.18,
        top=0.72,
    )
    draw_endpoints(grid_b[0, 0], fig_b, endpoints)
    figures["b_paired_endpoint_skill"] = fig_b

    fig_c, ax_c = plt.subplots(figsize=(4.65, 2.85))
    fig_c.subplots_adjust(left=0.25, right=0.97, bottom=0.20, top=0.80)
    draw_observation_quality(ax_c, observations)
    figures["c_observation_error_fingerprint"] = fig_c

    fig_d, ax_d = plt.subplots(figsize=(3.15, 3.0))
    fig_d.subplots_adjust(left=0.08, right=0.98, bottom=0.04, top=0.84)
    draw_exclusive_hits(ax_d, physics_hits, ai_hits)
    figures["d_exclusive_lowvis_hits"] = fig_d
    return figures


def parse_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [
        item.strip().lower()
        for item in value.replace(":", ",").split(",")
        if item.strip()
    ]
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported formats {unknown}; supported={sorted(supported)}")
    return ["svg", *[fmt for fmt in requested if fmt != "svg"]]


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    saved: List[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {"facecolor": WHITE}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    out_dir: Path,
    stem: str,
    endpoints: pd.DataFrame,
    observations: pd.DataFrame,
    events: pd.DataFrame,
    saved: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    source_frames = {
        "primary_endpoints": endpoints,
        "observation_quality": observations,
        "exclusive_hits": events,
    }
    source_files: Dict[str, str] = {}
    for key, frame in source_frames.items():
        path = out_dir / f"{stem}_source_{key}.csv"
        frame.to_csv(path, index=False)
        source_files[key] = path.name

    manifest = {
        "status": "completed",
        "figure_stem": stem,
        "figure_size_inches": list(FIGURE_SIZE),
        "primary_analysis_dir": str(Path(args.primary_analysis_dir).resolve()),
        "observation_quality_csv": str(Path(args.observation_quality_csv).resolve()),
        "exclusive_hit_source": str(args.exclusive_hit_source),
        "core_claim": (
            "With samples, variables, VisCast architecture, training and evaluation "
            "held fixed, the physics forecast source yields higher Low-vis AP, "
            "matched-FPR recall and CSI than the AI forecast source across all "
            "training seeds; the advantage aligns with lower near-surface station "
            "errors and more exclusive Low-vis hits."
        ),
        "interpretation_boundary": (
            "The figure supports a task-specific forecast-source information "
            "difference. It contains no SHAP or causal single-variable attribution."
        ),
        "exports": {
            path.suffix.lstrip("."): {
                "file": path.name,
                "sha256": sha256_file(path),
            }
            for path in saved
        },
        "source_data": source_files,
    }
    with (out_dir / f"{stem}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    configure_style()
    primary_dir = Path(args.primary_analysis_dir).expanduser().resolve()
    observation_path = Path(args.observation_quality_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, bootstrap = read_primary(primary_dir)
    endpoints = prepare_endpoint_source(metrics, bootstrap)
    observations = prepare_observation_source(observation_path)
    physics_hits, ai_hits, hit_source = resolve_exclusive_hits(args, primary_dir)
    args.exclusive_hit_source = hit_source
    fig, events = make_figure(
        endpoints,
        observations,
        physics_hits,
        ai_hits,
    )
    formats = parse_formats(args.formats)
    saved = save_figure(fig, out_dir, args.stem, formats, int(args.dpi))
    write_bundle(out_dir, args.stem, endpoints, observations, events, saved, args)
    panel_exports: Dict[str, List[str]] = {}
    if not args.no_panel_exports:
        for suffix, panel_figure in make_panel_figures(
            endpoints,
            observations,
            physics_hits,
            ai_hits,
        ).items():
            paths = save_figure(
                panel_figure,
                out_dir,
                f"{args.stem}_{suffix}",
                formats,
                int(args.dpi),
            )
            panel_exports[suffix] = [path.name for path in paths]
        (out_dir / f"{args.stem}_panel_exports.json").write_text(
            json.dumps(panel_exports, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"[done] wrote {len(saved)} formats to {out_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
