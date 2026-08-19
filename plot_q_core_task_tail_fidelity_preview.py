#!/usr/bin/env python3
"""Draw standalone manuscript-style previews for task-tail fidelity diagnostics.

The script does not modify the existing Tianji--Pangu composite figure.  It
creates three independent figures from the tail-fidelity analysis tables:

1. marginal Q05/Q95 fidelity in observed Low-vis samples;
2. variable-specific placement of validation-defined task tails;
3. recovery of the multivariable joint task tail.

Task-tail directions are made internally consistent before plotting.  The
monthly-standardized validation definitions used by the joint-tail analysis
take precedence; variables not present in that table use the validation-only
event-conditioned direction table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_source_palette import (
    SOURCE_COLORS,
    SOURCE_DARK_COLORS,
    SOURCE_PALE_COLORS,
)
from paper_figure_geometry import (
    INTERVAL_CAPSIZE,
    INTERVAL_CAPTHICK,
    INTERVAL_LINEWIDTH,
    INTERVAL_MARKERSIZE,
)


FIGURE_WIDTH = 7.60
TIANJI = SOURCE_COLORS["tianji"]
PANGU = SOURCE_COLORS["pangu"]
TIANJI_DARK = SOURCE_DARK_COLORS["tianji"]
PANGU_DARK = SOURCE_DARK_COLORS["pangu"]
NEUTRAL = SOURCE_COLORS["baseline"]
INK = "#17191B"
GRID = "#E8EAEB"
CONNECTOR = "#CDD2D6"
SURFACE_BAND = "#F3F6F8"
UPPER_AIR_BAND = "#F5F2F8"
# Band colors shared with plot_common_variable_error_regimes so the manuscript
# row d/e/f uses one consistent family shading scheme.
FAMILY_SURFACE_BAND = "#F5F8FA"
FAMILY_PRESSURE_BAND = "#F7F4FA"
# FEATURE_ORDER matches plot_common_variable_error_regimes.FEATURE_ORDER; the
# first three rows are station-observation references, the rest are ERA5.
SURFACE_SPLIT = 3

FEATURE_ORDER = [
    "T2M",
    "WSPD10",
    "MSLP",
    "T_925",
    "Q_1000",
    "Q_925",
    "UV_925_VECTOR",
]

FEATURE_LABELS = {
    "T2M": "2-m temperature",
    "WSPD10": "10-m wind speed",
    "MSLP": "Mean sea-level pressure",
    "T_925": "925-hPa temperature",
    "Q_1000": "1000-hPa specific humidity",
    "Q_925": "925-hPa specific humidity",
    "UV_925_VECTOR": "925-hPa vector wind",
}

METRIC_LABELS = {
    "pod": "Tail-event recall",
    "csi": "CSI",
    "precision": "Precision",
    "fpr": "False-positive rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--prefix", default="task_tail_fidelity")
    return parser.parse_args()


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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_columns(df: pd.DataFrame, names: Iterable[str], source: Path) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")


def load_table(analysis_dir: Path, name: str, columns: Iterable[str]) -> pd.DataFrame:
    path = analysis_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, columns, path)
    return frame


def style_axis(ax, xgrid: bool = True, ygrid: bool = False) -> None:
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=2.7, width=0.75, color=INK, pad=2.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)


def panel_label(ax, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def source_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PANGU,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6.2,
            label="Pangu (AI)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=TIANJI,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6.2,
            label="Tianji (physics)",
        ),
    ]


def add_reference_bands(ax, x_label_position: float) -> None:
    ax.axhspan(-0.5, 2.5, color=SURFACE_BAND, zorder=-4)
    ax.axhspan(2.5, 6.5, color=UPPER_AIR_BAND, zorder=-4)
    ax.axhline(2.5, color="#D7DBDE", linewidth=0.8, zorder=-1)
    ax.text(
        x_label_position,
        0.15,
        "Station observations",
        ha="right",
        va="center",
        color="#4F555A",
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        x_label_position,
        3.15,
        "ERA5 reference analysis",
        ha="right",
        va="center",
        color="#4F555A",
        fontsize=7.2,
        fontweight="bold",
    )


def direction_map(
    definitions: pd.DataFrame,
    shifts: pd.DataFrame,
) -> Tuple[Dict[str, str], list[dict[str, str]]]:
    directions: Dict[str, str] = {}
    for row in definitions.itertuples(index=False):
        directions[str(row.feature)] = str(row.task_tail_direction)
    for feature, part in shifts.groupby("feature", sort=False):
        values = sorted(set(part["task_tail_direction"].astype(str)))
        if len(values) != 1:
            raise ValueError(f"{feature}: ambiguous event-conditioned task-tail directions {values}")
        directions.setdefault(str(feature), values[0])

    mismatches = []
    for feature, part in shifts.groupby("feature", sort=False):
        shift_direction = str(part.iloc[0]["task_tail_direction"])
        selected = directions.get(str(feature))
        if selected is not None and selected != shift_direction:
            mismatches.append(
                {
                    "feature": str(feature),
                    "event_conditioned_direction": shift_direction,
                    "monthly_standardized_direction": selected,
                }
            )
    missing = sorted(set(FEATURE_ORDER) - set(directions))
    if missing:
        raise ValueError(f"No validation-defined task-tail direction for {missing}")
    return directions, mismatches


def export_figure(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"dpi": dpi} if ext in {"png", "tiff"} else {}
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def draw_marginal_fidelity(
    marginal: pd.DataFrame,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> pd.DataFrame:
    source = marginal[
        (marginal["scope"].astype(str) == "true_low_visibility")
        & marginal["feature"].astype(str).isin(FEATURE_ORDER)
        & marginal["source"].astype(str).isin(["pangu", "tianji"])
    ].copy()
    source["feature"] = pd.Categorical(source["feature"], FEATURE_ORDER, ordered=True)
    source = source.sort_values(["feature", "source"]).reset_index(drop=True)
    expected = len(FEATURE_ORDER) * 2
    if len(source) != expected:
        raise ValueError(f"Expected {expected} marginal rows, found {len(source)}")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.35))
    y = np.arange(len(FEATURE_ORDER), dtype=float)
    pivot = source.pivot(
        index="feature",
        columns="source",
        values="outer_quantile_abs_error_normalized_by_reference_central90_width",
    ).reindex(FEATURE_ORDER)
    max_value = float(np.nanmax(pivot.to_numpy(dtype=float)))
    xmax = max(0.16, np.ceil((max_value + 0.012) / 0.02) * 0.02)
    add_reference_bands(ax, xmax * 0.985)
    for yi, feature in enumerate(FEATURE_ORDER):
        pangu = float(pivot.loc[feature, "pangu"])
        tianji = float(pivot.loc[feature, "tianji"])
        ax.plot(
            [pangu, tianji],
            [yi, yi],
            color=CONNECTOR,
            linewidth=1.25,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            pangu,
            yi,
            s=52,
            color=PANGU,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            tianji,
            yi,
            s=52,
            marker="s",
            color=TIANJI,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yticks(y, [FEATURE_LABELS[value] for value in FEATURE_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Normalized Q05/Q95 error  (lower is better)")
    ax.set_title("Marginal tail fidelity is variable-specific", loc="left", fontweight="bold", pad=12)
    ax.legend(
        handles=source_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.72, 1.13),
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.285, right=0.985, top=0.82, bottom=0.16)
    export_figure(fig, out_dir, stem, dpi)
    return source.assign(panel="marginal_tail_fidelity")


def select_placement_rows(
    placement_ci: pd.DataFrame,
    directions: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_ORDER:
        selected = placement_ci[
            (placement_ci["feature"].astype(str) == feature)
            & (placement_ci["scope"].astype(str) == "all_paired")
            & (placement_ci["tail"].astype(str) == directions[feature])
            & (placement_ci["threshold_mode"].astype(str) == "quantile_matched")
            & (placement_ci["metric"].astype(str) == "csi")
        ]
        if len(selected) != 1:
            raise ValueError(f"{feature}: expected one quantile-matched CSI row, found {len(selected)}")
        row = selected.iloc[0].copy()
        row["selected_task_tail_direction"] = directions[feature]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def significance_style(row: pd.Series) -> Tuple[str, str]:
    lo = float(row["delta_tianji_minus_pangu_ci_low"])
    hi = float(row["delta_tianji_minus_pangu_ci_high"])
    if lo > 0.0:
        return TIANJI, "s"
    if hi < 0.0:
        return PANGU, "o"
    return NEUTRAL, "D"


def draw_tail_placement_panel(
    ax,
    selected: pd.DataFrame,
    *,
    show_tail_direction: bool = True,
    show_direction_labels: bool = True,
    show_y: bool = True,
    show_values: bool = True,
    shading: str = "sign",
    xgrid: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
) -> pd.DataFrame:
    """Draw the established task-tail placement panel on a supplied axis."""

    y = np.arange(len(FEATURE_ORDER), dtype=float)
    low = selected["delta_tianji_minus_pangu_ci_low"].astype(float).to_numpy()
    high = selected["delta_tianji_minus_pangu_ci_high"].astype(float).to_numpy()
    xmin = min(-0.08, np.floor((float(np.nanmin(low)) - 0.012) / 0.02) * 0.02)
    xmax = max(0.18, np.ceil((float(np.nanmax(high)) + 0.012) / 0.02) * 0.02)
    if shading == "family":
        ax.axhspan(-0.5, SURFACE_SPLIT - 0.5, color=FAMILY_SURFACE_BAND, zorder=-6)
        ax.axhspan(
            SURFACE_SPLIT - 0.5,
            len(FEATURE_ORDER) - 0.5,
            color=FAMILY_PRESSURE_BAND,
            zorder=-6,
        )
    elif shading == "sign":
        ax.axvspan(xmin, 0.0, color=SOURCE_PALE_COLORS["pangu"], alpha=0.72, zorder=-5)
        ax.axvspan(0.0, xmax, color=SOURCE_PALE_COLORS["tianji"], alpha=0.72, zorder=-5)
    elif shading == "none":
        pass
    else:
        raise ValueError(f"Unknown tail-placement shading: {shading}")
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=3)
    if shading == "none":
        ax.axhline(2.5, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    else:
        ax.axhline(2.5, color="#D7DBDE", linewidth=0.8, zorder=0)

    labels = []
    for yi, row in selected.iterrows():
        value = float(row["delta_tianji_minus_pangu"])
        lo = float(row["delta_tianji_minus_pangu_ci_low"])
        hi = float(row["delta_tianji_minus_pangu_ci_high"])
        color, _marker = significance_style(row)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            markersize=INTERVAL_MARKERSIZE,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            ecolor=color,
            elinewidth=INTERVAL_LINEWIDTH,
            capsize=INTERVAL_CAPSIZE,
            capthick=INTERVAL_CAPTHICK,
            zorder=3,
        )
        if show_values:
            ax.annotate(
                f"{value:+.3f}",
                xy=(value, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                fontsize=7.2,
                fontweight="bold",
                annotation_clip=False,
            )
        direction = str(row["selected_task_tail_direction"])
        feature_label = FEATURE_LABELS[str(row["feature"])]
        labels.append(
            f"{feature_label}  ({direction})"
            if show_tail_direction
            else feature_label
        )

    ax.set_yticks(y, labels if show_y else [])
    if not show_y:
        ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_ylim(len(FEATURE_ORDER) - 0.5, -0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else "Δ quantile-matched task-tail CSI  (Tianji − Pangu; 95% CI)"
    )
    if title is not None:
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            pad=12,
        )
    if show_direction_labels:
        ax.text(
            0.01,
            1.025,
            "Pangu higher",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=PANGU_DARK,
            fontsize=7.4,
            fontweight="bold",
        )
        ax.text(
            0.99,
            1.025,
            "Tianji higher",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=TIANJI_DARK,
            fontsize=7.4,
            fontweight="bold",
        )
    style_axis(ax, xgrid=xgrid)
    return selected.assign(panel="task_tail_placement")


def draw_tail_placement(
    selected: pd.DataFrame,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.55))
    source = draw_tail_placement_panel(
        ax,
        selected,
        title="Tail-placement gains concentrate in temperature and wind",
    )
    fig.subplots_adjust(left=0.335, right=0.985, top=0.83, bottom=0.16)
    export_figure(fig, out_dir, stem, dpi)
    return source


def select_joint_rows(joint_ci: pd.DataFrame, scope: str) -> pd.DataFrame:
    order = ["pod", "csi", "precision", "fpr"]
    selected = joint_ci[
        (joint_ci["scope"].astype(str) == scope)
        & joint_ci["metric"].astype(str).isin(order)
    ].copy()
    selected["metric"] = pd.Categorical(selected["metric"], order, ordered=True)
    selected = selected.sort_values("metric").reset_index(drop=True)
    if len(selected) != len(order):
        raise ValueError(f"{scope}: expected {len(order)} joint metric rows, found {len(selected)}")
    return selected


def draw_joint_scope(ax, selected: pd.DataFrame, title: str, joint_n: int, letter: str) -> None:
    y = np.arange(len(selected), dtype=float)
    ax.axvspan(-0.04, 0.0, color=SOURCE_PALE_COLORS["pangu"], alpha=0.72, zorder=-5)
    ax.axvspan(0.0, 0.11, color=SOURCE_PALE_COLORS["tianji"], alpha=0.72, zorder=-5)
    ax.axvline(0.0, color=INK, linewidth=0.85, zorder=0)
    for yi, row in selected.iterrows():
        value = float(row["delta_tianji_minus_pangu"])
        lo = float(row["delta_tianji_minus_pangu_ci_low"])
        hi = float(row["delta_tianji_minus_pangu_ci_high"])
        color, marker = significance_style(row)
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.3, solid_capstyle="round", zorder=2)
        ax.scatter(
            value,
            yi,
            s=58,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=7.2,
            fontweight="bold",
            annotation_clip=False,
        )

    ax.set_yticks(y, [METRIC_LABELS[str(value)] for value in selected["metric"].astype(str)])
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 0.11)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.text(
        0.0,
        1.02,
        f"{joint_n:,} reference joint-tail cases",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#4F555A",
        fontsize=7.2,
    )
    ax.set_xlabel("Δ joint-tail metric  (Tianji − Pangu)")
    style_axis(ax)
    panel_label(ax, letter)


def draw_joint_recovery(
    joint_ci: pd.DataFrame,
    joint_metrics: pd.DataFrame,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> pd.DataFrame:
    all_rows = select_joint_rows(joint_ci, "all_paired")
    low_rows = select_joint_rows(joint_ci, "true_low_visibility")
    counts = {}
    for scope in ("all_paired", "true_low_visibility"):
        values = pd.to_numeric(
            joint_metrics.loc[
                joint_metrics["scope"].astype(str) == scope,
                "reference_joint_tail_n",
            ],
            errors="coerce",
        ).dropna().unique()
        if len(values) != 1:
            raise ValueError(f"{scope}: expected one reference joint-tail count, found {values}")
        counts[scope] = int(values[0])

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.85), sharex=True)
    draw_joint_scope(axes[0], all_rows, "All paired samples", counts["all_paired"], "a")
    draw_joint_scope(
        axes[1],
        low_rows,
        "Observed low-visibility samples",
        counts["true_low_visibility"],
        "b",
    )
    fig.suptitle(
        "Tianji recovers more joint-tail events without a detectable precision or FPR penalty",
        x=0.50,
        y=0.995,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.50,
        0.015,
        "Joint tail: at least 2 of 6 variables enter validation-defined adverse tails; UTC-date bootstrap 95% CI",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#4F555A",
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.78, bottom=0.22, wspace=0.34)
    export_figure(fig, out_dir, stem, dpi)
    return pd.concat(
        [
            all_rows.assign(panel="joint_tail_recovery", display_scope="all_paired"),
            low_rows.assign(panel="joint_tail_recovery", display_scope="true_low_visibility"),
        ],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    setup_style()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    marginal = load_table(
        analysis_dir,
        "marginal_q05_q95_tail_fidelity.csv",
        [
            "feature",
            "scope",
            "source",
            "outer_quantile_abs_error_normalized_by_reference_central90_width",
        ],
    )
    shifts = load_table(
        analysis_dir,
        "event_conditioned_distribution_shift.csv",
        ["feature", "task_tail_direction"],
    )
    definitions = load_table(
        analysis_dir,
        "joint_tail_feature_definitions.csv",
        ["feature", "task_tail_direction"],
    )
    placement_ci = load_table(
        analysis_dir,
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
    joint_ci = load_table(
        analysis_dir,
        "joint_tail_placement_utc_date_bootstrap_ci.csv",
        [
            "scope",
            "metric",
            "delta_tianji_minus_pangu",
            "delta_tianji_minus_pangu_ci_low",
            "delta_tianji_minus_pangu_ci_high",
        ],
    )
    joint_metrics = load_table(
        analysis_dir,
        "joint_tail_placement_metrics.csv",
        ["scope", "reference_joint_tail_n"],
    )

    directions, mismatches = direction_map(definitions, shifts)
    placement = select_placement_rows(placement_ci, directions)
    frames = [
        draw_marginal_fidelity(
            marginal,
            out_dir,
            f"{args.prefix}_01_marginal_fidelity",
            args.dpi,
        ),
        draw_tail_placement(
            placement,
            out_dir,
            f"{args.prefix}_02_tail_placement",
            args.dpi,
        ),
        draw_joint_recovery(
            joint_ci,
            joint_metrics,
            out_dir,
            f"{args.prefix}_03_joint_recovery",
            args.dpi,
        ),
    ]
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        out_dir / f"{args.prefix}_visualization_source_data.csv",
        index=False,
        float_format="%.8f",
    )
    manifest = {
        "analysis_dir": str(analysis_dir),
        "out_dir": str(out_dir),
        "task_tail_directions": directions,
        "direction_mismatches_resolved_by_monthly_standardized_definition": mismatches,
        "existing_main_figure_modified": False,
        "exports": ["png", "pdf", "svg", "tiff"],
    }
    (out_dir / f"{args.prefix}_visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for path in sorted(out_dir.glob(f"{args.prefix}_*.png")):
        print(path)


if __name__ == "__main__":
    main()
