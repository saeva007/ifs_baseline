#!/usr/bin/env python3
"""Plot the four-source operational-ensemble analysis and every panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analyze_operational_source_ensembles import (
    ALL_FOUR_ENSEMBLE,
    CURRENT_ENSEMBLE,
    FAMILY_BALANCED_ENSEMBLE,
    MEMBERS,
    RECIPE_LABELS,
    RECIPES,
    TIANJI_PANGU_ENSEMBLE,
)


FIGURE_SIZE = (7.205, 4.95)
WHITE = "#FFFFFF"
INK = "#17191B"
MID = "#687076"
GRID = "#E7E9EA"
LIGHT = "#DDE1E3"
POSITIVE = "#2D8A52"
NEGATIVE = "#C7773C"
MEMBER_COLORS = {
    "tianji": "#2E5A87",
    "T2ND_rh2m_source_full": "#65A6A2",
    "ifs": "#777B7E",
    "pangu2025_source_full": "#8E6BBE",
}
MEMBER_SHORT = {
    "tianji": "Tianji",
    "T2ND_rh2m_source_full": "T2ND",
    "ifs": "IFS",
    "pangu2025_source_full": "Pangu",
}
RECIPE_ORDER = (
    CURRENT_ENSEMBLE,
    TIANJI_PANGU_ENSEMBLE,
    ALL_FOUR_ENSEMBLE,
    FAMILY_BALANCED_ENSEMBLE,
)
RECIPE_SHORT = {
    CURRENT_ENSEMBLE: "Current BE",
    TIANJI_PANGU_ENSEMBLE: "Tianji + Pangu",
    ALL_FOUR_ENSEMBLE: "All four",
    FAMILY_BALANCED_ENSEMBLE: "Family-balanced",
}
RECIPE_COLORS = {
    CURRENT_ENSEMBLE: "#C44E52",
    TIANJI_PANGU_ENSEMBLE: "#8062AD",
    ALL_FOUR_ENSEMBLE: "#A55C8C",
    FAMILY_BALANCED_ENSEMBLE: "#2E887A",
}
TRANSITION_ORDER = (
    "shared_success",
    "candidate_rescue",
    "candidate_harm",
    "shared_failure",
)
TRANSITION_LABELS = {
    "shared_success": "Shared success",
    "candidate_rescue": "Candidate rescue",
    "candidate_harm": "Candidate harm",
    "shared_failure": "Shared failure",
}
TRANSITION_COLORS = {
    "shared_success": "#68747A",
    "candidate_rescue": POSITIVE,
    "candidate_harm": NEGATIVE,
    "shared_failure": LIGHT,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
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
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out-dir", default="", help="Defaults to --analysis-dir.")
    parser.add_argument("--stem", default="fig_operational_source_ensembles")
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def read_tables(directory: Path) -> Dict[str, pd.DataFrame]:
    names = ("recipe_weights", "performance", "relative_gain", "anchor_delta", "transitions")
    tables = {}
    for name in names:
        path = directory / f"operational_ensemble_{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing operational-ensemble source table: {path}")
        tables[name] = pd.read_csv(path)
    require_columns(tables["relative_gain"], ("recipe", "metric_label", "relative_change_percent"), "relative gain")
    require_columns(tables["transitions"], ("scope", "candidate", "category", "share"), "transitions")
    return tables


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.06) -> None:
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


def title(ax: plt.Axes, text: str, x: float = 0.0, y: float = 1.045) -> None:
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


def style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    ax.tick_params(length=2.7, width=0.75, color=INK, pad=2.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.55, linestyle=(0, (2, 2)))
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.55, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)


def blend_with_white(hex_color: str, strength: float) -> Tuple[float, float, float]:
    raw = hex_color.lstrip("#")
    rgb = np.asarray([int(raw[index : index + 2], 16) / 255.0 for index in (0, 2, 4)])
    strength = float(np.clip(strength, 0.0, 1.0))
    mixed = 1.0 - strength * (1.0 - rgb)
    return tuple(float(value) for value in mixed)


def draw_recipe_matrix(ax: plt.Axes, weights: pd.DataFrame, label: str = "a") -> None:
    panel_label(ax, label, x=-0.14, y=1.08)
    title(ax, "Probability-fusion recipes", x=0.0, y=1.08)
    ax.set_xlim(0, len(MEMBERS))
    ax.set_ylim(0, len(RECIPE_ORDER))
    for row_index, recipe in enumerate(RECIPE_ORDER):
        y = len(RECIPE_ORDER) - 1 - row_index
        for column_index, source in enumerate(MEMBERS):
            selected = weights[(weights["recipe"] == recipe) & (weights["source"] == source)]
            value = float(selected.iloc[0]["weight"]) if not selected.empty else float(RECIPES[recipe][source])
            face = blend_with_white(MEMBER_COLORS[source], 0.16 + 0.78 * min(value / 0.5, 1.0))
            ax.add_patch(
                Rectangle(
                    (column_index + 0.04, y + 0.08),
                    0.92,
                    0.84,
                    facecolor=face,
                    edgecolor=WHITE,
                    linewidth=0.8,
                )
            )
            ax.text(
                column_index + 0.5,
                y + 0.5,
                f"{value:.2f}" if value > 0 else "–",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold" if value > 0 else "normal",
                color=INK if value > 0 else "#A8ADB0",
            )
    ax.set_xticks(np.arange(len(MEMBERS)) + 0.5, [MEMBER_SHORT[source] for source in MEMBERS])
    ax.set_yticks(
        np.arange(len(RECIPE_ORDER)) + 0.5,
        [RECIPE_SHORT[recipe] for recipe in RECIPE_ORDER[::-1]],
    )
    ax.tick_params(axis="both", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Member probability weight", labelpad=5.0)


def draw_relative_gain(ax: plt.Axes, source: pd.DataFrame, label: str = "b") -> None:
    panel_label(ax, label, x=-0.08, y=1.16)
    title(ax, "Skill relative to the best individual member", x=0.0, y=1.16)
    metric_order = ["Average precision", "Precision", "Recall", "CSI"]
    base = np.arange(len(metric_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(RECIPE_ORDER))
    ax.axvline(0.0, color=INK, linewidth=0.85, zorder=1)
    for offset, recipe in zip(offsets, RECIPE_ORDER):
        part = source[source["recipe"] == recipe].set_index("metric_label")
        values = np.asarray(
            [float(part.loc[metric, "relative_change_percent"]) for metric in metric_order],
            dtype=float,
        )
        y = base + offset
        ax.hlines(y, 0.0, values, color=RECIPE_COLORS[recipe], linewidth=1.15, alpha=0.85)
        ax.scatter(
            values,
            y,
            s=25 if recipe == FAMILY_BALANCED_ENSEMBLE else 20,
            color=RECIPE_COLORS[recipe],
            edgecolor=WHITE,
            linewidth=0.55,
            zorder=3,
            label=RECIPE_SHORT[recipe],
        )
    finite = pd.to_numeric(source["relative_change_percent"], errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lower = min(-2.0, float(finite.min()) - 1.3)
        upper = max(2.0, float(finite.max()) + 1.3)
        ax.set_xlim(lower, upper)
    ax.set_yticks(base, metric_order)
    ax.invert_yaxis()
    ax.set_xlabel("Relative change (%) · positive favours ensemble")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.025),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    style_axis(ax, xgrid=True)


def draw_transition(
    ax: plt.Axes,
    transitions: pd.DataFrame,
    scope: str,
    heading: str,
    label: str,
) -> None:
    panel_label(ax, label, x=-0.10, y=1.08)
    part = transitions[transitions["scope"] == scope].copy()
    if part.empty:
        raise ValueError(f"No transition rows for scope={scope}")
    total = int(part["scope_total"].max())
    title(ax, f"{heading}  (n={total:,})", x=0.0, y=1.08)
    candidates = (
        TIANJI_PANGU_ENSEMBLE,
        ALL_FOUR_ENSEMBLE,
        FAMILY_BALANCED_ENSEMBLE,
    )
    y = np.arange(len(candidates), dtype=float)
    left = np.zeros(len(candidates), dtype=float)
    for category in TRANSITION_ORDER:
        values = []
        for recipe in candidates:
            selected = part[(part["candidate"] == recipe) & (part["category"] == category)]
            values.append(100.0 * float(selected.iloc[0]["share"]) if not selected.empty else 0.0)
        values_array = np.asarray(values, dtype=float)
        bars = ax.barh(
            y,
            values_array,
            left=left,
            height=0.62,
            color=TRANSITION_COLORS[category],
            edgecolor=WHITE,
            linewidth=0.45,
            label=TRANSITION_LABELS[category],
        )
        if category in {"candidate_rescue", "candidate_harm"}:
            for bar, value in zip(bars, values_array):
                if value >= 2.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        fontweight="bold",
                        color=WHITE,
                    )
        left += values_array
    ax.set_yticks(y, [RECIPE_SHORT[recipe] for recipe in candidates])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of samples in the observed regime (%)")
    style_axis(ax, xgrid=True)


def parse_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [item.strip().lower() for item in value.replace(":", ",").split(",") if item.strip()]
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported formats {unknown}")
    return ["svg", *[item for item in requested if item != "svg"]]


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {
            "facecolor": WHITE,
            "bbox_inches": "tight",
            "pad_inches": 0.035,
        }
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = int(dpi)
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def make_composite(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = GridSpec(
        2,
        6,
        figure=fig,
        left=0.075,
        right=0.985,
        bottom=0.185,
        top=0.94,
        height_ratios=[1.0, 0.92],
        wspace=1.10,
        hspace=0.78,
    )
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2:6])
    ax_c = fig.add_subplot(grid[1, 0:3])
    ax_d = fig.add_subplot(grid[1, 3:6])
    draw_recipe_matrix(ax_a, tables["recipe_weights"], "a")
    draw_relative_gain(ax_b, tables["relative_gain"], "b")
    draw_transition(
        ax_c,
        tables["transitions"],
        "observed_low_vis_lt1km",
        "Observed Low-vis: rescue and harm",
        "c",
    )
    draw_transition(
        ax_d,
        tables["transitions"],
        "observed_non_low_vis_ge1km",
        "Non-Low-vis: false-alarm trade-off",
        "d",
    )
    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=4,
        columnspacing=1.2,
        handlelength=1.3,
        frameon=False,
    )
    return fig


def standalone_panel(
    draw: Callable[..., object],
    args: Tuple[object, ...],
    size: Tuple[float, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.20, top=0.82)
    draw(ax, *args)
    if draw is draw_transition:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            columnspacing=1.0,
            frameon=False,
        )
    return fig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_bundle(
    analysis_dir: Path | str,
    out_dir: Path | str,
    *,
    stem: str = "fig_operational_source_ensembles",
    formats: Sequence[str] = ("svg", "pdf", "png", "tiff"),
    dpi: int = 600,
) -> Dict[str, object]:
    configure_style()
    analysis = Path(analysis_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = read_tables(analysis)
    exports: Dict[str, List[Path]] = {
        "composite": save_figure(make_composite(tables), output, stem, formats, dpi)
    }
    panels = (
        (
            "a_recipe_weights",
            draw_recipe_matrix,
            (tables["recipe_weights"], "a"),
            (4.2, 3.0),
        ),
        (
            "b_relative_gain",
            draw_relative_gain,
            (tables["relative_gain"], "b"),
            (6.2, 3.2),
        ),
        (
            "c_lowvis_transitions",
            draw_transition,
            (
                tables["transitions"],
                "observed_low_vis_lt1km",
                "Observed Low-vis: rescue and harm",
                "c",
            ),
            (5.2, 3.1),
        ),
        (
            "d_nonlowvis_transitions",
            draw_transition,
            (
                tables["transitions"],
                "observed_non_low_vis_ge1km",
                "Non-Low-vis: false-alarm trade-off",
                "d",
            ),
            (5.2, 3.1),
        ),
    )
    for suffix, draw, draw_args, size in panels:
        panel = standalone_panel(draw, draw_args, size)
        exports[suffix] = save_figure(panel, output, f"{stem}_{suffix}", formats, dpi)
    manifest = {
        "status": "completed",
        "figure_stem": stem,
        "core_claim": (
            "Pangu's independent operational contribution is evaluated through "
            "predeclared ensemble recipes and paired rescue/harm transitions."
        ),
        "archetype": "asymmetric quantitative composite",
        "backend": "Python/matplotlib",
        "figure_size_inches": list(FIGURE_SIZE),
        "source_data": {
            name: f"operational_ensemble_{name}.csv" for name in tables
        },
        "exports": {
            key: [{"file": path.name, "sha256": sha256_file(path)} for path in paths]
            for key, paths in exports.items()
        },
    }
    manifest_path = output / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": manifest_path.name}


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    analysis = Path(args.analysis_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else analysis
    report = plot_bundle(
        analysis,
        output,
        stem=args.stem,
        formats=formats,
        dpi=int(args.dpi),
    )
    print(f"[OK] operational-ensemble figure bundle: {output / str(report['manifest'])}")


if __name__ == "__main__":
    main()
