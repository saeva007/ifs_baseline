#!/usr/bin/env python3
"""Plot the extended source-full best-effort evidence page and every panel.

The layout borrows the evidence hierarchy of recent high-impact forecast
comparison figures: a compact construction panel, one dominant quantitative
comparison, and quieter panels for discrimination, calibration, response and
complementarity.  It retains the manuscript source palette and typography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


FIGURE_SIZE = (7.205, 6.55)
WHITE = "#FFFFFF"
INK = "#17191B"
MID = "#697075"
LIGHT = "#D9DCDE"
GRID = "#E8EAEB"
PALE = "#F7F8F8"
LOWVIS_PALE = "#FBECEF"
POSITIVE = "#2D8A52"
NEGATIVE = "#B64B4B"
ENSEMBLE = "tianji_t2nd_ifs_mean_softmax"
MEMBERS = ("tianji", "T2ND_rh2m_source_full", "ifs")
SOURCE_ORDER = (*MEMBERS, ENSEMBLE)
SOURCE_LABELS = {
    "tianji": "Tianji",
    "T2ND_rh2m_source_full": "Tianji T2ND",
    "ifs": "IFS-trained",
    ENSEMBLE: "Best effort",
}
SOURCE_SHORT = {
    "tianji": "T",
    "T2ND_rh2m_source_full": "T2",
    "ifs": "I",
    ENSEMBLE: "BE",
}
SOURCE_COLORS = {
    "tianji": "#2E5A87",
    "T2ND_rh2m_source_full": "#65A6A2",
    "ifs": "#777B7E",
    ENSEMBLE: "#C44E52",
}
SOURCE_MARKERS = {
    "tianji": "s",
    "T2ND_rh2m_source_full": "^",
    "ifs": "D",
    ENSEMBLE: "o",
}
GAIN_METRICS = (
    ("low_vis_ap", "Average precision", "higher"),
    ("low_vis_precision", "Precision", "higher"),
    ("low_vis_recall", "Recall", "higher"),
    ("low_vis_csi", "CSI", "higher"),
)


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
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.8,
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
    parser.add_argument("--stem", default="fig_best_effort_extended")
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def read_tables(directory: Path) -> Dict[str, pd.DataFrame]:
    names = (
        "paired_metrics",
        "precision_recall",
        "reliability",
        "visibility_response",
        "member_complementarity",
    )
    tables = {}
    for name in names:
        path = directory / f"best_effort_{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing best-effort source table: {path}")
        tables[name] = pd.read_csv(path)
    require_columns(
        tables["paired_metrics"],
        ["source", *(metric for metric, _, _ in GAIN_METRICS)],
        "paired metrics",
    )
    return tables


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
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


def title(ax: plt.Axes, text: str, x: float = 0.0, y: float = 1.04) -> None:
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


def style_axis(
    ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False
) -> None:
    ax.tick_params(length=2.7, width=0.75, color=INK, pad=2.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)
    ax.grid(False)
    ax.set_axisbelow(True)


def rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    wh: Tuple[float, float],
    text: str,
    face: str,
    edge: str,
    fontsize: float = 6.8,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.028",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=edge,
        linespacing=1.1,
    )


def draw_construction(ax: plt.Axes, label: str = "a") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, label, x=-0.02, y=1.02)
    title(ax, "Source-full probability fusion", x=0.08, y=1.02)
    x_positions = (0.02, 0.345, 0.67)
    for source, x in zip(MEMBERS, x_positions):
        compact_label = {
            "tianji": "Tianji",
            "T2ND_rh2m_source_full": "T2ND",
            "ifs": "IFS",
        }[source]
        rounded_box(
            ax,
            (x, 0.64),
            (0.285, 0.19),
            f"{compact_label}\nsoftmax",
            f"{SOURCE_COLORS[source]}18",
            SOURCE_COLORS[source],
            fontsize=6.1,
        )
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.142, 0.64),
                (0.50, 0.45),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.9,
                color=SOURCE_COLORS[source],
            )
        )
    rounded_box(
        ax,
        (0.27, 0.27),
        (0.46, 0.18),
        "Equal-weight mean\nof probabilities",
        "#F9EDEF",
        SOURCE_COLORS[ENSEMBLE],
        fontsize=6.6,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.27),
            (0.50, 0.17),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1,
            color=SOURCE_COLORS[ENSEMBLE],
        )
    )
    ax.text(
        0.50,
        0.09,
        "one argmax",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
    )


def ensemble_gain_source(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("source")
    missing = [source for source in SOURCE_ORDER if source not in indexed.index]
    if missing:
        raise ValueError(f"paired metrics are missing {missing}")
    rows = []
    for metric, metric_label, direction in GAIN_METRICS:
        member_values = indexed.loc[list(MEMBERS), metric].astype(float)
        if direction == "higher":
            best_source = str(member_values.idxmax())
            best = float(member_values.max())
            ensemble = float(indexed.loc[ENSEMBLE, metric])
            gain = 100.0 * (ensemble - best) / best if best > 0.0 else math.nan
        else:
            best_source = str(member_values.idxmin())
            best = float(member_values.min())
            ensemble = float(indexed.loc[ENSEMBLE, metric])
            gain = 100.0 * (best - ensemble) / best if best > 0.0 else math.nan
        rows.append(
            {
                "metric": metric,
                "metric_label": metric_label,
                "direction": direction,
                "best_member": best_source,
                "best_member_label": SOURCE_LABELS[best_source],
                "best_member_value": best,
                "ensemble_value": ensemble,
                "relative_gain_percent": gain,
            }
        )
    return pd.DataFrame(rows)


def draw_gain(ax: plt.Axes, metrics: pd.DataFrame, label: str = "b") -> pd.DataFrame:
    source = ensemble_gain_source(metrics)
    panel_label(ax, label, x=-0.075, y=1.08)
    title(ax, "Best effort relative to the best individual member", x=0.02, y=1.08)
    source = source.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(source))
    gains = source["relative_gain_percent"].to_numpy(dtype=float)
    finite = np.isfinite(gains)
    max_abs = max(float(np.max(np.abs(gains[finite]))) if finite.any() else 1.0, 2.0)
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in gains]
    for yi, value, color in zip(y, gains, colors):
        ax.plot([0.0, value], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter(value, yi, s=34, color=color, edgecolor=WHITE, linewidth=0.65, zorder=3)
        ax.text(
            value + math.copysign(0.035 * max_abs, value if value != 0 else 1),
            yi,
            f"{value:+.1f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )
    ax.axvline(0.0, color=INK, linewidth=0.85)
    ax.set_yticks(y, source["metric_label"])
    ax.set_xlim(-1.27 * max_abs, 1.27 * max_abs)
    ax.set_ylim(-0.55, len(source) - 0.45)
    ax.set_xlabel("Relative improvement over best member (%)")
    style_axis(ax, xgrid=True)
    return source.iloc[::-1].reset_index(drop=True)


def draw_pr(ax: plt.Axes, source: pd.DataFrame, label: str = "c") -> None:
    panel_label(ax, label, x=-0.10, y=1.08)
    title(ax, "Threshold-free low-vis discrimination", x=0.02, y=1.08)
    for method in SOURCE_ORDER:
        part = source[source["source"] == method].sort_values("recall")
        ax.plot(
            part["recall"],
            part["precision"],
            color=SOURCE_COLORS[method],
            linewidth=2.1 if method == ENSEMBLE else 1.15,
            alpha=1.0 if method == ENSEMBLE else 0.82,
            label=SOURCE_LABELS[method],
        )
    ax.set_xlim(0.0, 1.0)
    ymax = min(
        1.0,
        max(0.25, float(source["precision"].replace([np.inf, -np.inf], np.nan).max()) * 1.05),
    )
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", ncol=1, borderaxespad=0.2, handlelength=1.8)
    style_axis(ax, xgrid=True, ygrid=True)


def draw_reliability(ax: plt.Axes, source: pd.DataFrame, label: str = "d") -> None:
    panel_label(ax, label, x=-0.10, y=1.08)
    title(ax, "Probability reliability", x=0.02, y=1.08)
    ax.plot([0, 1], [0, 1], color=LIGHT, linewidth=1.0, linestyle=(0, (3, 2)))
    for method in SOURCE_ORDER:
        part = source[source["source"] == method].sort_values("mean_probability")
        ax.plot(
            part["mean_probability"],
            part["observed_frequency"],
            color=SOURCE_COLORS[method],
            marker=SOURCE_MARKERS[method],
            markersize=3.0 if method == ENSEMBLE else 2.4,
            linewidth=1.8 if method == ENSEMBLE else 1.0,
            alpha=1.0 if method == ENSEMBLE else 0.78,
        )
    max_value = max(
        0.20,
        float(
            np.nanmax(
                source[["mean_probability", "observed_frequency"]].to_numpy(dtype=float)
            )
        )
        * 1.08,
    )
    max_value = min(max_value, 1.0)
    ax.set_xlim(0.0, max_value)
    ax.set_ylim(0.0, max_value)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    style_axis(ax, xgrid=True, ygrid=True)


def draw_response(ax: plt.Axes, source: pd.DataFrame, label: str = "e") -> None:
    panel_label(ax, label, x=-0.075, y=1.08)
    title(ax, "Response across the full observed-visibility distribution", x=0.02, y=1.08)
    labels = (
        source.sort_values("visibility_bin")
        .drop_duplicates("visibility_bin")["visibility_bin_km"]
        .astype(str)
        .tolist()
    )
    x = np.arange(len(labels), dtype=float)
    ax.axvspan(-0.5, 2.5, color=LOWVIS_PALE, zorder=0)
    ax.axvline(2.5, color="#B86B76", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    for method in SOURCE_ORDER:
        part = source[source["source"] == method].sort_values("visibility_bin")
        values = part["mean_low_vis_probability"].to_numpy(dtype=float)
        method_x = part["visibility_bin"].to_numpy(dtype=float)
        ax.plot(
            method_x,
            values,
            color=SOURCE_COLORS[method],
            marker=SOURCE_MARKERS[method],
            markersize=4.0 if method == ENSEMBLE else 3.1,
            linewidth=2.0 if method == ENSEMBLE else 1.05,
            alpha=1.0 if method == ENSEMBLE else 0.78,
            zorder=3,
        )
        if method == ENSEMBLE:
            ax.fill_between(
                method_x,
                part["q25_low_vis_probability"].to_numpy(dtype=float),
                part["q75_low_vis_probability"].to_numpy(dtype=float),
                color=SOURCE_COLORS[method],
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
    ax.text(
        0.04,
        0.95,
        "Observed Low-vis",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color="#9B4553",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Observed visibility (km)")
    ax.set_ylabel("Mean predicted Low-vis probability")
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.0, min(1.0, max(0.25, float(source["mean_low_vis_probability"].max()) * 1.10)))
    style_axis(ax, ygrid=True)


def short_pattern(text: str) -> str:
    if text == "No member":
        return "None"
    replacements = {
        "Tianji T2ND": "T2",
        "IFS-trained": "I",
        "Tianji": "T",
    }
    parts = [part.strip() for part in str(text).split("+")]
    return " + ".join(replacements.get(part, part) for part in parts)


def draw_complementarity(ax: plt.Axes, source: pd.DataFrame, label: str = "f") -> None:
    panel_label(ax, label, x=-0.11, y=1.08)
    title(ax, "Member-hit complementarity", x=0.02, y=1.08)
    part = source.sort_values("n", ascending=True, kind="stable").copy()
    y = np.arange(len(part))
    total = float(part["observed_low_vis_total"].max())
    hit = 100.0 * part["ensemble_hit_n"].to_numpy(dtype=float) / total
    miss = 100.0 * part["ensemble_miss_n"].to_numpy(dtype=float) / total
    ax.barh(y, hit, color=SOURCE_COLORS[ENSEMBLE], height=0.66, label="Ensemble hit")
    ax.barh(
        y,
        miss,
        left=hit,
        color="#D9DCDE",
        edgecolor=WHITE,
        linewidth=0.35,
        height=0.66,
        label="Ensemble miss",
    )
    ax.set_yticks(y, [short_pattern(value) for value in part["member_pattern"]])
    ax.set_xlabel("Share of observed Low-vis samples (%)")
    ax.set_ylim(-0.5, len(part) + 0.85)
    ax.legend(
        loc="upper right",
        ncol=2,
        borderaxespad=0.2,
        columnspacing=0.9,
        handlelength=1.2,
    )
    style_axis(ax, xgrid=True)


def parse_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [
        item.strip().lower()
        for item in value.replace(":", ",").split(",")
        if item.strip()
    ]
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
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def make_composite(tables: Mapping[str, pd.DataFrame]) -> Tuple[plt.Figure, pd.DataFrame]:
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = GridSpec(
        3,
        4,
        figure=fig,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.965,
        height_ratios=[0.84, 1.02, 1.10],
        wspace=0.74,
        hspace=0.66,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1:4])
    ax_c = fig.add_subplot(grid[1, 0:2])
    ax_d = fig.add_subplot(grid[1, 2:4])
    ax_e = fig.add_subplot(grid[2, 0:2])
    ax_f = fig.add_subplot(grid[2, 2:4])
    draw_construction(ax_a)
    gain = draw_gain(ax_b, tables["paired_metrics"])
    draw_pr(ax_c, tables["precision_recall"])
    draw_reliability(ax_d, tables["reliability"])
    draw_response(ax_e, tables["visibility_response"])
    draw_complementarity(ax_f, tables["member_complementarity"])
    return fig, gain


def standalone_panel(
    draw: Callable[..., object],
    data: pd.DataFrame | None,
    size: Tuple[float, float],
    label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.82)
    if data is None:
        draw(ax, label)
    else:
        draw(ax, data, label)
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
    stem: str = "fig_best_effort_extended",
    formats: Sequence[str] = ("svg", "pdf", "png", "tiff"),
    dpi: int = 600,
) -> Dict[str, object]:
    configure_style()
    analysis = Path(analysis_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = read_tables(analysis)
    fig, gain = make_composite(tables)
    gain_path = output / f"{stem}_source_relative_gain.csv"
    gain.to_csv(gain_path, index=False)
    exports: Dict[str, List[Path]] = {
        "composite": save_figure(fig, output, stem, formats, dpi)
    }
    panel_specs = (
        ("a_construction", draw_construction, None, (3.2, 2.6), "a"),
        ("b_relative_gain", draw_gain, tables["paired_metrics"], (6.4, 3.0), "b"),
        ("c_precision_recall", draw_pr, tables["precision_recall"], (3.6, 3.2), "c"),
        ("d_reliability", draw_reliability, tables["reliability"], (3.6, 3.2), "d"),
        ("e_visibility_response", draw_response, tables["visibility_response"], (4.8, 3.0), "e"),
        (
            "f_member_complementarity",
            draw_complementarity,
            tables["member_complementarity"],
            (4.2, 3.2),
            "f",
        ),
    )
    for suffix, draw, data, size, label in panel_specs:
        panel_fig = standalone_panel(draw, data, size, label)
        exports[suffix] = save_figure(
            panel_fig, output, f"{stem}_{suffix}", formats, dpi
        )
    manifest = {
        "status": "completed",
        "figure_stem": stem,
        "core_claim": (
            "The equal-weight probability ensemble is evaluated on the exact member "
            "intersection across discrimination, thresholded skill, calibration, the "
            "full visibility distribution and member-hit complementarity."
        ),
        "archetype": "asymmetric quantitative composite",
        "figure_size_inches": list(FIGURE_SIZE),
        "source_data": {
            name: f"best_effort_{name}.csv" for name in tables
        },
        "derived_gain_source": gain_path.name,
        "exports": {
            key: [
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ]
            for key, paths in exports.items()
        },
    }
    manifest_path = output / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": manifest_path.name}


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    analysis = Path(args.analysis_dir).expanduser().resolve()
    output = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else analysis
    )
    report = plot_bundle(
        analysis,
        output,
        stem=args.stem,
        formats=formats,
        dpi=int(args.dpi),
    )
    print(f"[OK] best-effort figure bundle: {output / str(report['manifest'])}")


if __name__ == "__main__":
    main()
