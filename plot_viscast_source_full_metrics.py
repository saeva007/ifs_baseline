#!/usr/bin/env python3
"""Redraw the source-specific full-input performance figure with VisCast naming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_WIDTH = 7.60
INK = "#17191B"
SOURCE_ORDER = [
    "tianji",
    "T2ND_rh2m_source_full",
    "ifs",
    "tianji_t2nd_ifs_mean_softmax",
    "era5_2025_source_full",
    "pangu2025_source_full",
    "ifs_diagnostic",
]
SOURCE_LABELS = {
    "tianji": "Tianji VisCast",
    "T2ND_rh2m_source_full": "T2ND RH2m VisCast",
    "ifs": "IFS VisCast",
    "tianji_t2nd_ifs_mean_softmax": "Mean-softmax ensemble",
    "era5_2025_source_full": "ERA5 VisCast",
    "pangu2025_source_full": "Pangu VisCast",
    "ifs_diagnostic": "IFS diagnostic VIS",
}
SOURCE_COLORS = {
    "tianji": "#386890",
    "T2ND_rh2m_source_full": "#2B9B7C",
    "ifs": "#85888B",
    "tianji_t2nd_ifs_mean_softmax": "#C45458",
    "era5_2025_source_full": "#DA6700",
    "pangu2025_source_full": "#8E6BBE",
    "ifs_diagnostic": "#3D3D3D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--figure-stem", default="fig_forecast_source_key_metrics_source_full_viscast")
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
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def require_columns(df: pd.DataFrame, names: Iterable[str], path: Path) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#31363B")
        spine.set_linewidth(0.75)


def select_rows(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["source"] = work["source"].astype(str)
    if "sample_scope" in work.columns:
        preferred = work[work["sample_scope"].astype(str).str.contains("independent_source", case=False, na=False)]
        if not preferred.empty:
            work = preferred
    rows = []
    for source in SOURCE_ORDER:
        part = work[work["source"] == source]
        if part.empty:
            continue
        rows.append(part.iloc[0])
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("No requested source-full configurations were found")
    return selected


def draw_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    title: str,
    metrics: list[tuple[str, str]],
    show_ylabel: bool,
) -> pd.DataFrame:
    x = np.arange(len(metrics), dtype=float)
    n_sources = len(source)
    width = min(0.13, 0.80 / max(n_sources, 1))
    records = []
    for idx, row in enumerate(source.itertuples(index=False)):
        source_key = str(row.source)
        values = [float(getattr(row, metric)) for metric, _ in metrics]
        offset = (idx - (n_sources - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width * 0.90,
            color=SOURCE_COLORS[source_key],
            label=SOURCE_LABELS[source_key],
            edgecolor="white",
            linewidth=0.3,
        )
        records.extend(
            {"panel": title, "source": source_key, "metric": metric, "value": value}
            for (metric, _), value in zip(metrics, values)
        )
    values = np.asarray([record["value"] for record in records], dtype=float)
    upper = min(1.0, max(0.10, float(values.max()) * 1.12))
    ax.set_ylim(0.0, upper)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    if show_ylabel:
        ax.set_ylabel("Score")
    style_axis(ax)
    return pd.DataFrame(records)


def export(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf", "svg", "tiff"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext in {"png", "tiff"} else None, facecolor="white")
        outputs.append(str(path))
    return outputs


def main() -> None:
    args = parse_args()
    setup_style()
    metrics_path = args.metrics_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    frame = pd.read_csv(metrics_path)
    required = [
        "source",
        "fog_precision",
        "mist_precision",
        "low_vis_precision",
        "fog_pod",
        "mist_pod",
        "low_vis_recall",
        "fog_f1",
        "mist_f1",
        "low_vis_f1",
        "fog_csi",
        "mist_csi",
        "low_vis_csi",
    ]
    require_columns(frame, required, metrics_path)
    source = select_rows(frame)
    categories = [
        ("fog", "Ultra-low\n(<500 m)"),
        ("mist", "Moderate-low\n(500–1000 m)"),
        ("low_vis", "Low-vis event\n(<1000 m)"),
    ]
    panels = [
        ("Precision", [(f"{prefix}_precision", label) for prefix, label in categories]),
        ("Recall", [("fog_pod", categories[0][1]), ("mist_pod", categories[1][1]), ("low_vis_recall", categories[2][1])]),
        ("F1 score", [(f"{prefix}_f1", label) for prefix, label in categories]),
        ("CSI", [(f"{prefix}_csi", label) for prefix, label in categories]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.35))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.105, hspace=0.48, wspace=0.18)
    frames = []
    for index, (ax, (title, specs)) in enumerate(zip(axes.flat, panels)):
        frames.append(draw_panel(ax, source, title, specs, index % 2 == 0))
        panel_label(ax, "abcd"[index])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.985),
        ncol=4,
        columnspacing=1.05,
        handlelength=1.2,
        handletextpad=0.45,
    )
    outputs = export(fig, out_dir, args.figure_stem, args.dpi)
    plt.close(fig)
    pd.concat(frames, ignore_index=True).to_csv(
        out_dir / f"{args.figure_stem}_source_data.csv", index=False, float_format="%.8f"
    )
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(
        json.dumps({"metrics_csv": str(metrics_path), "figures": outputs}, indent=2),
        encoding="utf-8",
    )
    print(out_dir / f"{args.figure_stem}.png")


if __name__ == "__main__":
    main()
