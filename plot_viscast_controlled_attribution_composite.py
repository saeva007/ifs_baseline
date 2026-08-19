#!/usr/bin/env python3
"""Draw the controlled attribution figure for VisCast operator and input effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_source_palette import SOURCE_COLORS, SOURCE_DARK_COLORS
from paper_figure_geometry import (
    ENDPOINT_BAR_WIDTH,
    INTERVAL_CAPSIZE,
    INTERVAL_CAPTHICK,
    INTERVAL_LINEWIDTH,
    INTERVAL_MARKERSIZE,
    TWO_SOURCE_BAR_WIDTH,
)


FIGURE_WIDTH = 7.60
INK = "#17191B"
TIANJI = SOURCE_COLORS["tianji"]
PANGU = SOURCE_COLORS["pangu"]
IFS = SOURCE_COLORS["baseline"]
IFS_VISCAST = "#202124"
TIANJI_DARK = SOURCE_DARK_COLORS["tianji"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ifs-comparison-dir",
        type=Path,
        required=True,
        help="Directory containing ifs_diagnostic_matched_metrics.csv.",
    )
    p.add_argument(
        "--endpoint-dir",
        type=Path,
        required=True,
        help="Q-Core endpoint directory containing metrics-by-seed and bootstrap draws.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--figure-stem", default="fig_viscast_controlled_attribution")
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
            "legend.fontsize": 7.2,
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


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#31363B")
        spine.set_linewidth(0.75)


def panel_label(ax: plt.Axes, letter: str, x: float = -0.16) -> None:
    ax.text(
        x,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def normalize_masks(values: pd.Series) -> tuple[pd.Series, int]:
    raw = values.astype(str).str.replace(r"\.0$", "", regex=True)
    width = int(raw.str.len().max())
    return raw.str.zfill(width), width


def endpoint_distributions(metrics: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    frame = metrics.copy()
    frame["mask"], width = normalize_masks(frame["mask"])
    pangu_mask, tianji_mask = "0" * width, "1" * width
    frame = frame[frame["mask"].isin([pangu_mask, tianji_mask])]
    pivot = frame.pivot(index="seed", columns="mask", values=metric).dropna()
    if pivot.empty or pangu_mask not in pivot or tianji_mask not in pivot:
        raise ValueError(f"Cannot resolve paired Tianji/Pangu endpoints for {metric}")
    return (
        pivot[pangu_mask].to_numpy(dtype=float),
        pivot[tianji_mask].to_numpy(dtype=float),
    )


def draw_operator_panel(
    ax: plt.Axes,
    matched: pd.DataFrame,
    title: str,
    specs: list[tuple[str, str]],
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> pd.DataFrame:
    indexed = matched.set_index(matched["source"].astype(str).str.lower())
    if "ifs" not in indexed.index or "ifs_diagnostic" not in indexed.index:
        raise ValueError("IFS comparison must contain source rows 'ifs' and 'ifs_diagnostic'")
    model = indexed.loc["ifs"]
    diagnostic = indexed.loc["ifs_diagnostic"]
    x = np.arange(len(specs), dtype=float)
    width = TWO_SOURCE_BAR_WIDTH
    model_values = np.asarray([float(model[key]) for key, _ in specs])
    diagnostic_values = np.asarray([float(diagnostic[key]) for key, _ in specs])
    ax.bar(x - width / 2, diagnostic_values, width, color=IFS, label="IFS diagnostic VIS")
    ax.bar(x + width / 2, model_values, width, color=IFS_VISCAST, label="IFS-driven VisCast")
    ax.set_xticks(x, [label for _, label in specs])
    ax.set_ylim(0.0, min(1.0, max(model_values.max(), diagnostic_values.max()) * 1.22))
    if show_ylabel:
        ax.set_ylabel("Score")
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    if show_legend:
        ax.legend(loc="upper left", fontsize=6.6)
    style_axis(ax)
    return pd.DataFrame(
        [
            {"comparison": "operator", "source": source, "metric": key, "value": value}
            for source, values in (
                ("IFS diagnostic VIS", diagnostic_values),
                ("IFS-driven VisCast", model_values),
            )
            for (key, _), value in zip(specs, values)
        ]
    )


def draw_endpoint_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> pd.DataFrame:
    pangu, tianji = endpoint_distributions(metrics, metric)
    values = np.asarray([pangu.mean(), tianji.mean()])
    ax.bar(
        [0, 1],
        values,
        width=ENDPOINT_BAR_WIDTH,
        color=[PANGU, TIANJI],
        edgecolor=[SOURCE_DARK_COLORS["pangu"], TIANJI_DARK],
        linewidth=0.8,
    )
    ax.set_xticks([0, 1], ["Pangu", "Tianji"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, min(1.0, max(0.10, values.max() * 1.20)))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    style_axis(ax)
    rows = []
    for source, distribution in (("Pangu", pangu), ("Tianji", tianji)):
        rows.extend(
            {"comparison": "input", "source": source, "metric": metric, "seed": seed, "value": value}
            for seed, value in enumerate(distribution)
        )
    return pd.DataFrame(rows)


def bootstrap_interval(gap: pd.DataFrame, metric: str) -> tuple[float, float]:
    draws = pd.to_numeric(
        gap.loc[gap["metric"].astype(str) == metric, "delta_all1_minus_all0"],
        errors="coerce",
    ).dropna()
    if draws.empty:
        raise ValueError(f"No bootstrap draws for {metric}")
    return tuple(np.percentile(draws.to_numpy(dtype=float), [2.5, 97.5]))


def draw_delta_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    gap: pd.DataFrame,
    *,
    title: Optional[str] = "Bootstrap differences",
) -> pd.DataFrame:
    specs = [
        ("low_vis_ap", "Low-vis AP"),
        ("low_vis_recall_matched_fpr", "Matched-FPR recall"),
    ]
    rows = []
    for key, label in specs:
        pangu, tianji = endpoint_distributions(metrics, key)
        delta = float(tianji.mean() - pangu.mean())
        lo, hi = bootstrap_interval(gap, key)
        rows.append({"metric": key, "label": label, "delta": delta, "ci_low": lo, "ci_high": hi})
    source = pd.DataFrame(rows)
    # Keep the two related contrasts visually grouped near the panel centre.
    y = np.asarray([0.62, 0.38], dtype=float)
    ax.axvline(0.0, color=INK, linewidth=0.85, zorder=0)
    for yi, (_, row) in zip(y, source.iterrows()):
        ax.errorbar(
            row.delta,
            yi,
            xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
            fmt="o",
            markersize=INTERVAL_MARKERSIZE,
            color=TIANJI_DARK,
            markerfacecolor=TIANJI,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=TIANJI_DARK,
            elinewidth=INTERVAL_LINEWIDTH,
            capsize=INTERVAL_CAPSIZE,
            capthick=INTERVAL_CAPTHICK,
            zorder=3,
        )
    display_labels = [
        "Matched-FPR\nrecall" if label == "Matched-FPR recall" else label
        for label in source["label"].astype(str)
    ]
    ax.set_yticks(y, display_labels)
    ax.tick_params(axis="y", labelsize=6.9, pad=2.0)
    ax.set_ylim(0.22, 0.78)
    extent = max(abs(float(source["ci_low"].min())), abs(float(source["ci_high"].max())))
    ax.set_xlim(min(-0.01, -0.18 * extent), 1.12 * extent)
    ax.set_xlabel("Difference (Tianji − Pangu)")
    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
    style_axis(ax)
    return source.assign(comparison="input_bootstrap")


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
    ifs_dir = args.ifs_comparison_dir.expanduser().resolve()
    endpoint_dir = args.endpoint_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    matched_path = ifs_dir / "ifs_diagnostic_matched_metrics.csv"
    metrics_path = endpoint_dir / "qcore_t925_metrics_by_seed.csv"
    gap_path = endpoint_dir / "qcore_t925_bootstrap_gap_draws.csv"
    for path in (matched_path, metrics_path, gap_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    matched = pd.read_csv(matched_path)
    metrics = pd.read_csv(metrics_path, dtype={"mask": str})
    gap = pd.read_csv(gap_path)
    operator_metrics = [
        "fog_csi",
        "fog_pod",
        "fog_precision",
        "mist_csi",
        "mist_pod",
        "mist_precision",
        "low_vis_csi",
        "low_vis_recall",
        "low_vis_precision",
    ]
    require_columns(matched, ["source", *operator_metrics], matched_path)
    require_columns(metrics, ["mask", "seed", "low_vis_ap", "low_vis_recall_matched_fpr"], metrics_path)
    require_columns(gap, ["metric", "delta_all1_minus_all0"], gap_path)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.45))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.105,
        hspace=0.56,
        wspace=0.50,
    )
    operator_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    lower_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    operator_panels = [
        (
            "Ultra-low",
            [("fog_csi", "CSI"), ("fog_pod", "Recall"), ("fog_precision", "Precision")],
        ),
        (
            "Moderate-low",
            [("mist_csi", "CSI"), ("mist_pod", "Recall"), ("mist_precision", "Precision")],
        ),
        (
            "Low-vis event",
            [("low_vis_csi", "CSI"), ("low_vis_recall", "Recall"), ("low_vis_precision", "Precision")],
        ),
    ]
    frames = []
    for index, (axis, (title_text, specs)) in enumerate(zip(operator_axes, operator_panels)):
        frames.append(
            draw_operator_panel(
                axis,
                matched,
                title_text,
                specs,
                show_ylabel=(index == 0),
                show_legend=(index == 0),
            )
        )
    frames.extend(
        [
            draw_endpoint_panel(lower_axes[0], metrics, "low_vis_ap", "Low-vis average precision", "Average precision"),
            draw_endpoint_panel(lower_axes[1], metrics, "low_vis_recall_matched_fpr", "Recall at matched FPR", "Low-vis recall"),
            draw_delta_panel(lower_axes[2], metrics, gap),
        ]
    )
    for letter, axis in zip("abc", operator_axes):
        panel_label(axis, letter, x=-0.24)
    for letter, axis in zip("def", lower_axes):
        panel_label(axis, letter, x=-0.24)
    outputs = export(fig, out_dir, args.figure_stem, args.dpi)
    plt.close(fig)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        out_dir / f"{args.figure_stem}_source_data.csv", index=False, float_format="%.8f"
    )
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(
        json.dumps(
            {
                "ifs_comparison_dir": str(ifs_dir),
                "endpoint_dir": str(endpoint_dir),
                "figures": outputs,
                "rendering": "all panels redrawn from source tables on one canvas",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir / f"{args.figure_stem}.png")


if __name__ == "__main__":
    main()
