#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the direct IFS-input VisGen versus IFS diagnostic comparison.

The input is the ``overall_metrics.csv`` written by the source-full evaluator.
Both rows must therefore come from the same matched station-time evaluation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS: Sequence[Tuple[str, str, str]] = (
    ("ifs", "VisGen (IFS inputs)", "#2E5A87"),
    ("ifs_diagnostic", "IFS diagnostic VIS", "#4A4A4A"),
)

PANELS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    (
        "Precision",
        (("fog_precision", "Ultra-low\n(<500 m)"), ("mist_precision", "Moderate-low\n(500–1,000 m)"), ("low_vis_precision", "Low-vis event\n(<1,000 m)")),
    ),
    (
        "Recall",
        (("fog_pod", "Ultra-low\n(<500 m)"), ("mist_pod", "Moderate-low\n(500–1,000 m)"), ("low_vis_recall", "Low-vis event\n(<1,000 m)")),
    ),
    (
        "F1 score",
        (("fog_f1", "Ultra-low\n(<500 m)"), ("mist_f1", "Moderate-low\n(500–1,000 m)"), ("low_vis_f1", "Low-vis event\n(<1,000 m)")),
    ),
    (
        "CSI",
        (("fog_csi", "Ultra-low\n(<500 m)"), ("mist_csi", "Moderate-low\n(500–1,000 m)"), ("low_vis_csi", "Low-vis event\n(<1,000 m)")),
    ),
)

ALIASES: Dict[str, Sequence[str]] = {
    "fog_precision": ("fog_precision", "Fog_P"),
    "fog_pod": ("fog_pod", "Fog_R"),
    "fog_f1": ("fog_f1", "Fog_F1"),
    "fog_csi": ("fog_csi", "Fog_CSI"),
    "mist_precision": ("mist_precision", "Mist_P"),
    "mist_pod": ("mist_pod", "Mist_R"),
    "mist_f1": ("mist_f1", "Mist_F1"),
    "mist_csi": ("mist_csi", "Mist_CSI"),
    "low_vis_precision": ("low_vis_precision",),
    "low_vis_recall": ("low_vis_recall", "low_vis_pod"),
    "low_vis_f1": ("low_vis_f1",),
    "low_vis_csi": ("low_vis_csi",),
    "low_vis_fpr": ("low_vis_fpr", "false_positive_rate"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--figure_stem", default="fig_ifs_trained_visgen_vs_ifs_diagnostic")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
        }
    )


def metric_value(row: pd.Series, name: str) -> float:
    for column in ALIASES[name]:
        if column not in row.index:
            continue
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return math.nan


def adaptive_upper(values: Iterable[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return 0.2
    vmax = float(finite.max())
    padded = vmax + max(0.025, 0.10 * vmax)
    step = 0.05 if padded > 0.20 else 0.02
    return min(1.0, max(step * 3, math.ceil(padded / step) * step))


def select_rows(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    if "source" not in frame.columns:
        raise ValueError("metrics table has no source column")
    rows: Dict[str, pd.Series] = {}
    for source, _label, _color in METHODS:
        candidates = ("ifs_diagnostic_matched_model", "ifs") if source == "ifs" else (source,)
        subset = pd.DataFrame()
        selected_source = ""
        for candidate in candidates:
            subset = frame[frame["source"].astype(str).eq(candidate)]
            if not subset.empty:
                selected_source = candidate
                break
        if subset.empty:
            raise ValueError(f"Required source row is missing: {','.join(candidates)}")
        rows[source] = subset.iloc[-1]
        rows[source].attrs["selected_source"] = selected_source
    n_values = []
    for row in rows.values():
        try:
            n_values.append(int(float(row.get("n", row.get("matched_rows", math.nan)))))
        except (TypeError, ValueError, OverflowError):
            pass
    if len(n_values) == 2 and n_values[0] != n_values[1]:
        raise ValueError(f"IFS-trained and diagnostic rows have different sample counts: {n_values}")
    return rows


def source_table(rows: Dict[str, pd.Series], metrics_csv: Path) -> pd.DataFrame:
    records = []
    for source, label, _color in METHODS:
        row = rows[source]
        record = {
            "source": source,
            "selected_source_row": row.attrs.get("selected_source", source),
            "display_label": label,
            "metrics_csv": str(metrics_csv),
        }
        for metric in ALIASES:
            record[metric] = metric_value(row, metric)
        for name in ("n", "matched_rows", "sample_scope", "threshold_source", "evaluation_mode"):
            record[name] = row.get(name, "")
        records.append(record)
    return pd.DataFrame(records)


def draw(rows: Dict[str, pd.Series], out_dir: Path, stem: str, dpi: int, metrics_csv: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.2))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.80, hspace=0.42, wspace=0.18)
    width = 0.32
    for panel_index, (ax, (title, metrics)) in enumerate(zip(axes.flat, PANELS)):
        x = np.arange(len(metrics), dtype=float)
        all_values = [metric_value(rows[source], metric) for source, _label, _color in METHODS for metric, _category in metrics]
        y_max = adaptive_upper(all_values)
        for method_index, (source, label, color) in enumerate(METHODS):
            values = [metric_value(rows[source], metric) for metric, _category in metrics]
            offset = (method_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width * 0.92,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                label=label if panel_index == 0 else None,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(value + y_max * 0.025, y_max * 0.97),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [category for _metric, category in metrics])
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.24)
        ax.text(-0.13, 1.06, "abcd"[panel_index], transform=ax.transAxes, fontsize=12, fontweight="bold")
        if panel_index % 2 == 0:
            ax.set_ylabel("Score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.91))
    fig.suptitle(
        "A learned visibility operator recovers low-visibility signals from IFS forecast states",
        fontsize=13,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.08,
        0.035,
        "Same matched station–time evaluation; zero-based panel-specific ranges. Low-vis FPR is retained in the source-data table.",
        fontsize=8,
        color="#555555",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    source = source_table(rows, metrics_csv)
    source.to_csv(out_dir / f"{stem}_source_data.csv", index=False, float_format="%.8f")
    for extension in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        print(f"[figure] {path}", flush=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    if not metrics_csv.is_file():
        raise FileNotFoundError(metrics_csv)
    frame = pd.read_csv(metrics_csv)
    rows = select_rows(frame)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else metrics_csv.parent
    draw(rows, out_dir, args.figure_stem, args.dpi, metrics_csv)


if __name__ == "__main__":
    main()
