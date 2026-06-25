#!/usr/bin/env python3
"""
Plot the Tianji/IFS mean-softmax overlap ensemble result.

The normal path reads ``overall_metrics.csv`` from an ensemble output directory.
For local manuscript drafting, ``--use_embedded_log_values`` recreates the figure
from the 2026-05-25 run log copied into this script.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENSEMBLE_SOURCE = "ensemble_mean_softmax"
DEFAULT_REMOTE_OUT_DIR = (
    "/public/home/putianshu/vis_mlp/"
    "paper_eval_results_pm10_pm25_journal/overlap_softmax_ensemble"
)
DEFAULT_LOCAL_OUT_DIR = r"C:\vis_code\static_rnn_eval_results_r13\overlap_softmax_ensemble_summary"

KEY_METRICS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("fog_pod", "Ultra-low recall"),
    ("fog_precision", "Ultra-low precision"),
    ("fog_csi", "Ultra-low CSI"),
    ("low_vis_recall", "Low-vis event recall"),
    ("low_vis_precision", "Low-vis event precision"),
    ("low_vis_fpr", "Low-vis event FPR"),
    ("mist_csi", "Moderate-low CSI"),
    ("mist_precision", "Moderate-low precision"),
]

LOWER_IS_BETTER = {"fog_far", "mist_far", "low_vis_fpr", "multiclass_brier", "low_vis_brier", "ece_low_vis", "ece_multiclass"}

EMBEDDED_LOG_VALUES: Dict[str, Dict[str, float]] = {
    ENSEMBLE_SOURCE: {
        "accuracy": 0.946771,
        "fog_csi": 0.165683,
        "fog_f1": 0.284268,
        "fog_far": 0.820445,
        "fog_pod": 0.681992,
        "fog_precision": 0.179555,
        "low_vis_csi": 0.201522,
        "low_vis_f1": 0.335445,
        "low_vis_fpr": 0.043731,
        "low_vis_precision": 0.227332,
        "low_vis_recall": 0.639638,
        "mist_csi": 0.082533,
        "mist_f1": 0.152482,
        "mist_far": 0.871046,
        "mist_pod": 0.186511,
        "mist_precision": 0.128954,
    },
    "tianji": {
        "accuracy": 0.950050,
        "fog_csi": 0.182742,
        "fog_f1": 0.309014,
        "fog_far": 0.795548,
        "fog_pod": 0.632482,
        "fog_precision": 0.204452,
        "low_vis_csi": 0.202911,
        "low_vis_f1": 0.337366,
        "low_vis_fpr": 0.039942,
        "low_vis_precision": 0.233775,
        "low_vis_recall": 0.605820,
        "mist_csi": 0.075248,
        "mist_f1": 0.139965,
        "mist_far": 0.893072,
        "mist_pod": 0.202543,
        "mist_precision": 0.106928,
    },
    "ifs": {
        "accuracy": 0.947191,
        "fog_csi": 0.162644,
        "fog_f1": 0.279783,
        "fog_far": 0.821294,
        "fog_pod": 0.644076,
        "fog_precision": 0.178706,
        "low_vis_csi": 0.197957,
        "low_vis_f1": 0.330491,
        "low_vis_fpr": 0.042896,
        "low_vis_precision": 0.225277,
        "low_vis_recall": 0.620111,
        "mist_csi": 0.080250,
        "mist_f1": 0.148576,
        "mist_far": 0.878400,
        "mist_pod": 0.190934,
        "mist_precision": 0.121600,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot overlap mean-softmax ensemble summary.")
    p.add_argument("--results_dir", default=DEFAULT_REMOTE_OUT_DIR, help="Directory containing overall_metrics.csv.")
    p.add_argument("--out_dir", default="", help="Figure output directory. Defaults to --results_dir.")
    p.add_argument(
        "--use_embedded_log_values",
        action="store_true",
        help="Use the 2026-05-25 log values embedded in this script for local figure drafting.",
    )
    p.add_argument("--basename", default="fig_overlap_softmax_ensemble_summary")
    return p.parse_args()


def load_metrics(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    if args.use_embedded_log_values:
        rows = []
        for source, metrics in EMBEDDED_LOG_VALUES.items():
            row = {"source": source}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "source_metrics_from_20260525_log.csv", index=False)
        return df

    path = Path(args.results_dir) / "overall_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {path}. Use --results_dir to point to the ensemble output "
            "or --use_embedded_log_values for the local pasted-log snapshot."
        )
    df = pd.read_csv(path)
    needed = {"tianji", "ifs", ENSEMBLE_SOURCE}
    have = set(df["source"].astype(str))
    missing = sorted(needed - have)
    if missing:
        raise ValueError(f"overall_metrics.csv is missing source rows: {missing}")
    return df


def metric_value(df: pd.DataFrame, source: str, metric: str) -> float:
    row = df.loc[df["source"].astype(str) == source]
    if row.empty or metric not in row.columns:
        return math.nan
    return float(row.iloc[0][metric])


def is_improvement(metric: str, delta: float) -> bool:
    if not math.isfinite(delta):
        return False
    return delta < 0 if metric in LOWER_IS_BETTER else delta > 0


def delta_frame(df: pd.DataFrame, comparator: str) -> pd.DataFrame:
    rows = []
    for metric, label in KEY_METRICS:
        ens = metric_value(df, ENSEMBLE_SOURCE, metric)
        comp = metric_value(df, comparator, metric)
        delta = ens - comp
        rows.append(
            {
                "metric": metric,
                "label": label,
                "ensemble": ens,
                "comparator": comp,
                "delta": delta,
                "delta_pp": delta * 100.0,
                "better": is_improvement(metric, delta),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "grid.linewidth": 0.45,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def plot_delta_panel(ax, ddf: pd.DataFrame, title: str) -> None:
    plot_df = ddf.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["better"].to_numpy(), "#2F7D50", "#B65A50")
    ax.barh(y, plot_df["delta_pp"], color=colors, height=0.58, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#222222", lw=0.7)
    max_abs = max(0.3, float(np.nanmax(np.abs(plot_df["delta_pp"]))) * 1.25)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Ensemble - comparator (percentage points)")
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)
    for yi, row in plot_df.iterrows():
        val = float(row["delta_pp"])
        if not math.isfinite(val):
            continue
        label = f"{val:+.2f}"
        offset = max_abs * 0.035
        x = val + offset if val >= 0 else val - offset
        ha = "left" if val >= 0 else "right"
        ax.text(x, yi, label, va="center", ha=ha, fontsize=6.3, color="#2D2D2D")
    ax.text(
        0.02,
        0.03,
        "green = favorable change",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#4B4B4B",
    )


def plot_precision_recall_panel(ax, df: pd.DataFrame) -> None:
    method_order = ["tianji", "ifs", ENSEMBLE_SOURCE]
    method_labels = {"tianji": "Tianji", "ifs": "IFS", ENSEMBLE_SOURCE: "Ensemble"}
    colors = {"tianji": "#2E5A87", "ifs": "#777777", ENSEMBLE_SOURCE: "#2F7D50"}
    endpoints = [
        ("Ultra-low", "fog_pod", "fog_precision", "o", -0.25),
        ("Low-vis event", "low_vis_recall", "low_vis_precision", "s", 0.25),
    ]
    for label, recall_metric, precision_metric, marker, label_jitter in endpoints:
        xs = [metric_value(df, m, recall_metric) * 100.0 for m in method_order]
        ys = [metric_value(df, m, precision_metric) * 100.0 for m in method_order]
        ax.plot(xs, ys, color="#B8B8B8", lw=0.8, zorder=1)
        for method, x, y in zip(method_order, xs, ys):
            ax.scatter(
                x,
                y,
                s=42,
                color=colors[method],
                marker=marker,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.text(
                x + 0.35,
                y + label_jitter,
                f"{method_labels[method]} {label}",
                color=colors[method],
                fontsize=6.2,
                va="center",
            )
    ax.set_xlabel("Recall / POD (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("C. Operating-point shift", loc="left", fontweight="bold", pad=4)
    ax.grid(alpha=0.22)
    ax.set_xlim(58, 70.5)
    ax.set_ylim(9.2, 25.6)
    ax.text(
        0.02,
        0.08,
        "The ensemble moves rightward: higher detection,\nwith a precision cost versus Tianji.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#3E3E3E",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir or (DEFAULT_LOCAL_OUT_DIR if args.use_embedded_log_values else args.results_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args, out_dir)
    delta_t = delta_frame(df, "tianji")
    delta_i = delta_frame(df, "ifs")
    delta_t.to_csv(out_dir / "source_delta_ensemble_minus_tianji.csv", index=False)
    delta_i.to_csv(out_dir / "source_delta_ensemble_minus_ifs.csv", index=False)

    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1.0], hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_delta_panel(ax_a, delta_t, "A. Ensemble compared with Tianji")
    plot_delta_panel(ax_b, delta_i, "B. Ensemble compared with IFS")
    plot_precision_recall_panel(ax_c, df)

    fig.suptitle(
        "Mean-softmax ensemble is recall-oriented rather than uniformly superior",
        x=0.02,
        y=0.99,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.948,
        "Positive bars are not always favorable: lower FPR is better. Values are percentage-point changes.",
        ha="left",
        va="top",
        fontsize=6.8,
        color="#4A4A4A",
    )
    fig.subplots_adjust(top=0.86, left=0.14, right=0.985, bottom=0.105)

    stem = out_dir / args.basename
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote figure set to {stem}.[png|pdf|svg|tiff]")


if __name__ == "__main__":
    main()
