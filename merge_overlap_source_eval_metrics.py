#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge split overlap/source evaluation metrics and redraw the key source figure."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge split overlap source evaluation outputs.")
    p.add_argument("--root_dir", default="", help="Directory searched recursively for overall_metrics.csv files.")
    p.add_argument(
        "--input_dirs",
        default="",
        help="Comma/semicolon-separated run directories containing overall_metrics.csv; overrides --root_dir.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_figures", action="store_true")
    return p.parse_args()


def split_dirs(value: str) -> List[Path]:
    out: List[Path] = []
    for raw in str(value or "").replace(";", ",").split(","):
        item = raw.strip()
        if item:
            out.append(Path(item))
    return out


def metric_paths(args: argparse.Namespace) -> List[Path]:
    dirs = split_dirs(args.input_dirs)
    if dirs:
        return [d / "overall_metrics.csv" for d in dirs]
    root = Path(args.root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--root_dir does not exist: {root}")
    out_dir = Path(args.out_dir).resolve()
    paths = []
    for path in sorted(root.rglob("overall_metrics.csv")):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == out_dir / "overall_metrics.csv" or out_dir in resolved.parents:
            continue
        paths.append(path)
    return paths


def read_table(paths: List[Path], name: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        table_path = path.parent / name
        if table_path.is_file():
            df = pd.read_csv(table_path)
            df["run_dir"] = str(path.parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def read_ifs_diagnostic_table(paths: List[Path]) -> pd.DataFrame:
    """Read IFS empirical VIS rows stored outside overall_metrics.csv."""
    rename = {
        "model_label": "source_label",
        "fog_th": "fog_threshold",
        "mist_th": "mist_threshold",
        "Fog_P": "fog_precision",
        "Fog_R": "fog_pod",
        "Fog_CSI": "fog_csi",
        "Fog_FAR": "fog_far",
        "Fog_support": "fog_support",
        "Mist_P": "mist_precision",
        "Mist_R": "mist_pod",
        "Mist_CSI": "mist_csi",
        "Mist_FAR": "mist_far",
        "Mist_support": "mist_support",
        "Clear_P": "clear_precision",
        "Clear_R": "clear_recall",
        "Clear_CSI": "clear_csi",
        "Clear_FAR": "clear_far",
        "Clear_support": "clear_support",
    }
    frames = []
    for path in paths:
        table_path = path.parent / "ifs_diagnostic_matched_metrics.csv"
        if not table_path.is_file():
            continue
        df = pd.read_csv(table_path)
        if "source" not in df:
            continue
        df = df[df["source"].astype(str).eq("ifs_diagnostic")].copy()
        if df.empty:
            continue
        df = df.rename(columns=rename)
        if "source_label" not in df or df["source_label"].isna().all() or (df["source_label"].astype(str).str.len() == 0).all():
            df["source_label"] = "IFS empirical VIS"
        else:
            df["source_label"] = df["source_label"].fillna("").replace("", "IFS empirical VIS")
        if "low_vis_fpr" not in df and "false_positive_rate" in df:
            df["low_vis_fpr"] = df["false_positive_rate"]
        df["run_dir"] = str(path.parent)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def dedupe_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source" not in df:
        return df
    preferred = [
        "tianji",
        "T2ND_rh2m_source_full",
        "T2ND_rh2m_common_core",
        "ifs",
        "pangu2021_source_full",
        "era5_2025_source_full",
        "era5_2025_common_core",
        "pangu2021_common_core",
        "ensemble_mean_softmax",
        "ifs_diagnostic",
    ]
    df = df.copy()
    df["_order"] = df["source"].astype(str).map({s: i for i, s in enumerate(preferred)}).fillna(len(preferred))
    df = df.sort_values(["_order", "source", "run_dir"], kind="stable")
    return df.drop_duplicates("source", keep="last").drop(columns=["_order"]).reset_index(drop=True)


def plot_key_metrics_figure(overall_df: pd.DataFrame, out_dir: Path) -> List[str]:
    """Redraw split key-metric figures without importing the torch evaluator."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting env dependent.
        print(f"[WARN] matplotlib unavailable; skip key-metrics figure: {exc}", flush=True)
        return []

    if overall_df.empty or "source" not in overall_df:
        return []

    available_sources = list(dict.fromkeys(overall_df["source"].astype(str).tolist()))
    available_set = set(available_sources)
    row_by_source = {str(row["source"]): row for _, row in overall_df.iterrows()}
    source_labels = {
        "tianji": "Tianji-trained",
        "ifs": "IFS-trained",
        "T2ND_rh2m_source_full": "T2ND RH2M source-full",
        "T2ND_rh2m_common_core": "T2ND RH2M",
        "pangu2021_source_full": "Pangu-2021 source-full",
        "pangu2021_common_core": "Pangu-2021",
        "era5_2025_source_full": "ERA5-2025 source-full",
        "era5_2025_common_core": "ERA5-2025",
        "ensemble_mean_softmax": "Tianji+IFS mean softmax",
        "ifs_diagnostic": "IFS diagnostic VIS",
    }
    for _, row in overall_df.iterrows():
        src = str(row.get("source", ""))
        label = str(row.get("source_label", "") or "").strip()
        if src and label and label != src:
            source_labels[src] = label
    source_colors = {
        "tianji": "#2E5A87",
        "ifs": "#6C6C6C",
        "T2ND_rh2m_source_full": "#1B9E77",
        "T2ND_rh2m_common_core": "#1B9E77",
        "pangu2021_source_full": "#8E6BBE",
        "pangu2021_common_core": "#8E6BBE",
        "era5_2025_source_full": "#D95F02",
        "era5_2025_common_core": "#D95F02",
        "ensemble_mean_softmax": "#4C78A8",
        "ifs_diagnostic": "#E69F00",
    }
    fallback_colors = ["#4C78A8", "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2", "#E15759"]

    panels = [
        (
            "Fog (0-500 m)",
            [
                ("fog_precision", "Precision"),
                ("fog_pod", "Recall"),
                ("fog_f1", "F1"),
                ("fog_csi", "CSI"),
            ],
        ),
        (
            "Mist (500-1000 m)",
            [
                ("mist_precision", "Precision"),
                ("mist_pod", "Recall"),
                ("mist_f1", "F1"),
                ("mist_csi", "CSI"),
            ],
        ),
        (
            "Low visibility (<1000 m)",
            [
                ("low_vis_precision", "Precision"),
                ("low_vis_recall", "Recall"),
                ("low_vis_f1", "F1"),
                ("low_vis_csi", "CSI"),
                ("low_vis_fpr", "FPR"),
            ],
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    panel_letters = ["a", "b", "c"]

    def _adaptive_score_ylim(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.20
        vmax = float(np.nanmax(arr))
        if vmax <= 0:
            return 0.10
        padded = min(1.0, vmax + max(0.025, 0.10 * vmax))
        step = 0.02 if padded <= 0.20 else 0.05
        upper = max(step * 3, math.ceil(padded / step) * step)
        if vmax < 0.80:
            upper = min(upper, 0.85)
        elif vmax < 0.90:
            upper = min(upper, 0.95)
        return min(1.0, upper)

    def _draw_group(source_order: List[str], stem: str) -> List[str]:
        source_order = [src for src in source_order if src in available_set and src in row_by_source]
        if not source_order:
            return []
        n_sources = len(source_order)
        fig_w = max(11.2, 10.2 + 0.74 * max(0, n_sources - 2))
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4.45), sharey=False, constrained_layout=False)
        fig.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.79, wspace=0.24)

        for ax_idx, (ax, (title, metrics)) in enumerate(zip(axes, panels)):
            x = np.arange(len(metrics), dtype=np.float64)
            width = min(0.34, 0.78 / max(n_sources, 1))
            panel_values: List[float] = []
            for source in source_order:
                row = row_by_source[source]
                for metric, _ in metrics:
                    try:
                        panel_values.append(float(row.get(metric, np.nan)))
                    except Exception:
                        panel_values.append(math.nan)
            y_max = _adaptive_score_ylim(panel_values)
            for src_idx, source in enumerate(source_order):
                row = row_by_source[source]
                vals: List[float] = []
                finite_flags: List[bool] = []
                for metric, _ in metrics:
                    val = row.get(metric, np.nan)
                    try:
                        val = float(val)
                    except Exception:
                        val = math.nan
                    finite_flags.append(bool(np.isfinite(val)))
                    vals.append(val if np.isfinite(val) else 0.0)

                offset = (src_idx - (n_sources - 1) / 2.0) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width * 0.92,
                    label=source_labels.get(source, source) if ax_idx == 0 else None,
                    color=source_colors.get(source, fallback_colors[src_idx % len(fallback_colors)]),
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.96,
                )
                for bar, val, ok in zip(bars, vals, finite_flags):
                    if ok:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            min(val + y_max * 0.025, y_max * 0.98),
                            f"{val:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=90,
                        )

            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", alpha=0.28)
            ax.grid(axis="x", visible=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.text(
                -0.13,
                1.04,
                f"({panel_letters[ax_idx]})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
            )
            if ax_idx == 0:
                ax.set_ylabel("Score")
            if title.startswith("Low visibility"):
                ax.text(
                    0.98,
                    0.96,
                    "FPR lower is better",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    color="#444444",
                )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(len(handles), 5),
                frameon=False,
            )

        out_paths = [
            out_dir / f"{stem}.png",
            out_dir / f"{stem}.pdf",
            out_dir / f"{stem}.svg",
        ]
        for path in out_paths:
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
            print(f"  [Fig] Saved -> {path}", flush=True)
        plt.close(fig)
        return [str(p) for p in out_paths]

    written: List[str] = []
    has_source_full = bool(
        available_set
        & {"T2ND_rh2m_source_full", "pangu2021_source_full", "era5_2025_source_full"}
    ) or any("source_full" in str(row.get("data_dir", "")) for row in row_by_source.values())
    if has_source_full:
        written.extend(
            _draw_group(
                [
                    "tianji",
                    "T2ND_rh2m_source_full",
                    "ifs",
                    "pangu2021_source_full",
                    "era5_2025_source_full",
                ],
                "fig_forecast_source_key_metrics_source_full",
            )
        )
    if available_set & {"pangu2021_source_full", "ensemble_mean_softmax"}:
        written.extend(
            _draw_group(
                ["pangu2021_source_full", "ifs_diagnostic", "ensemble_mean_softmax"],
                "fig_forecast_source_key_metrics_pangu_ifs_ensemble",
            )
        )
    written.extend(
        _draw_group(
            ["tianji", "T2ND_rh2m_common_core", "ifs", "era5_2025_common_core", "ifs_diagnostic"],
            "fig_forecast_source_key_metrics_numerical_models",
        )
    )
    written.extend(
        _draw_group(
            ["tianji", "pangu2021_common_core"],
            "fig_forecast_source_key_metrics_tianji_pangu",
        )
    )
    if not written:
        fallback_order = [
            "tianji",
            "T2ND_rh2m_common_core",
            "ifs",
            "era5_2025_common_core",
            "pangu2021_common_core",
            "ifs_diagnostic",
        ]
        fallback_order.extend([s for s in available_sources if s not in set(fallback_order)])
        written.extend(_draw_group(fallback_order, "fig_forecast_source_key_metrics"))
    return written


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in metric_paths(args) if p.is_file()]
    if not paths:
        raise FileNotFoundError("No overall_metrics.csv files were found.")

    overall_raw = read_table(paths, "overall_metrics.csv")
    ifs_diagnostic = read_ifs_diagnostic_table(paths)
    if not ifs_diagnostic.empty:
        overall_raw = pd.concat([overall_raw, ifs_diagnostic], ignore_index=True, sort=False)
    overall = dedupe_sources(overall_raw)
    validation = dedupe_sources(read_table(paths, "validation_metrics.csv"))
    overall.to_csv(out_dir / "overall_metrics.csv", index=False)
    if not validation.empty:
        validation.to_csv(out_dir / "validation_metrics.csv", index=False)
    if not args.no_figures:
        plot_key_metrics_figure(overall, out_dir)

    keep = [
        c
        for c in (
            "source",
            "source_label",
            "n",
            "fog_pod",
            "fog_csi",
            "mist_pod",
            "mist_csi",
            "low_vis_recall",
            "low_vis_csi",
            "low_vis_precision",
            "low_vis_fpr",
            "run_dir",
        )
        if c in overall.columns
    ]
    print(f"[OK] merged {len(paths)} metric file(s) into {out_dir}", flush=True)
    print(overall[keep].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
