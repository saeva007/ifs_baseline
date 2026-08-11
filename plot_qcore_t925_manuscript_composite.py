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
import plot_common_variable_error_regimes as quality_plot


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


def draw_endpoint(ax, metrics: pd.DataFrame, gap: pd.DataFrame, metric: str, title: str, ylabel: str) -> pd.DataFrame:
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
        width=0.56,
        color=[PANGU, TIANJI],
        edgecolor=[PANGU_DARK, TIANJI_DARK],
        linewidth=0.9,
        alpha=0.92,
        zorder=2,
    )
    for position, mean, low, high, color in zip(
        positions, means, lows, highs, (PANGU_DARK, TIANJI_DARK)
    ):
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            ecolor=color,
            elinewidth=1.15,
            capsize=6.0,
            capthick=1.0,
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
            markersize=5.3,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=1.35,
            capsize=3.0,
            capthick=1.0,
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
    bars = ax.barh(y, values, color=[item[2] for item in mapping], height=0.42)
    for bar, value, (_, _, color) in zip(bars, values, mapping):
        ax.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", ha="left", color=color, fontweight="bold")
    ax.set_yticks(y, [item[1] for item in mapping])
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Exclusive hits")
    ax.set_title("Source-exclusive Low-vis hits", loc="left", fontweight="bold", pad=6)
    style_axis(ax, xgrid=True, ygrid=False)
    return pd.DataFrame({"source": [item[1] for item in mapping], "exclusive_hits": values})


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
            markersize=4.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            ecolor=color,
            elinewidth=1.15,
            capsize=2.5,
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
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else analysis_dir / "manuscript_figures"
    tables = load_inputs(endpoint_dir)
    quality_source = quality_plot.prepare_source(paired_quality_dir)

    variants = (
        ("offset_ci", "offset_ci"),
        ("paired_connector", "paired_connector"),
    )
    rendered = []
    for suffix, bias_layout in variants:
        fig = plt.figure(figsize=(FIGURE_WIDTH, 8.75))
        outer = fig.add_gridspec(
            3,
            1,
            height_ratios=[0.92, 1.22, 1.22],
            left=0.205,
            right=0.985,
            top=0.93,
            bottom=0.065,
            hspace=0.66,
        )
        endpoint_grid = outer[0].subgridspec(1, 2, wspace=0.40)
        quality_all_grid = outer[1].subgridspec(1, 2, wspace=0.19)
        quality_low_grid = outer[2].subgridspec(1, 2, wspace=0.19)
        endpoint_axes = [
            fig.add_subplot(endpoint_grid[0, index]) for index in range(2)
        ]
        quality_all_axes = [
            fig.add_subplot(quality_all_grid[0, index]) for index in range(2)
        ]
        quality_low_axes = [
            fig.add_subplot(quality_low_grid[0, index]) for index in range(2)
        ]

        source_frames = [
            draw_endpoint(
                endpoint_axes[0],
                tables["metrics"],
                tables["gap"],
                "low_vis_ap",
                "Low-vis average precision",
                "Average precision",
            ).assign(panel_metric="low_vis_ap"),
            draw_endpoint(
                endpoint_axes[1],
                tables["metrics"],
                tables["gap"],
                "low_vis_recall_matched_fpr",
                "Recall at matched FPR",
                "Low-vis recall",
            ).assign(panel_metric="low_vis_recall_matched_fpr"),
        ]
        for letter, ax in zip("ab", endpoint_axes):
            panel_label(ax, letter)

        for axes, scope, letters in (
            (quality_all_axes, quality_plot.SCOPES[0], ("c", "d")),
            (quality_low_axes, quality_plot.SCOPES[1], ("e", "f")),
        ):
            quality_plot.rmse_ratio_panel(
                axes[0],
                quality_source,
                scope,
                letters[0],
                True,
            )
            quality_plot.bias_panel(
                axes[1],
                quality_source,
                scope,
                letters[1],
                False,
                show_reference_labels=False,
                layout=bias_layout,
            )
            source_frames.extend(
                [
                    quality_source[quality_source["scope"] == scope].assign(
                        panel_metric=f"rmse_{scope}",
                        bias_layout=bias_layout,
                    ),
                    quality_source[quality_source["scope"] == scope].assign(
                        panel_metric=f"bias_{scope}",
                        bias_layout=bias_layout,
                    ),
                ]
            )

        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
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
            bbox_to_anchor=(0.60, 0.992),
            ncol=2,
            handletextpad=0.4,
            columnspacing=1.2,
        )

        stem = f"{args.figure_stem}_{suffix}"
        export(fig, out_dir, stem, args.dpi)
        plt.close(fig)
        pd.concat(source_frames, ignore_index=True, sort=False).to_csv(
            out_dir / f"{stem}_source_data.csv",
            index=False,
            float_format="%.8f",
        )
        rendered.append(str(out_dir / f"{stem}.pdf"))

    manifest = {
        "analysis_dir": str(analysis_dir),
        "endpoint_dir": str(endpoint_dir),
        "paired_quality_dir": str(paired_quality_dir),
        "figures": rendered,
        "bias_layouts": [item[1] for item in variants],
        "rendering": "all panels redrawn from source tables on one canvas",
    }
    (out_dir / f"{args.figure_stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for figure in rendered:
        print(figure)


if __name__ == "__main__":
    main()
