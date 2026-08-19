#!/usr/bin/env python3
"""Restructured Fig. 2 and Fig. 3 manuscript composites (v2).

Fig. 2 changes:
  * a-c operator panels stay as-is;
  * d-g are shrunk into a single four-panel row;
  * per-panel titles are removed;
  * temporal x tick labels drop the month span (T1..T5);
  * the bottom label becomes Temporal/Spatial CV CSI or recall.

Fig. 3 changes:
  * a+b merge into one panel with AP and recall groups, Pangu/Tianji colors;
  * c moves to the supplement (not drawn here);
  * row 2 keeps only d and f, without background shading; a dashed line
    separates the station-observation (surface) and ERA5 (pressure) rows;
  * row 3 keeps only g;
  * no top titles anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import plot_common_variable_error_regimes as quality_plot
import plot_q_core_task_tail_fidelity_preview as tail_plot
import plot_qcore_t925_manuscript_composite as composite
import plot_viscast_controlled_attribution_composite as controlled
from paper_source_palette import SOURCE_COLORS, SOURCE_DARK_COLORS


PANGU = SOURCE_COLORS["pangu"]
TIANJI = SOURCE_COLORS["tianji"]
PANGU_DARK = SOURCE_DARK_COLORS["pangu"]
TIANJI_DARK = SOURCE_DARK_COLORS["tianji"]
INK = "#17191B"

# --------------------------------------------------------------------------
# Fig. 2 geometry
# --------------------------------------------------------------------------
FIG2_SIZE = (7.80, 5)
FIG2_GRID = dict(
    height_ratios=[0.85, 0.85],
    left=0.105,
    right=0.985,
    top=0.920,
    bottom=0.085,
    hspace=0.4,
)
FIG2_OPERATOR_WSPACE = 0.42
FIG2_CV_WSPACE = 0.12
FIG2_OPERATOR_WIDTH_RATIOS = [1.0, 1.0, 1.0]
FIG2_CV_WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0]

# --------------------------------------------------------------------------
# Fig. 3 geometry
# --------------------------------------------------------------------------
FIG3_SIZE = (6.15, 5.46)
FIG3_GRID = dict(
    height_ratios=[0.62, 1.34, 0.74],
    left=0.175,
    right=0.985,
    top=0.955,
    bottom=0.065,
    hspace=0.35,
)
FIG3_ROW1_WIDTH_RATIOS = [1.15, 1.0]
FIG3_ROW1_WSPACE = 0.2
FIG3_ROW2_WSPACE = 0.12
FIG3_ROW2_WIDTH_RATIOS = [1.0, 1.15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["both", "fig2", "fig3"], default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf,svg,tiff")

    parser.add_argument("--mapping-cv-dir", type=Path, default=Path("/public/home/putianshu/vis_mlp/train"))
    parser.add_argument("--fig2-ifs-comparison-dir", type=Path)
    parser.add_argument("--fig2-spatial-result-root", type=Path)
    parser.add_argument("--fig2-temporal-result-root", type=Path)
    parser.add_argument("--fig2-gru-label", default="VisCast")
    parser.add_argument("--fig2-stem", default="fig2_restructured_v2")

    parser.add_argument("--fig3-analysis-dir", type=Path)
    parser.add_argument("--fig3-endpoint-dir", type=Path)
    parser.add_argument("--fig3-tail-analysis-dir", type=Path)
    parser.add_argument("--fig3-paired-quality-dir", type=Path)
    parser.add_argument("--fig3-stem", default="fig3_restructured_v2")
    return parser.parse_args()


def export(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    for raw in formats:
        fmt = raw.strip().lower()
        if not fmt:
            continue
        if fmt not in {"svg", "pdf", "png", "tiff"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        path = output_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    if not outputs:
        raise ValueError("At least one output format is required")
    return outputs


# --------------------------------------------------------------------------
# Fig. 2
# --------------------------------------------------------------------------
def load_mapping_cv(
    mapping_cv_dir: Path,
    spatial_root: Path,
    temporal_root: Path,
    gru_label: str,
):
    mapping_dir = mapping_cv_dir.expanduser().resolve()
    if str(mapping_dir) not in sys.path:
        sys.path.insert(0, str(mapping_dir))
    import plot_mapping_cv_folds as mapping_plot

    spatial_folds, spatial_pooled, spatial_manifest, spatial_coverage = mapping_plot._load_cv_result(
        spatial_root.expanduser().resolve(), "spatial"
    )
    temporal_folds, temporal_pooled, temporal_manifest, temporal_coverage = mapping_plot._load_cv_result(
        temporal_root.expanduser().resolve(), "temporal"
    )
    spatial_models = tuple(
        model
        for model in mapping_plot.MODEL_ORDER
        if model in set(spatial_folds["model"].astype(str))
    )
    temporal_models = tuple(
        model
        for model in mapping_plot.MODEL_ORDER
        if model in set(temporal_folds["model"].astype(str))
    )
    if spatial_models != temporal_models:
        raise ValueError(
            "Spatial and temporal metric tables contain different models: "
            f"spatial={spatial_models}, temporal={temporal_models}"
        )
    if set(mapping_plot.DIRECT_MODEL_ORDER).issubset(set(spatial_models)):
        model_order = mapping_plot.DIRECT_MODEL_ORDER
        model_labels = dict(mapping_plot.DIRECT_MODEL_LABELS)
        model_labels["gru"] = str(gru_label)
    else:
        raise ValueError(f"Unsupported mapping-CV model set: {spatial_models}")

    n_temporal = int(temporal_manifest["n_folds"])
    n_spatial = int(spatial_manifest["n_folds"])
    temporal_labels = [f"T{fold + 1}" for fold in range(n_temporal)]
    spatial_labels = [f"S{fold + 1}" for fold in range(n_spatial)]
    return (
        mapping_plot,
        spatial_folds,
        temporal_folds,
        spatial_pooled,
        temporal_pooled,
        spatial_manifest,
        temporal_manifest,
        spatial_coverage,
        temporal_coverage,
        model_order,
        model_labels,
        temporal_labels,
        spatial_labels,
    )


def draw_fig2(
    fig: plt.Figure,
    mapping_plot,
    matched: pd.DataFrame,
    spatial_folds: pd.DataFrame,
    temporal_folds: pd.DataFrame,
    temporal_labels: Sequence[str],
    spatial_labels: Sequence[str],
    model_order: Sequence[str],
    model_labels: Dict[str, str],
) -> pd.DataFrame:
    outer = fig.add_gridspec(2, 1, **FIG2_GRID)
    operator_grid = outer[0].subgridspec(
        1, 3, width_ratios=FIG2_OPERATOR_WIDTH_RATIOS, wspace=FIG2_OPERATOR_WSPACE
    )
    cv_grid = outer[1].subgridspec(
        1, 4, width_ratios=FIG2_CV_WIDTH_RATIOS, wspace=FIG2_CV_WSPACE
    )
    operator_axes = [fig.add_subplot(operator_grid[0, index]) for index in range(3)]
    cv_axes = [fig.add_subplot(cv_grid[0, 0])]
    cv_axes.extend(
        fig.add_subplot(cv_grid[0, index], sharey=cv_axes[0])
        for index in range(1, 4)
    )

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
            [
                ("low_vis_csi", "CSI"),
                ("low_vis_recall", "Recall"),
                ("low_vis_precision", "Precision"),
            ],
        ),
    ]
    source_frames = []
    for index, (axis, (title, specs)) in enumerate(zip(operator_axes, operator_panels)):
        source_frames.append(
            controlled.draw_operator_panel(
                axis,
                matched,
                None,
                specs,
                show_ylabel=(index == 0),
                show_legend=(index == 0),
                bottom_title=f"{title} skill",
            )
        )
    for letter, axis in zip("abc", operator_axes):
        controlled.panel_label(axis, letter, x=-0.24)

    panel_specs = (
        (cv_axes[0], temporal_folds, "low_vis_csi", temporal_labels, "Temporal CV CSI", "Score", "d"),
        (cv_axes[1], temporal_folds, "low_vis_recall", temporal_labels, "Temporal CV recall", "", "e"),
        (cv_axes[2], spatial_folds, "low_vis_csi", spatial_labels, "Spatial CV CSI", "", "f"),
        (cv_axes[3], spatial_folds, "low_vis_recall", spatial_labels, "Spatial CV recall", "", "g"),
    )
    for axis, table, metric, tick_labels, xlabel, ylabel, letter in panel_specs:
        mapping_plot._panel(
            axis,
            table,
            metric,
            tick_labels,
            title="",
            xlabel=xlabel,
            ylabel=ylabel,
            letter=letter,
            model_order=model_order,
            model_labels=model_labels,
        )
    for axis in cv_axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    cv_axes[0].legend(
        loc="upper left",
        handlelength=1.8,
        labelspacing=0.35,
    )
    return pd.concat(source_frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------
# Fig. 3
# --------------------------------------------------------------------------
def draw_endpoint_ap_recall_panel(ax, metrics: pd.DataFrame) -> pd.DataFrame:
    """One AP + recall panel in the style of Fig. 2 panel a."""

    specs = [
        ("low_vis_ap", "AP"),
        ("low_vis_recall_matched_fpr", "Recall"),
    ]
    x = np.arange(len(specs), dtype=float)
    width = 0.34
    rows = []
    maxima = 0.0
    for xi, (key, label) in enumerate(specs):
        pangu, tianji = controlled.endpoint_distributions(metrics, key)
        pangu_mean, tianji_mean = float(np.mean(pangu)), float(np.mean(tianji))
        maxima = max(maxima, pangu_mean, tianji_mean)
        ax.bar(
            xi - width / 2,
            pangu_mean,
            width,
            color=PANGU,
            edgecolor=PANGU_DARK,
            linewidth=0.7,
            label="Pangu" if xi == 0 else None,
            yerr=[[pangu_mean - float(np.min(pangu))], [float(np.max(pangu)) - pangu_mean]],
            error_kw=dict(elinewidth=1.0, ecolor=PANGU_DARK, capsize=0),
        )
        ax.bar(
            xi + width / 2,
            tianji_mean,
            width,
            color=TIANJI,
            edgecolor=TIANJI_DARK,
            linewidth=0.7,
            label="Tianji" if xi == 0 else None,
            yerr=[[tianji_mean - float(np.min(tianji))], [float(np.max(tianji)) - tianji_mean]],
            error_kw=dict(elinewidth=1.0, ecolor=TIANJI_DARK, capsize=0),
        )
        rows.append({"metric": key, "label": label, "source": "Pangu", "value": pangu_mean})
        rows.append({"metric": key, "label": label, "source": "Tianji", "value": tianji_mean})
    ax.set_xticks(x, [label for _, label in specs])
    ax.set_xlim(-0.55, len(specs) - 0.45)
    top = min(1.0, max(0.40, maxima * 1.22))
    ax.set_ylim(0.25, top)
    y_ticks = [0.25]
    next_tick = 0.40
    while next_tick <= top:
        y_ticks.append(next_tick)
        next_tick += 0.15
    ax.set_yticks(y_ticks)
    ax.set_ylabel("Score")
    ax.set_xlabel("Skill")
    ax.legend(loc="upper left", frameon=False)
    composite.style_axis(ax)
    return pd.DataFrame(rows)


def draw_fig3(
    fig: plt.Figure,
    metrics: pd.DataFrame,
    gap: pd.DataFrame,
    joint_ci: pd.DataFrame,
    joint_metrics: pd.DataFrame,
    placement: pd.DataFrame,
    quality_source: pd.DataFrame,
) -> pd.DataFrame:
    outer = fig.add_gridspec(3, 1, **FIG3_GRID)
    row1 = outer[0].subgridspec(
        1, 2, width_ratios=FIG3_ROW1_WIDTH_RATIOS, wspace=FIG3_ROW1_WSPACE
    )
    a_ax = fig.add_subplot(row1[0, 0])
    source_frames = [draw_endpoint_ap_recall_panel(a_ax, metrics)]
    composite.panel_label(a_ax, "a")
    b_ax = fig.add_subplot(row1[0, 1])
    source_frames.append(
        controlled.draw_delta_panel(b_ax, metrics, gap, title=None).assign(
            panel_metric="bootstrap_differences"
        )
    )
    b_ax.set_yticklabels(["AP", "Recall"])
    composite.panel_label(b_ax, "b", dx=0.14, dy_frac=0.04)

    row2 = outer[1].subgridspec(
        1, 2, width_ratios=FIG3_ROW2_WIDTH_RATIOS, wspace=FIG3_ROW2_WSPACE
    )
    d_ax = fig.add_subplot(row2[0, 0])
    f_ax = fig.add_subplot(row2[0, 1])

    quality_plot.rmse_ratio_panel(
        d_ax,
        quality_source,
        quality_plot.SCOPES[0],
        "d",
        True,
        show_reference_labels=False,
        show_label=False,
        show_direction_labels=False,
        title_text="",
        show_bands=False,
    )
    d_ax.set_xlabel("Tianji / Pangu RMSE")
    d_ax.set_yticklabels(
        ["T2m", "WS10m", "SLP", "T925", "Q1000", "Q925", "UV925"]
    )
    d_ax.axhline(2.5, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    composite.panel_label(d_ax, "c")
    source_frames.append(
        quality_source[quality_source["scope"] == quality_plot.SCOPES[0]].assign(
            panel_metric="rmse_all_paired"
        )
    )

    tail_plot.draw_tail_placement_panel(
        f_ax,
        placement,
        show_tail_direction=False,
        show_direction_labels=False,
        show_y=False,
        show_values=False,
        shading="none",
        xgrid=False,
        title=None,
        xlabel="Δ task-tail CSI (Tianji − Pangu)",
    )
    f_ax.spines["left"].set_visible(True)
    composite.panel_label(f_ax, "d", dx=0.14, dy_frac=0.04)
    source_frames.append(placement.assign(panel_metric="task_tail_placement"))

    g_ax = fig.add_subplot(outer[2, 0])
    source_frames.append(
        composite.draw_joint_scope(
            g_ax,
            joint_ci,
            joint_metrics,
            "all_paired",
            "",
            True,
        ).assign(panel_metric="joint_tail_all_paired")
    )
    composite.panel_label(g_ax, "e")
    return pd.concat(source_frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_fig3_tables(args: argparse.Namespace):
    analysis_dir = Path(args.fig3_analysis_dir).expanduser().resolve()
    endpoint_dir = Path(args.fig3_endpoint_dir).expanduser().resolve()
    tail_dir = (
        Path(args.fig3_tail_analysis_dir).expanduser().resolve()
        if args.fig3_tail_analysis_dir
        else analysis_dir
    )
    paired_dir = composite.resolve_paired_quality_dir(
        analysis_dir,
        endpoint_dir,
        Path(args.fig3_paired_quality_dir) if args.fig3_paired_quality_dir else None,
    )
    endpoint_tables = composite.load_inputs(endpoint_dir)
    joint_ci, joint_metrics = composite.load_tail_inputs(tail_dir)
    quality_source = quality_plot.prepare_source(paired_dir)
    placement_metrics = tail_plot.load_table(
        tail_dir,
        "reference_tail_placement_metrics.csv",
        ["feature", "tail", "task_relevant_tail"],
    )
    definitions = tail_plot.load_table(
        tail_dir,
        "joint_tail_feature_definitions.csv",
        ["feature", "task_tail_direction"],
    )
    placement_ci = tail_plot.load_table(
        tail_dir,
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
    relevant = placement_metrics[
        placement_metrics["task_relevant_tail"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    ].copy()
    directions = {
        str(row.feature): str(row.task_tail_direction)
        for row in definitions.itertuples(index=False)
    }
    for feature in tail_plot.FEATURE_ORDER:
        if feature in directions:
            continue
        tails = sorted(
            set(
                relevant.loc[
                    relevant["feature"].astype(str) == feature,
                    "tail",
                ].astype(str)
            )
        )
        if len(tails) != 1:
            raise ValueError(
                f"{feature}: expected one task-relevant tail in reference_tail_placement_metrics.csv, "
                f"found {tails}"
            )
        directions[feature] = tails[0]
    placement = tail_plot.select_placement_rows(placement_ci, directions)
    return endpoint_tables, joint_ci, joint_metrics, quality_source, placement, directions


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    composite.setup_style()

    if args.which in {"both", "fig2"}:
        if not (
            args.fig2_ifs_comparison_dir
            and args.fig2_spatial_result_root
            and args.fig2_temporal_result_root
        ):
            raise SystemExit(
                "fig2 requires --fig2-ifs-comparison-dir, --fig2-spatial-result-root, "
                "--fig2-temporal-result-root"
            )
        ifs_dir = args.fig2_ifs_comparison_dir.expanduser().resolve()
        matched_path = ifs_dir / "ifs_diagnostic_matched_metrics.csv"
        if not matched_path.is_file():
            raise FileNotFoundError(matched_path)
        matched = pd.read_csv(matched_path)
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
        controlled.require_columns(matched, ["source", *operator_metrics], matched_path)

        (
            mapping_plot,
            spatial_folds,
            temporal_folds,
            _spatial_pooled,
            _temporal_pooled,
            spatial_manifest,
            temporal_manifest,
            spatial_coverage,
            temporal_coverage,
            model_order,
            model_labels,
            temporal_labels,
            spatial_labels,
        ) = load_mapping_cv(
            args.mapping_cv_dir,
            args.fig2_spatial_result_root,
            args.fig2_temporal_result_root,
            args.fig2_gru_label,
        )
        controlled.setup_style()
        fig2 = plt.figure(figsize=FIG2_SIZE)
        fig2_source = draw_fig2(
            fig2,
            mapping_plot,
            matched,
            spatial_folds,
            temporal_folds,
            temporal_labels,
            spatial_labels,
            model_order,
            model_labels,
        )
        fig2_outputs = export(fig2, output_dir, args.fig2_stem, formats, args.dpi)
        plt.close(fig2)
        fig2_source.to_csv(
            output_dir / f"{args.fig2_stem}_source_data.csv",
            index=False,
            float_format="%.8f",
        )
        (output_dir / f"{args.fig2_stem}_manifest.json").write_text(
            json.dumps(
                {
                    "panels": {
                        "a-c": "IFS diagnostic visibility versus IFS-driven VisCast",
                        "d-g": "temporal and spatial blocked CV, one row",
                    },
                    "inputs": {
                        "ifs_comparison_dir": str(ifs_dir),
                        "spatial_result_root": str(args.fig2_spatial_result_root.expanduser().resolve()),
                        "temporal_result_root": str(args.fig2_temporal_result_root.expanduser().resolve()),
                    },
                    "outputs": fig2_outputs,
                    "spatial_coverage": spatial_coverage,
                    "temporal_coverage": temporal_coverage,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.which in {"both", "fig3"}:
        if not (args.fig3_analysis_dir and args.fig3_endpoint_dir):
            raise SystemExit("fig3 requires --fig3-analysis-dir and --fig3-endpoint-dir")
        endpoint_tables, joint_ci, joint_metrics, quality_source, placement, directions = load_fig3_tables(
            args
        )
        fig3 = plt.figure(figsize=FIG3_SIZE)
        fig3_source = draw_fig3(
            fig3,
            endpoint_tables["metrics"],
            endpoint_tables["gap"],
            joint_ci,
            joint_metrics,
            placement,
            quality_source,
        )
        fig3_outputs = export(fig3, output_dir, args.fig3_stem, formats, args.dpi)
        plt.close(fig3)
        fig3_source.to_csv(
            output_dir / f"{args.fig3_stem}_source_data.csv",
            index=False,
            float_format="%.8f",
        )
        (output_dir / f"{args.fig3_stem}_manifest.json").write_text(
            json.dumps(
                {
                    "panels": {
                        "a": "AP and recall, Pangu versus Tianji",
                        "b": "bootstrap differences",
                        "c": "paired RMSE ratio, all samples",
                        "d": "task-tail placement",
                        "e": "joint-tail recovery, all paired samples",
                        "removed": {"f": "supplement (old c)", "g": "RMSE low-vis (old e)", "h": "joint low-vis"},
                    },
                    "task_tail_directions": directions,
                    "outputs": fig3_outputs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[fig2_fig3_v2] outputs in {output_dir}")


if __name__ == "__main__":
    main()
