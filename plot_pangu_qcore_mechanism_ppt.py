#!/usr/bin/env python3
"""Draw the complete Pangu--Tianji q-core evidence story as standalone figures.

The plotting entrypoint is CPU-only and intentionally independent of Torch.  It
combines the passed three-seed ``mt2pw`` factorial analysis with the separate
paired source-quality diagnosis.  Every exported figure carries one scientific
claim and is written as editable SVG/PDF plus high-resolution PNG/TIFF.

The default surface-quality scope is the model-consistent definition
``visibility < 1000 m``.  Old quality reports that used ``<= 1000 m`` are
rejected for formal low-visibility figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_source_palette import (
    SOURCE_COLORS as PAPER_SOURCE_COLORS,
    SOURCE_DARK_COLORS,
    SOURCE_LIGHT_COLORS,
    SOURCE_MARKERS,
    SOURCE_PALE_COLORS,
)


# Nature Communications column widths: 89 mm for compact two-endpoint figures
# and 183 mm for denser comparisons. Height follows information density so
# sparse figures do not inherit the empty space required by larger plots.
FIGURE_WIDTH = 7.205
SINGLE_COLUMN_WIDTH = 3.504
FIGURE_SIZES = {
    "flow": (FIGURE_WIDTH, 2.65),
    "endpoint": (SINGLE_COLUMN_WIDTH, 3.10),
    "argmax": (SINGLE_COLUMN_WIDTH, 3.05),
    "shapley": (FIGURE_WIDTH, 3.15),
    "surface": (FIGURE_WIDTH, 2.75),
    "pressure": (FIGURE_WIDTH, 2.90),
    "events": (FIGURE_WIDTH, 2.25),
    "event_quality": (FIGURE_WIDTH, 2.95),
    "qc": (FIGURE_WIDTH, 2.65),
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# Paper-wide source identity.  Tianji is dark blue, Pangu is a visibly lighter
# violet, and baseline/ERA5 is grey.  Do not reuse these colors for physical
# variable packages or workflow stages.
PANGU = PAPER_SOURCE_COLORS["pangu"]
PANGU_DARK = SOURCE_DARK_COLORS["pangu"]
PANGU_LIGHT = SOURCE_LIGHT_COLORS["pangu"]
PANGU_PALE = SOURCE_PALE_COLORS["pangu"]
TIANJI = PAPER_SOURCE_COLORS["tianji"]
TIANJI_DARK = SOURCE_DARK_COLORS["tianji"]
TIANJI_LIGHT = SOURCE_LIGHT_COLORS["tianji"]
TIANJI_PALE = SOURCE_PALE_COLORS["tianji"]
ERA5 = PAPER_SOURCE_COLORS["era5_reference_analysis"]
ERA5_DARK = SOURCE_DARK_COLORS["era5_reference_analysis"]
ERA5_LIGHT = SOURCE_LIGHT_COLORS["era5_reference_analysis"]
ERA5_PALE = SOURCE_PALE_COLORS["era5_reference_analysis"]
INK = "#17191B"
LIGHT_GREY = "#D9DCDE"
PALE_GREY = "#F7F8F8"
GRID_GREY = "#E8EAEB"

PACKAGE_COLORS = {
    "T2": "#C66A3D",
    "M": "#4D8C57",
    "W": "#C59A32",
    "P": "#70757A",
}
SOURCE_COLORS = {
    "pangu": PANGU,
    "tianji": TIANJI,
    "era5_reference_analysis": ERA5,
}
SOURCE_LABELS = {
    "pangu": "Pangu",
    "tianji": "Tianji",
    "era5_reference_analysis": "ERA5",
}

DEFAULT_EVAL_ROOT = Path(
    "/public/home/putianshu/vis_mlp/"
    "paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/"
    "qcore_hybrid_mt2pw_formal_v1_20260708"
)
DEFAULT_QUALITY_DIR = Path(
    "/public/home/putianshu/vis_mlp/"
    "paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/"
    "qcore_paired_quality_strictlt_v1_20260716/analysis"
)

FIGURE_SPECS: Mapping[str, Mapping[str, str]] = {
    "00_qcore_full_experiment_flow": {
        "placement": "main",
        "claim": "The mechanism claim follows a controlled chain from fair performance to source attribution and independent quality checks.",
    },
    "01_qcore_lowvis_ap": {
        "placement": "main",
        "claim": "Tianji-trained q-core models have higher threshold-free Low-vis AP than Pangu-trained models.",
    },
    "02_qcore_matched_fpr_recall": {
        "placement": "main",
        "claim": "At validation-matched false-alarm rates, Tianji-trained models recover more low-visibility cases.",
    },
    "02a_qcore_argmax_lowvis_overview": {
        "placement": "supplement_or_ppt",
        "claim": "Under argmax, Tianji gains Low-vis recall and CSI with a slightly higher false-positive rate than Pangu.",
    },
    "03_source_block_shapley": {
        "placement": "main",
        "claim": "T2M and moisture dominate the controlled source contribution, wind is smaller, and MSLP is near zero.",
    },
    "04_surface_t2m_quality": {
        "placement": "main",
        "claim": "Tianji 2-m temperature is closer to station observations overall and during observed low visibility.",
    },
    "05_surface_wspd10_quality": {
        "placement": "main",
        "claim": "Tianji 10-m wind speed is closer to station observations overall and during observed low visibility.",
    },
    "06_surface_mslp_quality": {
        "placement": "supplement",
        "claim": "The overall MSLP difference is non-robust, whereas the low-visibility subset differs; its Shapley contribution remains near zero.",
    },
    "07_pressure_level_quality": {
        "placement": "main",
        "claim": "Pressure-level quality is mixed: Pangu is closer for Q1000 while Tianji is closer for 925-hPa vector wind.",
    },
    "08_unique_event_hits": {
        "placement": "main_or_supplement",
        "claim": "Tianji uniquely detects more low-visibility samples than Pangu at validation-matched operating points.",
    },
    "09_observation_anchored_tianji_only_advantage": {
        "placement": "main",
        "claim": "Within Tianji-only Low-vis hits, Tianji has lower observation-referenced T2M and WSPD10 RMSE, whereas MSLP is non-robust.",
    },
    "10_pressure_qc": {
        "placement": "supplement",
        "claim": "Pangu contains a small but explicit fraction of non-physical negative pressure-level specific humidity values.",
    },
}

FIGURE_SIZE_KEYS = {
    "00_qcore_full_experiment_flow": "flow",
    "01_qcore_lowvis_ap": "endpoint",
    "02_qcore_matched_fpr_recall": "endpoint",
    "02a_qcore_argmax_lowvis_overview": "argmax",
    "03_source_block_shapley": "shapley",
    "04_surface_t2m_quality": "surface",
    "05_surface_wspd10_quality": "surface",
    "06_surface_mslp_quality": "surface",
    "07_pressure_level_quality": "pressure",
    "08_unique_event_hits": "events",
    "09_observation_anchored_tianji_only_advantage": "event_quality",
    "10_pressure_qc": "qc",
}

EVENT_OBSERVATION_COLUMNS = [
    "time_utc",
    "station_key",
    "case_category",
    "vis_raw_m",
    "T2M_pangu",
    "T2M_tianji",
    "tem",
    "WSPD10_pangu",
    "WSPD10_tianji",
    "win_s_avg_10mi",
    "MSLP_pangu",
    "MSLP_tianji",
    "prs_sea",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("QCORE_MT2PW_EVAL_ROOT", DEFAULT_EVAL_ROOT)),
        help="Formal mt2pw evaluation root containing analysis/ and artifact_audit/.",
    )
    parser.add_argument(
        "--paired-quality-dir",
        type=Path,
        default=Path(os.environ.get("QCORE_PAIRED_QUALITY_DIR", DEFAULT_QUALITY_DIR)),
        help="Paired quality analysis directory, or its parent containing analysis/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <eval-root>/evidence_story_figures_nc_v6).",
    )
    parser.add_argument(
        "--surface-view",
        choices=("both", "all", "low"),
        default="both",
        help="Show all samples, observed low visibility, or both in station-quality figures.",
    )
    parser.add_argument(
        "--upper-scope",
        choices=("all_paired_test", "true_low_visibility", "elevation_le_500m"),
        default="all_paired_test",
        help="Scope for the pressure-level quality summary.",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png,tiff",
        help="Comma- or colon-separated formats; SVG is always emitted first.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG/TIFF raster resolution.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Skip the MSLP negative-control and pressure-QC supplement figures.",
    )
    parser.add_argument(
        "--allow-missing-artifact-audit",
        action="store_true",
        help="Permit plotting without the formal 51/51 artifact audit (preview only).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every input and stop before drawing.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)


def resolve_quality_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / "paired_source_quality_report.json").is_file():
        return resolved
    if (resolved / "analysis" / "paired_source_quality_report.json").is_file():
        return resolved / "analysis"
    raise FileNotFoundError(
        f"Could not find paired_source_quality_report.json under {resolved} or {resolved / 'analysis'}"
    )


def load_formal(
    eval_root: Path,
    allow_missing_artifact_audit: bool,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, pd.DataFrame]]:
    analysis = eval_root / "analysis"
    report = load_json(analysis / "hybrid_factorial_analysis_report.json")
    if report.get("status") != "passed":
        raise ValueError(f"Formal analysis status is not passed: {report.get('status')!r}")
    if report.get("group_profile") != "mt2pw":
        raise ValueError(f"Expected group_profile='mt2pw', got {report.get('group_profile')!r}")
    if list(report.get("seeds", [])) != [42, 2025, 20260702]:
        raise ValueError(f"Expected formal seeds [42, 2025, 20260702], got {report.get('seeds')!r}")
    masks = [str(mask).zfill(4) for mask in report.get("masks", [])]
    if len(masks) != 16 or set(masks) != {f"{i:04b}" for i in range(16)}:
        raise ValueError("Formal mt2pw analysis must contain all 16 source masks")
    bootstrap = report.get("bootstrap", {})
    if not isinstance(bootstrap, dict) or bootstrap.get("unit") != "UTC_valid_date":
        raise ValueError("Formal analysis must use UTC_valid_date block bootstrap")
    if int(bootstrap.get("iterations", 0)) != 1000:
        raise ValueError("Formal analysis must use 1000 bootstrap iterations")

    audit_path = eval_root / "artifact_audit" / "primary_training_artifact_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        expected = int(audit.get("expected_primary_models", -1))
        if (
            audit.get("status") != "passed"
            or audit.get("formal_matrix") is not True
            or audit.get("group_profile") != "mt2pw"
            or expected != 51
            or int(audit.get("found_triplets", -1)) != expected
            or int(audit.get("passed_triplets", -1)) != expected
        ):
            raise ValueError(f"Artifact audit is not the passed formal 51/51 matrix: {audit}")
    elif allow_missing_artifact_audit:
        audit = {"status": "missing_by_explicit_override"}
    else:
        raise FileNotFoundError(
            f"Missing {audit_path}; use --allow-missing-artifact-audit only for preview work"
        )

    filenames = {
        "metrics": "hybrid_factorial_metrics_by_seed.csv",
        "shapley": "hybrid_exact_shapley_effects.csv",
        "gap_draws": "hybrid_total_gap_bootstrap_draws.csv.gz",
        "events": "event_case_control_environment_summary.csv",
        "event_samples": "event_case_control_samples.csv.gz",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        path = analysis / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if key == "metrics":
            kwargs = {"dtype": {"mask": str}}
        elif key == "event_samples":
            kwargs = {"usecols": EVENT_OBSERVATION_COLUMNS}
        else:
            kwargs = {}
        tables[key] = pd.read_csv(path, **kwargs)

    metrics = tables["metrics"]
    require_columns(
        metrics,
        [
            "seed",
            "mask",
            "low_vis_ap",
            "low_vis_recall_matched_fpr",
            "low_vis_csi_matched_fpr",
            "low_vis_fpr_matched_fpr",
            "low_vis_precision_argmax",
            "low_vis_recall_argmax",
            "low_vis_csi_argmax",
            "low_vis_fpr_argmax",
        ],
        filenames["metrics"],
    )
    metrics["mask"] = normalized_mask(metrics["mask"])
    expected_pairs = {(seed, mask) for seed in (42, 2025, 20260702) for mask in masks}
    actual_pairs = set(zip(metrics["seed"].astype(int), metrics["mask"]))
    if actual_pairs != expected_pairs:
        raise ValueError("Metrics do not contain the exact 3-seed x 16-mask formal matrix")

    require_columns(
        tables["shapley"],
        ["metric", "group", "group_label", "shapley_mean", "ci_low", "ci_high"],
        filenames["shapley"],
    )
    require_columns(
        tables["gap_draws"],
        ["iteration", "metric", "delta_all1_minus_all0"],
        filenames["gap_draws"],
    )
    if tables["gap_draws"]["iteration"].nunique() != 1000:
        raise ValueError("Total-gap draw table must contain 1000 bootstrap iterations")
    require_columns(tables["events"], ["case_category", "n"], filenames["events"])
    require_columns(
        tables["event_samples"], EVENT_OBSERVATION_COLUMNS, filenames["event_samples"]
    )
    return report, audit, tables


def load_quality(
    quality_dir: Path,
    require_strict_low_visibility: bool,
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    report = load_json(quality_dir / "paired_source_quality_report.json")
    if report.get("status") != "completed":
        raise ValueError(f"Paired quality status is not completed: {report.get('status')!r}")
    if report.get("analysis_type") != "diagnostic_only_no_training":
        raise ValueError("Paired quality input is not the expected zero-training diagnosis")
    if int(report.get("test_rows", 0)) <= 0 or int(report.get("represented_utc_dates", 0)) <= 1:
        raise ValueError("Paired quality report has no usable paired test sample")
    if require_strict_low_visibility:
        definition = report.get("low_visibility_definition")
        if not isinstance(definition, dict):
            raise ValueError(
                "Low-visibility figure requires a rebuilt report with low_visibility_definition; "
                "legacy <=1000 m outputs are not publication-safe"
            )
        if definition.get("operator") != "<" or float(definition.get("threshold_m", math.nan)) != 1000.0:
            raise ValueError(
                f"Low-visibility quality must use visibility < 1000 m, got {definition!r}"
            )
        if definition.get("boundary_value_is_clear") is not True:
            raise ValueError("Quality report does not explicitly classify the 1000-m boundary as Clear")

    filenames = {
        "pressure": "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv",
        "qc": "pressure_level_source_qc.csv",
        "surface": "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv",
        "surface_pairs": "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv",
    }
    tables = {key: pd.read_csv(quality_dir / filename) for key, filename in filenames.items()}
    require_columns(
        tables["pressure"],
        [
            "feature",
            "scope",
            "pangu",
            "tianji",
            "delta_pangu_minus_tianji",
            "delta_ci_low",
            "delta_ci_high",
            "pangu_ci_low",
            "pangu_ci_high",
            "tianji_ci_low",
            "tianji_ci_high",
        ],
        filenames["pressure"],
    )
    require_columns(
        tables["surface"],
        ["feature", "source", "scope", "n", "represented_utc_dates", "rmse", "ci_low", "ci_high"],
        filenames["surface"],
    )
    require_columns(
        tables["surface_pairs"],
        [
            "feature",
            "left_source",
            "right_source",
            "scope",
            "delta_left_minus_right",
            "delta_ci_low",
            "delta_ci_high",
        ],
        filenames["surface_pairs"],
    )
    require_columns(
        tables["qc"],
        ["feature", "source", "rows", "outside_broad_range_rows", "outside_broad_range_fraction", "minimum"],
        filenames["qc"],
    )
    required_surface_scopes = ["all_paired_test"]
    if require_strict_low_visibility:
        required_surface_scopes.append("true_low_visibility")
    for scope in required_surface_scopes:
        if scope not in set(tables["surface"]["scope"]):
            raise ValueError(f"Surface quality table lacks scope={scope}")
    return report, tables


def ordered_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [
        item.strip().lower()
        for item in value.replace(":", ",").split(",")
        if item.strip()
    ]
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported formats {unknown}; supported={sorted(supported)}")
    result = ["svg"]
    result.extend(fmt for fmt in requested if fmt != "svg")
    return list(dict.fromkeys(result))


def save_figure(
    fig: plt.Figure,
    base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    saved: List[str] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        # Preserve the exact journal column width and information-aware height
        # selected by each plot. Explicit margins prevent unpredictable
        # tight-bbox resizing between SVG, PDF and TIFF exports.
        kwargs: Dict[str, object] = {"facecolor": "white"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path.name)
    plt.close(fig)
    return saved


def style_axis(
    ax: plt.Axes,
    *,
    horizontal_grid: bool = True,
    vertical_grid: bool = False,
) -> None:
    ax.tick_params(length=3.2, width=0.8, color=INK, pad=3)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if horizontal_grid:
        ax.yaxis.grid(True, color=GRID_GREY, linewidth=0.65, alpha=0.9)
    if vertical_grid:
        ax.xaxis.grid(
            True,
            color=GRID_GREY,
            linewidth=0.58,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.95,
        )
    ax.set_axisbelow(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scope_label(scope: str) -> str:
    return {
        "all_paired_test": "all paired test samples",
        "true_low_visibility": "observed Low-vis samples (<1000 m)",
        "elevation_le_500m": "stations at elevation ≤500 m",
    }[scope]


def flow_step(
    ax: plt.Axes,
    x: float,
    number: str,
    title: str,
    lines: Sequence[str],
    color: str,
) -> None:
    ax.scatter(x, 0.68, s=260, color="white", edgecolor=color, linewidth=1.5, zorder=3)
    ax.text(x, 0.68, number, ha="center", va="center", fontsize=8.4, fontweight="bold", color=color)
    ax.text(x, 0.55, title, ha="center", va="top", fontsize=9.0, fontweight="bold", color=INK)
    ax.text(
        x,
        0.47,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=7.2,
        linespacing=1.35,
        color=INK,
    )


def plot_flow(out_dir: Path, formats: Sequence[str], dpi: int) -> Tuple[pd.DataFrame, List[str]]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["flow"])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.055)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.94,
        "Controlled evidence chain",
        fontsize=12.3,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    xs = [0.07, 0.28, 0.49, 0.70, 0.91]
    ax.plot([xs[0], xs[-1]], [0.68, 0.68], color=LIGHT_GREY, linewidth=1.1, zorder=0)
    stage_color = "#4F5963"
    flow_step(ax, xs[0], "1", "Fair design", ["same rows and labels", "same q-core / model / loss", "Pangu 12–23 h"], stage_color)
    flow_step(ax, xs[1], "2", "Performance", ["Low-vis AP", "matched-FPR recall", "3 training seeds"], stage_color)
    flow_step(ax, xs[2], "3", "Attribution", ["16 source combinations", "T2M / moisture / wind / MSLP", "exact group Shapley"], stage_color)
    flow_step(ax, xs[3], "4", "Quality checks", ["surface → observations", "pressure levels → ERA5", "paired UTC-date CIs"], stage_color)
    flow_step(ax, xs[4], "5", "Bounded inference", ["performance + attribution", "+ quality + hit / miss", "task-specific conclusion"], stage_color)

    ax.plot([0.03, 0.97], [0.17, 0.17], color=INK, linewidth=1.2)
    ax.text(0.03, 0.135, "Supported: task-relevant source-information differences", ha="left", va="top", fontsize=6.9, fontweight="bold", color=INK)
    ax.text(0.56, 0.135, "Not supported: a universal physical-inconsistency claim", ha="left", va="top", fontsize=6.7, color=INK)

    source = pd.DataFrame(
        [
            {"stage": "1", "title": "Fair q-core comparison", "evidence": "controlled design"},
            {"stage": "2", "title": "Primary performance", "evidence": "AP and validation-matched FPR"},
            {"stage": "3", "title": "Controlled source attribution", "evidence": "16 masks x 3 seeds; exact group Shapley"},
            {"stage": "4a", "title": "Near-surface quality", "evidence": "automatic-station observations"},
            {"stage": "4b", "title": "Pressure-level quality", "evidence": "ERA5 reference analysis and QC"},
            {"stage": "5", "title": "Bounded mechanism conclusion", "evidence": "event closure and claim boundary"},
        ]
    )
    return source, save_figure(fig, out_dir / "00_qcore_full_experiment_flow", formats, dpi)


def endpoint_source(
    metrics: pd.DataFrame,
    gap_draws: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    part = metrics[metrics["mask"].isin(["0000", "1111"])][["seed", "mask", metric]].copy()
    part["source"] = part["mask"].map({"0000": "pangu", "1111": "tianji"})
    part = part.rename(columns={metric: "value"})
    draws = pd.to_numeric(
        gap_draws.loc[gap_draws["metric"] == metric, "delta_all1_minus_all0"],
        errors="coerce",
    ).dropna()
    if len(draws) != 1000:
        raise ValueError(f"{metric}: expected 1000 finite total-gap draws, got {len(draws)}")
    means = part.groupby("source", sort=False)["value"].mean()
    summary = pd.DataFrame(
        [
            {
                "seed": "mean",
                "mask": "0000",
                "source": "pangu",
                "value": float(means["pangu"]),
                "delta_tianji_minus_pangu": float(means["tianji"] - means["pangu"]),
                "delta_ci_low": float(draws.quantile(0.025)),
                "delta_ci_high": float(draws.quantile(0.975)),
            },
            {
                "seed": "mean",
                "mask": "1111",
                "source": "tianji",
                "value": float(means["tianji"]),
                "delta_tianji_minus_pangu": float(means["tianji"] - means["pangu"]),
                "delta_ci_low": float(draws.quantile(0.025)),
                "delta_ci_high": float(draws.quantile(0.975)),
            },
        ]
    )
    part["delta_tianji_minus_pangu"] = np.nan
    part["delta_ci_low"] = np.nan
    part["delta_ci_high"] = np.nan
    return pd.concat([part, summary], ignore_index=True)


def plot_endpoint(
    source: pd.DataFrame,
    metric_label: str,
    title: str,
    output_name: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    protocol_note: str,
) -> List[str]:
    seed_rows = source[source["seed"].astype(str) != "mean"].copy()
    mean_rows = source[source["seed"].astype(str) == "mean"].set_index("source")
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["endpoint"])
    fig.subplots_adjust(left=0.25, right=0.96, top=0.70, bottom=0.22)
    for seed in sorted(seed_rows["seed"].astype(int).unique()):
        pair = seed_rows[seed_rows["seed"].astype(int) == seed].set_index("source")
        ax.plot(
            [0, 1],
            [pair.loc["pangu", "value"], pair.loc["tianji", "value"]],
            color=LIGHT_GREY,
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(0, pair.loc["pangu", "value"], s=27, marker=SOURCE_MARKERS["pangu"], color=PANGU_LIGHT, edgecolor=PANGU, linewidth=0.8, zorder=2)
        ax.scatter(1, pair.loc["tianji", "value"], s=27, marker=SOURCE_MARKERS["tianji"], color=TIANJI_LIGHT, edgecolor=TIANJI, linewidth=0.8, zorder=2)

    pangu = float(mean_rows.loc["pangu", "value"])
    tianji = float(mean_rows.loc["tianji", "value"])
    ax.scatter(0, pangu, s=84, marker=SOURCE_MARKERS["pangu"], color=PANGU, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(1, tianji, s=84, marker=SOURCE_MARKERS["tianji"], color=TIANJI, edgecolor="white", linewidth=0.8, zorder=4)
    # Place labels toward the centre of the sparse two-point plot. This avoids
    # collisions with the y-axis tick labels and the right edge at 89-mm width.
    ax.annotate(f"{pangu:.3f}", (0, pangu), xytext=(10, -1), textcoords="offset points", ha="left", va="center", fontsize=9.2, fontweight="bold", color=PANGU_DARK)
    ax.annotate(f"{tianji:.3f}", (1, tianji), xytext=(-10, -1), textcoords="offset points", ha="right", va="center", fontsize=9.2, fontweight="bold", color=TIANJI)

    delta = float(mean_rows.loc["tianji", "delta_tianji_minus_pangu"])
    ci_low = float(mean_rows.loc["tianji", "delta_ci_low"])
    ci_high = float(mean_rows.loc["tianji", "delta_ci_high"])
    values = seed_rows["value"].to_numpy(dtype=float)
    span = max(float(values.max() - values.min()), 0.02)
    ymin = max(0.0, float(values.min() - 0.22 * span))
    ymax = float(values.max() + 0.38 * span)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.18, 1.18)
    ax.text(
        0.0,
        1.075,
        f"Δ Tianji−Pangu {delta:+.3f}  [{ci_low:+.3f}, {ci_high:+.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=TIANJI if delta > 0 else PANGU_DARK,
    )
    ax.text(
        0.0,
        1.205,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.set_xticks([0, 1], ["Pangu\ntrained", "Tianji\ntrained"])
    ax.get_xticklabels()[0].set_color(PANGU_DARK)
    ax.get_xticklabels()[1].set_color(TIANJI)
    ax.set_ylabel(metric_label)
    style_axis(ax)
    return save_figure(fig, out_dir / output_name, formats, dpi)


def qcore_argmax_source(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the two fair q-core endpoints in long form for argmax display."""

    metric_specs = [
        ("Precision", "low_vis_precision_argmax", "higher"),
        ("Recall", "low_vis_recall_argmax", "higher"),
        ("CSI", "low_vis_csi_argmax", "higher"),
        ("FPR", "low_vis_fpr_argmax", "lower"),
    ]
    source_specs = [("tianji", "1111"), ("pangu", "0000")]
    rows: List[Dict[str, object]] = []
    for source, mask in source_specs:
        endpoint = metrics[metrics["mask"] == mask].copy()
        if set(endpoint["seed"].astype(int)) != {42, 2025, 20260702}:
            raise ValueError(f"Argmax endpoint {mask} does not contain the three formal seeds")
        for label, column, direction in metric_specs:
            values = pd.to_numeric(endpoint[column], errors="raise")
            for seed, value in zip(endpoint["seed"].astype(int), values):
                rows.append(
                    {
                        "source": source,
                        "mask": mask,
                        "seed": int(seed),
                        "metric": label,
                        "metric_column": column,
                        "preferred_direction": direction,
                        "value": float(value),
                    }
                )
            rows.append(
                {
                    "source": source,
                    "mask": mask,
                    "seed": "mean",
                    "metric": label,
                    "metric_column": column,
                    "preferred_direction": direction,
                    "value": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_qcore_argmax_overview(
    source: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    """Draw a compact fair-input argmax overview with seed-level transparency."""

    metric_order = ["Precision", "Recall", "CSI", "FPR"]
    source_order = ["tianji", "pangu"]
    mean_rows = source[source["seed"].astype(str) == "mean"].copy()
    seed_rows = source[source["seed"].astype(str) != "mean"].copy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["argmax"])
    fig.subplots_adjust(left=0.18, right=0.98, top=0.68, bottom=0.20)
    x = np.arange(len(metric_order), dtype=float)
    width = 0.32
    offsets = {"tianji": -width / 2, "pangu": width / 2}
    jitter = {-1: -0.035, 0: 0.0, 1: 0.035}
    seed_order = [42, 2025, 20260702]

    for source_key in source_order:
        means = (
            mean_rows[mean_rows["source"] == source_key]
            .set_index("metric")
            .reindex(metric_order)
        )
        xpos = x + offsets[source_key]
        bars = ax.bar(
            xpos,
            means["value"].to_numpy(dtype=float),
            width=width * 0.88,
            color=SOURCE_COLORS[source_key],
            label=SOURCE_LABELS[source_key],
            zorder=2,
        )
        for seed_index, seed in enumerate(seed_order):
            values = (
                seed_rows[
                    (seed_rows["source"] == source_key)
                    & (seed_rows["seed"].astype(int) == seed)
                ]
                .set_index("metric")
                .reindex(metric_order)["value"]
                .to_numpy(dtype=float)
            )
            ax.scatter(
                xpos + jitter[seed_index - 1],
                values,
                s=13,
                marker="o",
                facecolor="white",
                edgecolor=SOURCE_DARK_COLORS[source_key],
                linewidth=0.65,
                zorder=3,
            )
        for metric, bar, value in zip(metric_order, bars, means["value"]):
            seed_max = float(
                seed_rows[
                    (seed_rows["source"] == source_key)
                    & (seed_rows["metric"] == metric)
                ]["value"].max()
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(float(value), seed_max) + 0.012,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color=SOURCE_DARK_COLORS[source_key],
                rotation=90,
            )

    ymax = max(0.84, float(source["value"].max()) + 0.09)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x, ["Precision ↑", "Recall ↑", "CSI ↑", "FPR ↓"])
    ax.set_ylabel("Score on paired test samples")
    fig.text(
        0.18,
        0.95,
        "Fair q-core argmax performance",
        ha="left",
        va="top",
        fontsize=10.7,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.18, 0.86),
        bbox_transform=fig.transFigure,
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.45,
    )
    style_axis(ax)
    return save_figure(
        fig,
        out_dir / "02a_qcore_argmax_lowvis_overview",
        formats,
        dpi,
    )


def shapley_source(shapley: pd.DataFrame) -> pd.DataFrame:
    order = ["T2", "M", "W", "P"]
    part = shapley[shapley["metric"] == "low_vis_ap"].copy()
    if set(part["group"].astype(str)) != set(order):
        raise ValueError(f"Low-vis AP Shapley rows must be {order}")
    part = part.set_index("group").loc[order].reset_index()
    total = float(part["shapley_mean"].sum())
    part["share_of_total_gap_percent"] = 100.0 * part["shapley_mean"] / total
    return part


def plot_shapley(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["shapley"])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.82, bottom=0.19)
    y = np.arange(len(source))[::-1]
    labels = {
        "T2": "2-m temperature",
        "M": "Moisture package",
        "W": "Wind package",
        "P": "Mean sea-level pressure",
    }
    for yi, row in zip(y, source.itertuples(index=False)):
        color = PACKAGE_COLORS[str(row.group)]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter(row.shapley_mean, yi, s=52, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(
            max(0.043, float(row.ci_high) + 0.0015),
            yi,
            f"{row.shapley_mean:+.3f}",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color=INK,
        )
    ax.axvline(0.0, color=INK, linewidth=0.8)
    ax.set_yticks(y, [labels[group] for group in source["group"]])
    xmin = min(-0.006, float(source["ci_low"].min()) - 0.002)
    xmax = max(0.052, float(source["ci_high"].max()) + 0.014)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Exact source-block Shapley contribution to Low-vis AP")
    ax.set_title("Source-block contributions to Low-vis AP", loc="left", pad=10, fontweight="bold")
    style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return save_figure(fig, out_dir / "03_source_block_shapley", formats, dpi)


def surface_feature_source(
    surface: pd.DataFrame,
    pairs: pd.DataFrame,
    feature: str,
    scopes: Sequence[str],
) -> pd.DataFrame:
    order = ["pangu", "tianji", "era5_reference_analysis"]
    frames: List[pd.DataFrame] = []
    for scope in scopes:
        part = surface[(surface["feature"] == feature) & (surface["scope"] == scope)].copy()
        part = part.set_index("source").reindex(order).reset_index()
        if part[["rmse", "ci_low", "ci_high"]].isna().any().any():
            raise ValueError(f"Missing three-source surface RMSE rows for feature={feature}, scope={scope}")
        pair = pairs[
            (pairs["feature"] == feature)
            & (pairs["scope"] == scope)
            & (pairs["left_source"] == "pangu")
            & (pairs["right_source"] == "tianji")
        ]
        if len(pair) != 1:
            raise ValueError(f"Expected one paired Pangu-minus-Tianji row for {feature}/{scope}")
        row = pair.iloc[0]
        part["pangu_minus_tianji"] = float(row["delta_left_minus_right"])
        part["pair_delta_ci_low"] = float(row["delta_ci_low"])
        part["pair_delta_ci_high"] = float(row["delta_ci_high"])
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def plot_surface_feature(
    source: pd.DataFrame,
    feature_label: str,
    unit: str,
    scopes: Sequence[str],
    output_name: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["surface"])
    fig.subplots_adjust(left=0.20, right=0.76, top=0.79, bottom=0.20)
    base_y = {scope: float(len(scopes) - 1 - index) for index, scope in enumerate(scopes)}
    offsets = {"pangu": 0.17, "tianji": 0.0, "era5_reference_analysis": -0.17}
    markers = {key: SOURCE_MARKERS[key] for key in offsets}
    for source_key in ("pangu", "tianji", "era5_reference_analysis"):
        rows = source[source["source"] == source_key]
        for row in rows.itertuples(index=False):
            yi = base_y[str(row.scope)] + offsets[source_key]
            xerr = np.array([[max(float(row.rmse - row.ci_low), 0.0)], [max(float(row.ci_high - row.rmse), 0.0)]])
            ax.errorbar(
                float(row.rmse),
                yi,
                xerr=xerr,
                fmt=markers[source_key],
                markersize=5.7,
                color=SOURCE_COLORS[source_key],
                ecolor=SOURCE_COLORS[source_key],
                elinewidth=1.25,
                capsize=2.6,
                capthick=0.9,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
                label=SOURCE_LABELS[source_key] if str(row.scope) == scopes[0] else None,
            )

    min_ci = float(source["ci_low"].min())
    max_ci = float(source["ci_high"].max())
    ci_span = max(max_ci - min_ci, max_ci * 0.08, 1e-6)
    ax.set_xlim(max(0.0, min_ci - 0.18 * ci_span), max_ci + 0.22 * ci_span)
    ax.set_ylim(-0.46, max(base_y.values()) + 0.46)
    scope_names = {
        "all_paired_test": "All test samples",
        "true_low_visibility": "Observed visibility <1 km",
        "elevation_le_500m": "Elevation ≤500 m",
    }
    ax.set_yticks([base_y[scope] for scope in scopes], [scope_names[scope] for scope in scopes])
    ax.set_xlabel(f"RMSE vs station observations ({unit})")
    title_map = {
        "2-m temperature": "2-m temperature RMSE",
        "10-m wind speed": "10-m wind-speed RMSE",
        "Mean sea-level pressure": "Mean sea-level pressure RMSE",
    }
    ax.set_title(title_map.get(feature_label, feature_label), loc="left", pad=10, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, handletextpad=0.4, columnspacing=1.1)

    ax.text(1.03, 1.02, "Pangu−Tianji ΔRMSE [95% CI]", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=INK, ha="left", va="bottom", clip_on=False)
    for scope in scopes:
        row = source[source["scope"] == scope].iloc[0]
        delta = float(row["pangu_minus_tianji"])
        lo = float(row["pair_delta_ci_low"])
        hi = float(row["pair_delta_ci_high"])
        color = TIANJI if delta > 0 else PANGU_DARK
        ax.text(
            1.03,
            base_y[scope],
            f"{delta:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
            transform=ax.get_yaxis_transform(),
            fontsize=7.7,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            clip_on=False,
        )
    style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return save_figure(fig, out_dir / output_name, formats, dpi)


def pressure_source(pressure: pd.DataFrame, scope: str) -> pd.DataFrame:
    wanted = [
        ("Q_1000", "Specific humidity at 1000 hPa"),
        ("Q_925", "Specific humidity at 925 hPa"),
        ("UV_925_VECTOR", "925-hPa vector wind"),
    ]
    rows: List[Dict[str, object]] = []
    for feature, label in wanted:
        part = pressure[(pressure["feature"] == feature) & (pressure["scope"] == scope)]
        if len(part) != 1:
            raise ValueError(f"Expected one pressure row for {feature}/{scope}")
        row = part.iloc[0]
        pangu = float(row["pangu"])
        rows.append(
            {
                "feature": feature,
                "label": label,
                "scope": scope,
                "pangu_rmse": pangu,
                "tianji_rmse": float(row["tianji"]),
                "delta_pangu_minus_tianji": float(row["delta_pangu_minus_tianji"]),
                "delta_ci_low": float(row["delta_ci_low"]),
                "delta_ci_high": float(row["delta_ci_high"]),
                "normalized_delta_percent": 100.0 * float(row["delta_pangu_minus_tianji"]) / pangu,
                "normalized_ci_low_percent": 100.0 * float(row["delta_ci_low"]) / pangu,
                "normalized_ci_high_percent": 100.0 * float(row["delta_ci_high"]) / pangu,
                "n": int(row["n"]),
                "represented_utc_dates": int(row["represented_utc_dates"]),
            }
        )
    return pd.DataFrame(rows)


def plot_pressure(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["pressure"])
    fig.subplots_adjust(left=0.31, right=0.95, top=0.79, bottom=0.20)
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        value = float(row.normalized_delta_percent)
        lo = float(row.normalized_ci_low_percent)
        hi = float(row.normalized_ci_high_percent)
        color = TIANJI if value > 0 else PANGU_DARK
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter(value, yi, s=52, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        label_x = hi + 0.6 if value >= 0 else lo - 0.6
        align = "left" if value >= 0 else "right"
        ax.text(label_x, yi, f"{value:+.1f}%", ha=align, va="center", fontsize=8.2, fontweight="bold", color=color)
    ax.axvline(0.0, color=INK, linewidth=0.85)
    ax.set_yticks(y, source["label"])
    xmin = min(-20.0, float(source["normalized_ci_low_percent"].min()) - 5.0)
    xmax = max(25.0, float(source["normalized_ci_high_percent"].max()) + 5.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Paired RMSE difference relative to Pangu (%)")
    ax.set_title("Pressure-level error relative to ERA5", loc="left", pad=10, fontweight="bold")
    ax.text(0.01, 0.985, "← Pangu closer", transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=PANGU_DARK, ha="left", va="top")
    ax.text(0.99, 0.985, "Tianji closer →", transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=TIANJI, ha="right", va="top")
    style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return save_figure(fig, out_dir / "07_pressure_level_quality", formats, dpi)


def event_source(events: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("tianji_hit_pangu_miss", "Tianji-only hit", TIANJI),
        ("pangu_hit_tianji_miss", "Pangu-only hit", PANGU),
    ]
    indexed = events.set_index("case_category")
    rows = []
    for key, label, color in wanted:
        if key not in indexed.index:
            raise ValueError(f"Missing event category {key}")
        rows.append({"case_category": key, "label": label, "n": int(indexed.loc[key, "n"]), "color": color})
    result = pd.DataFrame(rows)
    result["tianji_to_pangu_unique_hit_ratio"] = float(result.iloc[0]["n"] / result.iloc[1]["n"])
    return result


def plot_events(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["events"])
    fig.subplots_adjust(left=0.25, right=0.95, top=0.74, bottom=0.25)
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        ax.plot([0, row.n], [yi, yi], color=row.color, linewidth=3.0, solid_capstyle="round")
        ax.scatter(row.n, yi, s=58, color=row.color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(row.n + 100, yi, f"{int(row.n):,}", ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)
    ratio = float(source["tianji_to_pangu_unique_hit_ratio"].iloc[0])
    ax.text(0.995, 1.035, f"Tianji / Pangu = {ratio:.2f}×", transform=ax.transAxes, fontsize=8.3, fontweight="bold", color=TIANJI, ha="right", va="bottom")
    ax.set_yticks(y, source["label"])
    ax.set_xlim(0, max(source["n"]) * 1.30)
    ax.set_xlabel("True Low-vis samples detected by only one endpoint model")
    ax.set_title("Unique Low-vis hits at matched FPR", loc="left", pad=10, fontweight="bold")
    style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return save_figure(fig, out_dir / "08_unique_event_hits", formats, dpi)


def event_observation_advantage_source(
    samples: pd.DataFrame,
    iterations: int = 1000,
    seed: int = 20260702,
) -> pd.DataFrame:
    """Observation-anchored source RMSE within Tianji-hit/Pangu-miss samples.

    Dates, rather than station-time rows, are resampled to retain the spatial
    dependence shared by stations within the same weather day.  The same date
    draw is used for all three variables in every bootstrap iteration.
    """

    target_category = "tianji_hit_pangu_miss"
    target = samples[samples["case_category"] == target_category].copy()
    if target.empty:
        raise ValueError(f"No event samples for category={target_category}")
    target["time_utc"] = pd.to_datetime(target["time_utc"], errors="raise", utc=True)
    target["utc_valid_date"] = target["time_utc"].dt.floor("D")
    if target[["time_utc", "station_key"]].duplicated().any():
        raise ValueError("Tianji-only event samples contain duplicate station-time rows")
    visibility = pd.to_numeric(target["vis_raw_m"], errors="coerce")
    if visibility.isna().any() or bool((visibility >= 1000.0).any()):
        raise ValueError("Tianji-only event samples must all satisfy observed visibility <1000 m")

    dates = pd.Index(sorted(target["utc_valid_date"].unique()), name="utc_valid_date")
    if len(dates) < 10:
        raise ValueError(f"Event analysis requires at least 10 UTC dates, got {len(dates)}")
    if iterations < 200:
        raise ValueError("Use at least 200 UTC-date bootstrap iterations")
    rng = np.random.default_rng(seed)
    date_draws = rng.integers(0, len(dates), size=(iterations, len(dates)))

    feature_specs = [
        {
            "feature": "T2M",
            "label": "2-m temperature",
            "unit": "°C",
            "pangu": "T2M_pangu",
            "tianji": "T2M_tianji",
            "observation": "tem",
            "conversion": "forecast K minus 273.15; station observation in degC",
        },
        {
            "feature": "WSPD10",
            "label": "10-m wind speed",
            "unit": "m s-1",
            "pangu": "WSPD10_pangu",
            "tianji": "WSPD10_tianji",
            "observation": "win_s_avg_10mi",
            "conversion": "forecast and station observation in m s-1",
        },
        {
            "feature": "MSLP",
            "label": "Mean sea-level pressure",
            "unit": "hPa",
            "pangu": "MSLP_pangu",
            "tianji": "MSLP_tianji",
            "observation": "prs_sea",
            "conversion": "forecast Pa divided by 100; station observation in hPa",
        },
    ]

    rows: List[Dict[str, object]] = []
    for spec in feature_specs:
        frame = target[["utc_valid_date", spec["pangu"], spec["tianji"], spec["observation"]]].copy()
        pangu = pd.to_numeric(frame[spec["pangu"]], errors="coerce").to_numpy(dtype=float)
        tianji = pd.to_numeric(frame[spec["tianji"]], errors="coerce").to_numpy(dtype=float)
        observation = pd.to_numeric(frame[spec["observation"]], errors="coerce").to_numpy(dtype=float)
        if spec["feature"] == "T2M":
            pangu = pangu - 273.15
            tianji = tianji - 273.15
        elif spec["feature"] == "MSLP":
            pangu = pangu / 100.0
            tianji = tianji / 100.0
        finite = np.isfinite(pangu) & np.isfinite(tianji) & np.isfinite(observation)
        if int(finite.sum()) < 100:
            raise ValueError(f"{spec['feature']}: fewer than 100 complete event-observation rows")
        errors = pd.DataFrame(
            {
                "utc_valid_date": frame.loc[finite, "utc_valid_date"].to_numpy(),
                "pangu_squared_error": np.square(pangu[finite] - observation[finite]),
                "tianji_squared_error": np.square(tianji[finite] - observation[finite]),
            }
        )
        daily = errors.groupby("utc_valid_date", sort=True).agg(
            pangu_squared_error=("pangu_squared_error", "sum"),
            tianji_squared_error=("tianji_squared_error", "sum"),
            n=("pangu_squared_error", "size"),
        ).reindex(dates, fill_value=0)
        pangu_sum = daily["pangu_squared_error"].to_numpy(dtype=float)
        tianji_sum = daily["tianji_squared_error"].to_numpy(dtype=float)
        counts = daily["n"].to_numpy(dtype=float)

        def statistic(indices: np.ndarray) -> Tuple[float, float, float]:
            total_n = float(counts[indices].sum())
            if total_n <= 0:
                raise ValueError(f"{spec['feature']}: empty UTC-date bootstrap draw")
            pangu_rmse = math.sqrt(float(pangu_sum[indices].sum()) / total_n)
            tianji_rmse = math.sqrt(float(tianji_sum[indices].sum()) / total_n)
            if pangu_rmse <= 0:
                raise ValueError(f"{spec['feature']}: Pangu RMSE must be positive")
            reduction = 100.0 * (pangu_rmse - tianji_rmse) / pangu_rmse
            return pangu_rmse, tianji_rmse, reduction

        all_indices = np.arange(len(dates), dtype=int)
        pangu_rmse, tianji_rmse, reduction = statistic(all_indices)
        draws = np.array([statistic(indices)[2] for indices in date_draws], dtype=float)
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "feature": spec["feature"],
                "label": spec["label"],
                "unit": spec["unit"],
                "category": target_category,
                "target_category_rows": int(len(target)),
                "n_complete": int(finite.sum()),
                "represented_utc_dates": int(np.count_nonzero(counts)),
                "pangu_rmse": pangu_rmse,
                "tianji_rmse": tianji_rmse,
                "relative_rmse_reduction_percent": reduction,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                "bootstrap_unit": "UTC_valid_date",
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(seed),
                "reference": "automatic-station observation",
                "unit_conversion": spec["conversion"],
                "formula": "100 * (RMSE_Pangu - RMSE_Tianji) / RMSE_Pangu",
                "interpretation": "descriptive endpoint-conditioned association; not a causal source intervention",
            }
        )
    return pd.DataFrame(rows)


def plot_event_observation_advantage(
    source: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["event_quality"])
    fig.subplots_adjust(left=0.29, right=0.96, top=0.77, bottom=0.22)
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        supported = bool(row.ci_low > 0.0)
        line_color = TIANJI if supported else ERA5_DARK
        marker_color = TIANJI if supported else ERA5
        ax.plot(
            [row.ci_low, row.ci_high],
            [yi, yi],
            color=line_color,
            linewidth=2.2,
            solid_capstyle="round",
        )
        ax.scatter(
            row.relative_rmse_reduction_percent,
            yi,
            s=55,
            marker=SOURCE_MARKERS["tianji"] if supported else SOURCE_MARKERS["baseline"],
            color=marker_color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            float(row.ci_high) + 1.1,
            yi,
            f"{row.relative_rmse_reduction_percent:.1f}%",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color=line_color,
        )
    ax.axvline(0.0, color=INK, linewidth=0.85)
    ax.set_yticks(y, source["label"])
    xmin = min(-15.0, float(source["ci_low"].min()) - 3.0)
    xmax = max(60.0, float(source["ci_high"].max()) + 8.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.52, len(source) - 0.28)
    ax.set_xlabel("Tianji RMSE reduction relative to Pangu (%)")
    ax.set_title(
        "Observation-anchored source advantage within Tianji-only hits",
        loc="left",
        pad=10,
        fontweight="bold",
        fontsize=11.0,
    )
    ax.text(
        0.01,
        0.985,
        "← Pangu closer",
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=PANGU_DARK,
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        0.985,
        "Tianji closer →",
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=TIANJI,
        ha="right",
        va="top",
    )
    style_axis(ax, horizontal_grid=False, vertical_grid=True)
    return save_figure(
        fig,
        out_dir / "09_observation_anchored_tianji_only_advantage",
        formats,
        dpi,
    )


def qc_source(qc: pd.DataFrame) -> pd.DataFrame:
    order = ["Q_1000", "Q_925"]
    sources = ["pangu", "tianji", "era5_reference_analysis"]
    part = qc[qc["feature"].isin(order) & qc["source"].isin(sources)].copy()
    part["outside_percent"] = 100.0 * pd.to_numeric(part["outside_broad_range_fraction"], errors="raise")
    part["feature"] = pd.Categorical(part["feature"], categories=order, ordered=True)
    part["source"] = pd.Categorical(part["source"], categories=sources, ordered=True)
    return part.sort_values(["feature", "source"]).reset_index(drop=True)


def plot_qc(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["qc"])
    fig.subplots_adjust(left=0.14, right=0.95, top=0.77, bottom=0.21)
    features = ["Q_1000", "Q_925"]
    sources = ["pangu", "tianji", "era5_reference_analysis"]
    x = np.arange(len(features))
    width = 0.22
    for offset, source_key in zip((-width, 0.0, width), sources):
        item = source[source["source"].astype(str) == source_key].copy()
        item["feature_key"] = item["feature"].astype(str)
        item = item.set_index("feature_key")
        values = np.array([float(item.loc[feature, "outside_percent"]) for feature in features])
        bars = ax.bar(x + offset, values, width=width * 0.88, color=SOURCE_COLORS[source_key], label=SOURCE_LABELS[source_key])
        for bar, value in zip(bars, values):
            label = f"{value:.3f}%" if value > 0 else "0"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, label, ha="center", va="bottom", fontsize=7.2, color=SOURCE_COLORS[source_key])
    ymax = max(float(source["outside_percent"].max()) * 1.35, 0.08)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x, ["Q at 1000 hPa", "Q at 925 hPa"])
    ax.set_ylabel(r"Values outside 0–80 g kg$^{-1}$ (%)")
    ax.set_title("Pangu contains rare out-of-range specific humidity", loc="left", pad=10, fontweight="bold")
    ax.legend(loc="upper right", ncol=3, columnspacing=1.0, handletextpad=0.4)
    style_axis(ax)
    return save_figure(fig, out_dir / "10_pressure_qc", formats, dpi)


def write_guide(
    out_dir: Path,
    formal_report: Mapping[str, object],
    quality_report: Mapping[str, object],
    event_quality: pd.DataFrame,
    surface_scopes: Sequence[str],
    upper_scope: str,
) -> None:
    text = f"""# Q-core evidence-story figure guide

Every output is a standalone one-claim figure. Use SVG in PowerPoint for
editable text, PDF for manuscript assembly, TIFF for submission, and PNG for
quick review.

## Recommended story order

1. `00_qcore_full_experiment_flow`: define the controlled evidence chain.
2. `01_qcore_lowvis_ap`: establish the threshold-free endpoint gap.
3. `02_qcore_matched_fpr_recall`: show that the gap remains at a validation-matched false-alarm protocol.
4. `03_source_block_shapley`: attribute the gap to source packages through retraining.
5. `04_surface_t2m_quality`: connect the largest package contribution to station-observed T2M quality.
6. `05_surface_wspd10_quality`: connect the wind contribution to station-observed wind quality.
7. `07_pressure_level_quality`: show the mixed pressure-level ranking and prevent a uniform-RMSE overclaim.
8. `08_unique_event_hits`: establish the asymmetric number of endpoint-specific Low-vis hits.
9. `09_observation_anchored_tianji_only_advantage`: show that Tianji T2M and WSPD10 are closer to station observations within Tianji-only hits, with MSLP as a non-robust control.

Use `02a_qcore_argmax_lowvis_overview` in a methods/results presentation, or in
the supplement, when readers need the familiar argmax Precision/Recall/CSI/FPR
overview. It is not a replacement for the threshold-free AP and matched-FPR
primary endpoints.

Move `06_surface_mslp_quality`, `08_unique_event_hits`, and `10_pressure_qc` to
the supplement if the event-conditioned observation figure is used in the main
text, unless a reviewer specifically asks for the hit counts, negative control,
or QC rate in the main text.

## Caption essentials

- `01` and `02`: small points are individual training seeds, large points are
  three-seed means, and the reported difference CI is a joint UTC-date block
  bootstrap. The AP endpoint is threshold-free; matched-FPR thresholds were
  selected on validation and frozen before test evaluation.
- `02a`: bars are three-seed means and open circles are individual seeds on
  identical paired q-core test rows. All values use argmax. Arrows encode the
  preferred direction; F1 is omitted because it is a monotonic transformation
  of CSI for the same binary event and would duplicate information.
- `03`: horizontal intervals are 95% UTC-date bootstrap CIs from exact group
  Shapley attribution over all 16 retrained source-block combinations. Interpret
  these as package contributions, not single-variable causal effects.
- `04`--`06`: automatic-station observations are the reference. Pangu, Tianji,
  and ERA5 use identical station-time rows. The right column reports the paired
  Pangu-minus-Tianji RMSE difference and 95% UTC-date bootstrap CI for both the
  complete test set and observed visibility below 1 km. ERA5 is an analysis
  benchmark, not a third forecast.
- `07`: ERA5 is a reference analysis. Positive normalized differences indicate
  lower Tianji RMSE; negative values indicate lower Pangu RMSE. ERA5 humidity is
  derived and is not independent truth.
- `08`: endpoint probabilities are averaged across seeds; source-specific
  operating thresholds were fixed on validation at the common target FPR.
- `09`: the analysis is restricted to
  `tianji_hit_pangu_miss` station-time samples defined by the validation-frozen
  endpoint thresholds. Values are `100 × (RMSE_Pangu − RMSE_Tianji) /
  RMSE_Pangu`; horizontal intervals are 95% CIs from 1000 joint
  UTC-valid-date bootstrap draws. Forecast T2M is converted from K to °C and
  MSLP from Pa to hPa before comparison with automatic-station observations.
  This is a descriptive endpoint-conditioned association, not an independent
  source intervention or a causal estimate.

## Formal claim boundary

The experiment supports a task-specific, source-dependent multivariate
information-quality difference for Pangu-2025, 2025, 12--23 h lead and the
current low-visibility task. It does not prove that Pangu violates governing
equations, and it must not be generalized to all AI weather models.

## Figure contract

- `01`, `02`, and `02a`: 3.504 in (89-mm Nature single-column width)
- all denser comparison figures: at most 7.205 in (183-mm two-column width)
- height is tightened by information density rather than padded to one master
  aspect ratio
- subtle vertical dashed major grids are used only for horizontal numerical
  comparisons (`03`--`09`); endpoint plots retain horizontal value grids, and
  workflow/QC figures keep their semantically appropriate treatment
- typography: editable sans-serif text in SVG/PDF
- source palette: Tianji `#2E5A87` (dark blue), Pangu `#8E6BBE`
  (mid-light violet), baseline/ERA5 `#9A9A9A` (grey); marker shapes remain a
  secondary cue
- station-observation scopes: `{list(surface_scopes)}`
- pressure-level scope: `{upper_scope}` ({scope_label(upper_scope)})
- surface uncertainty: 95% UTC-valid-date block bootstrap
- performance variability: three training seeds plus UTC-date bootstrap
- Tianji-only observation analysis: `{int(event_quality['target_category_rows'].iloc[0])}`
  station-time samples, `{int(event_quality['represented_utc_dates'].max())}` UTC dates,
  1000 joint UTC-date bootstrap draws
- ERA5 role: reference-analysis benchmark, not truth or a third forecast

## Provenance summary

- formal analysis status: `{formal_report.get('status')}`
- formal group profile: `{formal_report.get('group_profile')}`
- formal seeds: `{formal_report.get('seeds')}`
- paired quality status: `{quality_report.get('status')}`
- paired quality rows: `{quality_report.get('test_rows')}`
- represented UTC dates: `{quality_report.get('represented_utc_dates')}`
- Low-vis definition: `{quality_report.get('low_visibility_definition')}`
"""
    (out_dir / "QCORE_EVIDENCE_STORY_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    quality_dir = resolve_quality_dir(args.paired_quality_dir)
    out_dir = (args.out_dir or (eval_root / "evidence_story_figures_nc_v6")).expanduser().resolve()
    formats = ordered_formats(args.formats)
    if args.dpi < 300:
        raise ValueError("Use --dpi >= 300; 600 is recommended for paper TIFF export")

    surface_scopes = {
        "both": ["all_paired_test", "true_low_visibility"],
        "all": ["all_paired_test"],
        "low": ["true_low_visibility"],
    }[args.surface_view]
    formal_report, audit, formal = load_formal(eval_root, args.allow_missing_artifact_audit)
    quality_report, quality = load_quality(
        quality_dir, "true_low_visibility" in surface_scopes
    )
    event_quality = event_observation_advantage_source(formal["event_samples"])
    expected_target_rows = int(
        formal["events"].set_index("case_category").loc["tianji_hit_pangu_miss", "n"]
    )
    if int(event_quality["target_category_rows"].iloc[0]) != expected_target_rows:
        raise ValueError(
            "Tianji-only event sample count differs between event summary and sample table"
        )
    print(
        "[validation] formal and paired-quality inputs passed: "
        f"mt2pw={formal_report.get('status')} artifact_audit={audit.get('status')} "
        f"quality={quality_report.get('status')} rows={quality_report.get('test_rows')} "
        f"tianji_only_rows={expected_target_rows}"
    )
    if args.validate_only:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, List[str]] = {}

    flow, files = plot_flow(out_dir, formats, args.dpi)
    flow.to_csv(source_dir / "00_qcore_full_experiment_flow.csv", index=False)
    generated["00_qcore_full_experiment_flow"] = files

    ap = endpoint_source(formal["metrics"], formal["gap_draws"], "low_vis_ap")
    ap.to_csv(source_dir / "01_qcore_lowvis_ap.csv", index=False)
    generated["01_qcore_lowvis_ap"] = plot_endpoint(
        ap,
        "Low-vis average precision (AP)",
        "Low-visibility average precision",
        "01_qcore_lowvis_ap",
        out_dir,
        formats,
        args.dpi,
        "AP uses no decision threshold.",
    )

    recall = endpoint_source(formal["metrics"], formal["gap_draws"], "low_vis_recall_matched_fpr")
    recall.to_csv(source_dir / "02_qcore_matched_fpr_recall.csv", index=False)
    generated["02_qcore_matched_fpr_recall"] = plot_endpoint(
        recall,
        "Low-vis recall",
        "Low-visibility recall at matched FPR",
        "02_qcore_matched_fpr_recall",
        out_dir,
        formats,
        args.dpi,
        "Thresholds were selected on validation at one target FPR and then frozen for test.",
    )

    argmax = qcore_argmax_source(formal["metrics"])
    argmax.to_csv(
        source_dir / "02a_qcore_argmax_lowvis_overview.csv", index=False
    )
    generated["02a_qcore_argmax_lowvis_overview"] = (
        plot_qcore_argmax_overview(argmax, out_dir, formats, args.dpi)
    )

    shapley = shapley_source(formal["shapley"])
    shapley.to_csv(source_dir / "03_source_block_shapley.csv", index=False)
    generated["03_source_block_shapley"] = plot_shapley(shapley, out_dir, formats, args.dpi)

    surface_specs = [
        ("T2M", "2-m temperature", "°C", "04_surface_t2m_quality"),
        ("WSPD10", "10-m wind speed", r"m s$^{-1}$", "05_surface_wspd10_quality"),
    ]
    if not args.main_only:
        surface_specs.append(("MSLP", "Mean sea-level pressure", "hPa", "06_surface_mslp_quality"))
    for feature, label, unit, output_name in surface_specs:
        source = surface_feature_source(
            quality["surface"], quality["surface_pairs"], feature, surface_scopes
        )
        source.to_csv(source_dir / f"{output_name}.csv", index=False)
        generated[output_name] = plot_surface_feature(
            source,
            label,
            unit,
            surface_scopes,
            output_name,
            out_dir,
            formats,
            args.dpi,
        )

    pressure = pressure_source(quality["pressure"], args.upper_scope)
    pressure.to_csv(source_dir / "07_pressure_level_quality.csv", index=False)
    generated["07_pressure_level_quality"] = plot_pressure(pressure, out_dir, formats, args.dpi)

    events = event_source(formal["events"])
    events.drop(columns=["color"]).to_csv(source_dir / "08_unique_event_hits.csv", index=False)
    generated["08_unique_event_hits"] = plot_events(events, out_dir, formats, args.dpi)

    event_quality.to_csv(
        source_dir / "09_observation_anchored_tianji_only_advantage.csv", index=False
    )
    generated["09_observation_anchored_tianji_only_advantage"] = (
        plot_event_observation_advantage(event_quality, out_dir, formats, args.dpi)
    )

    if not args.main_only:
        qc = qc_source(quality["qc"])
        qc.to_csv(source_dir / "10_pressure_qc.csv", index=False)
        generated["10_pressure_qc"] = plot_qc(qc, out_dir, formats, args.dpi)

    write_guide(
        out_dir,
        formal_report,
        quality_report,
        event_quality,
        surface_scopes,
        args.upper_scope,
    )
    manifest = {
        "status": "passed",
        "eval_root": str(eval_root),
        "paired_quality_dir": str(quality_dir),
        "formal_analysis_status": formal_report.get("status"),
        "artifact_audit_status": audit.get("status"),
        "paired_quality_status": quality_report.get("status"),
        "surface_scopes": surface_scopes,
        "upper_scope": args.upper_scope,
        "formats": formats,
        "dpi": args.dpi,
        "maximum_figure_width_inches": FIGURE_WIDTH,
        "single_column_width_inches": SINGLE_COLUMN_WIDTH,
        "source_palette": PAPER_SOURCE_COLORS,
        "figures": {
            key: {
                **FIGURE_SPECS[key],
                "size_inches": list(FIGURE_SIZES[FIGURE_SIZE_KEYS[key]]),
                "files": files,
                "source_data": f"source_data/{key}.csv",
                "file_sha256": {
                    name: sha256_file(out_dir / name) for name in files
                },
                "source_data_sha256": sha256_file(source_dir / f"{key}.csv"),
            }
            for key, files in generated.items()
        },
        "claim_boundary": (
            "Task-specific source-dependent multivariate information quality; not a direct proof "
            "of governing-equation or conservation-law violations in Pangu."
        ),
    }
    (out_dir / "qcore_evidence_story_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {len(generated)} standalone figures to {out_dir}")
    for key, files in generated.items():
        print(f"  {key}: {', '.join(files)}")


if __name__ == "__main__":
    main()
