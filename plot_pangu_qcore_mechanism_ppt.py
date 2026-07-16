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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


# A 13.33 x 7.50 in master scales to a 183-mm two-column paper figure while
# retaining roughly 9--14 pt text, and is also native to a 16:9 presentation.
MASTER_SIZE = (13.333, 7.50)
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
        "font.size": 18,
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
        "axes.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# Source colors deliberately differ in hue and lightness.  The same mapping is
# used in every figure so the story remains visually stable.
PANGU = "#A98DCE"          # light violet: AI forecast source
PANGU_DARK = "#654A8A"
PANGU_PALE = "#F0EBF7"
TIANJI = "#006C75"         # dark teal: numerical-model forecast source
TIANJI_LIGHT = "#65AEB4"
TIANJI_PALE = "#E3F1F2"
ERA5 = "#777777"           # neutral reference-analysis benchmark
ERA5_LIGHT = "#B8B8B8"
ERA5_PALE = "#EEEEEE"
INK = "#202124"
MID_GREY = "#686B70"
LIGHT_GREY = "#D7D9DC"
PALE_GREY = "#F4F5F6"
GRID_GREY = "#E4E6E8"
POSITIVE = "#0B6E4F"

PACKAGE_COLORS = {
    "T2": "#174A7E",
    "M": "#007C82",
    "W": "#7AA6C2",
    "P": "#9A9A9A",
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
    "03_source_block_shapley": {
        "placement": "main",
        "claim": "T2M and moisture dominate the controlled source contribution, wind is smaller, and MSLP is near zero.",
    },
    "04_surface_t2m_quality": {
        "placement": "main",
        "claim": "Tianji 2-m temperature is closer to station observations in the selected quality scope.",
    },
    "05_surface_wspd10_quality": {
        "placement": "main",
        "claim": "Tianji 10-m wind speed is closer to station observations in the selected quality scope.",
    },
    "06_surface_mslp_quality": {
        "placement": "supplement",
        "claim": "MSLP provides a source-quality negative control because its contribution to performance is not robust.",
    },
    "07_pressure_level_quality": {
        "placement": "main",
        "claim": "Pressure-level quality is mixed: Pangu is closer for Q1000 while Tianji is closer for 925-hPa vector wind.",
    },
    "08_unique_event_hits": {
        "placement": "main_or_supplement",
        "claim": "Tianji uniquely detects more low-visibility samples than Pangu at validation-matched operating points.",
    },
    "09_pressure_qc": {
        "placement": "supplement",
        "claim": "Pangu contains a small but explicit fraction of non-physical negative pressure-level specific humidity values.",
    },
}


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
        help="Output directory (default: <eval-root>/evidence_story_figures).",
    )
    parser.add_argument(
        "--quality-scope",
        choices=("all_paired_test", "true_low_visibility", "elevation_le_500m"),
        default="true_low_visibility",
        help="Scope for the station-observation figures.",
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
        help="Comma-separated formats; SVG is always emitted first.",
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
    }
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        path = analysis / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs = {"dtype": {"mask": str}} if key == "metrics" else {}
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
    return report, audit, tables


def load_quality(
    quality_dir: Path,
    quality_scope: str,
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    report = load_json(quality_dir / "paired_source_quality_report.json")
    if report.get("status") != "completed":
        raise ValueError(f"Paired quality status is not completed: {report.get('status')!r}")
    if report.get("analysis_type") != "diagnostic_only_no_training":
        raise ValueError("Paired quality input is not the expected zero-training diagnosis")
    if int(report.get("test_rows", 0)) <= 0 or int(report.get("represented_utc_dates", 0)) <= 1:
        raise ValueError("Paired quality report has no usable paired test sample")
    if quality_scope == "true_low_visibility":
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
    for scope in (quality_scope,):
        if scope not in set(tables["surface"]["scope"]):
            raise ValueError(f"Surface quality table lacks scope={scope}")
    return report, tables


def ordered_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
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
        # Preserve the exact 16:9 master canvas for predictable PPT placement.
        # All layouts reserve explicit margins, so tight-bbox cropping is not
        # needed and would make the exported aspect ratios inconsistent.
        kwargs: Dict[str, object] = {"facecolor": "white"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path.name)
    plt.close(fig)
    return saved


def style_axis(ax: plt.Axes, *, horizontal_grid: bool = True) -> None:
    ax.tick_params(length=5.5, width=1.4, color=INK)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if horizontal_grid:
        ax.yaxis.grid(True, color=GRID_GREY, linewidth=1.0, alpha=0.8)
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


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    edge: str,
    fill: str,
    number: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=2.2,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.add_patch(plt.Circle((x + 0.035, y + height - 0.043), 0.023, color=edge, zorder=3))
    ax.text(
        x + 0.035,
        y + height - 0.043,
        number,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    ax.text(
        x + 0.072,
        y + height - 0.045,
        title,
        ha="left",
        va="center",
        fontsize=16.5,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.03,
        y + height - 0.092,
        body,
        ha="left",
        va="top",
        fontsize=12.5,
        linespacing=1.28,
        color=INK,
    )


def arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = MID_GREY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def plot_flow(out_dir: Path, formats: Sequence[str], dpi: int) -> Tuple[pd.DataFrame, List[str]]:
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.975, bottom=0.025)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.035,
        0.955,
        "Evidence chain for the Pangu–Tianji q-core experiment",
        fontsize=27,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    ax.text(
        0.035,
        0.910,
        "One controlled question per stage; performance attribution is separated from source-quality validation.",
        fontsize=15.5,
        color=MID_GREY,
        ha="left",
        va="center",
    )

    draw_box(
        ax,
        0.035,
        0.625,
        0.29,
        0.215,
        "Fair q-core design",
        "Identical station–time pairs\nIdentical q-core inputs, model and loss\nCanonical Pangu 12–23 h leads",
        "#4F5965",
        PALE_GREY,
        "1",
    )
    draw_box(
        ax,
        0.355,
        0.625,
        0.29,
        0.215,
        "Performance gap",
        "Threshold-free Low-vis AP\nValidation-matched-FPR recall and CSI\n3 seeds + UTC-date bootstrap",
        "#3F5F78",
        "#E9F0F5",
        "2",
    )
    draw_box(
        ax,
        0.675,
        0.625,
        0.29,
        0.215,
        "Factorial attribution",
        "16 retrained source-block combinations\nMoisture / T2M / wind / MSLP\nExact group Shapley contributions",
        "#1C6A70",
        TIANJI_PALE,
        "3",
    )
    arrow(ax, (0.327, 0.733), (0.347, 0.733))
    arrow(ax, (0.647, 0.733), (0.667, 0.733))

    draw_box(
        ax,
        0.115,
        0.315,
        0.34,
        0.205,
        "Near-surface quality",
        "T2M / 10-m wind / MSLP\nMatched automatic-station observations\nSame paired rows + UTC-date CIs",
        TIANJI,
        TIANJI_PALE,
        "4a",
    )
    draw_box(
        ax,
        0.545,
        0.315,
        0.34,
        0.205,
        "Pressure-level quality",
        "Q1000 / Q925 / 925-hPa wind\nPointwise ERA5 reference analysis\nElevation sensitivity + physical QC",
        ERA5,
        ERA5_PALE,
        "4b",
    )
    arrow(ax, (0.83, 0.618), (0.345, 0.528), color="#5B7F82")
    arrow(ax, (0.83, 0.618), (0.715, 0.528), color="#7C7C7C")

    draw_box(
        ax,
        0.255,
        0.075,
        0.58,
        0.145,
        "Bounded mechanism conclusion",
        "Performance, attribution, quality and hit/miss evidence support task-relevant\nsource-information differences—not universal physical-inconsistency proof.",
        PANGU_DARK,
        PANGU_PALE,
        "5",
    )
    arrow(ax, (0.285, 0.307), (0.365, 0.225), color="#5B7F82")
    arrow(ax, (0.715, 0.307), (0.635, 0.225), color="#7C7C7C")

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
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.22)
    for seed in sorted(seed_rows["seed"].astype(int).unique()):
        pair = seed_rows[seed_rows["seed"].astype(int) == seed].set_index("source")
        ax.plot(
            [0, 1],
            [pair.loc["pangu", "value"], pair.loc["tianji", "value"]],
            color=LIGHT_GREY,
            linewidth=2.3,
            zorder=1,
        )
        ax.scatter(0, pair.loc["pangu", "value"], s=115, color=PANGU, alpha=0.62, edgecolor="white", linewidth=1.1, zorder=2)
        ax.scatter(1, pair.loc["tianji", "value"], s=115, color=TIANJI, alpha=0.62, edgecolor="white", linewidth=1.1, zorder=2)

    pangu = float(mean_rows.loc["pangu", "value"])
    tianji = float(mean_rows.loc["tianji", "value"])
    ax.scatter(0, pangu, s=310, marker="D", color=PANGU_DARK, edgecolor="white", linewidth=1.8, zorder=4)
    ax.scatter(1, tianji, s=310, marker="D", color=TIANJI, edgecolor="white", linewidth=1.8, zorder=4)
    ax.text(0, pangu, f"  {pangu:.3f}", ha="left", va="center", fontsize=21, fontweight="bold", color=PANGU_DARK)
    ax.text(1, tianji, f"  {tianji:.3f}", ha="left", va="center", fontsize=21, fontweight="bold", color=TIANJI)

    delta = float(mean_rows.loc["tianji", "delta_tianji_minus_pangu"])
    ci_low = float(mean_rows.loc["tianji", "delta_ci_low"])
    ci_high = float(mean_rows.loc["tianji", "delta_ci_high"])
    values = seed_rows["value"].to_numpy(dtype=float)
    span = max(float(values.max() - values.min()), 0.02)
    ymin = max(0.0, float(values.min() - 0.55 * span))
    ymax = float(values.max() + 1.15 * span)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.45, 1.55)
    annotation_y = ymax - 0.12 * (ymax - ymin)
    ax.text(
        0.5,
        annotation_y,
        f"Tianji − Pangu = {delta:+.3f}   (95% CI {ci_low:+.3f} to {ci_high:+.3f})",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=POSITIVE if delta > 0 else PANGU_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F8F7", edgecolor="#BFD8CF", linewidth=1.2),
    )
    ax.set_xticks([0, 1], ["Pangu-trained", "Tianji-trained"])
    ax.get_xticklabels()[0].set_color(PANGU_DARK)
    ax.get_xticklabels()[1].set_color(TIANJI)
    ax.set_ylabel(metric_label)
    ax.set_title(title, loc="left", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.17,
        f"Small circles: individual training seeds; diamonds: 3-seed means. {protocol_note}\n"
        "95% CI from joint UTC-valid-date block bootstrap; higher is better.",
        transform=ax.transAxes,
        fontsize=14.2,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return save_figure(fig, out_dir / output_name, formats, dpi)


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
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.83, bottom=0.22)
    y = np.arange(len(source))[::-1]
    labels = {
        "T2": "2-m temperature",
        "M": "Moisture package",
        "W": "Wind package",
        "P": "Mean sea-level pressure",
    }
    for yi, row in zip(y, source.itertuples(index=False)):
        color = PACKAGE_COLORS[str(row.group)]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, linewidth=5.0, solid_capstyle="round")
        ax.scatter(row.shapley_mean, yi, s=245, color=color, edgecolor="white", linewidth=1.6, zorder=3)
        ax.text(
            max(0.043, float(row.ci_high) + 0.0025),
            yi,
            f"{row.shapley_mean:+.3f}",
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
    ax.axvline(0.0, color=MID_GREY, linestyle="--", linewidth=1.8)
    ax.set_yticks(y, [labels[group] for group in source["group"]])
    xmin = min(-0.006, float(source["ci_low"].min()) - 0.002)
    xmax = max(0.052, float(source["ci_high"].max()) + 0.014)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Exact source-block Shapley contribution to Low-vis AP")
    ax.set_title("T2M and moisture explain most of the controlled source gap", loc="left", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.17,
        "Dots: 3-seed point estimates; lines: 95% UTC-date block-bootstrap CIs.\n"
        "All 16 source-block combinations were retrained; contributions are package-level attribution, not single-variable causality.",
        transform=ax.transAxes,
        fontsize=14.2,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax, horizontal_grid=False)
    return save_figure(fig, out_dir / "03_source_block_shapley", formats, dpi)


def surface_feature_source(
    surface: pd.DataFrame,
    pairs: pd.DataFrame,
    feature: str,
    scope: str,
) -> pd.DataFrame:
    order = ["pangu", "tianji", "era5_reference_analysis"]
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
    return part


def plot_surface_feature(
    source: pd.DataFrame,
    feature_label: str,
    unit: str,
    scope: str,
    output_name: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.23)
    x = np.arange(len(source))
    for xi, row in zip(x, source.itertuples(index=False)):
        color = SOURCE_COLORS[str(row.source)]
        lo = float(row.rmse - row.ci_low)
        hi = float(row.ci_high - row.rmse)
        ax.errorbar(
            xi,
            row.rmse,
            yerr=np.array([[max(lo, 0.0)], [max(hi, 0.0)]]),
            fmt="o",
            markersize=15,
            color=color,
            ecolor=color,
            elinewidth=3.2,
            capsize=8,
            capthick=2.6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.text(xi, float(row.ci_high) + 0.045 * max(float(source["ci_high"].max()), 1.0), f"{row.rmse:.2f}", ha="center", va="bottom", fontsize=19, fontweight="bold", color=color)

    ymax = float(source["ci_high"].max()) * 1.30
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(-0.55, len(source) - 0.45)
    ax.set_xticks(x, [SOURCE_LABELS[str(item)] for item in source["source"]])
    for tick, key in zip(ax.get_xticklabels(), source["source"]):
        tick.set_color(SOURCE_COLORS[str(key)])
        tick.set_fontweight("bold")
    ax.set_ylabel(f"RMSE against station observations ({unit})")
    ax.set_title(f"{feature_label} quality — {scope_label(scope)}", loc="left", pad=18, fontweight="bold")

    delta = float(source["pangu_minus_tianji"].iloc[0])
    lo = float(source["pair_delta_ci_low"].iloc[0])
    hi = float(source["pair_delta_ci_high"].iloc[0])
    n = int(source["n"].min())
    dates = int(source["represented_utc_dates"].min())
    ax.text(
        0.0,
        -0.18,
        f"Pangu − Tianji ΔRMSE = {delta:+.2f} {unit} (95% CI {lo:+.2f} to {hi:+.2f}).  "
        f"n = {n:,} paired rows; {dates} UTC dates.\n"
        "Automatic-station observations are the reference; ERA5 is an analysis benchmark, not a third forecast.",
        transform=ax.transAxes,
        fontsize=14.0,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
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
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.28, right=0.95, top=0.82, bottom=0.23)
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        value = float(row.normalized_delta_percent)
        lo = float(row.normalized_ci_low_percent)
        hi = float(row.normalized_ci_high_percent)
        color = TIANJI if value > 0 else PANGU_DARK
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=5.0, solid_capstyle="round")
        ax.scatter(value, yi, s=245, color=color, edgecolor="white", linewidth=1.6, zorder=3)
        label_x = hi + 0.9 if value >= 0 else lo - 0.9
        align = "left" if value >= 0 else "right"
        ax.text(label_x, yi, f"{value:+.1f}%", ha=align, va="center", fontsize=18, fontweight="bold", color=color)
    ax.axvline(0.0, color=INK, linewidth=1.8)
    ax.set_yticks(y, source["label"])
    xmin = min(-20.0, float(source["normalized_ci_low_percent"].min()) - 5.0)
    xmax = max(25.0, float(source["normalized_ci_high_percent"].max()) + 5.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Paired RMSE difference relative to Pangu (%)")
    ax.set_title("Pressure-level quality is source- and variable-dependent", loc="left", pad=24, fontweight="bold")
    ax.text(0.17, 1.012, "← Pangu closer", transform=ax.transAxes, fontsize=16, fontweight="bold", color=PANGU_DARK, ha="center")
    ax.text(0.83, 1.012, "Tianji closer →", transform=ax.transAxes, fontsize=16, fontweight="bold", color=TIANJI, ha="center")
    scope = str(source["scope"].iloc[0])
    n = int(source["n"].min())
    dates = int(source["represented_utc_dates"].min())
    ax.text(
        0.0,
        -0.18,
        f"{scope_label(scope)}; n = {n:,}; {dates} UTC dates; 95% UTC-date block-bootstrap CIs.\n"
        "Positive values mean lower Tianji RMSE; ERA5 is a reference analysis.\n"
        "ERA5 Q is derived and is not independent truth.",
        transform=ax.transAxes,
        fontsize=14.0,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax, horizontal_grid=False)
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
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.83, bottom=0.22)
    y = np.arange(len(source))[::-1]
    bars = ax.barh(y, source["n"], color=source["color"], height=0.54, edgecolor="none")
    for bar, n in zip(bars, source["n"]):
        ax.text(n + 110, bar.get_y() + bar.get_height() / 2, f"{int(n):,}", ha="left", va="center", fontsize=22, fontweight="bold", color=INK)
    ratio = float(source["tianji_to_pangu_unique_hit_ratio"].iloc[0])
    ax.text(0.74, 0.46, f"{ratio:.2f}×", transform=ax.transAxes, fontsize=40, fontweight="bold", color=TIANJI, ha="center")
    ax.text(0.74, 0.37, "more unique hits", transform=ax.transAxes, fontsize=18, color=INK, ha="center")
    ax.set_yticks(y, source["label"])
    ax.set_xlim(0, max(source["n"]) * 1.30)
    ax.set_xlabel("True Low-vis samples detected by only one endpoint model")
    ax.set_title("Tianji uniquely detects more low-visibility cases than Pangu", loc="left", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.16,
        "Seed-mean endpoint probabilities; source-specific thresholds were fixed on validation at a common target FPR.",
        transform=ax.transAxes,
        fontsize=14.2,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return save_figure(fig, out_dir / "08_unique_event_hits", formats, dpi)


def qc_source(qc: pd.DataFrame) -> pd.DataFrame:
    order = ["Q_1000", "Q_925"]
    sources = ["pangu", "tianji", "era5_reference_analysis"]
    part = qc[qc["feature"].isin(order) & qc["source"].isin(sources)].copy()
    part["outside_percent"] = 100.0 * pd.to_numeric(part["outside_broad_range_fraction"], errors="raise")
    part["feature"] = pd.Categorical(part["feature"], categories=order, ordered=True)
    part["source"] = pd.Categorical(part["source"], categories=sources, ordered=True)
    return part.sort_values(["feature", "source"]).reset_index(drop=True)


def plot_qc(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=MASTER_SIZE)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.23)
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
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, label, ha="center", va="bottom", fontsize=15, color=SOURCE_COLORS[source_key])
    ymax = max(float(source["outside_percent"].max()) * 1.35, 0.08)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x, ["Q at 1000 hPa", "Q at 925 hPa"])
    ax.set_ylabel(r"Values outside 0–80 g kg$^{-1}$ (%)")
    ax.set_title("Non-physical negative specific humidity is rare but explicit in Pangu", loc="left", pad=18, fontweight="bold")
    ax.legend(loc="upper right", ncol=3, columnspacing=1.4, handletextpad=0.5)
    ax.text(
        0.0,
        -0.17,
        "All broad-range failures are retained in the primary RMSE and reported here; derived dew-point failures are not double-counted.\n"
        "This QC documents a source defect but does not by itself establish the cause of the model-performance gap.",
        transform=ax.transAxes,
        fontsize=14.0,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return save_figure(fig, out_dir / "09_pressure_qc", formats, dpi)


def write_guide(
    out_dir: Path,
    formal_report: Mapping[str, object],
    quality_report: Mapping[str, object],
    quality_scope: str,
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
8. `08_unique_event_hits`: optional event-level closure.

Move `06_surface_mslp_quality` and `09_pressure_qc` to the supplement unless a
reviewer specifically asks for the negative control or QC rate in the main text.

## Formal claim boundary

The experiment supports a task-specific, source-dependent multivariate
information-quality difference for Pangu-2025, 2025, 12--23 h lead and the
current low-visibility task. It does not prove that Pangu violates governing
equations, and it must not be generalized to all AI weather models.

## Figure contract

- master size: 13.333 x 7.50 in (16:9; scalable to 183-mm paper width)
- typography: editable sans-serif text in SVG/PDF
- station-observation scope: `{quality_scope}` ({scope_label(quality_scope)})
- pressure-level scope: `{upper_scope}` ({scope_label(upper_scope)})
- surface uncertainty: 95% UTC-valid-date block bootstrap
- performance variability: three training seeds plus UTC-date bootstrap
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
    out_dir = (args.out_dir or (eval_root / "evidence_story_figures")).expanduser().resolve()
    formats = ordered_formats(args.formats)
    if args.dpi < 300:
        raise ValueError("Use --dpi >= 300; 600 is recommended for paper TIFF export")

    formal_report, audit, formal = load_formal(eval_root, args.allow_missing_artifact_audit)
    quality_report, quality = load_quality(quality_dir, args.quality_scope)
    print(
        "[validation] formal and paired-quality inputs passed: "
        f"mt2pw={formal_report.get('status')} artifact_audit={audit.get('status')} "
        f"quality={quality_report.get('status')} rows={quality_report.get('test_rows')}"
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
        "Tianji-trained q-core models achieve higher threshold-free Low-vis AP",
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
        "Tianji recovers more Low-vis cases at a validation-matched false-alarm protocol",
        "02_qcore_matched_fpr_recall",
        out_dir,
        formats,
        args.dpi,
        "Thresholds were selected on validation at one target FPR and then frozen for test.",
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
            quality["surface"], quality["surface_pairs"], feature, args.quality_scope
        )
        source.to_csv(source_dir / f"{output_name}.csv", index=False)
        generated[output_name] = plot_surface_feature(
            source,
            label,
            unit,
            args.quality_scope,
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

    if not args.main_only:
        qc = qc_source(quality["qc"])
        qc.to_csv(source_dir / "09_pressure_qc.csv", index=False)
        generated["09_pressure_qc"] = plot_qc(qc, out_dir, formats, args.dpi)

    write_guide(out_dir, formal_report, quality_report, args.quality_scope, args.upper_scope)
    manifest = {
        "status": "passed",
        "eval_root": str(eval_root),
        "paired_quality_dir": str(quality_dir),
        "formal_analysis_status": formal_report.get("status"),
        "artifact_audit_status": audit.get("status"),
        "paired_quality_status": quality_report.get("status"),
        "quality_scope": args.quality_scope,
        "upper_scope": args.upper_scope,
        "formats": formats,
        "dpi": args.dpi,
        "master_size_inches": list(MASTER_SIZE),
        "figures": {
            key: {
                **FIGURE_SPECS[key],
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
