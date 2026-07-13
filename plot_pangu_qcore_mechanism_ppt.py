#!/usr/bin/env python3
"""Create one-claim-per-figure PPT graphics for the formal Pangu q-core study.

The script is intentionally independent of Torch.  It consumes only the formal
``mt2pw`` analysis tables and artifact audit, rejects smoke/incomplete runs, and
exports editable SVG plus PDF/PNG previews.  Each output is a standalone figure
with one scientific message; no composite subplot is produced.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


# Mandatory editable-vector and presentation typography settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Liberation Sans",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 19
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 17
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


PANGU = "#8E78B7"       # lighter violet: AI forecast source
PANGU_DARK = "#5D477F"
TIANJI = "#0F6B78"      # darker teal: NWP-derived forecast source
TIANJI_LIGHT = "#58A6AE"
INK = "#252525"
MID_GREY = "#747474"
LIGHT_GREY = "#D8D8D8"
PALE_GREY = "#F2F3F5"
POSITIVE = "#166F5B"
NEGATIVE = "#8C5A63"

DEFAULT_EVAL_ROOT = Path(
    "/public/home/putianshu/vis_mlp/"
    "paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/"
    "qcore_hybrid_mt2pw_formal_v1_20260708"
)

PRIMARY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("low_vis_ap", "Low-vis AP"),
    ("low_vis_csi_matched_fpr", "Low-vis CSI"),
    ("low_vis_recall_matched_fpr", "Low-vis recall"),
)

FIGURE_SPECS: Mapping[str, Mapping[str, str]] = {
    "00_weather_model_principle": {
        "claim": "NWP constrains atmospheric evolution explicitly, whereas Pangu learns the state transition statistically.",
        "role": "Background mechanism hypothesis; not direct evidence from this experiment.",
    },
    "01_qcore_fair_performance": {
        "claim": "Tianji-trained q-core models outperform Pangu-trained models under the same predictors and operating-point protocol.",
        "role": "Primary fair-performance evidence.",
    },
    "02_source_block_shapley": {
        "claim": "T2M and moisture are the two dominant positive source blocks; wind is smaller and MSLP is near zero.",
        "role": "Controlled hybrid-retraining attribution.",
    },
    "03_lowvis_observation_quality": {
        "claim": "During observed low-visibility cases, Tianji has lower T2M, 10-m wind-speed and MSLP MAE than Pangu.",
        "role": "Observation-anchored source-quality evidence.",
    },
    "04_unique_event_hits": {
        "claim": "At validation-matched FPR, Tianji uniquely detects more than twice as many low-visibility samples as Pangu.",
        "role": "Event-level closure of the performance gap.",
    },
    "05_moisture_reference_caveat": {
        "claim": "Marginal Q error against ERA5 reference analysis does not uniformly favour Tianji.",
        "role": "Limitation: moisture attribution is not explained by marginal Q MAE alone.",
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
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <eval-root>/ppt_figures).",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="Comma-separated output formats. SVG is always emitted first.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--skip-principle",
        action="store_true",
        help="Skip the conceptual NWP-versus-Pangu mechanism schematic.",
    )
    parser.add_argument(
        "--allow-missing-artifact-audit",
        action="store_true",
        help="Permit plotting without artifact_audit (not recommended for formal figures).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all formal-input checks without drawing figures.",
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


def load_and_validate(
    eval_root: Path,
    allow_missing_artifact_audit: bool,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, pd.DataFrame]]:
    analysis = eval_root / "analysis"
    required_files = {
        "metrics": "hybrid_factorial_metrics_by_seed.csv",
        "shapley": "hybrid_exact_shapley_effects.csv",
        "bootstrap": "hybrid_date_block_bootstrap_ci.csv",
        "gap_draws": "hybrid_total_gap_bootstrap_draws.csv.gz",
        "quality": "observation_anchored_source_quality.csv",
        "events": "event_case_control_environment_summary.csv",
        "q_reference": "q_reference_analysis_quality_and_extreme_placement.csv",
    }
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
    if report.get("all0_mask") != "0000" or report.get("all1_mask") != "1111":
        raise ValueError("Formal endpoint masks must be 0000 and 1111")
    bootstrap_info = report.get("bootstrap", {})
    if not isinstance(bootstrap_info, dict) or bootstrap_info.get("unit") != "UTC_valid_date":
        raise ValueError("Formal uncertainty must use UTC_valid_date block bootstrap")
    if int(bootstrap_info.get("iterations", 0)) != 1000:
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
            raise ValueError(f"Artifact audit is not the passed 51/51 formal mt2pw matrix: {audit}")
    elif allow_missing_artifact_audit:
        audit = {"status": "missing_by_explicit_override"}
    else:
        raise FileNotFoundError(
            f"Missing {audit_path}; use --allow-missing-artifact-audit only for non-formal previews"
        )

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in required_files.items():
        path = analysis / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs = {"dtype": {"mask": str}} if key == "metrics" else {}
        tables[key] = pd.read_csv(path, **kwargs)

    metrics = tables["metrics"]
    require_columns(
        metrics,
        ["seed", "mask", *(metric for metric, _ in PRIMARY_METRICS)],
        "hybrid_factorial_metrics_by_seed.csv",
    )
    metrics["mask"] = normalized_mask(metrics["mask"])
    if set(metrics["seed"].astype(int)) != {42, 2025, 20260702}:
        raise ValueError("Metrics do not contain the three formal seeds")
    expected_pairs = {(seed, mask) for seed in (42, 2025, 20260702) for mask in masks}
    actual_pairs = set(zip(metrics["seed"].astype(int), metrics["mask"]))
    if actual_pairs != expected_pairs:
        raise ValueError("Metrics do not contain exactly one complete 3-seed x 16-mask matrix")

    require_columns(
        tables["shapley"],
        ["metric", "group", "group_label", "shapley_mean", "ci_low", "ci_high"],
        "hybrid_exact_shapley_effects.csv",
    )
    require_columns(
        tables["gap_draws"],
        ["iteration", "metric", "delta_all1_minus_all0"],
        "hybrid_total_gap_bootstrap_draws.csv.gz",
    )
    if tables["gap_draws"]["iteration"].nunique() != 1000:
        raise ValueError("Gap-draw table must contain 1000 bootstrap iterations")
    require_columns(
        tables["quality"],
        ["feature", "source", "scope", "n", "bias", "mae", "rmse", "correlation"],
        "observation_anchored_source_quality.csv",
    )
    require_columns(tables["events"], ["case_category", "n"], "event_case_control_environment_summary.csv")
    require_columns(
        tables["q_reference"],
        ["feature", "source", "analysis", "mae", "correlation"],
        "q_reference_analysis_quality_and_extreme_placement.csv",
    )
    return report, audit, tables


def ordered_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png"}
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported formats: {unknown}; supported={sorted(supported)}")
    result = ["svg"]
    result.extend(fmt for fmt in requested if fmt != "svg")
    return list(dict.fromkeys(result))


def finish(fig: plt.Figure, base: Path, formats: Sequence[str], dpi: int, *, tight: bool = True) -> List[str]:
    if tight:
        fig.tight_layout(pad=1.4)
    saved: List[str] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path.name)
    plt.close(fig)
    return saved


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(length=5, width=1.3, color=INK)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.title.set_color(INK)


def plot_principle(out_dir: Path, formats: Sequence[str], dpi: int) -> Tuple[pd.DataFrame, List[str]]:
    fig, ax = plt.subplots(figsize=(13.0, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.93,
        "NWP and Pangu advance the atmosphere in fundamentally different ways",
        fontsize=27,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )

    lanes = [
        {
            "y": 0.62,
            "label": "Numerical weather prediction",
            "color": TIANJI,
            "fill": "#E3F0F1",
            "middle": "Dynamical core\n+ physical parameterizations",
            "callout": "Explicitly integrates governing equations; consistency is constrained, not perfect.",
        },
        {
            "y": 0.29,
            "label": "Pangu-Weather",
            "color": PANGU_DARK,
            "fill": "#EEE9F5",
            "middle": "3D Earth-specific\nTransformer",
            "callout": "Learns joint ERA5 transitions; no explicit PDE integration or conservation guarantee.",
        },
    ]
    for lane in lanes:
        y = float(lane["y"])
        color = str(lane["color"])
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y - 0.125),
                0.93,
                0.245,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0,
                facecolor=str(lane["fill"]),
                zorder=0,
            )
        )
        ax.text(0.055, y + 0.092, str(lane["label"]), fontsize=20, fontweight="bold", color=color, va="center")
        boxes = [
            (0.225, "Initial atmospheric\nstate (analysis)"),
            (0.515, str(lane["middle"])),
            (0.805, "Forecast atmospheric\nstate"),
        ]
        for x, text in boxes:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.102, y - 0.070),
                    0.204,
                    0.096,
                    boxstyle="round,pad=0.011,rounding_size=0.012",
                    linewidth=1.8,
                    edgecolor=color,
                    facecolor="white",
                )
            )
            ax.text(x, y - 0.022, text, fontsize=17, color=INK, ha="center", va="center", linespacing=1.2)
        for x0, x1 in ((0.332, 0.407), (0.622, 0.697)):
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y - 0.022),
                    (x1, y - 0.022),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=2.1,
                    color=color,
                )
            )
        ax.text(0.515, y - 0.103, str(lane["callout"]), fontsize=13.2, color=MID_GREY, ha="center", va="center")

    ax.text(
        0.5,
        0.055,
        "Mechanism hypothesis for this study: low-visibility prediction depends on the joint T–q–wind structure, not only marginal errors.",
        fontsize=16,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
    )
    source = pd.DataFrame(
        [
            {
                "system": "Numerical weather prediction",
                "state_transition": "explicit numerical integration of governing equations plus parameterizations",
                "physical_consistency": "explicitly constrained but not perfect",
            },
            {
                "system": "Pangu-Weather",
                "state_transition": "3D Earth-specific Transformer learned from ERA5 state pairs",
                "physical_consistency": "implicit/statistical; no explicit PDE integration or conservation guarantee",
            },
        ]
    )
    files = finish(fig, out_dir / "00_weather_model_principle", formats, dpi, tight=False)
    return source, files


def performance_source(metrics: pd.DataFrame, gap_draws: pd.DataFrame) -> pd.DataFrame:
    endpoint = metrics.groupby("mask", sort=True)[[metric for metric, _ in PRIMARY_METRICS]].mean()
    rows: List[Dict[str, object]] = []
    for metric, label in PRIMARY_METRICS:
        draws = pd.to_numeric(
            gap_draws.loc[gap_draws["metric"] == metric, "delta_all1_minus_all0"], errors="coerce"
        ).dropna()
        if len(draws) != 1000:
            raise ValueError(f"{metric}: expected 1000 finite total-gap bootstrap draws, got {len(draws)}")
        pangu = float(endpoint.loc["0000", metric])
        tianji = float(endpoint.loc["1111", metric])
        rows.append(
            {
                "metric": metric,
                "label": label,
                "pangu_mean": pangu,
                "tianji_mean": tianji,
                "delta_tianji_minus_pangu": tianji - pangu,
                "delta_ci_low": float(draws.quantile(0.025)),
                "delta_ci_high": float(draws.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def plot_performance(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        pangu = float(row.pangu_mean)
        tianji = float(row.tianji_mean)
        ci_lo = pangu + float(row.delta_ci_low)
        ci_hi = pangu + float(row.delta_ci_high)
        ax.plot([pangu, tianji], [yi, yi], color=LIGHT_GREY, linewidth=8, solid_capstyle="round", zorder=1)
        ax.plot([ci_lo, ci_hi], [yi, yi], color=TIANJI_LIGHT, linewidth=3.5, solid_capstyle="round", zorder=2)
        ax.scatter(pangu, yi, s=180, color=PANGU, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(tianji, yi, s=180, color=TIANJI, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(pangu - 0.012, yi + 0.17, f"{pangu:.3f}", ha="right", va="center", fontsize=15, color=PANGU_DARK)
        ax.text(tianji + 0.012, yi + 0.17, f"{tianji:.3f}", ha="left", va="center", fontsize=15, color=TIANJI)
        ax.text(
            0.825,
            yi,
            f"Δ {row.delta_tianji_minus_pangu:+.3f}\n[{row.delta_ci_low:.3f}, {row.delta_ci_high:.3f}]",
            ha="left",
            va="center",
            fontsize=15,
            color=INK,
            linespacing=1.25,
        )
    ax.set_yticks(y, source["label"])
    ax.set_xlim(0.13, 0.96)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Test-set score (higher is better)")
    ax.set_title("Tianji improves all primary q-core endpoints at comparable false-alarm rates", loc="left", pad=18, fontweight="bold")
    ax.scatter([], [], s=150, color=PANGU, label="Pangu-trained")
    ax.scatter([], [], s=150, color=TIANJI, label="Tianji-trained")
    ax.legend(loc="lower right", ncol=2, handletextpad=0.5, columnspacing=1.4)
    ax.text(
        0.0,
        -0.22,
        "Mean across 3 seeds; brackets are 95% UTC-date block-bootstrap CIs for Tianji − Pangu.\n"
        "CSI and recall use thresholds selected on validation at a common target FPR.",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return finish(fig, out_dir / "01_qcore_fair_performance", formats, dpi)


def shapley_source(shapley: pd.DataFrame) -> pd.DataFrame:
    order = ["T2", "M", "W", "P"]
    part = shapley[shapley["metric"] == "low_vis_ap"].copy()
    if set(part["group"]) != set(order):
        raise ValueError(f"Low-vis AP Shapley rows must be {order}, got {sorted(part['group'].astype(str))}")
    part = part.set_index("group").loc[order].reset_index()
    total_gap = float(part["shapley_mean"].sum())
    part["share_of_total_gap_percent"] = 100.0 * part["shapley_mean"] / total_gap
    return part[
        [
            "group",
            "group_label",
            "shapley_mean",
            "shapley_seed_sd",
            "ci_low",
            "ci_high",
            "share_of_total_gap_percent",
        ]
    ]


def plot_shapley(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    y = np.arange(len(source))[::-1]
    colors = ["#0F4D92", "#42949E", "#8FBAD9", "#9B9B9B"]
    for yi, row, color in zip(y, source.itertuples(index=False), colors):
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, linewidth=4.2, solid_capstyle="round")
        ax.scatter(row.shapley_mean, yi, s=210, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(
            0.047,
            yi,
            f"{row.shapley_mean:+.3f}  ({row.share_of_total_gap_percent:+.1f}%)",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
        )
    ax.axvline(0.0, color=MID_GREY, linestyle="--", linewidth=1.6)
    ax.set_yticks(y, ["2-m temperature", "Moisture", "Wind", "Mean sea-level pressure"])
    ax.set_xlim(-0.007, 0.069)
    ax.set_ylim(-0.55, len(source) - 0.35)
    ax.set_xlabel("Exact group Shapley contribution to Low-vis AP")
    ax.set_title("T2M and moisture jointly explain most of the source-performance gap", loc="left", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.19,
        "Dots: 3-seed point estimates; lines: 95% UTC-date block-bootstrap CIs.\n"
        "Controlled block replacement and retraining; contributions are not single-variable causal effects.",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return finish(fig, out_dir / "02_source_block_shapley", formats, dpi)


def quality_source(quality: pd.DataFrame) -> pd.DataFrame:
    features = [
        ("T2M", "2-m temperature", "K"),
        ("WSPD10", "10-m wind speed", r"m s$^{-1}$"),
        ("MSLP", "Mean sea-level pressure", "hPa"),
    ]
    part = quality[quality["scope"] == "true_low_visibility"].copy()
    rows: List[Dict[str, object]] = []
    for feature, label, unit in features:
        item = part[part["feature"] == feature].set_index("source")
        if not {"pangu", "tianji"}.issubset(item.index):
            raise ValueError(f"Missing paired low-visibility quality rows for {feature}")
        pangu = float(item.loc["pangu", "mae"])
        tianji = float(item.loc["tianji", "mae"])
        rows.append(
            {
                "feature": feature,
                "label": label,
                "unit": unit,
                "n_pangu": int(item.loc["pangu", "n"]),
                "n_tianji": int(item.loc["tianji", "n"]),
                "pangu_mae": pangu,
                "tianji_mae": tianji,
                "tianji_relative_mae_percent": 100.0 * tianji / pangu,
                "tianji_mae_reduction_percent": 100.0 * (pangu - tianji) / pangu,
            }
        )
    return pd.DataFrame(rows)


def plot_quality(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    y = np.arange(len(source))[::-1]
    for yi, row in zip(y, source.itertuples(index=False)):
        ratio = float(row.tianji_relative_mae_percent)
        ax.plot([ratio, 100.0], [yi, yi], color=LIGHT_GREY, linewidth=8, solid_capstyle="round", zorder=1)
        ax.scatter(100.0, yi, s=180, color=PANGU, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(ratio, yi, s=180, color=TIANJI, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(
            104.0,
            yi,
            f"−{row.tianji_mae_reduction_percent:.1f}%",
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=POSITIVE,
        )
        ax.text(
            38.0,
            yi - 0.25,
            f"MAE: {row.pangu_mae:.2f} vs {row.tianji_mae:.2f} {row.unit}",
            ha="left",
            va="center",
            fontsize=14.5,
            color=MID_GREY,
        )
    ax.axvline(100.0, color=PANGU_DARK, linestyle=":", linewidth=1.4, alpha=0.7)
    ax.set_yticks(y, source["label"])
    ax.set_xlim(35, 128)
    ax.set_ylim(-0.65, len(source) - 0.3)
    ax.set_xlabel("MAE relative to Pangu (%)   —   lower is better")
    ax.set_title("Tianji is closer to station observations during low-visibility conditions", loc="left", pad=18, fontweight="bold")
    ax.scatter([], [], s=150, color=PANGU, label="Pangu = 100%")
    ax.scatter([], [], s=150, color=TIANJI, label="Tianji")
    ax.legend(loc="lower right", ncol=2, handletextpad=0.5, columnspacing=1.4)
    ax.text(
        0.0,
        -0.19,
        "Paired 2025 test samples with observed visibility < 1 km. Error reductions are descriptive source-quality comparisons.",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return finish(fig, out_dir / "03_lowvis_observation_quality", formats, dpi)


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
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    y = np.arange(len(source))[::-1]
    bars = ax.barh(y, source["n"], color=source["color"], height=0.56, edgecolor="none")
    for bar, n in zip(bars, source["n"]):
        ax.text(n + 110, bar.get_y() + bar.get_height() / 2, f"{int(n):,}", ha="left", va="center", fontsize=21, fontweight="bold", color=INK)
    ratio = float(source["tianji_to_pangu_unique_hit_ratio"].iloc[0])
    ax.text(
        0.69,
        0.50,
        f"{ratio:.2f}×",
        transform=ax.transAxes,
        fontsize=34,
        fontweight="bold",
        color=TIANJI,
        ha="center",
        va="center",
    )
    ax.text(0.69, 0.39, "more unique hits", transform=ax.transAxes, fontsize=17, color=INK, ha="center", va="center")
    ax.set_yticks(y, source["label"])
    ax.set_xlim(0, 5200)
    ax.set_xlabel("True Low-vis test samples detected by only one source model")
    ax.set_title("Tianji recovers substantially more low-visibility cases missed by Pangu", loc="left", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.20,
        "Seed-mean probabilities; source-specific thresholds fixed on validation to a common target FPR.",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return finish(fig, out_dir / "04_unique_event_hits", formats, dpi)


def q_source(q_reference: pd.DataFrame) -> pd.DataFrame:
    part = q_reference[q_reference["analysis"] == "continuous_error_vs_reference_analysis"].copy()
    wanted = ["Q_1000", "Q_925"]
    rows: List[Dict[str, object]] = []
    for feature in wanted:
        item = part[part["feature"] == feature].set_index("source")
        if not {"pangu", "tianji"}.issubset(item.index):
            raise ValueError(f"Missing Pangu/Tianji ERA5-reference continuous-error rows for {feature}")
        rows.append(
            {
                "feature": feature,
                "label": feature.replace("_", " at ") + " hPa" if "_" in feature else feature,
                "pangu_mae_g_kg": float(item.loc["pangu", "mae"]),
                "tianji_mae_g_kg": float(item.loc["tianji", "mae"]),
                "pangu_correlation": float(item.loc["pangu", "correlation"]),
                "tianji_correlation": float(item.loc["tianji", "correlation"]),
            }
        )
    result = pd.DataFrame(rows)
    result["label"] = ["Specific humidity at 1000 hPa", "Specific humidity at 925 hPa"]
    return result


def plot_q_caveat(source: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    y = np.arange(len(source))[::-1]
    offset = 0.13
    ax.barh(y + offset, source["pangu_mae_g_kg"], height=0.24, color=PANGU, label="Pangu")
    ax.barh(y - offset, source["tianji_mae_g_kg"], height=0.24, color=TIANJI, label="Tianji")
    for yi, row in zip(y, source.itertuples(index=False)):
        ax.text(row.pangu_mae_g_kg + 0.035, yi + offset, f"{row.pangu_mae_g_kg:.3f}", va="center", ha="left", fontsize=16, color=PANGU_DARK)
        ax.text(row.tianji_mae_g_kg + 0.035, yi - offset, f"{row.tianji_mae_g_kg:.3f}", va="center", ha="left", fontsize=16, color=TIANJI)
    ax.set_yticks(y, source["label"])
    ax.set_xlim(0, 2.55)
    ax.set_xlabel(r"MAE against ERA5 reference analysis (g kg$^{-1}$)   —   lower is better")
    ax.set_title("Marginal moisture error does not uniformly favour Tianji", loc="left", pad=18, fontweight="bold")
    ax.legend(loc="lower right", ncol=2, columnspacing=1.4)
    ax.text(
        0.0,
        -0.20,
        "ERA5 is a reference analysis, not truth. Q1000 favours Pangu slightly; Q925 is nearly tied.\n"
        "Therefore the positive moisture Shapley effect cannot be attributed to marginal Q MAE alone.",
        transform=ax.transAxes,
        fontsize=13.5,
        color=MID_GREY,
        va="top",
    )
    style_axis(ax)
    return finish(fig, out_dir / "05_moisture_reference_caveat", formats, dpi)


def write_guide(out_dir: Path, report: Mapping[str, object], audit: Mapping[str, object]) -> None:
    text = """# Pangu q-core PPT figure guide

All figures are standalone (no composite subplots). Use SVG in PowerPoint when
editability matters; use PNG for quick preview and PDF for print/archive.

## Recommended story order

1. `00_weather_model_principle`: architecture context and the physical-consistency hypothesis. Do not present this as an experimentally proven mechanism.
2. `01_qcore_fair_performance`: common q-core inputs still leave a significant Tianji advantage in AP, matched-FPR CSI and recall.
3. `02_source_block_shapley`: controlled hybrid retraining localizes the gain mainly to T2M and moisture, with a smaller wind contribution and no robust MSLP contribution.
4. `03_lowvis_observation_quality`: during observed low-visibility conditions Tianji is closer to station observations for T2M, 10-m wind speed and MSLP.
5. `04_unique_event_hits`: the aggregate gain closes at event level; Tianji-only hits outnumber Pangu-only hits by about 2.16 to 1.
6. `05_moisture_reference_caveat` (discussion/supplement): marginal Q errors against ERA5 reference analysis do not uniformly favour Tianji, so the joint-structure hypothesis remains plausible but unproven.

## Claim boundary

The formal experiment supports a source-dependent multivariate information-quality
gap for this 2025, 12--23 h, low-visibility task. It does not by itself prove that
Pangu violates governing equations, nor that all AI weather models have weaker
physical consistency. A direct physical-consistency claim would require additional
joint-balance or conservation diagnostics on gridded fields.

## Provenance

"""
    text += f"- analysis status: `{report.get('status')}`\n"
    text += f"- group profile: `{report.get('group_profile')}`\n"
    text += f"- seeds: `{report.get('seeds')}`\n"
    text += f"- bootstrap: `{report.get('bootstrap')}`\n"
    text += f"- artifact audit: `{audit}`\n"
    (out_dir / "PPT_FIGURE_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    out_dir = (args.out_dir or (eval_root / "ppt_figures")).expanduser().resolve()
    formats = ordered_formats(args.formats)
    if args.dpi < 150:
        raise ValueError("Use --dpi >= 150; 300 is recommended for PPT previews")
    report, audit, tables = load_and_validate(eval_root, args.allow_missing_artifact_audit)
    print(
        "[validation] formal mt2pw inputs passed: "
        f"status={report['status']} seeds={report['seeds']} masks={len(report['masks'])} "
        f"artifact_audit={audit.get('status')}"
    )
    if args.validate_only:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, List[str]] = {}

    if not args.skip_principle:
        principle, files = plot_principle(out_dir, formats, args.dpi)
        principle.to_csv(source_dir / "00_weather_model_principle.csv", index=False)
        generated["00_weather_model_principle"] = files

    performance = performance_source(tables["metrics"], tables["gap_draws"])
    performance.to_csv(source_dir / "01_qcore_fair_performance.csv", index=False)
    generated["01_qcore_fair_performance"] = plot_performance(performance, out_dir, formats, args.dpi)

    shapley = shapley_source(tables["shapley"])
    shapley.to_csv(source_dir / "02_source_block_shapley.csv", index=False)
    generated["02_source_block_shapley"] = plot_shapley(shapley, out_dir, formats, args.dpi)

    quality = quality_source(tables["quality"])
    quality.to_csv(source_dir / "03_lowvis_observation_quality.csv", index=False)
    generated["03_lowvis_observation_quality"] = plot_quality(quality, out_dir, formats, args.dpi)

    events = event_source(tables["events"])
    events.drop(columns=["color"]).to_csv(source_dir / "04_unique_event_hits.csv", index=False)
    generated["04_unique_event_hits"] = plot_events(events, out_dir, formats, args.dpi)

    q_reference = q_source(tables["q_reference"])
    q_reference.to_csv(source_dir / "05_moisture_reference_caveat.csv", index=False)
    generated["05_moisture_reference_caveat"] = plot_q_caveat(q_reference, out_dir, formats, args.dpi)

    write_guide(out_dir, report, audit)
    manifest = {
        "status": "passed",
        "eval_root": str(eval_root),
        "formal_analysis_status": report.get("status"),
        "artifact_audit_status": audit.get("status"),
        "group_profile": report.get("group_profile"),
        "seeds": report.get("seeds"),
        "target_validation_fpr": report.get("target_validation_fpr"),
        "formats": formats,
        "dpi": args.dpi,
        "figures": {
            key: {**FIGURE_SPECS[key], "files": files, "source_data": f"source_data/{key}.csv"}
            for key, files in generated.items()
        },
        "literature_context": {
            "pangu_architecture": "https://doi.org/10.1038/s41586-023-06185-3",
            "mlwp_physical_consistency_limitations": "https://doi.org/10.1029/2023GL107377",
            "hybrid_dynamical_core_contrast": "https://doi.org/10.1038/s41586-024-07744-y",
        },
        "claim_boundary": (
            "The current experiment supports source-dependent multivariate information quality, "
            "not a direct proof of governing-equation or conservation-law violations in Pangu."
        ),
    }
    (out_dir / "ppt_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {len(generated)} standalone figures to {out_dir}")
    for key, files in generated.items():
        print(f"  {key}: {', '.join(files)}")


if __name__ == "__main__":
    main()
