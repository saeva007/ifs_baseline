#!/usr/bin/env python3
"""Plot common-variable RMSE and bias for all-test and Low-vis regimes.

Different variables retain their native units in the exported source table.
The main figure uses two unitless, paired quantities so all variables can be
shown without visually privileging variables with large numerical units:

* Tianji RMSE / Pangu RMSE, with paired UTC-date bootstrap intervals;
* signed bias / RMSE for each source, with UTC-date bootstrap bias intervals.

Every panel is exported independently as well as in the full four-panel page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_source_palette import (
    SOURCE_COLORS,
    SOURCE_DARK_COLORS,
    SOURCE_LIGHT_COLORS,
    SOURCE_MARKERS,
)


FIGURE_SIZE = (7.205, 5.25)
WHITE = "#FFFFFF"
INK = "#17191B"
MID = "#697075"
LIGHT = "#D9DCDE"
GRID = "#E8EAEB"
SURFACE_BAND = "#F5F8FA"
PRESSURE_BAND = "#F7F4FA"
SCOPES = ("all_paired_test", "true_low_visibility")
SCOPE_LABELS = {
    "all_paired_test": "All paired test samples",
    "true_low_visibility": "Observed Low-vis (<1 km)",
}
FEATURE_SPECS: Tuple[Mapping[str, str], ...] = (
    {
        "feature": "T2M",
        "label": "2-m temperature",
        "short": "T2M",
        "family": "surface",
        "reference": "Station observations",
    },
    {
        "feature": "WSPD10",
        "label": "10-m wind speed",
        "short": "WSPD10",
        "family": "surface",
        "reference": "Station observations",
    },
    {
        "feature": "MSLP",
        "label": "Mean sea-level pressure",
        "short": "MSLP",
        "family": "surface",
        "reference": "Station observations",
    },
    {
        "feature": "T_925",
        "label": "925-hPa temperature",
        "short": "T925",
        "family": "pressure",
        "reference": "ERA5 analysis",
    },
    {
        "feature": "Q_1000",
        "label": "1000-hPa specific humidity",
        "short": "Q1000",
        "family": "pressure",
        "reference": "ERA5 analysis",
    },
    {
        "feature": "Q_925",
        "label": "925-hPa specific humidity",
        "short": "Q925",
        "family": "pressure",
        "reference": "ERA5 analysis",
    },
    {
        "feature": "UV_925_VECTOR",
        "label": "925-hPa vector wind",
        "short": "UV925",
        "family": "pressure",
        "reference": "ERA5 analysis",
    },
)
FEATURE_ORDER = tuple(str(spec["feature"]) for spec in FEATURE_SPECS)
FEATURE_LABELS = {str(spec["feature"]): str(spec["label"]) for spec in FEATURE_SPECS}
FEATURE_SHORT = {str(spec["feature"]): str(spec["short"]) for spec in FEATURE_SPECS}
FEATURE_FAMILY = {str(spec["feature"]): str(spec["family"]) for spec in FEATURE_SPECS}
FEATURE_REFERENCE = {str(spec["feature"]): str(spec["reference"]) for spec in FEATURE_SPECS}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-quality-dir",
        required=True,
        help="Directory containing pressure- and surface-quality CSV files.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="fig_common_variable_error_regimes")
    parser.add_argument("--formats", default="svg,pdf,png,tiff")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def resolve_quality_dir(path: Path) -> Path:
    required = (
        "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv",
        "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv",
        "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv",
    )
    candidates = (path, path / "analysis")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        f"Could not find the three paired-quality CSV files under {path} or {path / 'analysis'}"
    )


def pressure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "feature",
        "scope",
        "unit",
        "pangu",
        "tianji",
        "pangu_ci_low",
        "pangu_ci_high",
        "tianji_ci_low",
        "tianji_ci_high",
        "pangu_bias",
        "tianji_bias",
        "pangu_bias_ci_low",
        "pangu_bias_ci_high",
        "tianji_bias_ci_low",
        "tianji_bias_ci_high",
        "tianji_to_pangu_ratio",
        "tianji_to_pangu_ratio_ci_low",
        "tianji_to_pangu_ratio_ci_high",
        "n",
        "represented_utc_dates",
    ]
    require_columns(frame, required, "pressure-level quality")
    rows = []
    for feature in FEATURE_ORDER:
        if FEATURE_FAMILY[feature] != "pressure":
            continue
        for scope in SCOPES:
            part = frame[(frame["feature"] == feature) & (frame["scope"] == scope)]
            if len(part) != 1:
                raise ValueError(f"Expected one pressure row for {feature}/{scope}")
            row = part.iloc[0]
            denominator = float(row["pangu"])
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"Invalid Pangu RMSE for {feature}/{scope}: {denominator}")
            ratio_value = float(row["tianji_to_pangu_ratio"])
            ratio_ci_low = float(row["tianji_to_pangu_ratio_ci_low"])
            ratio_ci_high = float(row["tianji_to_pangu_ratio_ci_high"])
            for source in ("pangu", "tianji"):
                rows.append(
                    {
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "feature_short": FEATURE_SHORT[feature],
                        "family": "pressure",
                        "reference": FEATURE_REFERENCE[feature],
                        "unit": row["unit"],
                        "scope": scope,
                        "scope_label": SCOPE_LABELS[scope],
                        "source": source,
                        "rmse": float(row[source]),
                        "rmse_ci_low": float(row[f"{source}_ci_low"]),
                        "rmse_ci_high": float(row[f"{source}_ci_high"]),
                        "bias": float(row[f"{source}_bias"]),
                        "bias_ci_low": float(row[f"{source}_bias_ci_low"]),
                        "bias_ci_high": float(row[f"{source}_bias_ci_high"]),
                        "bias_ci_available": True,
                        "bias_ci_method": "paired UTC-date bootstrap",
                        "rmse_ratio_tianji_over_pangu": ratio_value,
                        "rmse_ratio_ci_low": ratio_ci_low,
                        "rmse_ratio_ci_high": ratio_ci_high,
                        "rmse_ratio_ci_available": True,
                        "n": int(row["n"]),
                        "represented_utc_dates": int(row["represented_utc_dates"]),
                        "rmse_ratio_ci_method": "paired UTC-date bootstrap",
                    }
                )
    return pd.DataFrame(rows)


def surface_rows(surface: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        surface,
        [
            "feature",
            "scope",
            "source",
            "unit",
            "rmse",
            "ci_low",
            "ci_high",
            "bias",
            "bias_ci_low",
            "bias_ci_high",
            "n",
            "represented_utc_dates",
        ],
        "surface observation quality",
    )
    require_columns(
        pairs,
        [
            "feature",
            "scope",
            "left_source",
            "right_source",
            "ratio_right_over_left",
            "ratio_right_over_left_ci_low",
            "ratio_right_over_left_ci_high",
        ],
        "surface pairwise quality",
    )
    rows = []
    for feature in FEATURE_ORDER:
        if FEATURE_FAMILY[feature] != "surface":
            continue
        for scope in SCOPES:
            ratio = pairs[
                (pairs["feature"] == feature)
                & (pairs["scope"] == scope)
                & (pairs["left_source"] == "pangu")
                & (pairs["right_source"] == "tianji")
            ]
            if len(ratio) != 1:
                raise ValueError(f"Expected one Pangu/Tianji pair for {feature}/{scope}")
            ratio_row = ratio.iloc[0]
            ratio_value = float(ratio_row["ratio_right_over_left"])
            ratio_ci_low = float(ratio_row["ratio_right_over_left_ci_low"])
            ratio_ci_high = float(ratio_row["ratio_right_over_left_ci_high"])
            for source in ("pangu", "tianji"):
                part = surface[
                    (surface["feature"] == feature)
                    & (surface["scope"] == scope)
                    & (surface["source"] == source)
                ]
                if len(part) != 1:
                    raise ValueError(f"Expected one surface row for {feature}/{scope}/{source}")
                row = part.iloc[0]
                rows.append(
                    {
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "feature_short": FEATURE_SHORT[feature],
                        "family": "surface",
                        "reference": FEATURE_REFERENCE[feature],
                        "unit": row["unit"],
                        "scope": scope,
                        "scope_label": SCOPE_LABELS[scope],
                        "source": source,
                        "rmse": float(row["rmse"]),
                        "rmse_ci_low": float(row["ci_low"]),
                        "rmse_ci_high": float(row["ci_high"]),
                        "bias": float(row["bias"]),
                        "bias_ci_low": float(row["bias_ci_low"]),
                        "bias_ci_high": float(row["bias_ci_high"]),
                        "bias_ci_available": True,
                        "bias_ci_method": "paired UTC-date bootstrap",
                        "rmse_ratio_tianji_over_pangu": ratio_value,
                        "rmse_ratio_ci_low": ratio_ci_low,
                        "rmse_ratio_ci_high": ratio_ci_high,
                        "rmse_ratio_ci_available": True,
                        "n": int(row["n"]),
                        "represented_utc_dates": int(row["represented_utc_dates"]),
                        "rmse_ratio_ci_method": "paired UTC-date bootstrap",
                    }
                )
    return pd.DataFrame(rows)


def prepare_source(directory: Path) -> pd.DataFrame:
    pressure = pd.read_csv(
        directory / "pressure_level_paired_rmse_utc_date_bootstrap_ci.csv"
    )
    surface = pd.read_csv(
        directory / "surface_observation_three_source_rmse_utc_date_bootstrap_ci.csv"
    )
    pairs = pd.read_csv(
        directory / "surface_observation_pairwise_rmse_delta_utc_date_bootstrap_ci.csv"
    )
    source = pd.concat(
        [surface_rows(surface, pairs), pressure_rows(pressure)],
        ignore_index=True,
        sort=False,
    )
    source["feature_order"] = source["feature"].map(
        {feature: index for index, feature in enumerate(FEATURE_ORDER)}
    )
    source["normalized_bias"] = source["bias"] / source["rmse"]
    source["normalized_bias_ci_low"] = source["bias_ci_low"] / source["rmse"]
    source["normalized_bias_ci_high"] = source["bias_ci_high"] / source["rmse"]
    interval_columns = [
        "rmse_ratio_tianji_over_pangu",
        "rmse_ratio_ci_low",
        "rmse_ratio_ci_high",
        "normalized_bias",
        "normalized_bias_ci_low",
        "normalized_bias_ci_high",
    ]
    finite = np.isfinite(source[interval_columns].to_numpy(dtype=float)).all(axis=1)
    if not bool(np.all(finite)):
        bad = source.loc[~finite, ["feature", "scope", "source"]].to_dict("records")
        raise ValueError(
            "Paired variable-quality confidence intervals are required for the "
            f"manuscript figure; incomplete rows: {bad}"
        )
    return source.sort_values(
        ["feature_order", "scope", "source"], kind="stable"
    ).reset_index(drop=True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def title(ax: plt.Axes, text: str, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def style_axis(ax: plt.Axes, *, xgrid: bool = True) -> None:
    ax.tick_params(length=2.7, width=0.75, color=INK, pad=2.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.75)
    ax.grid(False)
    ax.set_axisbelow(True)


def add_family_bands(
    ax: plt.Axes,
    *,
    show_reference_labels: bool = True,
) -> None:
    # y runs top-to-bottom after inversion: the first three rows are surface.
    ax.axhspan(-0.5, 2.5, color=SURFACE_BAND, zorder=0)
    ax.axhspan(2.5, 6.5, color=PRESSURE_BAND, zorder=0)
    ax.axhline(2.5, color=LIGHT, linewidth=0.7, zorder=1)
    if not show_reference_labels:
        return
    ax.text(
        0.99,
        0.975,
        "station observations",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=SOURCE_DARK_COLORS["baseline"],
    )
    ax.text(
        0.99,
        0.48,
        "ERA5 analysis",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=SOURCE_DARK_COLORS["pangu"],
    )


def rmse_ratio_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    scope: str,
    label: str,
    show_y: bool,
    show_reference_labels: bool = True,
) -> None:
    panel_label(ax, label, x=-0.16 if show_y else -0.08, y=1.08)
    title(ax, f"RMSE · {SCOPE_LABELS[scope]}", x=0.02, y=1.08)
    part = (
        source[(source["scope"] == scope) & (source["source"] == "pangu")]
        .set_index("feature")
        .reindex(FEATURE_ORDER)
        .reset_index()
    )
    y = np.arange(len(part), dtype=float)
    for yi, row in zip(y, part.itertuples(index=False)):
        estimate = float(row.rmse_ratio_tianji_over_pangu)
        ci_low = float(row.rmse_ratio_ci_low)
        ci_high = float(row.rmse_ratio_ci_high)
        has_ci = np.isfinite(ci_low) and np.isfinite(ci_high)
        low = min(ci_low, estimate) if has_ci else estimate
        high = max(ci_high, estimate) if has_ci else estimate
        if has_ci and low <= 1.0 <= high:
            color = MID
        elif estimate < 1.0:
            color = SOURCE_COLORS["tianji"]
        else:
            color = SOURCE_DARK_COLORS["pangu"]
        if has_ci:
            ax.errorbar(
                estimate,
                yi,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                markersize=5.0,
                color=color,
                markerfacecolor=color,
                markeredgecolor=WHITE,
                markeredgewidth=0.65,
                ecolor=color,
                elinewidth=1.15,
                capsize=2.8,
                capthick=0.9,
                zorder=3,
            )
        else:
            ax.scatter(
                estimate,
                yi,
                s=34,
                color=color,
                edgecolor=WHITE,
                linewidth=0.65,
                zorder=3,
            )
    ax.axvline(1.0, color=INK, linewidth=0.85)
    positive = part[
        [
            "rmse_ratio_tianji_over_pangu",
            "rmse_ratio_ci_low",
            "rmse_ratio_ci_high",
        ]
    ].to_numpy(dtype=float).ravel()
    positive = positive[np.isfinite(positive)]
    if positive.size == 0 or np.any(positive <= 0):
        raise ValueError("RMSE-ratio estimates and available confidence intervals must be positive")
    lo = float(np.log2(positive.min()))
    hi = float(np.log2(positive.max()))
    span = max(hi - lo, 0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2 ** (lo - 0.12 * span), 2 ** (hi + 0.12 * span))
    candidate_ticks = np.asarray([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0])
    visible = candidate_ticks[
        (candidate_ticks >= ax.get_xlim()[0]) & (candidate_ticks <= ax.get_xlim()[1])
    ]
    ax.set_xticks(visible, [f"{value:g}" for value in visible])
    ax.set_ylim(len(FEATURE_ORDER) - 0.5, -0.5)
    ax.set_yticks(
        y,
        [FEATURE_LABELS[feature] for feature in FEATURE_ORDER] if show_y else [],
    )
    if not show_y:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Tianji RMSE / Pangu RMSE")
    ax.text(
        0.01,
        1.01,
        "← Tianji closer",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=SOURCE_DARK_COLORS["tianji"],
    )
    ax.text(
        0.99,
        1.01,
        "Pangu closer →",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=SOURCE_DARK_COLORS["pangu"],
    )
    add_family_bands(ax, show_reference_labels=show_reference_labels)
    style_axis(ax)


def bias_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    scope: str,
    label: str,
    show_y: bool,
    show_reference_labels: bool = True,
    layout: str = "offset_ci",
) -> None:
    if layout not in {"offset_ci", "paired_connector"}:
        raise ValueError(f"Unknown systematic-bias layout: {layout}")
    panel_label(ax, label, x=-0.16 if show_y else -0.08, y=1.08)
    title(ax, f"Systematic bias · {SCOPE_LABELS[scope]}", x=0.02, y=1.08)
    part = source[source["scope"] == scope].copy()
    y_base = {
        feature: float(index) for index, feature in enumerate(FEATURE_ORDER)
    }
    offsets = (
        {"pangu": -0.11, "tianji": 0.11}
        if layout == "offset_ci"
        else {"pangu": 0.0, "tianji": 0.0}
    )

    if layout == "paired_connector":
        estimates = part.pivot(
            index="feature",
            columns="source",
            values="normalized_bias",
        ).reindex(FEATURE_ORDER)
        if estimates[["pangu", "tianji"]].isna().any().any():
            missing = estimates[
                estimates[["pangu", "tianji"]].isna().any(axis=1)
            ].index.tolist()
            raise ValueError(f"Missing paired systematic-bias estimates for {missing}")
        for feature in FEATURE_ORDER:
            yi = y_base[feature]
            ax.plot(
                [
                    float(estimates.loc[feature, "pangu"]),
                    float(estimates.loc[feature, "tianji"]),
                ],
                [yi, yi],
                color=LIGHT,
                linewidth=1.25,
                solid_capstyle="round",
                zorder=1,
            )

    for source_key in ("pangu", "tianji"):
        rows = (
            part[part["source"] == source_key]
            .set_index("feature")
            .reindex(FEATURE_ORDER)
            .reset_index()
        )
        for row in rows.itertuples(index=False):
            estimate = float(row.normalized_bias)
            yi = y_base[str(row.feature)] + offsets[source_key]
            ci_low = float(row.normalized_bias_ci_low)
            ci_high = float(row.normalized_bias_ci_high)
            if layout == "paired_connector":
                ax.scatter(
                    estimate,
                    yi,
                    s=38,
                    marker=SOURCE_MARKERS[source_key],
                    facecolor=SOURCE_COLORS[source_key],
                    edgecolor=WHITE,
                    linewidth=0.7,
                    zorder=3,
                )
            elif np.isfinite(ci_low) and np.isfinite(ci_high):
                low = min(ci_low, estimate)
                high = max(ci_high, estimate)
                ax.errorbar(
                    estimate,
                    yi,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=SOURCE_MARKERS[source_key],
                    markersize=4.8,
                    color=SOURCE_COLORS[source_key],
                    markerfacecolor=SOURCE_COLORS[source_key],
                    markeredgecolor=WHITE,
                    markeredgewidth=0.6,
                    ecolor=SOURCE_COLORS[source_key],
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=0.85,
                    alpha=0.92,
                    zorder=3,
                )
            else:
                ax.scatter(
                    estimate,
                    yi,
                    s=30,
                    marker=SOURCE_MARKERS[source_key],
                    facecolor=SOURCE_COLORS[source_key],
                    edgecolor=WHITE,
                    linewidth=0.6,
                    zorder=3,
                )
    ax.axvline(0.0, color=INK, linewidth=0.85)
    extent_columns = (
        ["normalized_bias"]
        if layout == "paired_connector"
        else [
            "normalized_bias",
            "normalized_bias_ci_low",
            "normalized_bias_ci_high",
        ]
    )
    extent = float(
        np.nanmax(np.abs(part[extent_columns].to_numpy(dtype=float)))
    )
    extent = max(0.25, 1.12 * extent)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(len(FEATURE_ORDER) - 0.5, -0.5)
    ax.set_yticks(
        np.arange(len(FEATURE_ORDER)),
        [FEATURE_LABELS[feature] for feature in FEATURE_ORDER] if show_y else [],
    )
    if not show_y:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Signed bias / RMSE")
    ax.text(
        0.01,
        1.01,
        "negative",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MID,
    )
    ax.text(
        0.99,
        1.01,
        "positive",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=MID,
    )
    add_family_bands(ax, show_reference_labels=show_reference_labels)
    style_axis(ax)


def make_composite(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.subplots_adjust(
        left=0.205,
        right=0.985,
        bottom=0.105,
        top=0.89,
        wspace=0.19,
        hspace=0.44,
    )
    rmse_ratio_panel(axes[0, 0], source, SCOPES[0], "a", True)
    rmse_ratio_panel(
        axes[0, 1],
        source,
        SCOPES[1],
        "b",
        False,
        show_reference_labels=False,
    )
    bias_panel(axes[1, 0], source, SCOPES[0], "c", True)
    bias_panel(
        axes[1, 1],
        source,
        SCOPES[1],
        "d",
        False,
        show_reference_labels=False,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=SOURCE_MARKERS["pangu"],
                color=SOURCE_COLORS["pangu"],
                markerfacecolor=SOURCE_COLORS["pangu"],
                linestyle="none",
                label="Pangu (AI)",
            ),
            Line2D(
                [0],
                [0],
                marker=SOURCE_MARKERS["tianji"],
                color=SOURCE_COLORS["tianji"],
                markerfacecolor=SOURCE_COLORS["tianji"],
                linestyle="none",
                label="Tianji (physics)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.60, 0.985),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    return fig


def standalone_panel(
    source: pd.DataFrame,
    *,
    kind: str,
    scope: str,
    label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.55, 3.65))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.18, top=0.82)
    if kind == "rmse":
        rmse_ratio_panel(ax, source, scope, label, True)
    else:
        bias_panel(ax, source, scope, label, True)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker=SOURCE_MARKERS["pangu"],
                    color=SOURCE_COLORS["pangu"],
                    linestyle="none",
                    label="Pangu (AI)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=SOURCE_MARKERS["tianji"],
                    color=SOURCE_COLORS["tianji"],
                    linestyle="none",
                    label="Tianji (physics)",
                ),
            ],
            loc="lower right",
            ncol=1,
        )
    return fig


def parse_formats(value: str) -> List[str]:
    supported = {"svg", "pdf", "png", "tiff"}
    requested = [
        item.strip().lower()
        for item in value.replace(":", ",").split(",")
        if item.strip()
    ]
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported formats {unknown}")
    return ["svg", *[item for item in requested if item != "svg"]]


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {
            "facecolor": WHITE,
            "bbox_inches": "tight",
            "pad_inches": 0.035,
        }
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    configure_style()
    quality_dir = resolve_quality_dir(
        Path(args.paired_quality_dir).expanduser().resolve()
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    source = prepare_source(quality_dir)
    source_path = out_dir / f"{args.stem}_source_data.csv"
    source.to_csv(source_path, index=False)

    exports: Dict[str, List[Path]] = {}
    exports["composite"] = save_figure(
        make_composite(source), out_dir, args.stem, formats, int(args.dpi)
    )
    panel_specs = (
        ("a_rmse_all", "rmse", SCOPES[0], "a"),
        ("b_rmse_lowvis", "rmse", SCOPES[1], "b"),
        ("c_bias_all", "bias", SCOPES[0], "c"),
        ("d_bias_lowvis", "bias", SCOPES[1], "d"),
    )
    for suffix, kind, scope, label in panel_specs:
        exports[suffix] = save_figure(
            standalone_panel(source, kind=kind, scope=scope, label=label),
            out_dir,
            f"{args.stem}_{suffix}",
            formats,
            int(args.dpi),
        )

    manifest = {
        "status": "completed",
        "figure_stem": args.stem,
        "core_claim": (
            "Source errors are compared on all paired samples and observed Low-vis "
            "samples; regime dependence is separated from overall accuracy."
        ),
        "archetype": "quantitative grid",
        "figure_size_inches": list(FIGURE_SIZE),
        "source_data": source_path.name,
        "reference_policy": {
            "surface": "automatic station observations",
            "pressure_level": "ERA5 reference analysis",
        },
        "metrics": {
            "rmse": "Tianji RMSE divided by Pangu RMSE; paired UTC-date bootstrap CI",
            "bias": "signed source bias divided by its own RMSE; bias CI divided by point RMSE",
        },
        "exports": {
            key: [
                {"file": path.name, "sha256": sha256_file(path)}
                for path in paths
            ]
            for key, paths in exports.items()
        },
    }
    manifest_path = out_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] common-variable error figure bundle: {manifest_path}")


if __name__ == "__main__":
    main()
