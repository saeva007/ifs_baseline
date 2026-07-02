#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First-order trajectory ALE for the Pangu/Tianji q-core endpoint models.

For a dynamic predictor, the conditioning variable is its 12-hour trajectory
mean.  Within each empirical quantile bin, the complete trajectory is shifted
to the lower and upper bin edges while its within-window anomaly pattern is
preserved.  This is an accumulated-local-effects diagnostic, not a synthetic
source substitution and not a causal effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import analyze_multi_source_feature_importance as importance


DEFAULT_FEATURES = "Q_1000,DP_1000,Q_925,DP_925,RH_925,T2M,MSLP,U10,V10,WSPD10,U_925,V_925,WSPD925"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trajectory-mean ALE for trained q-core endpoint models")
    ap.add_argument("--sources", required=True, help="tag=data|checkpoint|scaler-or-AUTO|label;...")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference-source", default="pangu")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--sample-size", type=int, default=50000)
    ap.add_argument("--min-low-vis", type=int, default=200)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--static-rnn-train-dir", default=os.environ.get("STATIC_RNN_TRAIN_DIR", str(Path(__file__).resolve().parent.parent / "train")))
    ap.add_argument("--allow-legacy-time-alignment", action="store_true")
    ap.add_argument("--allow-partial-load", action="store_true")
    ap.add_argument("--checkpoint-tag", default="S2_PhaseB_best_score")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def centered_curve(local_effect: np.ndarray, weights: np.ndarray) -> np.ndarray:
    local = np.asarray(local_effect, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    curve = np.cumsum(local)
    valid = np.isfinite(curve) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return np.full_like(curve, np.nan)
    curve -= np.average(curve[valid], weights=weight[valid])
    return curve


def daily_bin_stats(effects: np.ndarray, bins: np.ndarray, dates: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_dates, date_codes = np.unique(dates.astype(str), return_inverse=True)
    sums = np.zeros((len(unique_dates), n_bins), dtype=np.float64)
    counts = np.zeros((len(unique_dates), n_bins), dtype=np.int64)
    np.add.at(sums, (date_codes, bins), effects)
    np.add.at(counts, (date_codes, bins), 1)
    return sums, counts


def ale_for_feature(
    rows: np.ndarray,
    dates: np.ndarray,
    source_eval,
    feature: str,
    train_mod,
    ev,
    eval_args: argparse.Namespace,
    cli: argparse.Namespace,
) -> pd.DataFrame:
    order = importance.canonical_order(source_eval.dynamic_feature_order, source_eval.dyn_vars_count)
    lookup = {name.upper(): idx for idx, name in enumerate(order)}
    if feature.upper() not in lookup:
        return pd.DataFrame()
    feature_idx = lookup[feature.upper()]
    columns = np.asarray(
        [time_idx * source_eval.dyn_vars_count + feature_idx for time_idx in range(eval_args.window)],
        dtype=np.int64,
    )
    trajectory = np.asarray(rows[:, columns], dtype=np.float64)
    summary = np.nanmean(trajectory, axis=1)
    finite = np.isfinite(summary) & np.all(np.isfinite(trajectory), axis=1)
    if int(finite.sum()) < max(100, cli.bins * 10):
        raise RuntimeError(f"{source_eval.source}/{feature}: only {int(finite.sum())} finite trajectories")
    working = np.asarray(rows[finite], dtype=np.float32)
    z = summary[finite]
    working_dates = dates[finite]
    edges = np.unique(np.quantile(z, np.linspace(0.0, 1.0, int(cli.bins) + 1)))
    if len(edges) < 4:
        raise RuntimeError(f"{source_eval.source}/{feature}: fewer than three non-degenerate ALE bins")
    n_bins = len(edges) - 1
    bin_code = np.searchsorted(edges, z, side="right") - 1
    bin_code = np.clip(bin_code, 0, n_bins - 1)
    effects = np.full(len(z), np.nan, dtype=np.float64)
    device = ev.resolve_device(cli.device)
    for bin_idx in range(n_bins):
        positions = np.flatnonzero(bin_code == bin_idx)
        if not len(positions):
            continue
        base = working[positions]
        lower = base.copy()
        upper = base.copy()
        lower[:, columns] += (edges[bin_idx] - z[positions])[:, None].astype(np.float32)
        upper[:, columns] += (edges[bin_idx + 1] - z[positions])[:, None].astype(np.float32)
        p_lower = ev.predict_static_rows_for_swap(lower, source_eval, train_mod, eval_args, device)[:, :2].sum(axis=1)
        p_upper = ev.predict_static_rows_for_swap(upper, source_eval, train_mod, eval_args, device)[:, :2].sum(axis=1)
        effects[positions] = p_upper - p_lower
    if np.any(~np.isfinite(effects)):
        raise RuntimeError(f"{source_eval.source}/{feature}: non-finite local effects")
    counts = np.bincount(bin_code, minlength=n_bins)
    local = np.asarray([effects[bin_code == idx].mean() for idx in range(n_bins)])
    curve = centered_curve(local, counts)
    ci_low = np.full(n_bins, np.nan)
    ci_high = np.full(n_bins, np.nan)
    if cli.bootstrap_iters > 0:
        sums, daily_counts = daily_bin_stats(effects, bin_code, working_dates, n_bins)
        rng = np.random.default_rng(cli.seed + sum(ord(char) for char in f"{source_eval.source}:{feature}"))
        draws = np.full((cli.bootstrap_iters, n_bins), np.nan, dtype=np.float64)
        for iteration in range(cli.bootstrap_iters):
            selected = rng.integers(0, len(sums), size=len(sums))
            draw_sums = sums[selected].sum(axis=0)
            draw_counts = daily_counts[selected].sum(axis=0)
            draw_local = np.divide(draw_sums, draw_counts, out=np.full(n_bins, np.nan), where=draw_counts > 0)
            draws[iteration] = centered_curve(draw_local, draw_counts)
        ci_low = np.nanpercentile(draws, 2.5, axis=0)
        ci_high = np.nanpercentile(draws, 97.5, axis=0)
    return pd.DataFrame(
        {
            "source": source_eval.source,
            "feature": order[feature_idx],
            "bin": np.arange(n_bins),
            "lower": edges[:-1],
            "upper": edges[1:],
            "midpoint": (edges[:-1] + edges[1:]) / 2.0,
            "n": counts,
            "local_effect": local,
            "ale_low_vis_probability": curve,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )


def plot_ale(frame: pd.DataFrame, labels: Dict[str, str], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] ALE plot skipped: {exc}")
        return
    features = list(dict.fromkeys(frame["feature"].astype(str)))
    columns = 3
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11.0, 3.0 * rows), constrained_layout=True, squeeze=False)
    colors = ["#684A9B", "#176B87", "#B65C19", "#555B66"]
    for ax, feature in zip(axes.ravel(), features):
        part = frame[frame["feature"] == feature]
        for color, (source, source_part) in zip(colors, part.groupby("source", sort=False)):
            source_part = source_part.sort_values("midpoint")
            x = source_part["midpoint"].to_numpy(dtype=float)
            y = source_part["ale_low_vis_probability"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", ms=3, color=color, label=labels.get(source, source))
            if source_part["ci_low"].notna().any():
                ax.fill_between(x, source_part["ci_low"], source_part["ci_high"], color=color, alpha=0.15)
        ax.axhline(0.0, lw=0.8, color="#777777")
        ax.set(title=feature, xlabel="12 h trajectory mean", ylabel="ALE of Low-vis probability")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    if features:
        axes.ravel()[0].legend(frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig_q_core_trajectory_ale.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    if cli.bins < 3:
        raise ValueError("--bins must be at least 3")
    out_dir = Path(cli.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = importance.load_source_evaluator()
    eval_args = ev.ensure_static_rnn_dataset_args(importance.evaluator_args(cli))
    specs, labels = importance.parse_sources(cli.sources, ev, cli)
    ev.fill_auto_scaler_paths(specs, eval_args)
    for tag, spec in specs.items():
        ev.require_file(spec.ckpt_path, f"{tag} checkpoint")
        ev.require_file(spec.scaler_path, f"{tag} scaler")
    configs = {tag: ev.read_build_config(spec.data_dir) for tag, spec in specs.items()}
    ev.validate_build_time_alignment(configs, cli.allow_legacy_time_alignment)
    train_mod = ev.import_training_module(eval_args)
    device = ev.resolve_device(cli.device)
    source_order = [cli.reference_source] + [tag for tag in specs if tag != cli.reference_source]
    source_evals = {
        tag: importance.load_source_for_importance(tag, specs[tag], train_mod, eval_args, device, ev)
        for tag in source_order
    }
    aligned, meta = ev.align_sources_to_reference(source_evals, source_order, strict_meta=False)
    y = source_evals[cli.reference_source].test_targets[aligned[cli.reference_source]]
    sample_pos = importance.uniform_common_sample(y, cli.sample_size, cli.seed, cli.min_low_vis)
    meta = meta.iloc[sample_pos].reset_index(drop=True)
    dates = pd.to_datetime(meta["time"], errors="raise").dt.strftime("%Y-%m-%d").to_numpy()
    requested = parse_csv(cli.features)
    frames: List[pd.DataFrame] = []
    for tag in source_order:
        source_eval = source_evals[tag]
        x = np.load(Path(source_eval.spec.data_dir) / "X_test.npy", mmap_mode="r")
        rows = np.asarray(x[aligned[tag][sample_pos]], dtype=np.float32)
        for feature in requested:
            result = ale_for_feature(rows, dates, source_eval, feature, train_mod, ev, eval_args, cli)
            if not result.empty:
                result.insert(1, "source_label", labels[tag])
                frames.append(result)
    if not frames:
        raise RuntimeError("None of the requested ALE features were available")
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out_dir / "q_core_trajectory_ale.csv", index=False)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "first-order ALE conditioned on the empirical 12 h trajectory mean",
                "trajectory_intervention": "shift all 12 values to a bin edge while preserving within-window anomalies",
                "interpretation": "within-model response diagnostic; not a causal source-substitution effect",
                "features": requested,
                "bins": cli.bins,
                "sample_rows": int(len(sample_pos)),
                "bootstrap_unit": "UTC_valid_date",
                "bootstrap_iterations": cli.bootstrap_iters,
                "sources": {tag: {"data_dir": specs[tag].data_dir, "checkpoint": specs[tag].ckpt_path, "scaler": specs[tag].scaler_path} for tag in source_order},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    if not cli.no_plot:
        plot_ale(result, labels, out_dir)
    print(f"[OK] ALE outputs written to {out_dir}")


if __name__ == "__main__":
    main()
