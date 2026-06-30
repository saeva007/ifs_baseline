#!/usr/bin/env python3
"""Summarize q-core models on one common, paired 2025 test-sample intersection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_METRICS = (
    "fog_pod",
    "fog_precision",
    "fog_csi",
    "fog_f1",
    "mist_pod",
    "mist_precision",
    "mist_csi",
    "mist_f1",
    "low_vis_recall",
    "low_vis_precision",
    "low_vis_csi",
    "low_vis_f1",
    "low_vis_fpr",
    "accuracy",
    "macro_f1",
)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray | None = None) -> Dict[str, float]:
    """Metric definitions mirrored from the main overlap evaluator."""
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = np.zeros((3, 3), dtype=np.int64)
    valid = (targets >= 0) & (targets <= 2) & (preds >= 0) & (preds <= 2)
    np.add.at(cm, (targets[valid], preds[valid]), 1)
    n = int(cm.sum())
    metrics: Dict[str, float] = {"n": float(n)}
    f1_values: List[float] = []
    weighted_f1_num = 0.0
    for cid, short in enumerate(("fog", "mist", "clear")):
        tp = float(cm[cid, cid])
        fp = float(cm[:, cid].sum() - cm[cid, cid])
        fn = float(cm[cid, :].sum() - cm[cid, cid])
        support = float(cm[cid, :].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics[f"{short}_precision"] = precision
        metrics[f"{short}_{'recall' if short == 'clear' else 'pod'}"] = recall
        metrics[f"{short}_f1"] = f1
        metrics[f"{short}_csi"] = safe_div(tp, tp + fp + fn)
        metrics[f"{short}_far"] = safe_div(fp, tp + fp)
        metrics[f"{short}_support"] = support
        metrics[f"pred_{short}"] = float(cm[:, cid].sum())
        f1_values.append(f1)
        weighted_f1_num += f1 * support
    metrics["accuracy"] = safe_div(float(np.trace(cm)), float(n))
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else math.nan
    metrics["weighted_f1"] = safe_div(weighted_f1_num, float(n))
    pred_low = preds <= 1
    true_low = targets <= 1
    is_clear = targets == 2
    low_tp = float((pred_low & true_low).sum())
    low_fp = float((pred_low & ~true_low).sum())
    low_fn = float((~pred_low & true_low).sum())
    low_precision = safe_div(low_tp, low_tp + low_fp)
    low_recall = safe_div(low_tp, low_tp + low_fn)
    metrics["low_vis_precision"] = low_precision
    metrics["low_vis_pod"] = low_recall
    metrics["low_vis_recall"] = low_recall
    metrics["low_vis_f1"] = safe_div(2.0 * low_precision * low_recall, low_precision + low_recall)
    metrics["low_vis_csi"] = safe_div(low_tp, low_tp + low_fp + low_fn)
    metrics["low_vis_far"] = safe_div(low_fp, low_tp + low_fp)
    metrics["low_vis_fpr"] = safe_div(float((pred_low & is_clear).sum()), float(is_clear.sum()))
    if probs is not None:
        one_hot = np.eye(3, dtype=np.float64)[targets]
        metrics["multiclass_brier"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        low_prob = probs[:, 0] + probs[:, 1]
        metrics["low_vis_brier"] = float(np.mean((low_prob - true_low.astype(np.float64)) ** 2))
    return metrics


def metric_direction(metric: str) -> str:
    return "lower" if metric.endswith(("_fpr", "_far", "_brier", "_ece")) else "higher"


def parse_sources(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in str(text or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            tag, label = raw.split("=", 1)
        else:
            tag = label = raw
        tag, label = tag.strip(), label.strip()
        if not tag or tag in out:
            raise ValueError(f"Empty or duplicate source tag in {raw!r}")
        out[tag] = label or tag
    if len(out) < 2:
        raise ValueError("At least two evaluated sources are required.")
    return out


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def load_source(path: Path, tag: str) -> pd.DataFrame:
    csv_path = path / f"per_sample_{tag}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing per-sample evaluation: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"time", "station_id", "y_cls", "vis_raw_m", "pred", "p_fog", "p_mist", "p_clear"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path} is missing columns {sorted(missing)}")
    time = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError(f"{csv_path} contains invalid timestamps")
    out = df.copy()
    out["time_utc"] = time
    out["time_key"] = time.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["station_key"] = normalize_station(out["station_id"])
    if out[["time_key", "station_key"]].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate (time, station_id) rows")
    out.index = pd.MultiIndex.from_frame(out[["time_key", "station_key"]])
    return out


def align_sources(frames: Mapping[str, pd.DataFrame]) -> Tuple[pd.MultiIndex, Dict[str, pd.DataFrame]]:
    common: pd.MultiIndex | None = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index, sort=False)
    assert common is not None
    common = common.sort_values()
    if len(common) == 0:
        raise RuntimeError("No common test samples across q-core sources")
    aligned = {tag: frame.reindex(common) for tag, frame in frames.items()}
    ref_tag = next(iter(aligned))
    ref = aligned[ref_tag]
    for tag, frame in aligned.items():
        if not np.array_equal(ref["y_cls"].to_numpy(dtype=np.int64), frame["y_cls"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Class labels differ between {ref_tag} and {tag} on the paired intersection")
        if not np.allclose(
            ref["vis_raw_m"].to_numpy(dtype=np.float64),
            frame["vis_raw_m"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-3,
        ):
            raise ValueError(f"Raw visibility labels differ between {ref_tag} and {tag}")
    return common, aligned


def arrays(frame: pd.DataFrame, idx: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if idx is None:
        idx = np.arange(len(frame), dtype=np.int64)
    y = frame["y_cls"].to_numpy(dtype=np.int64)[idx]
    pred = frame["pred"].to_numpy(dtype=np.int64)[idx]
    probs = frame[["p_fog", "p_mist", "p_clear"]].to_numpy(dtype=np.float64)[idx]
    return y, pred, probs


def metric_rows(aligned: Mapping[str, pd.DataFrame], labels: Mapping[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for tag, frame in aligned.items():
        y, pred, probs = arrays(frame)
        metrics = compute_metrics(y, pred, probs=probs)
        row: Dict[str, object] = {
            "source": tag,
            "source_label": labels[tag],
            "sample_scope": "four_source_paired_test_intersection",
            "n": int(len(y)),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def date_groups(frame: pd.DataFrame) -> List[np.ndarray]:
    date = frame["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    groups = [np.flatnonzero(date == value) for value in pd.unique(date)]
    return [g.astype(np.int64) for g in groups if len(g)]


def paired_block_bootstrap(
    aligned: Mapping[str, pd.DataFrame],
    reference: str,
    metrics: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    ref = aligned[reference]
    groups = date_groups(ref)
    if not groups:
        raise RuntimeError("No UTC date groups available for paired bootstrap")
    rng = np.random.default_rng(seed)
    point: Dict[str, Dict[str, float]] = {}
    for tag, frame in aligned.items():
        y, pred, probs = arrays(frame)
        point[tag] = compute_metrics(y, pred, probs=probs)

    draws: Dict[Tuple[str, str], List[float]] = {
        (tag, metric): []
        for tag in aligned
        if tag != reference
        for metric in metrics
    }
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[int(i)] for i in chosen])
        boot_metrics: Dict[str, Dict[str, float]] = {}
        for tag, frame in aligned.items():
            y, pred, probs = arrays(frame, idx)
            boot_metrics[tag] = compute_metrics(y, pred, probs=probs)
        for tag in aligned:
            if tag == reference:
                continue
            for metric in metrics:
                draws[(tag, metric)].append(float(boot_metrics[tag][metric]) - float(boot_metrics[reference][metric]))

    rows: List[Dict[str, object]] = []
    for tag in aligned:
        if tag == reference:
            continue
        for metric in metrics:
            values = np.asarray(draws[(tag, metric)], dtype=np.float64)
            direction = metric_direction(metric)
            p_better = float(np.mean(values < 0.0)) if direction == "lower" else float(np.mean(values > 0.0))
            rows.append(
                {
                    "source": tag,
                    "reference": reference,
                    "metric": metric,
                    "preferred_direction": direction,
                    "source_value": float(point[tag][metric]),
                    "reference_value": float(point[reference][metric]),
                    "delta_source_minus_reference": float(point[tag][metric]) - float(point[reference][metric]),
                    "delta_ci95_low": float(np.percentile(values, 2.5)),
                    "delta_ci95_high": float(np.percentile(values, 97.5)),
                    "bootstrap_probability_source_better": p_better,
                    "bootstrap_unit": "UTC_valid_date",
                    "bootstrap_reps": int(n_bootstrap),
                    "n_date_blocks": int(len(groups)),
                }
            )
    return pd.DataFrame(rows)


def write_paired_samples(path: Path, aligned: Mapping[str, pd.DataFrame], labels: Mapping[str, str]) -> None:
    ref_tag = next(iter(aligned))
    ref = aligned[ref_tag]
    out = ref[["time", "station_id", "y_cls", "vis_raw_m"]].reset_index(drop=True).copy()
    for tag, frame in aligned.items():
        out[f"{tag}_pred"] = frame["pred"].to_numpy(dtype=np.int64)
        out[f"{tag}_p_fog"] = frame["p_fog"].to_numpy(dtype=np.float64)
        out[f"{tag}_p_mist"] = frame["p_mist"].to_numpy(dtype=np.float64)
        out[f"{tag}_p_clear"] = frame["p_clear"].to_numpy(dtype=np.float64)
    out.to_csv(path / "q_core_paired_common_predictions.csv.gz", index=False, compression="gzip")
    with (path / "q_core_source_labels.json").open("w", encoding="utf-8") as f:
        json.dump(dict(labels), f, ensure_ascii=False, indent=2)


def make_figures(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[WARN] figures skipped: {exc}")
        return

    colors = ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00"]
    skill_metrics = ["low_vis_precision", "low_vis_recall", "low_vis_csi", "low_vis_f1"]
    labels = metrics_df["source_label"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for j, metric in enumerate(skill_metrics):
        ax.bar(x + (j - 1.5) * width, metrics_df[metric], width, label=metric.replace("low_vis_", "").upper(), color=colors[j])
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score on paired test samples")
    ax.legend(ncol=4, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_q_core_paired_low_vis_skill.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.bar(x, metrics_df["low_vis_fpr"], color=colors[: len(labels)])
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Low-visibility false-positive rate")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_q_core_paired_low_vis_fpr.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--sources", required=True, help="Semicolon-separated tag=display label specs")
    ap.add_argument("--reference", default="pangu2025_q_core_no_rh2m")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20250629)
    ap.add_argument("--no-paired-samples", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    labels = parse_sources(args.sources)
    if args.reference not in labels:
        raise KeyError(f"Reference {args.reference!r} is not in --sources")
    frames = {tag: load_source(eval_dir, tag) for tag in labels}
    common, aligned = align_sources(frames)
    metrics_df = metric_rows(aligned, labels)
    metrics_df.to_csv(eval_dir / "q_core_paired_common_metrics.csv", index=False)

    coverage = pd.DataFrame(
        [
            {
                "source": tag,
                "source_label": labels[tag],
                "source_test_rows": int(len(frames[tag])),
                "paired_common_rows": int(len(common)),
                "paired_common_fraction": float(len(common) / max(len(frames[tag]), 1)),
            }
            for tag in labels
        ]
    )
    coverage.to_csv(eval_dir / "q_core_paired_common_coverage.csv", index=False)

    metrics = [metric for metric in DEFAULT_METRICS if metric in metrics_df.columns]
    delta_df = paired_block_bootstrap(
        aligned,
        args.reference,
        metrics,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    delta_df.to_csv(eval_dir / "q_core_paired_deltas_vs_pangu2025.csv", index=False)
    if not args.no_paired_samples:
        write_paired_samples(eval_dir, aligned, labels)
    if not args.no_figures:
        make_figures(metrics_df, eval_dir)

    report = {
        "status": "passed",
        "sample_scope": "intersection of identical 2025 (valid_time, station_id) test samples across all sources",
        "threshold_mode": "argmax",
        "controlled_dimensions": [
            "dynamic variable order",
            "model architecture and training protocol",
            "visibility labels",
            "paired test samples",
        ],
        "interpretation_caveats": [
            "Pangu-2025 uses a 24 h ONNX product while Tianji/IFS use their existing 12 <= lead_hour < 24 stitching convention.",
            "ERA5 is a reference analysis, not an operational forecast source.",
            "Results are a common-input product comparison, not pure forecast-source causal attribution.",
        ],
        "reference": args.reference,
        "paired_common_rows": int(len(common)),
        "source_rows": {tag: int(len(frame)) for tag, frame in frames.items()},
        "bootstrap": {
            "unit": "UTC_valid_date",
            "reps": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
        },
    }
    with (eval_dir / "q_core_paired_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(metrics_df[["source", "n", "low_vis_precision", "low_vis_recall", "low_vis_csi", "low_vis_fpr"]].to_string(index=False))
    print(f"[OK] paired q-core outputs written to {eval_dir}")


if __name__ == "__main__":
    main()
