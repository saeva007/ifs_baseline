#!/usr/bin/env python3
"""Build compact, paired diagnostics for the source-full best-effort ensemble.

The formal best-effort product is the equal-weight mean of post-softmax
probabilities from the Tianji, Tianji-T2ND and IFS-trained source-full models,
followed by one argmax.  This module deliberately evaluates every member on
the exact ensemble intersection.  It writes compact source-data tables for:

* paired low-visibility discrimination and probability metrics;
* precision-recall curves;
* reliability;
* response across the complete observed-visibility distribution; and
* member-hit overlap within observed low-visibility samples.

It can be imported by the evaluator (avoiding large per-sample CSV files) or
run as a CLI against an existing source-full result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve
except Exception:  # pragma: no cover - lightweight fallback for plot-only runtimes.
    def average_precision_score(target, score):
        observed = np.asarray(target, dtype=np.int8)
        probability = np.asarray(score, dtype=np.float64)
        order = np.argsort(-probability, kind="stable")
        observed = observed[order]
        positives = int(observed.sum())
        if positives <= 0:
            return math.nan
        cumulative = np.cumsum(observed)
        precision = cumulative / np.arange(1, len(observed) + 1)
        return float(np.sum(precision * observed) / positives)

    def precision_recall_curve(target, score):
        observed = np.asarray(target, dtype=np.int8)
        probability = np.asarray(score, dtype=np.float64)
        order = np.argsort(-probability, kind="stable")
        observed = observed[order]
        probability = probability[order]
        distinct = np.r_[np.flatnonzero(np.diff(probability)), len(probability) - 1]
        true_positive = np.cumsum(observed)[distinct].astype(np.float64)
        predicted_positive = (distinct + 1).astype(np.float64)
        positives = max(float(observed.sum()), 1.0)
        precision = true_positive / predicted_positive
        recall = true_positive / positives
        return (
            np.r_[precision, 1.0],
            np.r_[recall, 0.0],
            probability[distinct],
        )


ENSEMBLE_SOURCE = "tianji_t2nd_ifs_mean_softmax"
DEFAULT_MEMBERS = ("tianji", "T2ND_rh2m_source_full", "ifs")
DEFAULT_LABELS = {
    "tianji": "Tianji",
    "T2ND_rh2m_source_full": "Tianji T2ND",
    "ifs": "IFS-trained",
    ENSEMBLE_SOURCE: "Best effort",
}
REQUIRED_SAMPLE_COLUMNS = (
    "time",
    "station_id",
    "y_cls",
    "vis_raw_m",
    "pred",
    "p_fog",
    "p_mist",
    "p_clear",
)
VISIBILITY_BINS = (
    (-np.inf, 200.0, "<0.2"),
    (200.0, 500.0, "0.2–0.5"),
    (500.0, 1000.0, "0.5–1"),
    (1000.0, 2000.0, "1–2"),
    (2000.0, 5000.0, "2–5"),
    (5000.0, 10000.0, "5–10"),
    (10000.0, np.inf, "≥10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="Directory containing per_sample_<source>.csv files.",
    )
    parser.add_argument("--out-dir", default="", help="Defaults to --sample-dir.")
    parser.add_argument(
        "--members",
        default=",".join(DEFAULT_MEMBERS),
        help="Comma-separated ensemble member source tags.",
    )
    parser.add_argument("--ensemble-source", default=ENSEMBLE_SOURCE)
    parser.add_argument("--reliability-bins", type=int, default=12)
    parser.add_argument("--pr-max-points", type=int, default=401)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


def normalize_station(values: Iterable[object]) -> pd.Series:
    raw = pd.Series(values)
    text = raw.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    valid = numeric.notna()
    if valid.any():
        text.loc[valid] = numeric.loc[valid].astype(np.int64).astype(str)
    return text


def alignment_keys(frame: pd.DataFrame, row_column: str) -> pd.DataFrame:
    require_columns(frame, ("time", "station_id"), row_column)
    out = frame.copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    if time.isna().any():
        raise ValueError(f"{row_column}: unparseable time rows={int(time.isna().sum())}")
    out["__time"] = time.dt.floor("s")
    out["__station"] = normalize_station(out["station_id"])
    out["__dup"] = out.groupby(["__time", "__station"], sort=False).cumcount()
    out[row_column] = np.arange(len(out), dtype=np.int64)
    return out


def payload_from_frame(frame: pd.DataFrame, label: str) -> Dict[str, object]:
    require_columns(frame, REQUIRED_SAMPLE_COLUMNS, label)
    probs = frame[["p_fog", "p_mist", "p_clear"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{label}: probability columns contain non-finite values")
    sums = probs.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError(f"{label}: probability rows must have positive sums")
    probs = probs / sums
    return {
        "label": label,
        "probs": probs,
        "preds": frame["pred"].to_numpy(dtype=np.int64),
        "targets": frame["y_cls"].to_numpy(dtype=np.int64),
        "raw_visibility_m": frame["vis_raw_m"].to_numpy(dtype=np.float64),
    }


def load_aligned_payloads(
    sample_dir: Path,
    members: Sequence[str],
    ensemble_source: str,
) -> Dict[str, Dict[str, object]]:
    ensemble_path = sample_dir / f"per_sample_{ensemble_source}.csv"
    if not ensemble_path.is_file():
        raise FileNotFoundError(
            f"Missing {ensemble_path}; rerun source-full evaluation without --no_per_sample_csv "
            "or use the evaluator-integrated compact analysis."
        )
    reference = pd.read_csv(ensemble_path)
    require_columns(reference, REQUIRED_SAMPLE_COLUMNS, ensemble_path.name)
    reference = alignment_keys(reference, "__reference_row")
    key_columns = ["__time", "__station", "__dup"]
    payloads: Dict[str, Dict[str, object]] = {
        ensemble_source: payload_from_frame(
            reference.sort_values("__reference_row"), DEFAULT_LABELS.get(ensemble_source, ensemble_source)
        )
    }

    for source in members:
        path = sample_dir / f"per_sample_{source}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ensemble-member sample table: {path}")
        member = alignment_keys(pd.read_csv(path), "__member_row")
        joined = reference[key_columns + ["__reference_row"]].merge(
            member,
            on=key_columns,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        missing = joined["__member_row"].isna()
        if missing.any():
            raise RuntimeError(
                f"{source}: {int(missing.sum())} ensemble rows were absent from the member sample table"
            )
        joined = joined.sort_values("__reference_row").reset_index(drop=True)
        targets = joined["y_cls"].to_numpy(dtype=np.int64)
        reference_targets = np.asarray(payloads[ensemble_source]["targets"], dtype=np.int64)
        if not np.array_equal(targets, reference_targets):
            raise RuntimeError(f"{source}: observed class labels differ after alignment")
        payloads[source] = payload_from_frame(joined, DEFAULT_LABELS.get(source, source))
    return payloads


def ece_binary(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    total = max(len(probability), 1)
    value = 0.0
    for bin_index in range(bins):
        selected = index == bin_index
        if not selected.any():
            continue
        value += (
            float(selected.sum())
            / total
            * abs(float(probability[selected].mean()) - float(target[selected].mean()))
        )
    return float(value)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else math.nan


def metric_table(
    payloads: Mapping[str, Mapping[str, object]],
    source_order: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for source in source_order:
        payload = payloads[source]
        targets = np.asarray(payload["targets"], dtype=np.int64)
        preds = np.asarray(payload["preds"], dtype=np.int64)
        probs = np.asarray(payload["probs"], dtype=np.float64)
        true_low = targets <= 1
        pred_low = preds <= 1
        clear = targets == 2
        tp = float(np.sum(true_low & pred_low))
        fp = float(np.sum(~true_low & pred_low))
        fn = float(np.sum(true_low & ~pred_low))
        low_probability = probs[:, 0] + probs[:, 1]
        rows.append(
            {
                "source": source,
                "source_label": str(payload["label"]),
                "n": int(len(targets)),
                "observed_low_vis_n": int(true_low.sum()),
                "predicted_low_vis_n": int(pred_low.sum()),
                "low_vis_ap": float(
                    average_precision_score(true_low.astype(np.int8), low_probability)
                ),
                "low_vis_precision": safe_div(tp, tp + fp),
                "low_vis_recall": safe_div(tp, tp + fn),
                "low_vis_csi": safe_div(tp, tp + fp + fn),
                "low_vis_fpr": safe_div(float(np.sum(clear & pred_low)), float(clear.sum())),
                "low_vis_brier": float(
                    np.mean((low_probability - true_low.astype(np.float64)) ** 2)
                ),
                "ece_low_vis": ece_binary(
                    low_probability, true_low.astype(np.float64)
                ),
                "comparison_scope": "exact ensemble-member intersection",
                "decision_rule": (
                    "equal-weight mean post-softmax then argmax"
                    if source == ENSEMBLE_SOURCE
                    else "member prediction on ensemble intersection"
                ),
            }
        )
    return pd.DataFrame(rows)


def reliability_table(
    payloads: Mapping[str, Mapping[str, object]],
    source_order: Sequence[str],
    bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for source in source_order:
        payload = payloads[source]
        targets = np.asarray(payload["targets"], dtype=np.int64)
        probability = np.asarray(payload["probs"], dtype=np.float64)[:, :2].sum(axis=1)
        observed = (targets <= 1).astype(np.float64)
        index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
        for bin_index in range(bins):
            selected = index == bin_index
            if not selected.any():
                continue
            rows.append(
                {
                    "source": source,
                    "source_label": str(payload["label"]),
                    "bin": bin_index,
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "mean_probability": float(probability[selected].mean()),
                    "observed_frequency": float(observed[selected].mean()),
                    "n": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows)


def pr_table(
    payloads: Mapping[str, Mapping[str, object]],
    source_order: Sequence[str],
    max_points: int,
) -> pd.DataFrame:
    rows = []
    for source in source_order:
        payload = payloads[source]
        targets = np.asarray(payload["targets"], dtype=np.int64)
        probability = np.asarray(payload["probs"], dtype=np.float64)[:, :2].sum(axis=1)
        precision, recall, thresholds = precision_recall_curve(
            (targets <= 1).astype(np.int8), probability
        )
        threshold_full = np.r_[thresholds, np.nan]
        order = np.argsort(recall, kind="stable")
        precision = precision[order]
        recall = recall[order]
        threshold_full = threshold_full[order]
        if len(recall) > int(max_points):
            keep = np.unique(
                np.linspace(0, len(recall) - 1, int(max_points)).round().astype(np.int64)
            )
            precision = precision[keep]
            recall = recall[keep]
            threshold_full = threshold_full[keep]
        for point_index, (p_value, r_value, threshold) in enumerate(
            zip(precision, recall, threshold_full)
        ):
            rows.append(
                {
                    "source": source,
                    "source_label": str(payload["label"]),
                    "point_index": point_index,
                    "recall": float(r_value),
                    "precision": float(p_value),
                    "threshold": float(threshold) if np.isfinite(threshold) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def visibility_response_table(
    payloads: Mapping[str, Mapping[str, object]],
    source_order: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for source in source_order:
        payload = payloads[source]
        raw = np.asarray(payload["raw_visibility_m"], dtype=np.float64)
        probability = np.asarray(payload["probs"], dtype=np.float64)[:, :2].sum(axis=1)
        for bin_index, (left, right, label) in enumerate(VISIBILITY_BINS):
            selected = np.isfinite(raw) & (raw >= left) & (raw < right)
            if not selected.any():
                continue
            values = probability[selected]
            rows.append(
                {
                    "source": source,
                    "source_label": str(payload["label"]),
                    "visibility_bin": bin_index,
                    "visibility_bin_km": label,
                    "visibility_left_m": float(left),
                    "visibility_right_m": float(right),
                    "scope": (
                        "observed_low_visibility" if right <= 1000.0 else "all_sample_context"
                    ),
                    "mean_low_vis_probability": float(values.mean()),
                    "median_low_vis_probability": float(np.median(values)),
                    "q25_low_vis_probability": float(np.quantile(values, 0.25)),
                    "q75_low_vis_probability": float(np.quantile(values, 0.75)),
                    "n": int(selected.sum()),
                }
            )
    return pd.DataFrame(rows)


def complementarity_table(
    payloads: Mapping[str, Mapping[str, object]],
    members: Sequence[str],
    ensemble_source: str,
) -> pd.DataFrame:
    targets = np.asarray(payloads[ensemble_source]["targets"], dtype=np.int64)
    true_low = targets <= 1
    if not true_low.any():
        raise RuntimeError("No observed low-visibility samples in the ensemble intersection")
    member_hits = np.column_stack(
        [np.asarray(payloads[source]["preds"], dtype=np.int64) <= 1 for source in members]
    )[true_low]
    ensemble_hit = (
        np.asarray(payloads[ensemble_source]["preds"], dtype=np.int64)[true_low] <= 1
    )
    rows = []
    total = int(true_low.sum())
    for mask in range(2 ** len(members)):
        bits = np.asarray([(mask >> idx) & 1 for idx in range(len(members))], dtype=bool)
        selected = np.all(member_hits == bits[None, :], axis=1)
        n = int(selected.sum())
        if n == 0:
            continue
        names = [
            DEFAULT_LABELS.get(source, source)
            for source, enabled in zip(members, bits)
            if enabled
        ]
        ensemble_n = int(np.sum(ensemble_hit[selected]))
        rows.append(
            {
                "pattern_mask": format(mask, f"0{len(members)}b"),
                "member_pattern": " + ".join(names) if names else "No member",
                "member_hit_count": int(bits.sum()),
                "n": n,
                "share_of_observed_low_vis": n / total,
                "ensemble_hit_n": ensemble_n,
                "ensemble_miss_n": n - ensemble_n,
                "ensemble_hit_rate": ensemble_n / n,
                "observed_low_vis_total": total,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n", "member_hit_count"], ascending=[False, False], kind="stable"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_payloads(
    payloads: Mapping[str, Mapping[str, object]],
    source_order: Sequence[str],
) -> None:
    missing = [source for source in source_order if source not in payloads]
    if missing:
        raise KeyError(f"Missing payloads for {missing}")
    reference = source_order[-1]
    y_ref = np.asarray(payloads[reference]["targets"], dtype=np.int64)
    raw_ref = np.asarray(payloads[reference]["raw_visibility_m"], dtype=np.float64)
    for source in source_order:
        payload = payloads[source]
        probs = np.asarray(payload["probs"], dtype=np.float64)
        preds = np.asarray(payload["preds"], dtype=np.int64)
        targets = np.asarray(payload["targets"], dtype=np.int64)
        raw = np.asarray(payload["raw_visibility_m"], dtype=np.float64)
        if probs.shape != (len(y_ref), 3) or len(preds) != len(y_ref):
            raise ValueError(f"{source}: payload shape is not aligned to the ensemble")
        if not np.array_equal(targets, y_ref):
            raise ValueError(f"{source}: target labels differ from the ensemble reference")
        finite = np.isfinite(raw) & np.isfinite(raw_ref)
        if finite.any() and not np.allclose(raw[finite], raw_ref[finite], atol=1.0e-3):
            raise ValueError(f"{source}: raw visibility differs from the ensemble reference")


def write_analysis_bundle(
    payloads: Mapping[str, Mapping[str, object]],
    out_dir: Path | str,
    *,
    members: Sequence[str] = DEFAULT_MEMBERS,
    ensemble_source: str = ENSEMBLE_SOURCE,
    reliability_bins: int = 12,
    pr_max_points: int = 401,
) -> Dict[str, object]:
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_order = [*members, ensemble_source]
    validate_payloads(payloads, source_order)
    tables = {
        "paired_metrics": metric_table(payloads, source_order),
        "precision_recall": pr_table(payloads, source_order, pr_max_points),
        "reliability": reliability_table(payloads, source_order, reliability_bins),
        "visibility_response": visibility_response_table(payloads, source_order),
        "member_complementarity": complementarity_table(
            payloads, members, ensemble_source
        ),
    }
    files: MutableMapping[str, str] = {}
    for name, frame in tables.items():
        path = output / f"best_effort_{name}.csv"
        frame.to_csv(path, index=False)
        files[name] = path.name
    manifest = {
        "status": "completed",
        "ensemble_source": ensemble_source,
        "members": list(members),
        "source_order": source_order,
        "paired_scope": "exact common intersection used by the ensemble",
        "decision": "equal-weight mean post-softmax probabilities, then one argmax",
        "sample_scopes": {
            "performance_and_probability": "all common test samples",
            "visibility_response": "full observed-visibility distribution with <1 km bins retained",
            "member_complementarity": "observed visibility <1 km",
        },
        "outputs": {
            name: {"file": filename, "sha256": sha256_file(output / filename)}
            for name, filename in files.items()
        },
    }
    manifest_path = output / "best_effort_extended_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": manifest_path.name}


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else sample_dir
    )
    members = tuple(item.strip() for item in args.members.split(",") if item.strip())
    payloads = load_aligned_payloads(sample_dir, members, args.ensemble_source)
    report = write_analysis_bundle(
        payloads,
        out_dir,
        members=members,
        ensemble_source=args.ensemble_source,
        reliability_bins=args.reliability_bins,
        pr_max_points=args.pr_max_points,
    )
    print(
        f"[OK] best-effort paired diagnostics: {out_dir / str(report['manifest'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
