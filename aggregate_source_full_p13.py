#!/usr/bin/env python3
"""Aggregate source-specific P13 seed predictions without overwriting legacy rows.

Each seed has already been evaluated independently on the same source-specific
test split.  This script verifies the station-time identity, observed labels,
and raw visibility before computing the formal P13 decision:

    equal-weight mean(post-softmax class probabilities) -> argmax

The output is deliberately a new source-tagged bundle so Figure 1/4 consumers
can opt in without changing historical source-full results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    # Prefer this checked-out evaluator over a same-named module on PATH.
    sys.path.insert(0, str(SCRIPT_DIR))

from test_PMST_overlap_forecast_source_s2 import compute_metrics  # noqa: E402


PROBABILITY_COLUMNS = ("p_fog", "p_mist", "p_clear")


def parse_seeds(value: str) -> List[int]:
    tokens = [token.strip() for token in str(value).replace(":", ",").split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one seed is required")
    try:
        seeds = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid seed list: {value!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def canonical_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def time_station_keys(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"time", "station_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{source} lacks alignment columns: {missing}")
    time = pd.to_datetime(frame["time"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("h")
    if time.isna().any():
        raise ValueError(f"{source} has {int(time.isna().sum())} invalid timestamps")
    keys = pd.DataFrame(
        {
            "time": time,
            "station_key": canonical_station(frame["station_id"]),
        }
    )
    keys["duplicate_index_within_time_station"] = keys.groupby(
        ["time", "station_key"], sort=False
    ).cumcount()
    return keys


def read_seed_prediction(path: Path, source: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    label_column = "y_cls" if "y_cls" in frame.columns else "y_true"
    required = {"time", "station_id", label_column, "vis_raw_m", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} lacks required columns: {missing}")
    keys = time_station_keys(frame, str(path))
    table = pd.DataFrame(
        {
            "time": keys["time"],
            "station_key": keys["station_key"],
            "duplicate_index_within_time_station": keys["duplicate_index_within_time_station"],
            "station_id": frame["station_id"].astype(str),
            "y_cls": pd.to_numeric(frame[label_column], errors="coerce"),
            "vis_raw_m": pd.to_numeric(frame["vis_raw_m"], errors="coerce"),
            **{
                column: pd.to_numeric(frame[column], errors="coerce")
                for column in PROBABILITY_COLUMNS
            },
        }
    ).sort_values(
        ["time", "station_key", "duplicate_index_within_time_station"], kind="stable"
    ).reset_index(drop=True)
    if table[["time", "station_key", "duplicate_index_within_time_station"]].duplicated().any():
        raise ValueError(f"{path} has duplicate station-time alignment keys")
    numeric_columns = ["y_cls", "vis_raw_m", *PROBABILITY_COLUMNS]
    if not np.isfinite(table[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} has non-finite observed values or probabilities")
    y = table["y_cls"].to_numpy(dtype=float)
    if not np.array_equal(y, y.astype(np.int64).astype(float)):
        raise ValueError(f"{path} has non-integer class labels")
    probs = table[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if np.any(probs < 0) or np.any(probs.sum(axis=1) <= 0):
        raise ValueError(f"{path} has invalid post-softmax probabilities")
    return table


def validate_same_observed(reference: pd.DataFrame, candidate: pd.DataFrame, source: str) -> None:
    key_columns = ["time", "station_key", "duplicate_index_within_time_station"]
    if len(reference) != len(candidate) or not reference[key_columns].equals(candidate[key_columns]):
        raise ValueError(f"{source} does not share the exact station-time rows with the first seed")
    if not np.array_equal(
        reference["y_cls"].to_numpy(dtype=np.int64), candidate["y_cls"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(f"{source} observed class labels differ across seeds")
    if not np.allclose(
        reference["vis_raw_m"].to_numpy(dtype=float),
        candidate["vis_raw_m"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
        equal_nan=False,
    ):
        raise ValueError(f"{source} raw visibility differs across seeds")


def validate_against_mainline(table: pd.DataFrame, mainline_path: Path) -> Dict[str, object]:
    if not mainline_path.is_file():
        raise FileNotFoundError(mainline_path)
    main = pd.read_csv(mainline_path)
    label_column = "y_true" if "y_true" in main.columns else "y_cls"
    if "vis_raw_m" not in main.columns:
        raise KeyError(f"{mainline_path} lacks vis_raw_m")
    main_keys = time_station_keys(main, str(mainline_path))
    main_table = pd.DataFrame(
        {
            "time": main_keys["time"],
            "station_key": main_keys["station_key"],
            "duplicate_index_within_time_station": main_keys["duplicate_index_within_time_station"],
            "main_y_cls": pd.to_numeric(main[label_column], errors="coerce"),
            "main_vis_raw_m": pd.to_numeric(main["vis_raw_m"], errors="coerce"),
        }
    )
    if main_table[["time", "station_key", "duplicate_index_within_time_station"]].duplicated().any():
        raise ValueError(f"{mainline_path} has duplicate station-time alignment keys")
    keys = ["time", "station_key", "duplicate_index_within_time_station"]
    joined = table.merge(main_table, on=keys, how="left", validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    if not matched.any():
        raise ValueError("Source P13 output has no rows in common with the mainline P13 test table")
    observed = joined.loc[matched]
    labels_match = np.isclose(
        observed["y_cls"].to_numpy(dtype=float),
        observed["main_y_cls"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=False,
    )
    visibility_match = np.isclose(
        observed["vis_raw_m"].to_numpy(dtype=float),
        observed["main_vis_raw_m"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
        equal_nan=False,
    )
    label_mismatches = int(np.count_nonzero(~labels_match))
    visibility_mismatches = int(np.count_nonzero(~visibility_match))
    if label_mismatches or visibility_mismatches:
        raise ValueError(
            "Source P13 observed data disagree with the mainline P13 test table: "
            f"label_mismatches={label_mismatches}/{int(matched.sum())}; "
            f"raw_visibility_mismatches={visibility_mismatches}/{int(matched.sum())}"
        )
    return {
        "path": str(mainline_path),
        "mainline_rows": int(len(main_table)),
        "source_rows": int(len(table)),
        "matched_rows": int(matched.sum()),
        "source_coverage_of_mainline": float(matched.sum() / max(len(main_table), 1)),
        "observed_data_audit": "passed",
    }


def run(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args.seeds)
    seed_root = Path(args.seed_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    output_csv = out_dir / f"per_sample_{args.source_tag}_p13.csv"
    output_metrics = out_dir / "overall_metrics.csv"
    output_manifest = out_dir / "run_config.json"
    for target in (output_csv, output_metrics, output_manifest):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing P13 output: {target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_paths = [seed_root / f"seed_{seed}" / f"per_sample_{args.source_tag}.csv" for seed in seeds]
    seed_tables = [read_seed_prediction(path, f"seed={seed}") for seed, path in zip(seeds, seed_paths)]
    reference = seed_tables[0]
    for seed, candidate in zip(seeds[1:], seed_tables[1:]):
        validate_same_observed(reference, candidate, f"seed={seed}")

    probabilities = np.mean(
        np.stack(
            [table[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float) for table in seed_tables], axis=0
        ),
        axis=0,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("P13 mean post-softmax probabilities are non-finite")
    pred = np.argmax(probabilities, axis=1).astype(np.int64)
    y_true = reference["y_cls"].to_numpy(dtype=np.int64)
    metrics = compute_metrics(y_true, pred, probs=probabilities)

    output = reference[
        ["time", "station_id", "station_key", "duplicate_index_within_time_station", "y_cls", "vis_raw_m"]
    ].copy()
    output["pred"] = pred
    output["p_fog"] = probabilities[:, 0]
    output["p_mist"] = probabilities[:, 1]
    output["p_clear"] = probabilities[:, 2]
    output["correct"] = pred == y_true
    output["ensemble_seeds"] = ":".join(str(seed) for seed in seeds)
    output["decision_rule"] = "argmax_mean_post_softmax"

    mainline_audit: Dict[str, object] = {"status": "not_requested"}
    if args.mainline_p13_per_sample:
        mainline_audit = validate_against_mainline(
            output, Path(args.mainline_p13_per_sample).expanduser().resolve()
        )

    output.to_csv(output_csv, index=False)
    metric_row = {
        "source": f"{args.source_tag}_p13",
        "label": args.label,
        "n": int(len(output)),
        "decision_rule": "argmax_mean_post_softmax",
        "threshold_source": "argmax_mean_post_softmax",
        "ensemble_size": int(len(seeds)),
        "member_seeds": ",".join(str(seed) for seed in seeds),
        "member_prediction_paths": ";".join(str(path) for path in seed_paths),
        **metrics,
    }
    pd.DataFrame([metric_row]).to_csv(output_metrics, index=False)
    manifest = {
        "schema_version": 1,
        "source_tag": args.source_tag,
        "label": args.label,
        "member_seeds": seeds,
        "member_prediction_paths": [str(path) for path in seed_paths],
        "combination": "equal_weight_mean_post_softmax_then_argmax",
        "output_per_sample": str(output_csv),
        "output_metrics": str(output_metrics),
        "rows": int(len(output)),
        "mainline_p13_alignment": mainline_audit,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mainline-p13-per-sample", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
