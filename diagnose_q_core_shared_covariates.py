#!/usr/bin/env python3
"""Locate source-independent q-core differences at logical-column resolution.

This is a read-only diagnostic for already-built datasets.  It aligns rows by
UTC valid time and station ID, then compares zenith, PM10, PM2.5, the six
static inputs, and the four cyclical time encodings.  It does not modify data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


DYNAMIC_SHARED = ("ZENITH", "PM10_ugm3", "PM25_ugm3")
STATIC_NAMES = (
    "LAT_NORM",
    "LON_NORM",
    "OROGRAPHY_CENTER_M",
    "OROGRAPHY_LOCAL_ANOMALY_M",
    "OROGRAPHY_LOCAL_STD_M",
    "VEGETATION_ID",
)
CYCLICAL_NAMES = ("MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS")


def load_config(path: Path) -> Dict[str, object]:
    cfg_path = path / "dataset_build_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a JSON object: {cfg_path}")
    return cfg


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def metadata_frame(path: Path, split: str) -> pd.DataFrame:
    meta_path = path / f"meta_{split}.csv"
    meta = pd.read_csv(meta_path, usecols=["time", "station_id"])
    times = pd.to_datetime(meta["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(f"{meta_path}: {int(times.isna().sum())} invalid timestamps")
    out = pd.DataFrame(
        {
            "time_key": times.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "station_key": normalize_station(meta["station_id"]),
            "_row_pos": np.arange(len(meta), dtype=np.int64),
        }
    )
    if out[["time_key", "station_key"]].duplicated().any():
        raise ValueError(f"{meta_path}: duplicate (time, station_id) rows")
    out.index = pd.MultiIndex.from_frame(out[["time_key", "station_key"]])
    return out


def shared_columns(
    cfg: Mapping[str, object],
    row_width: int,
) -> Tuple[List[str], List[str], List[int]]:
    order = [str(v) for v in cfg.get("dynamic_feature_order", [])]
    window = int(cfg.get("window", 12))
    fe_dim = int(cfg.get("fe_dim", -1))
    if window < 1 or fe_dim < 4:
        raise ValueError(f"Invalid window/fe_dim: window={window}, fe_dim={fe_dim}")
    missing = [name for name in DYNAMIC_SHARED if name not in order]
    if missing:
        raise ValueError(f"Missing shared dynamic features: {missing}")
    dyn_vars = len(order)
    dynamic_width = window * dyn_vars
    static_dim = row_width - dynamic_width - fe_dim
    if static_dim != len(STATIC_NAMES):
        raise ValueError(f"Expected {len(STATIC_NAMES)} static columns, got {static_dim}")

    groups: List[str] = []
    names: List[str] = []
    columns: List[int] = []
    for step in range(window):
        lag = step - (window - 1)
        for feature in DYNAMIC_SHARED:
            groups.append(feature)
            names.append(f"{feature}[lag={lag:+d}step]")
            columns.append(step * dyn_vars + order.index(feature))
    static_start = dynamic_width
    for offset, name in enumerate(STATIC_NAMES):
        groups.append(name)
        names.append(name)
        columns.append(static_start + offset)
    for offset, name in enumerate(CYCLICAL_NAMES):
        groups.append(name)
        names.append(name)
        columns.append(row_width - len(CYCLICAL_NAMES) + offset)
    return groups, names, columns


def summarize_columns(
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[str],
    names: Sequence[str],
    rtol: float,
    atol: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for j, (group, name) in enumerate(zip(groups, names)):
        a = left[:, j]
        b = right[:, j]
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        finite = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a - b)
        finite_diff = diff[finite]
        rows.append(
            {
                "group": group,
                "column": name,
                "rows_checked": int(len(a)),
                "bad_count": int((~close).sum()),
                "bad_fraction": float((~close).mean()),
                "nonfinite_pair_count": int((~finite).sum()),
                "max_abs_diff": float(np.max(finite_diff)) if finite_diff.size else np.nan,
                "mean_abs_diff": float(np.mean(finite_diff)) if finite_diff.size else np.nan,
                "left_mean": float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else np.nan,
                "right_mean": float(np.mean(b[np.isfinite(b)])) if np.isfinite(b).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def group_summary(columns: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group, part in columns.groupby("group", sort=False):
        denom = int(part["rows_checked"].sum())
        bad = int(part["bad_count"].sum())
        rows.append(
            {
                "group": group,
                "logical_columns": int(len(part)),
                "values_checked": denom,
                "bad_count": bad,
                "bad_fraction": float(bad / max(denom, 1)),
                "max_abs_diff": float(part["max_abs_diff"].max()),
                "mean_abs_diff_across_columns": float(part["mean_abs_diff"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["bad_count", "max_abs_diff"], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left-dir", required=True)
    ap.add_argument("--right-dir", required=True)
    ap.add_argument("--left-tag", default="tianji")
    ap.add_argument("--right-tag", default="pangu2025")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--max-common-rows", type=int, default=50000)
    ap.add_argument("--rtol", type=float, default=1.0e-5)
    ap.add_argument("--atol", type=float, default=1.0e-4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    left_dir = Path(args.left_dir).expanduser().resolve()
    right_dir = Path(args.right_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    left_meta = metadata_frame(left_dir, args.split)
    right_meta = metadata_frame(right_dir, args.split)
    common = left_meta.index.intersection(right_meta.index, sort=False).sort_values()
    if args.max_common_rows > 0:
        common = common[: args.max_common_rows]
    if len(common) == 0:
        raise RuntimeError("No common (time, station_id) rows")

    left_pos = left_meta.loc[common, "_row_pos"].to_numpy(dtype=np.int64)
    right_pos = right_meta.loc[common, "_row_pos"].to_numpy(dtype=np.int64)
    left_x = np.load(left_dir / f"X_{args.split}.npy", mmap_mode="r")
    right_x = np.load(right_dir / f"X_{args.split}.npy", mmap_mode="r")
    left_groups, left_names, left_columns = shared_columns(load_config(left_dir), int(left_x.shape[1]))
    right_groups, right_names, right_columns = shared_columns(load_config(right_dir), int(right_x.shape[1]))
    if left_groups != right_groups or left_names != right_names:
        raise ValueError("Shared logical-column layouts differ between datasets")

    left_values = np.asarray(left_x[left_pos][:, left_columns], dtype=np.float64)
    right_values = np.asarray(right_x[right_pos][:, right_columns], dtype=np.float64)
    columns = summarize_columns(
        left_values, right_values, left_groups, left_names, args.rtol, args.atol
    )
    groups = group_summary(columns)
    columns.to_csv(out_dir / "shared_covariate_column_summary.csv", index=False)
    groups.to_csv(out_dir / "shared_covariate_group_summary.csv", index=False)

    payload = {
        "left_tag": args.left_tag,
        "right_tag": args.right_tag,
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "split": args.split,
        "common_rows_checked": int(len(common)),
        "rtol": args.rtol,
        "atol": args.atol,
        "groups_with_differences": groups[groups["bad_count"] > 0].to_dict(orient="records"),
    }
    with (out_dir / "shared_covariate_diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Compared {len(common)} common rows: {args.left_tag} vs {args.right_tag}")
    print(groups.to_string(index=False))
    print("\nTop differing logical columns:")
    print(columns.sort_values(["bad_count", "max_abs_diff"], ascending=False).head(20).to_string(index=False))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
