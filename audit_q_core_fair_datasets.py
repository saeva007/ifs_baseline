#!/usr/bin/env python3
"""Fail-fast audit for the Pangu-2025 q-core fair-comparison datasets.

The audit is deliberately stricter than the training loader.  The loader can
replace non-finite inputs after scaling, which is useful operationally but can
hide a broken source field.  This script therefore checks the declared layout,
actual arrays, physical sanity, split identity, and label identity before any
S1/S2 training job is allowed to start.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_ORDER = [
    "T2M",
    "MSLP",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "ZENITH",
    "PM10_ugm3",
    "PM25_ugm3",
]
EXPECTED_FEATURE_SET = "q_core_no_rh2m"
S2_SPLITS = ("train", "val", "test")
S1_SPLITS = ("train", "val")


def parse_specs(text: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in str(text or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid source spec {raw!r}; expected tag=/path")
        tag, path = raw.split("=", 1)
        tag = tag.strip()
        if not tag or tag in out:
            raise ValueError(f"Empty or duplicate source tag in {raw!r}")
        out[tag] = Path(path.strip()).expanduser().resolve()
    if len(out) < 2:
        raise ValueError("At least two source datasets are required for a fair-comparison audit.")
    return out


def load_config(path: Path) -> Dict[str, object]:
    cfg_path = path / "dataset_build_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing dataset config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Dataset config must be a JSON object: {cfg_path}")
    return cfg


def normalized_feature_set(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def normalized_station(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )


def metadata_frame(path: Path, split: str) -> pd.DataFrame:
    meta_path = path / f"meta_{split}.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {meta_path}")
    meta = pd.read_csv(meta_path)
    required = {"time", "station_id"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"{meta_path} is missing columns {sorted(missing)}")
    times = pd.to_datetime(meta["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(f"{meta_path} contains {int(times.isna().sum())} invalid timestamps")
    out = meta.copy()
    out["time_key"] = times.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["station_key"] = normalized_station(meta["station_id"])
    out["_row_pos"] = np.arange(len(out), dtype=np.int64)
    if out[["time_key", "station_key"]].duplicated().any():
        n_dup = int(out[["time_key", "station_key"]].duplicated(keep=False).sum())
        raise ValueError(f"{meta_path} has {n_dup} duplicate (time, station_id) rows")
    out.index = pd.MultiIndex.from_frame(out[["time_key", "station_key"]])
    return out


def iter_row_slices(n_rows: int, chunk_rows: int, max_rows: int) -> Iterable[slice]:
    n = n_rows if max_rows <= 0 else min(n_rows, max_rows)
    for start in range(0, n, chunk_rows):
        yield slice(start, min(start + chunk_rows, n))


def plausible_bounds(feature: str) -> Tuple[float, float] | None:
    return {
        "T2M": (180.0, 340.0),
        "MSLP": (500.0, 120000.0),  # accepts hPa or Pa; cross-source units are checked separately
        "U10": (-150.0, 150.0),
        "WSPD10": (0.0, 150.0),
        "V10": (-150.0, 150.0),
        "WDIR10": (0.0, 360.0),
        "RH_925": (0.0, 100.5),
        "U_925": (-150.0, 150.0),
        "WSPD925": (0.0, 150.0),
        "V_925": (-150.0, 150.0),
        "DP_1000": (150.0, 340.0),
        "DP_925": (150.0, 340.0),
        "Q_1000": (0.0, 0.08),
        "Q_925": (0.0, 0.08),
        "ZENITH": (0.0, 180.0),
        "PM10_ugm3": (0.0, 10000.0),
        "PM25_ugm3": (0.0, 10000.0),
    }.get(feature)


def scan_dynamic_features(
    x: np.ndarray,
    window: int,
    order: Sequence[str],
    chunk_rows: int,
    max_rows: int,
) -> List[Dict[str, object]]:
    n_features = len(order)
    accum = {
        name: {
            "n": 0,
            "finite": 0,
            "nonzero": 0,
            "outside": 0,
            "sum": 0.0,
            "min": math.inf,
            "max": -math.inf,
        }
        for name in order
    }
    for sl in iter_row_slices(int(x.shape[0]), chunk_rows, max_rows):
        dyn = np.asarray(x[sl, : window * n_features], dtype=np.float64).reshape(-1, window, n_features)
        for j, name in enumerate(order):
            vals = dyn[:, :, j].reshape(-1)
            finite = np.isfinite(vals)
            a = accum[name]
            a["n"] += int(vals.size)
            a["finite"] += int(finite.sum())
            if finite.any():
                fv = vals[finite]
                a["nonzero"] += int((np.abs(fv) > 1e-12).sum())
                a["sum"] += float(np.sum(fv, dtype=np.float64))
                a["min"] = min(float(a["min"]), float(np.min(fv)))
                a["max"] = max(float(a["max"]), float(np.max(fv)))
                bounds = plausible_bounds(name)
                if bounds is not None:
                    a["outside"] += int(((fv < bounds[0]) | (fv > bounds[1])).sum())
    rows: List[Dict[str, object]] = []
    for name in order:
        a = accum[name]
        finite_n = int(a["finite"])
        total_n = int(a["n"])
        rows.append(
            {
                "feature": name,
                "values_checked": total_n,
                "finite_fraction": finite_n / max(total_n, 1),
                "nonzero_fraction_of_finite": int(a["nonzero"]) / max(finite_n, 1),
                "outside_plausible_fraction": int(a["outside"]) / max(finite_n, 1),
                "mean": float(a["sum"]) / max(finite_n, 1),
                "min": float(a["min"]) if finite_n else math.nan,
                "max": float(a["max"]) if finite_n else math.nan,
            }
        )
    return rows


def audit_dataset(
    tag: str,
    path: Path,
    splits: Sequence[str],
    min_finite: float,
    max_outside: float,
    chunk_rows: int,
    max_rows: int,
    require_meta: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    cfg = load_config(path)
    feature_set = normalized_feature_set(cfg.get("feature_set"))
    if feature_set != EXPECTED_FEATURE_SET:
        raise ValueError(f"{tag}: feature_set={feature_set!r}, expected {EXPECTED_FEATURE_SET!r}")
    order = [str(v) for v in cfg.get("dynamic_feature_order", [])]
    if order != EXPECTED_ORDER:
        raise ValueError(f"{tag}: dynamic_feature_order mismatch\nactual={order}\nexpected={EXPECTED_ORDER}")
    if int(cfg.get("dyn_vars", -1)) != len(EXPECTED_ORDER):
        raise ValueError(f"{tag}: dyn_vars={cfg.get('dyn_vars')} is not {len(EXPECTED_ORDER)}")
    if list(cfg.get("zero_filled_pmst_features", [])):
        raise ValueError(f"{tag}: zero_filled_pmst_features must be empty for a native common-input run")
    if require_meta and str(cfg.get("time_coordinate", "")).upper() != "UTC":
        raise ValueError(f"{tag}: time_coordinate must be explicitly recorded as UTC")
    if require_meta and tag.lower() in {"pangu2025", "pangu_2025"}:
        source_inputs = cfg.get("source_inputs", [])
        if isinstance(source_inputs, str):
            source_inputs = [source_inputs]
        provenance = " ".join(str(v) for v in source_inputs)
        if "lead12_23h" not in provenance.lower():
            raise ValueError(
                f"{tag}: source_inputs do not identify the current 12 <= lead < 24 h Pangu product: {source_inputs}"
            )
    window = int(cfg.get("window", 12))
    fe_dim = int(cfg.get("fe_dim", -1))
    if window != 12 or fe_dim < 0:
        raise ValueError(f"{tag}: unexpected window/fe_dim: window={window}, fe_dim={fe_dim}")

    split_summary: Dict[str, object] = {}
    feature_rows: List[Dict[str, object]] = []
    for split in splits:
        x_path = path / f"X_{split}.npy"
        y_path = path / f"y_{split}.npy"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"{tag}: missing {x_path.name} or {y_path.name} under {path}")
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
            raise ValueError(f"{tag}/{split}: invalid shapes X={x.shape}, y={y.shape}")
        static_dim = int(x.shape[1]) - window * len(order) - fe_dim
        if static_dim != 6:
            raise ValueError(
                f"{tag}/{split}: row width {x.shape[1]} implies static_dim={static_dim}; expected 6"
            )
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y_arr).all() or (y_arr < 0).any() or (y_arr > 10000).any():
            raise ValueError(f"{tag}/{split}: visibility labels contain non-finite or out-of-range values")
        if require_meta:
            meta = metadata_frame(path, split)
            if len(meta) != len(y):
                raise ValueError(f"{tag}/{split}: metadata rows={len(meta)} but labels={len(y)}")
            years = pd.to_datetime(meta["time_key"], utc=True).dt.year.unique().tolist()
            if years != [2025]:
                raise ValueError(f"{tag}/{split}: expected 2025 valid times only, got years={years}")
        rows = scan_dynamic_features(x, window, order, chunk_rows, max_rows)
        for row in rows:
            row.update({"source": tag, "split": split, "dataset_dir": str(path)})
            if float(row["finite_fraction"]) < min_finite:
                raise ValueError(
                    f"{tag}/{split}/{row['feature']}: finite_fraction={row['finite_fraction']:.6f} "
                    f"is below {min_finite:.6f}"
                )
            if float(row["nonzero_fraction_of_finite"]) <= 1e-6:
                raise ValueError(f"{tag}/{split}/{row['feature']}: channel is effectively all zero")
            if float(row["outside_plausible_fraction"]) > max_outside:
                raise ValueError(
                    f"{tag}/{split}/{row['feature']}: outside-plausible fraction "
                    f"{row['outside_plausible_fraction']:.6f} exceeds {max_outside:.6f}"
                )
        feature_rows.extend(rows)
        split_summary[split] = {
            "rows": int(len(y)),
            "row_width": int(x.shape[1]),
            "static_dim": static_dim,
            "label_min_m": float(np.min(y_arr)),
            "label_max_m": float(np.max(y_arr)),
        }
    return {"path": str(path), "config": cfg, "splits": split_summary}, feature_rows


def audit_paired_splits(
    sources: Mapping[str, Path],
    splits: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Tuple[str, str]]]]:
    coverage_rows: List[Dict[str, object]] = []
    common_keys_by_split: Dict[str, List[Tuple[str, str]]] = {}
    for split in splits:
        frames: Dict[str, pd.DataFrame] = {}
        labels: Dict[str, pd.Series] = {}
        common_index: pd.MultiIndex | None = None
        for tag, path in sources.items():
            frame = metadata_frame(path, split)
            y = np.load(path / f"y_{split}.npy", mmap_mode="r")
            frames[tag] = frame
            labels[tag] = pd.Series(np.asarray(y, dtype=np.float64), index=frame.index)
            common_index = frame.index if common_index is None else common_index.intersection(frame.index, sort=False)
        assert common_index is not None
        if len(common_index) == 0:
            raise RuntimeError(f"{split}: no common (time, station_id) rows across sources")
        common_index = common_index.sort_values()
        ref_tag = next(iter(sources))
        ref_y = labels[ref_tag].reindex(common_index).to_numpy()
        for tag, frame in frames.items():
            y = labels[tag].reindex(common_index).to_numpy()
            if not np.allclose(ref_y, y, rtol=0.0, atol=1e-3, equal_nan=False):
                mismatch = int((np.abs(ref_y - y) > 1e-3).sum())
                raise ValueError(f"{split}: {tag} has {mismatch} labels inconsistent with {ref_tag}")
            coverage_rows.append(
                {
                    "split": split,
                    "source": tag,
                    "source_rows": int(len(frame)),
                    "common_rows": int(len(common_index)),
                    "common_fraction": float(len(common_index) / max(len(frame), 1)),
                }
            )
        validate_shared_covariates(sources, frames, common_index, split, ref_tag)
        common_keys_by_split[split] = [(str(a), str(b)) for a, b in common_index.tolist()]

    seen: set[Tuple[str, str]] = set()
    for split in splits:
        keys = set(common_keys_by_split[split])
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"Split leakage: {split} shares {len(overlap)} sample keys with an earlier split")
        seen.update(keys)
    return coverage_rows, common_keys_by_split


def shared_covariate_columns(cfg: Mapping[str, object], row_width: int) -> List[int]:
    window = int(cfg.get("window", 12))
    dyn_vars = int(cfg.get("dyn_vars", len(EXPECTED_ORDER)))
    fe_dim = int(cfg.get("fe_dim", -1))
    if dyn_vars != len(EXPECTED_ORDER) or fe_dim < 4:
        raise ValueError(f"Cannot locate shared columns from dyn_vars={dyn_vars}, fe_dim={fe_dim}")
    dyn_cols: List[int] = []
    for step in range(window):
        base = step * dyn_vars
        dyn_cols.extend(base + EXPECTED_ORDER.index(name) for name in ("ZENITH", "PM10_ugm3", "PM25_ugm3"))
    static_start = window * dyn_vars
    static_cols = list(range(static_start, static_start + 6))
    cyc_cols = list(range(row_width - 4, row_width))
    return [*dyn_cols, *static_cols, *cyc_cols]


def validate_shared_covariates(
    sources: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    split: str,
    ref_tag: str,
    chunk_rows: int = 50000,
) -> None:
    """Verify source-independent inputs are identical on the paired rows.

    Zenith, PM10, PM2.5, station/static attributes, and cyclical valid-time
    encodings should not change with forecast source.  A mismatch usually means
    a time-zone, station-order, PM-year, or metadata-alignment error.
    """
    ref_x = np.load(sources[ref_tag] / f"X_{split}.npy", mmap_mode="r")
    ref_cfg = load_config(sources[ref_tag])
    ref_cols = shared_covariate_columns(ref_cfg, int(ref_x.shape[1]))
    ref_pos = frames[ref_tag].loc[common_index, "_row_pos"].to_numpy(dtype=np.int64)
    for tag, path in sources.items():
        if tag == ref_tag:
            continue
        x = np.load(path / f"X_{split}.npy", mmap_mode="r")
        cfg = load_config(path)
        cols = shared_covariate_columns(cfg, int(x.shape[1]))
        if len(cols) != len(ref_cols):
            raise ValueError(f"{split}: shared-column count differs for {ref_tag} and {tag}")
        pos = frames[tag].loc[common_index, "_row_pos"].to_numpy(dtype=np.int64)
        for start in range(0, len(common_index), chunk_rows):
            end = min(start + chunk_rows, len(common_index))
            left = np.asarray(ref_x[ref_pos[start:end]][:, ref_cols], dtype=np.float64)
            right = np.asarray(x[pos[start:end]][:, cols], dtype=np.float64)
            if not np.allclose(left, right, rtol=1e-5, atol=1e-4, equal_nan=False):
                diff = np.abs(left - right)
                finite = np.isfinite(diff)
                max_diff = float(np.max(diff[finite])) if finite.any() else math.inf
                n_bad = int((~np.isclose(left, right, rtol=1e-5, atol=1e-4, equal_nan=False)).sum())
                raise ValueError(
                    f"{split}: source-independent covariates differ between {ref_tag} and {tag}; "
                    f"bad_values={n_bad}, max_abs_diff={max_diff:.6g}. Check UTC/station/PM alignment."
                )


def validate_shared_protocol(dataset_results: Mapping[str, Dict[str, object]]) -> None:
    fields = ("window", "step", "split", "val_last_days", "test_last_days", "gap_hours")
    tags = list(dataset_results)
    ref_tag = tags[0]
    ref_cfg = dataset_results[ref_tag]["config"]
    for tag in tags[1:]:
        cfg = dataset_results[tag]["config"]
        mismatches = {field: (ref_cfg.get(field), cfg.get(field)) for field in fields if ref_cfg.get(field) != cfg.get(field)}
        if mismatches:
            raise ValueError(f"Protocol mismatch {ref_tag} vs {tag}: {mismatches}")


def validate_cross_source_units(feature_df: pd.DataFrame, source_tags: Sequence[str]) -> None:
    s2 = feature_df[feature_df["source"].isin(source_tags)].copy()
    for split in S2_SPLITS:
        part = s2[s2["split"] == split]
        mslp = part[part["feature"] == "MSLP"].set_index("source")["mean"]
        if len(mslp) != len(source_tags):
            raise ValueError(f"{split}: missing MSLP audit rows for one or more sources")
        unit_family = {tag: ("Pa" if float(value) > 20000.0 else "hPa") for tag, value in mslp.items()}
        if len(set(unit_family.values())) != 1:
            raise ValueError(f"{split}: cross-source MSLP unit mismatch detected: {unit_family}")
        for feature in ("Q_1000", "Q_925"):
            q = part[part["feature"] == feature].set_index("source")["mean"]
            bad = {tag: float(value) for tag, value in q.items() if not (1e-5 < float(value) < 0.04)}
            if bad:
                raise ValueError(f"{split}: {feature} does not look like kg kg-1 for sources {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", required=True, help="Semicolon-separated tag=/dataset/dir specs")
    ap.add_argument("--s1-dir", required=True, help="q-core S1 dataset directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-feature-finite-fraction", type=float, default=0.99)
    ap.add_argument("--max-outside-plausible-fraction", type=float, default=0.01)
    ap.add_argument("--chunk-rows", type=int, default=50000)
    ap.add_argument("--max-rows-per-split", type=int, default=0, help="0 scans every row")
    args = ap.parse_args()

    sources = parse_specs(args.sources)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}
    feature_rows: List[Dict[str, object]] = []
    for tag, path in sources.items():
        result, rows = audit_dataset(
            tag,
            path,
            S2_SPLITS,
            args.min_feature_finite_fraction,
            args.max_outside_plausible_fraction,
            args.chunk_rows,
            args.max_rows_per_split,
            require_meta=True,
        )
        results[tag] = result
        feature_rows.extend(rows)

    s1_result, s1_rows = audit_dataset(
        "s1_q_core_no_rh2m",
        Path(args.s1_dir).expanduser().resolve(),
        S1_SPLITS,
        args.min_feature_finite_fraction,
        args.max_outside_plausible_fraction,
        args.chunk_rows,
        args.max_rows_per_split,
        require_meta=False,
    )
    feature_rows.extend(s1_rows)
    validate_shared_protocol(results)
    coverage_rows, common_keys = audit_paired_splits(sources, S2_SPLITS)

    feature_df = pd.DataFrame(feature_rows)
    validate_cross_source_units(feature_df, list(sources))
    feature_df.to_csv(out_dir / "q_core_feature_quality.csv", index=False)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "q_core_common_sample_coverage.csv", index=False)
    summary = {
        "status": "passed",
        "feature_set": EXPECTED_FEATURE_SET,
        "expected_dynamic_feature_order": EXPECTED_ORDER,
        "sources": results,
        "s1": s1_result,
        "common_rows": {split: len(keys) for split, keys in common_keys.items()},
        "thresholds": {
            "min_feature_finite_fraction": args.min_feature_finite_fraction,
            "max_outside_plausible_fraction": args.max_outside_plausible_fraction,
            "max_rows_per_split": args.max_rows_per_split,
        },
    }
    with (out_dir / "q_core_data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"status": "passed", "common_rows": summary["common_rows"]}, indent=2))


if __name__ == "__main__":
    main()
