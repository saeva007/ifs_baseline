#!/usr/bin/env python3
"""Audit canonical-unit source-full datasets before any GPU training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from pmst_overlap_common import PM_QC_POLICY_VERSION


EXPECTED_POLICY = "pmst_canonical_units_v2_20260630"
REQUIRED_UNITS = {
    "MSLP": "Pa",
    "PM10_ugm3": "ug m-3",
    "PM25_ugm3": "ug m-3",
}
S2_SPLITS = ("train", "val", "test")
S1_SPLITS = ("train", "val")


def parse_specs(raw: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in raw.split(";"):
        if not item.strip():
            continue
        tag, sep, path = item.partition("=")
        if not sep or not tag.strip() or not path.strip() or tag.strip() in out:
            raise ValueError(f"Invalid or duplicate dataset spec: {item!r}")
        out[tag.strip()] = Path(path.strip()).expanduser().resolve()
    if not out:
        raise ValueError("No dataset specs were supplied")
    return out


def load_config(path: Path) -> Dict[str, object]:
    cfg_path = path / "dataset_build_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"Dataset config is not an object: {cfg_path}")
    return cfg


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def metadata_frame(path: Path, split: str) -> pd.DataFrame:
    meta = pd.read_csv(path / f"meta_{split}.csv")
    missing = {"time", "station_id"} - set(meta.columns)
    if missing:
        raise ValueError(f"{path}/meta_{split}.csv missing columns={sorted(missing)}")
    times = pd.to_datetime(meta["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError(f"{path}/meta_{split}.csv contains invalid timestamps")
    out = pd.DataFrame(
        {
            "time_key": times.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "station_key": normalize_station(meta["station_id"]),
            "_row_pos": np.arange(len(meta), dtype=np.int64),
        }
    )
    if out[["time_key", "station_key"]].duplicated().any():
        raise ValueError(f"{path}/meta_{split}.csv has duplicate time/station rows")
    out.index = pd.MultiIndex.from_frame(out[["time_key", "station_key"]])
    return out


def bounds(feature: str) -> Tuple[float, float] | None:
    return {
        "T2M": (180.0, 340.0),
        "MSLP": (50000.0, 120000.0),
        "U10": (-150.0, 150.0),
        "WSPD10": (0.0, 150.0),
        "V10": (-150.0, 150.0),
        "WDIR10": (0.0, 360.0),
        "RH2M": (0.0, 100.5),
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


def row_slices(n: int, chunk: int, max_rows: int) -> Iterable[slice]:
    stop = n if max_rows <= 0 else min(n, max_rows)
    for start in range(0, stop, chunk):
        yield slice(start, min(start + chunk, stop))


def scan_features(
    x: np.ndarray,
    window: int,
    order: Sequence[str],
    chunk_rows: int,
    max_rows: int,
) -> list[Dict[str, object]]:
    accum = {
        name: {"n": 0, "finite": 0, "outside": 0, "nonzero": 0, "sum": 0.0, "min": math.inf, "max": -math.inf}
        for name in order
    }
    for sl in row_slices(len(x), chunk_rows, max_rows):
        dyn = np.asarray(x[sl, : window * len(order)], dtype=np.float64).reshape(-1, window, len(order))
        for idx, name in enumerate(order):
            vals = dyn[:, :, idx].reshape(-1)
            finite = np.isfinite(vals)
            state = accum[name]
            state["n"] += int(vals.size)
            state["finite"] += int(finite.sum())
            if finite.any():
                fv = vals[finite]
                state["sum"] += float(fv.sum(dtype=np.float64))
                state["min"] = min(float(state["min"]), float(fv.min()))
                state["max"] = max(float(state["max"]), float(fv.max()))
                state["nonzero"] += int((np.abs(fv) > 1e-12).sum())
                limit = bounds(name)
                if limit is not None:
                    state["outside"] += int(((fv < limit[0]) | (fv > limit[1])).sum())
    rows = []
    for name, state in accum.items():
        finite_n = int(state["finite"])
        rows.append(
            {
                "feature": name,
                "values_checked": int(state["n"]),
                "finite_fraction": finite_n / max(int(state["n"]), 1),
                "outside_plausible_fraction": int(state["outside"]) / max(finite_n, 1),
                "nonzero_fraction": int(state["nonzero"]) / max(finite_n, 1),
                "mean": float(state["sum"]) / max(finite_n, 1),
                "min": float(state["min"]) if finite_n else math.nan,
                "max": float(state["max"]) if finite_n else math.nan,
            }
        )
    return rows


def audit_dataset(
    tag: str,
    path: Path,
    splits: Sequence[str],
    require_meta: bool,
    chunk_rows: int,
    max_rows: int,
    min_finite: float,
    max_outside: float,
) -> Tuple[Dict[str, object], list[Dict[str, object]], Dict[str, pd.DataFrame]]:
    cfg = load_config(path)
    if str(cfg.get("canonical_unit_policy", "")) != EXPECTED_POLICY:
        raise ValueError(f"{tag}: missing canonical unit policy {EXPECTED_POLICY}")
    if tag.startswith("s1_") and str(cfg.get("pm_qc_policy", "")) != PM_QC_POLICY_VERSION:
        raise ValueError(
            f"{tag}: missing S1 PM QC policy {PM_QC_POLICY_VERSION}; rebuild this S1 profile"
        )
    units = cfg.get("canonical_dynamic_units")
    if not isinstance(units, Mapping):
        raise ValueError(f"{tag}: canonical_dynamic_units is missing")
    bad_units = {name: units.get(name) for name, expected in REQUIRED_UNITS.items() if units.get(name) != expected}
    if bad_units:
        raise ValueError(f"{tag}: canonical units mismatch: {bad_units}")
    if str(cfg.get("feature_set", "")).lower().replace("-", "_") != "source_full":
        raise ValueError(f"{tag}: feature_set is not source_full")
    order = [str(v) for v in cfg.get("dynamic_feature_order", [])]
    if not order or int(cfg.get("dyn_vars", -1)) != len(order):
        raise ValueError(f"{tag}: invalid dynamic feature order/dyn_vars")
    required = {"MSLP", "ZENITH", "PM10_ugm3", "PM25_ugm3"}
    if not required.issubset(order):
        raise ValueError(f"{tag}: source-full layout misses {sorted(required - set(order))}")
    if list(cfg.get("zero_filled_pmst_features", [])):
        raise ValueError(f"{tag}: source-full data contains zero-filled placeholder features")
    window = int(cfg.get("window", 12))
    fe_dim = int(cfg.get("fe_dim", -1))
    if window != 12 or fe_dim < 4:
        raise ValueError(f"{tag}: invalid window/fe_dim={window}/{fe_dim}")

    feature_rows: list[Dict[str, object]] = []
    frames: Dict[str, pd.DataFrame] = {}
    split_summary: Dict[str, object] = {}
    for split in splits:
        x = np.load(path / f"X_{split}.npy", mmap_mode="r")
        y = np.load(path / f"y_{split}.npy", mmap_mode="r")
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
            raise ValueError(f"{tag}/{split}: invalid X/y shapes {x.shape}/{y.shape}")
        static_dim = int(x.shape[1]) - window * len(order) - fe_dim
        if static_dim != 6:
            raise ValueError(f"{tag}/{split}: static_dim={static_dim}, expected 6")
        upper = float(cfg.get("max_vis_threshold", 30000.0 if require_meta else 90000.0))
        if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > upper):
            raise ValueError(f"{tag}/{split}: invalid visibility labels")
        if require_meta:
            frame = metadata_frame(path, split)
            if len(frame) != len(y):
                raise ValueError(f"{tag}/{split}: metadata/label row mismatch")
            frames[split] = frame
        rows = scan_features(x, window, order, chunk_rows, max_rows)
        for row in rows:
            row.update({"source": tag, "split": split})
            feature_rows.append(row)
            if float(row["finite_fraction"]) < min_finite:
                raise ValueError(f"{tag}/{split}/{row['feature']}: finite fraction too low: {row['finite_fraction']}")
            if bounds(str(row["feature"])) is not None and float(row["outside_plausible_fraction"]) > max_outside:
                raise ValueError(
                    f"{tag}/{split}/{row['feature']}: outside-plausible fraction "
                    f"{row['outside_plausible_fraction']:.6f} exceeds {max_outside:.6f}"
                )
        critical = {row["feature"]: row for row in rows if row["feature"] in required}
        if any(float(critical[name]["nonzero_fraction"]) <= 0 for name in required):
            raise ValueError(f"{tag}/{split}: a required dynamic channel is all zero")
        split_summary[split] = {"rows": int(len(y)), "row_width": int(x.shape[1]), "static_dim": static_dim}
    return {"path": str(path), "config": cfg, "splits": split_summary}, feature_rows, frames


def shared_columns(cfg: Mapping[str, object], width: int) -> list[int]:
    order = [str(v) for v in cfg["dynamic_feature_order"]]
    window = int(cfg["window"])
    cols = []
    for step in range(window):
        base = step * len(order)
        cols.extend(base + order.index(name) for name in ("ZENITH", "PM10_ugm3", "PM25_ugm3"))
    static_start = window * len(order)
    return [*cols, *range(static_start, static_start + 6), *range(width - 4, width)]


def audit_pairing(
    sources: Mapping[str, Path],
    results: Mapping[str, Dict[str, object]],
    frames: Mapping[str, Mapping[str, pd.DataFrame]],
) -> Dict[str, int]:
    coverage: Dict[str, int] = {}
    ref_tag = next(iter(sources))
    for split in S2_SPLITS:
        common = frames[ref_tag][split].index
        for tag in sources:
            common = common.intersection(frames[tag][split].index, sort=False)
        if len(common) == 0:
            raise ValueError(f"{split}: no common source-full samples")
        coverage[split] = int(len(common))
        ref_x = np.load(sources[ref_tag] / f"X_{split}.npy", mmap_mode="r")
        ref_y = np.load(sources[ref_tag] / f"y_{split}.npy", mmap_mode="r")
        ref_pos = frames[ref_tag][split].loc[common, "_row_pos"].to_numpy(dtype=np.int64)
        ref_cols = shared_columns(results[ref_tag]["config"], int(ref_x.shape[1]))
        for tag in list(sources)[1:]:
            x = np.load(sources[tag] / f"X_{split}.npy", mmap_mode="r")
            y = np.load(sources[tag] / f"y_{split}.npy", mmap_mode="r")
            pos = frames[tag][split].loc[common, "_row_pos"].to_numpy(dtype=np.int64)
            if not np.allclose(ref_y[ref_pos], y[pos], rtol=0.0, atol=1e-3):
                raise ValueError(f"{split}: labels differ between {ref_tag} and {tag}")
            cols = shared_columns(results[tag]["config"], int(x.shape[1]))
            for start in range(0, len(common), 50000):
                end = min(start + 50000, len(common))
                left = np.asarray(ref_x[ref_pos[start:end]][:, ref_cols], dtype=np.float64)
                right = np.asarray(x[pos[start:end]][:, cols], dtype=np.float64)
                if not np.allclose(left, right, rtol=1e-5, atol=1e-4):
                    diff = np.abs(left - right)
                    raise ValueError(
                        f"{split}: shared covariates differ between {ref_tag} and {tag}; "
                        f"max_abs_diff={float(np.nanmax(diff)):.6g}"
                    )
    return coverage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--s1-profiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-rows", type=int, default=50000)
    ap.add_argument("--max-rows-per-split", type=int, default=0)
    ap.add_argument("--min-feature-finite-fraction", type=float, default=0.99)
    ap.add_argument("--max-outside-plausible-fraction", type=float, default=0.01)
    args = ap.parse_args()

    sources = parse_specs(args.sources)
    s1_profiles = parse_specs(args.s1_profiles)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    rows: list[Dict[str, object]] = []
    issues: list[str] = []
    for tag, path in sources.items():
        try:
            result, feature_rows, source_frames = audit_dataset(
                tag, path, S2_SPLITS, True, args.chunk_rows, args.max_rows_per_split,
                args.min_feature_finite_fraction, args.max_outside_plausible_fraction,
            )
            results[tag] = result
            frames[tag] = source_frames
            rows.extend(feature_rows)
        except Exception as exc:
            issues.append(f"{tag}: {type(exc).__name__}: {exc}")
    s1_results: Dict[str, Dict[str, object]] = {}
    for tag, path in s1_profiles.items():
        try:
            result, feature_rows, _ = audit_dataset(
                f"s1_{tag}", path, S1_SPLITS, False, args.chunk_rows, args.max_rows_per_split,
                args.min_feature_finite_fraction, args.max_outside_plausible_fraction,
            )
            s1_results[tag] = result
            rows.extend(feature_rows)
        except Exception as exc:
            issues.append(f"s1_{tag}: {type(exc).__name__}: {exc}")

    pd.DataFrame(rows).to_csv(out_dir / "source_full_feature_quality.csv", index=False)
    if issues:
        failure = {"status": "failed", "stage": "per_dataset_checks", "issues": issues}
        (out_dir / "source_full_data_audit_failed.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError("source-full audit failed:\n- " + "\n- ".join(issues))

    profile_map = {"tianji": "tianji", "t2nd": "tianji", "ifs": "ifs", "pangu2025": "pangu2025", "era5_2025": "tianji"}
    for source, profile in profile_map.items():
        s2_order = results[source]["config"]["dynamic_feature_order"]
        s1_order = s1_results[profile]["config"]["dynamic_feature_order"]
        if s2_order != s1_order:
            issues.append(f"S1/S2 dynamic layout mismatch for {source} using profile {profile}")

    try:
        common_rows = audit_pairing(sources, results, frames)
    except Exception as exc:
        common_rows = {}
        issues.append(f"paired-source checks: {type(exc).__name__}: {exc}")
    if issues:
        failure = {"status": "failed", "stage": "layout_and_pairing_checks", "issues": issues}
        (out_dir / "source_full_data_audit_failed.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError("source-full audit failed:\n- " + "\n- ".join(issues))
    summary = {
        "status": "passed",
        "canonical_unit_policy": EXPECTED_POLICY,
        "sources": results,
        "s1_profiles": s1_results,
        "paired_common_rows": common_rows,
    }
    (out_dir / "source_full_data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "paired_common_rows": common_rows}, indent=2))


if __name__ == "__main__":
    main()
