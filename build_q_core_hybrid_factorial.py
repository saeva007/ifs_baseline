#!/usr/bin/env python3
"""Build and audit Pangu/Tianji q-core hybrid factorial datasets.

Each output starts from the Pangu q-core rows and replaces complete 12-hour
sequences from Tianji for the selected physical packages.  Fog-derived
features are then recomputed from the hybrid dynamic tensor.  Labels, sample
metadata, static inputs, cyclical time encodings, zenith and aerosol channels
must be identical across the two source datasets; a mismatch is a hard error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

try:
    import pvlib  # noqa: F401
except ModuleNotFoundError:
    # This builder reuses only deterministic feature-engineering helpers; it
    # never computes solar geometry. Keep the source dataset builders strict.
    sys.modules["pvlib"] = types.ModuleType("pvlib")

from pmst_overlap_common import (
    CANONICAL_UNIT_POLICY_VERSION,
    PM_QC_POLICY_VERSION,
    Q_CORE_NO_RH2M_DYN_FEATURES,
    compute_fog_features_pmst,
)


EXPECTED_ORDER = list(Q_CORE_NO_RH2M_DYN_FEATURES)
GROUPS: Dict[str, Tuple[str, ...]] = {
    "M": ("Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"),
    "T": ("T2M", "MSLP"),
    "W": ("U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925"),
}
GROUP_ORDER = ("M", "T", "W")
SHARED_DYNAMIC = ("ZENITH", "PM10_ugm3", "PM25_ugm3")
STATIC_DIM = 6
CYCLICAL_DIM = 4
DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-dir", required=True)
    ap.add_argument("--tianji-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--masks", default="all", help="all or comma-separated MTW masks, e.g. 000,100,111")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--limit-rows", type=int, default=0, help="Smoke-test row limit per split; 0 uses all rows.")
    ap.add_argument("--rtol", type=float, default=1.0e-6)
    ap.add_argument("--atol", type=float, default=1.0e-6)
    ap.add_argument("--audit-only", action="store_true")
    return ap.parse_args()


def parse_csv(value: str) -> List[str]:
    normalized = str(value).replace(";", ",").replace(":", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def parse_masks(value: str) -> List[str]:
    masks = [f"{i:03b}" for i in range(8)] if str(value).strip().lower() == "all" else parse_csv(value)
    bad = [mask for mask in masks if len(mask) != 3 or any(bit not in "01" for bit in mask)]
    if bad:
        raise ValueError(f"Invalid MTW mask(s): {bad}")
    return list(dict.fromkeys(masks))


def read_config(data_dir: Path) -> Dict[str, object]:
    path = data_dir / "dataset_build_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cfg


def require_layout(data_dir: Path) -> Dict[str, object]:
    cfg = read_config(data_dir)
    order = [str(v) for v in cfg.get("dynamic_feature_order", [])]
    if str(cfg.get("feature_set")) != "q_core_no_rh2m":
        raise ValueError(f"{data_dir}: feature_set must be q_core_no_rh2m")
    if order != EXPECTED_ORDER:
        raise ValueError(f"{data_dir}: dynamic_feature_order mismatch\nactual={order}\nexpected={EXPECTED_ORDER}")
    if int(cfg.get("dyn_vars", -1)) != len(EXPECTED_ORDER):
        raise ValueError(f"{data_dir}: dyn_vars must be {len(EXPECTED_ORDER)}")
    if int(cfg.get("window", -1)) != 12:
        raise ValueError(f"{data_dir}: window must be 12")
    if str(cfg.get("canonical_unit_policy", "")) != CANONICAL_UNIT_POLICY_VERSION:
        raise ValueError(f"{data_dir}: canonical unit policy is not {CANONICAL_UNIT_POLICY_VERSION}")
    pm_policy = cfg.get("pm_qc_policy")
    if pm_policy not in (None, "") and str(pm_policy) != PM_QC_POLICY_VERSION:
        raise ValueError(f"{data_dir}: PM QC policy is not {PM_QC_POLICY_VERSION}")
    if cfg.get("zero_filled_pmst_features") not in (None, []):
        raise ValueError(f"{data_dir}: q-core must not contain zero-filled feature slots")
    return cfg


def require_canonical_pangu_lead(cfg: Mapping[str, object], data_dir: Path) -> None:
    lead = cfg.get("source_forecast_lead")
    if not isinstance(lead, Mapping):
        raise ValueError(f"{data_dir}: missing source_forecast_lead provenance")
    lo, hi = lead.get("min_hours"), lead.get("max_hours")
    if lo is None or hi is None:
        raise ValueError(f"{data_dir}: source_forecast_lead lacks min_hours/max_hours")
    if not math.isclose(float(lo), 12.0, rel_tol=0.0, abs_tol=1.0e-6) or not math.isclose(
        float(hi), 23.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(f"{data_dir}: Pangu lead range must be canonical 12--23 h, got {lo}--{hi} h")


def normalize_station(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def canonical_meta(path: Path, limit: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "station_id"}
    if not required.issubset(df.columns):
        raise KeyError(f"{path}: missing {sorted(required - set(df.columns))}")
    if limit > 0:
        df = df.iloc[:limit].copy()
    time = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError(f"{path}: invalid timestamps")
    keys = pd.DataFrame(
        {
            "time": time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "station_id": normalize_station(df["station_id"]),
        }
    )
    if keys.duplicated().any():
        raise ValueError(f"{path}: duplicate (time, station_id) rows")
    return keys


def row_alignment_hash(keys: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for time_value, station in keys[["time", "station_id"]].itertuples(index=False, name=None):
        h.update(str(time_value).encode("utf-8"))
        h.update(b"|")
        h.update(str(station).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_close(name: str, left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name}: shape mismatch {left.shape} != {right.shape}")
    if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        finite = diff[np.isfinite(diff)]
        maximum = float(finite.max()) if finite.size else math.nan
        raise ValueError(f"{name}: values differ (max_abs_diff={maximum})")


def mask_groups(mask: str) -> List[str]:
    return [name for name, bit in zip(GROUP_ORDER, mask) if bit == "1"]


def group_columns(order: Sequence[str], selected_groups: Iterable[str]) -> List[int]:
    selected = set(selected_groups)
    names = [feature for group in GROUP_ORDER if group in selected for feature in GROUPS[group]]
    return [order.index(name) for name in names]


def verify_source_pair(
    pangu_dir: Path,
    tianji_dir: Path,
    split: str,
    cfg: Mapping[str, object],
    limit_rows: int,
    chunk_rows: int,
    rtol: float,
    atol: float,
) -> Tuple[int, str]:
    p_meta = canonical_meta(pangu_dir / f"meta_{split}.csv", limit_rows)
    t_meta = canonical_meta(tianji_dir / f"meta_{split}.csv", limit_rows)
    if not p_meta.equals(t_meta):
        raise ValueError(f"{split}: Pangu and Tianji metadata/order differ; hybrids require exact paired rows")
    n = len(p_meta)
    p_y = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")[:n]
    t_y = np.load(tianji_dir / f"y_{split}.npy", mmap_mode="r")[:n]
    assert_close(f"{split}: labels", p_y, t_y, rtol=0.0, atol=1.0e-6)

    p_x = np.load(pangu_dir / f"X_{split}.npy", mmap_mode="r")
    t_x = np.load(tianji_dir / f"X_{split}.npy", mmap_mode="r")
    if p_x.shape[0] < n or t_x.shape[0] < n or p_x.shape[1:] != t_x.shape[1:]:
        raise ValueError(f"{split}: feature matrix shapes differ: {p_x.shape} vs {t_x.shape}")
    dyn = int(cfg["dyn_vars"])
    window = int(cfg["window"])
    fe_dim = int(cfg["fe_dim"])
    expected_width = window * dyn + STATIC_DIM + fe_dim
    if int(p_x.shape[1]) != expected_width:
        raise ValueError(f"{split}: row width {p_x.shape[1]} != expected {expected_width}")
    order = [str(v) for v in cfg["dynamic_feature_order"]]
    shared_dyn_cols = [step * dyn + order.index(name) for step in range(window) for name in SHARED_DYNAMIC]
    static_start = window * dyn
    shared_cols = shared_dyn_cols + list(range(static_start, static_start + STATIC_DIM))
    shared_cols += list(range(expected_width - CYCLICAL_DIM, expected_width))
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        assert_close(
            f"{split}: shared source-independent columns rows {start}:{end}",
            np.asarray(p_x[start:end, shared_cols]),
            np.asarray(t_x[start:end, shared_cols]),
            rtol,
            atol,
        )
    return n, row_alignment_hash(p_meta)


def build_one_split(
    pangu_dir: Path,
    tianji_dir: Path,
    out_dir: Path,
    split: str,
    mask: str,
    cfg: Mapping[str, object],
    n: int,
    chunk_rows: int,
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    p_x = np.load(pangu_dir / f"X_{split}.npy", mmap_mode="r")
    t_x = np.load(tianji_dir / f"X_{split}.npy", mmap_mode="r")
    out_path = out_dir / f"X_{split}.npy"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")
    out_x = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, int(p_x.shape[1])))
    order = [str(v) for v in cfg["dynamic_feature_order"]]
    dyn = int(cfg["dyn_vars"])
    window = int(cfg["window"])
    fe_dim = int(cfg["fe_dim"])
    fog_dim = fe_dim - CYCLICAL_DIM
    if fog_dim < 1:
        raise ValueError(f"{split}: fe_dim={fe_dim} leaves no fog-derived features")
    dyn_width = window * dyn
    donor_idx = group_columns(order, mask_groups(mask))
    max_abs_endpoint = 0.0

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        base_rows = np.asarray(p_x[start:end], dtype=np.float32)
        donor_rows = np.asarray(t_x[start:end], dtype=np.float32)
        hybrid = base_rows[:, :dyn_width].reshape(-1, window, dyn).copy()
        if donor_idx:
            donor_dyn = donor_rows[:, :dyn_width].reshape(-1, window, dyn)
            hybrid[:, :, donor_idx] = donor_dyn[:, :, donor_idx]
        fog = compute_fog_features_pmst(hybrid, window, dyn, order)
        if fog.shape[1] != fog_dim:
            raise ValueError(
                f"{split}/{mask}: recomputed fog feature dim {fog.shape[1]} != expected {fog_dim}; "
                "the source datasets were not built with the current shared FE implementation"
            )
        static = base_rows[:, dyn_width : dyn_width + STATIC_DIM]
        cyc = base_rows[:, -CYCLICAL_DIM:]
        rows = np.concatenate([hybrid.reshape(end - start, dyn_width), static, fog, cyc], axis=1).astype(np.float32)
        out_x[start:end] = rows
        if mask in {"000", "111"}:
            expected = base_rows if mask == "000" else donor_rows
            diff = np.abs(rows.astype(np.float64) - expected.astype(np.float64))
            finite = diff[np.isfinite(diff)]
            max_abs_endpoint = max(max_abs_endpoint, float(finite.max()) if finite.size else 0.0)
            assert_close(f"{split}/{mask}: endpoint identity", rows, expected, rtol, atol)
    out_x.flush()
    del out_x
    shutil.copy2(pangu_dir / f"y_{split}.npy", out_dir / f"y_{split}.npy")
    if n != int(np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r").shape[0]):
        y = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")[:n]
        np.save(out_dir / f"y_{split}.npy", np.asarray(y))
    meta = pd.read_csv(pangu_dir / f"meta_{split}.csv").iloc[:n]
    meta.to_csv(out_dir / f"meta_{split}.csv", index=False)
    return {
        "split": split,
        "rows": n,
        "mask": mask,
        "replaced_groups": mask_groups(mask),
        "replaced_features": [order[i] for i in donor_idx],
        "endpoint_max_abs_diff": max_abs_endpoint if mask in {"000", "111"} else None,
    }


def config_for_mask(
    base_cfg: Mapping[str, object],
    pangu_dir: Path,
    tianji_dir: Path,
    mask: str,
    split_hashes: Mapping[str, str],
    limit_rows: int,
) -> Dict[str, object]:
    cfg = dict(base_cfg)
    selected_groups = mask_groups(mask)
    cfg.update(
        {
            "dataset": "pangu_tianji_q_core_hybrid_factorial",
            "feature_set": "q_core_no_rh2m",
            "hybrid_base_source": "pangu2025",
            "hybrid_donor_source": "tianji",
            "hybrid_group_order": list(GROUP_ORDER),
            "hybrid_group_mask": mask,
            "hybrid_group_definitions": {key: list(value) for key, value in GROUPS.items()},
            "replaced_feature_groups": selected_groups,
            "replaced_features": [feature for group in selected_groups for feature in GROUPS[group]],
            "base_dataset_dir": str(pangu_dir),
            "donor_dataset_dir": str(tianji_dir),
            "row_alignment_sha256": dict(split_hashes),
            "recomputed_fog_features": True,
            "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
            "pm_qc_policy": PM_QC_POLICY_VERSION,
            "hybrid_smoke_limit_rows": int(limit_rows),
            "scientific_role": "controlled source-block retraining attribution; not a single-variable causal effect",
        }
    )
    return cfg


def audit_outputs(
    pangu_dir: Path,
    tianji_dir: Path,
    out_root: Path,
    masks: Sequence[str],
    splits: Sequence[str],
    chunk_rows: int,
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    base_cfg = require_layout(pangu_dir)
    require_canonical_pangu_lead(base_cfg, pangu_dir)
    require_layout(tianji_dir)
    order = [str(v) for v in base_cfg["dynamic_feature_order"]]
    dyn = int(base_cfg["dyn_vars"])
    window = int(base_cfg["window"])
    dyn_width = dyn * window
    rows: List[Dict[str, object]] = []
    for mask in masks:
        data_dir = out_root / f"mtw_{mask}"
        cfg = require_layout(data_dir)
        if str(cfg.get("hybrid_group_mask")) != mask or not bool(cfg.get("recomputed_fog_features")):
            raise ValueError(f"{data_dir}: hybrid provenance is missing or inconsistent")
        selected_idx = set(group_columns(order, mask_groups(mask)))
        for split in splits:
            x = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
            p = np.load(pangu_dir / f"X_{split}.npy", mmap_mode="r")
            t = np.load(tianji_dir / f"X_{split}.npy", mmap_mode="r")
            n = len(x)
            out_meta = canonical_meta(data_dir / f"meta_{split}.csv")
            base_meta = canonical_meta(pangu_dir / f"meta_{split}.csv", n)
            if not out_meta.equals(base_meta):
                raise ValueError(f"audit {mask}/{split}: metadata differ from the Pangu base")
            out_y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
            base_y = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")[:n]
            assert_close(f"audit {mask}/{split}: labels", np.asarray(out_y), np.asarray(base_y), 0.0, atol)
            fe_dim = int(cfg["fe_dim"])
            fog_dim = fe_dim - CYCLICAL_DIM
            static_start = dyn_width
            fog_start = static_start + STATIC_DIM
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                hybrid_dyn = np.asarray(x[start:end, :dyn_width]).reshape(end - start, window, dyn)
                expected_dyn = np.asarray(p[start:end, :dyn_width]).reshape(end - start, window, dyn).copy()
                if selected_idx:
                    donor_dyn = np.asarray(t[start:end, :dyn_width]).reshape(end - start, window, dyn)
                    donor_columns = sorted(selected_idx)
                    expected_dyn[:, :, donor_columns] = donor_dyn[:, :, donor_columns]
                assert_close(
                    f"audit {mask}/{split} dynamic rows {start}:{end}",
                    hybrid_dyn,
                    expected_dyn,
                    rtol,
                    atol,
                )
                expected_fog = compute_fog_features_pmst(hybrid_dyn, window, dyn, order)
                assert_close(
                    f"audit {mask}/{split} recomputed FE rows {start}:{end}",
                    np.asarray(x[start:end, fog_start : fog_start + fog_dim]),
                    expected_fog,
                    rtol,
                    atol,
                )
                shared_tail_cols = list(range(static_start, static_start + STATIC_DIM)) + list(
                    range(int(x.shape[1]) - CYCLICAL_DIM, int(x.shape[1]))
                )
                assert_close(
                    f"audit {mask}/{split} static/cyclical rows {start}:{end}",
                    np.asarray(x[start:end, shared_tail_cols]),
                    np.asarray(p[start:end, shared_tail_cols]),
                    rtol,
                    atol,
                )
                if mask == "000":
                    assert_close(
                        f"audit {mask}/{split} full endpoint rows {start}:{end}",
                        np.asarray(x[start:end]),
                        np.asarray(p[start:end]),
                        rtol,
                        atol,
                    )
                if mask == "111":
                    assert_close(
                        f"audit {mask}/{split} full endpoint rows {start}:{end}",
                        np.asarray(x[start:end]),
                        np.asarray(t[start:end]),
                        rtol,
                        atol,
                    )
            rows.append({"mask": mask, "split": split, "rows": n, "status": "passed"})
    return {
        "status": "passed",
        "masks": list(masks),
        "splits": list(splits),
        "checks": rows,
        "group_order": list(GROUP_ORDER),
        "group_definitions": {key: list(value) for key, value in GROUPS.items()},
    }


def main() -> None:
    args = parse_args()
    pangu_dir = Path(args.pangu_dir).expanduser().resolve()
    tianji_dir = Path(args.tianji_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    masks = parse_masks(args.masks)
    splits = parse_csv(args.splits)
    if not splits or any(split not in DEFAULT_SPLITS for split in splits):
        raise ValueError(f"splits must be selected from {DEFAULT_SPLITS}")
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")

    if args.audit_only:
        if (out_root / "BUILD_INCOMPLETE").exists():
            raise RuntimeError(f"Hybrid build is incomplete: {out_root / 'BUILD_INCOMPLETE'}")
        payload = audit_outputs(
            pangu_dir, tianji_dir, out_root, masks, splits, args.chunk_rows, args.rtol, args.atol
        )
        with (out_root / "hybrid_factorial_audit.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    p_cfg = require_layout(pangu_dir)
    t_cfg = require_layout(tianji_dir)
    require_canonical_pangu_lead(p_cfg, pangu_dir)
    for key in ("dynamic_feature_order", "dyn_vars", "window", "fe_dim"):
        if p_cfg.get(key) != t_cfg.get(key):
            raise ValueError(f"Pangu/Tianji config mismatch for {key}: {p_cfg.get(key)!r} != {t_cfg.get(key)!r}")
    out_root.mkdir(parents=True, exist_ok=True)
    incomplete = out_root / "BUILD_INCOMPLETE"
    incomplete.write_text("hybrid factorial build in progress\n", encoding="utf-8")
    split_rows: Dict[str, int] = {}
    split_hashes: Dict[str, str] = {}
    for split in splits:
        n, digest = verify_source_pair(
            pangu_dir,
            tianji_dir,
            split,
            p_cfg,
            args.limit_rows,
            args.chunk_rows,
            args.rtol,
            args.atol,
        )
        split_rows[split] = n
        split_hashes[split] = digest

    build_records: List[Dict[str, object]] = []
    for mask in masks:
        out_dir = out_root / f"mtw_{mask}"
        out_dir.mkdir(parents=True, exist_ok=False)
        cfg = config_for_mask(p_cfg, pangu_dir, tianji_dir, mask, split_hashes, args.limit_rows)
        for split in splits:
            build_records.append(
                build_one_split(
                    pangu_dir,
                    tianji_dir,
                    out_dir,
                    split,
                    mask,
                    p_cfg,
                    split_rows[split],
                    args.chunk_rows,
                    args.rtol,
                    args.atol,
                )
            )
        with (out_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    audit = audit_outputs(
        pangu_dir, tianji_dir, out_root, masks, splits, args.chunk_rows, args.rtol, args.atol
    )
    manifest = {
        "status": "passed",
        "pangu_dir": str(pangu_dir),
        "tianji_dir": str(tianji_dir),
        "out_root": str(out_root),
        "masks": masks,
        "split_rows": split_rows,
        "row_alignment_sha256": split_hashes,
        "build_records": build_records,
        "audit": audit,
    }
    with (out_root / "hybrid_factorial_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (out_root / "hybrid_factorial_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    incomplete.unlink()
    print(f"[OK] built and audited q-core hybrid factorial datasets: {out_root}")


if __name__ == "__main__":
    main()
