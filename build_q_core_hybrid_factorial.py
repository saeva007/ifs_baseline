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
import sys
import types
from dataclasses import dataclass
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
    Q_CORE_T925_NO_RH2M_DYN_FEATURES,
    compute_fog_features_pmst,
)


GROUP_PROFILES: Dict[str, Dict[str, object]] = {
    "mtw": {
        "dataset_prefix": "mtw",
        "description": "Original three-package factorial: moisture, thermal/pressure, wind.",
        "order": ("M", "T", "W"),
        "feature_set": "q_core_no_rh2m",
        "dynamic_order": tuple(Q_CORE_NO_RH2M_DYN_FEATURES),
        "groups": {
            "M": ("Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"),
            "T": ("T2M", "MSLP"),
            "W": ("U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925"),
        },
    },
    "mt2pw": {
        "dataset_prefix": "mt2pw",
        "description": "T-package follow-up factorial: split T into T2M and MSLP.",
        "order": ("M", "T2", "P", "W"),
        "feature_set": "q_core_no_rh2m",
        "dynamic_order": tuple(Q_CORE_NO_RH2M_DYN_FEATURES),
        "groups": {
            "M": ("Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"),
            "T2": ("T2M",),
            "P": ("MSLP",),
            "W": ("U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925"),
        },
    },
    "m925b": {
        "dataset_prefix": "m925b",
        "description": (
            "Three-package joint-structure factorial: low-level moisture/thermodynamic state, "
            "explicit T925, and the remaining q-core background."
        ),
        "order": ("M", "H", "B"),
        "feature_set": "q_core_t925_no_rh2m",
        "dynamic_order": tuple(Q_CORE_T925_NO_RH2M_DYN_FEATURES),
        "groups": {
            "M": ("Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"),
            "H": ("T_925",),
            "B": (
                "T2M",
                "MSLP",
                "U10",
                "V10",
                "WSPD10",
                "WDIR10",
                "U_925",
                "V_925",
                "WSPD925",
            ),
        },
    },
}
GROUP_PROFILE = "mtw"
FEATURE_SET = str(GROUP_PROFILES[GROUP_PROFILE]["feature_set"])
EXPECTED_ORDER = list(GROUP_PROFILES[GROUP_PROFILE]["dynamic_order"])  # type: ignore[arg-type]
GROUPS: Dict[str, Tuple[str, ...]] = dict(GROUP_PROFILES[GROUP_PROFILE]["groups"])  # type: ignore[arg-type]
GROUP_ORDER = tuple(GROUP_PROFILES[GROUP_PROFILE]["order"])  # type: ignore[arg-type]
DATASET_PREFIX = str(GROUP_PROFILES[GROUP_PROFILE]["dataset_prefix"])
SHARED_DYNAMIC = ("ZENITH", "PM10_ugm3", "PM25_ugm3")
STATIC_DIM = 6
CYCLICAL_DIM = 4
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_ENDPOINT_FOG_ATOL = 5.0e-5


def set_group_profile(profile: str, dataset_prefix: str | None = None) -> None:
    global GROUP_PROFILE, FEATURE_SET, EXPECTED_ORDER, GROUPS, GROUP_ORDER, DATASET_PREFIX
    key = str(profile).strip().lower()
    if key not in GROUP_PROFILES:
        raise ValueError(f"Unknown group profile {profile!r}; choose from {sorted(GROUP_PROFILES)}")
    spec = GROUP_PROFILES[key]
    groups = {str(name): tuple(features) for name, features in dict(spec["groups"]).items()}
    order = tuple(str(name) for name in tuple(spec["order"]))
    dynamic_order = [str(name) for name in tuple(spec["dynamic_order"])]
    feature_set = str(spec["feature_set"])
    missing = [name for name in order if name not in groups]
    if missing:
        raise ValueError(f"Group profile {profile!r} has missing group definitions: {missing}")
    for group, features in groups.items():
        absent = [feature for feature in features if feature not in dynamic_order]
        if absent:
            raise ValueError(f"Group profile {profile!r}/{group} uses unknown dynamic features: {absent}")
    flat_features = [feature for features in groups.values() for feature in features]
    duplicates = sorted({feature for feature in flat_features if flat_features.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Group profile {profile!r} assigns features to more than one group: {duplicates}")
    covered = set(flat_features)
    expected_source_features = set(dynamic_order) - set(SHARED_DYNAMIC)
    if covered != expected_source_features:
        raise ValueError(
            f"Group profile {profile!r} must partition every source-dependent dynamic feature; "
            f"missing={sorted(expected_source_features - covered)}, extra={sorted(covered - expected_source_features)}"
        )
    prefix = str(dataset_prefix).strip() if dataset_prefix else str(spec["dataset_prefix"])
    if not prefix or any(ch.isspace() for ch in prefix) or "/" in prefix or "\\" in prefix:
        raise ValueError(f"Invalid dataset prefix: {prefix!r}")
    GROUP_PROFILE = key
    FEATURE_SET = feature_set
    EXPECTED_ORDER = dynamic_order
    GROUPS = groups
    GROUP_ORDER = order
    DATASET_PREFIX = prefix


def all_masks() -> List[str]:
    width = len(GROUP_ORDER)
    return [f"{value:0{width}b}" for value in range(1 << width)]


def zero_mask() -> str:
    return "0" * len(GROUP_ORDER)


def one_mask() -> str:
    return "1" * len(GROUP_ORDER)


def dataset_dir_name(mask: str) -> str:
    return f"{DATASET_PREFIX}_{mask}"


@dataclass(frozen=True)
class PairAlignment:
    keys: pd.DataFrame
    pangu_rows: np.ndarray
    tianji_rows: np.ndarray
    pangu_source_rows: int
    tianji_source_rows: int

    @property
    def n(self) -> int:
        return int(len(self.keys))

    @property
    def digest(self) -> str:
        return row_alignment_hash(self.keys)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-dir", required=True)
    ap.add_argument("--tianji-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument(
        "--group-profile",
        default="mtw",
        choices=sorted(GROUP_PROFILES),
        help="Physical package profile: mtw keeps the original 3 groups; mt2pw splits T into T2M and MSLP.",
    )
    ap.add_argument(
        "--dataset-prefix",
        default="",
        help="Optional output directory prefix override; defaults to the selected group profile prefix.",
    )
    ap.add_argument("--masks", default="all", help="all or comma-separated binary masks for the selected group profile")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--limit-rows", type=int, default=0, help="Smoke-test row limit per split; 0 uses all rows.")
    ap.add_argument("--rtol", type=float, default=1.0e-6)
    ap.add_argument("--atol", type=float, default=1.0e-6)
    ap.add_argument(
        "--endpoint-fog-atol",
        type=float,
        default=DEFAULT_ENDPOINT_FOG_ATOL,
        help=(
            "Absolute compatibility tolerance between recomputed float32 fog features and the source endpoint. "
            "Dynamic, label, metadata and unchanged-column checks retain --rtol/--atol."
        ),
    )
    ap.add_argument("--audit-only", action="store_true")
    return ap.parse_args()


def parse_csv(value: str) -> List[str]:
    normalized = str(value).replace(";", ",").replace(":", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def parse_masks(value: str) -> List[str]:
    masks = all_masks() if str(value).strip().lower() == "all" else parse_csv(value)
    width = len(GROUP_ORDER)
    bad = [mask for mask in masks if len(mask) != width or any(bit not in "01" for bit in mask)]
    if bad:
        raise ValueError(f"Invalid {GROUP_PROFILE} mask(s): {bad}; expected {width}-bit binary masks")
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
    if str(cfg.get("feature_set")) != FEATURE_SET:
        raise ValueError(f"{data_dir}: feature_set must be {FEATURE_SET}")
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


def key_index(keys: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(keys[["time", "station_id"]])


def positions_for_keys(source_keys: pd.DataFrame, wanted_keys: pd.DataFrame, label: str) -> np.ndarray:
    lookup = pd.Series(np.arange(len(source_keys), dtype=np.int64), index=key_index(source_keys))
    positions = lookup.reindex(key_index(wanted_keys))
    if positions.isna().any():
        examples = wanted_keys.loc[positions.isna(), ["time", "station_id"]].head(3).to_dict("records")
        raise ValueError(f"{label}: aligned keys are absent from the source metadata; examples={examples}")
    return positions.to_numpy(dtype=np.int64)


def align_source_pair(pangu_dir: Path, tianji_dir: Path, split: str, limit_rows: int) -> PairAlignment:
    p_meta = canonical_meta(pangu_dir / f"meta_{split}.csv")
    t_meta = canonical_meta(tianji_dir / f"meta_{split}.csv")
    t_index = key_index(t_meta)
    common_mask = key_index(p_meta).isin(t_index)
    pangu_rows = np.flatnonzero(common_mask).astype(np.int64)
    if limit_rows > 0:
        pangu_rows = pangu_rows[: int(limit_rows)]
    keys = p_meta.iloc[pangu_rows].reset_index(drop=True)
    if keys.empty:
        raise ValueError(f"{split}: Pangu and Tianji have no common (time, station_id) rows")
    tianji_rows = positions_for_keys(t_meta, keys, f"{split}/Tianji")
    return PairAlignment(
        keys=keys,
        pangu_rows=pangu_rows,
        tianji_rows=tianji_rows,
        pangu_source_rows=len(p_meta),
        tianji_source_rows=len(t_meta),
    )


def assert_close(name: str, left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name}: shape mismatch {left.shape} != {right.shape}")
    if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        finite = diff[np.isfinite(diff)]
        maximum = float(finite.max()) if finite.size else math.nan
        raise ValueError(f"{name}: values differ (max_abs_diff={maximum})")


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    finite = diff[np.isfinite(diff)]
    return float(finite.max()) if finite.size else math.nan


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
) -> PairAlignment:
    alignment = align_source_pair(pangu_dir, tianji_dir, split, limit_rows)
    n = alignment.n
    p_y_all = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")
    t_y_all = np.load(tianji_dir / f"y_{split}.npy", mmap_mode="r")
    p_y = p_y_all[alignment.pangu_rows]
    t_y = t_y_all[alignment.tianji_rows]
    assert_close(f"{split}: labels", p_y, t_y, rtol=0.0, atol=1.0e-6)

    p_x = np.load(pangu_dir / f"X_{split}.npy", mmap_mode="r")
    t_x = np.load(tianji_dir / f"X_{split}.npy", mmap_mode="r")
    if (
        p_x.shape[0] != alignment.pangu_source_rows
        or t_x.shape[0] != alignment.tianji_source_rows
        or len(p_y_all) != alignment.pangu_source_rows
        or len(t_y_all) != alignment.tianji_source_rows
        or p_x.shape[1:] != t_x.shape[1:]
    ):
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
        p_rows = alignment.pangu_rows[start:end]
        t_rows = alignment.tianji_rows[start:end]
        assert_close(
            f"{split}: shared source-independent columns rows {start}:{end}",
            np.asarray(p_x[p_rows][:, shared_cols]),
            np.asarray(t_x[t_rows][:, shared_cols]),
            rtol,
            atol,
        )
    return alignment


def build_one_split(
    pangu_dir: Path,
    tianji_dir: Path,
    out_dir: Path,
    split: str,
    mask: str,
    cfg: Mapping[str, object],
    alignment: PairAlignment,
    chunk_rows: int,
    rtol: float,
    atol: float,
    endpoint_fog_atol: float,
) -> Dict[str, object]:
    p_x = np.load(pangu_dir / f"X_{split}.npy", mmap_mode="r")
    t_x = np.load(tianji_dir / f"X_{split}.npy", mmap_mode="r")
    n = alignment.n
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
    fog_start = dyn_width + STATIC_DIM
    donor_idx = group_columns(order, mask_groups(mask))
    max_abs_endpoint = 0.0

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        base_rows = np.asarray(p_x[alignment.pangu_rows[start:end]], dtype=np.float32)
        donor_rows = np.asarray(t_x[alignment.tianji_rows[start:end]], dtype=np.float32)
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
        if mask in {zero_mask(), one_mask()}:
            expected = base_rows if mask == zero_mask() else donor_rows
            source_fog = expected[:, fog_start : fog_start + fog_dim]
            recomputed_fog = rows[:, fog_start : fog_start + fog_dim]
            max_abs_endpoint = max(max_abs_endpoint, max_abs_difference(recomputed_fog, source_fog))
            assert_close(
                f"{split}/{mask}: recomputed/source fog compatibility",
                recomputed_fog,
                source_fog,
                rtol,
                endpoint_fog_atol,
            )
    out_x.flush()
    del out_x
    y = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")[alignment.pangu_rows]
    np.save(out_dir / f"y_{split}.npy", np.asarray(y))
    meta = pd.read_csv(pangu_dir / f"meta_{split}.csv").iloc[alignment.pangu_rows]
    meta.to_csv(out_dir / f"meta_{split}.csv", index=False)
    return {
        "split": split,
        "rows": n,
        "mask": mask,
        "replaced_groups": mask_groups(mask),
        "replaced_features": [order[i] for i in donor_idx],
        "endpoint_recomputed_fog_max_abs_diff": max_abs_endpoint if mask in {zero_mask(), one_mask()} else None,
        "endpoint_recomputed_fog_atol": endpoint_fog_atol if mask in {zero_mask(), one_mask()} else None,
        "pangu_source_rows": alignment.pangu_source_rows,
        "tianji_source_rows": alignment.tianji_source_rows,
        "common_rows": n,
    }


def config_for_mask(
    base_cfg: Mapping[str, object],
    pangu_dir: Path,
    tianji_dir: Path,
    mask: str,
    split_hashes: Mapping[str, str],
    split_alignment: Mapping[str, Mapping[str, int]],
    limit_rows: int,
    endpoint_fog_atol: float,
) -> Dict[str, object]:
    cfg = dict(base_cfg)
    selected_groups = mask_groups(mask)
    cfg.update(
        {
            "dataset": "pangu_tianji_q_core_hybrid_factorial",
            "feature_set": FEATURE_SET,
            "hybrid_base_source": "pangu2025",
            "hybrid_donor_source": "tianji",
            "hybrid_group_profile": GROUP_PROFILE,
            "hybrid_dataset_prefix": DATASET_PREFIX,
            "hybrid_group_order": list(GROUP_ORDER),
            "hybrid_group_mask": mask,
            "hybrid_group_definitions": {key: list(value) for key, value in GROUPS.items()},
            "replaced_feature_groups": selected_groups,
            "replaced_features": [feature for group in selected_groups for feature in GROUPS[group]],
            "base_dataset_dir": str(pangu_dir),
            "donor_dataset_dir": str(tianji_dir),
            "row_alignment_sha256": dict(split_hashes),
            "row_alignment_policy": "ordered intersection of canonical (valid_time, station_id) keys in Pangu row order",
            "source_pair_coverage": {key: dict(value) for key, value in split_alignment.items()},
            "recomputed_fog_features": True,
            "thermodynamic_source_channels_recomputed": False,
            "thermodynamic_cross_source_policy": (
                "For m925b, T_925 and the M channels are independently assigned by the factorial mask; "
                "RH_925/DP_925/Q_925 are not re-derived after mixing because their cross-source consistency "
                "is the experimental factor. Only downstream fog-engineered features are recomputed."
                if GROUP_PROFILE == "m925b"
                else "not_applicable"
            ),
            "endpoint_source_fog_compatibility_atol": endpoint_fog_atol,
            "endpoint_identity_policy": (
                "dynamic/static/time fields retain source checks; recomputed float32 fog features use a separately "
                "recorded numerical compatibility tolerance"
            ),
            "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
            "pm_qc_policy": PM_QC_POLICY_VERSION,
            "hybrid_smoke_limit_rows": int(limit_rows),
            "scientific_role": "controlled source-block retraining attribution; not a single-variable causal effect",
            "joint_structure_interpretation": (
                "For m925b, the M:H interaction is evidence of predictive complementarity under retraining; "
                "it is not by itself proof of dynamical-equation consistency."
                if GROUP_PROFILE == "m925b"
                else None
            ),
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
    endpoint_fog_atol: float,
) -> Dict[str, object]:
    base_cfg = require_layout(pangu_dir)
    require_canonical_pangu_lead(base_cfg, pangu_dir)
    require_layout(tianji_dir)
    order = [str(v) for v in base_cfg["dynamic_feature_order"]]
    dyn = int(base_cfg["dyn_vars"])
    window = int(base_cfg["window"])
    dyn_width = dyn * window
    rows: List[Dict[str, object]] = []
    alignment_cache: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    reference_mask = masks[0]
    for split in splits:
        reference_meta = canonical_meta(out_root / dataset_dir_name(reference_mask) / f"meta_{split}.csv")
        base_meta = canonical_meta(pangu_dir / f"meta_{split}.csv")
        donor_meta = canonical_meta(tianji_dir / f"meta_{split}.csv")
        alignment_cache[split] = (
            reference_meta,
            positions_for_keys(base_meta, reference_meta, f"audit {split}/Pangu"),
            positions_for_keys(donor_meta, reference_meta, f"audit {split}/Tianji"),
        )
    for mask in masks:
        data_dir = out_root / dataset_dir_name(mask)
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
            reference_meta, p_rows, t_rows = alignment_cache[split]
            if not out_meta.equals(reference_meta):
                raise ValueError(f"audit {mask}/{split}: output metadata differ across hybrid masks")
            out_y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
            base_y = np.load(pangu_dir / f"y_{split}.npy", mmap_mode="r")[p_rows]
            assert_close(f"audit {mask}/{split}: labels", np.asarray(out_y), np.asarray(base_y), 0.0, atol)
            fe_dim = int(cfg["fe_dim"])
            fog_dim = fe_dim - CYCLICAL_DIM
            static_start = dyn_width
            fog_start = static_start + STATIC_DIM
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                p_chunk = np.asarray(p[p_rows[start:end]])
                t_chunk = np.asarray(t[t_rows[start:end]])
                hybrid_dyn = np.asarray(x[start:end, :dyn_width]).reshape(end - start, window, dyn)
                expected_dyn = p_chunk[:, :dyn_width].reshape(end - start, window, dyn).copy()
                if selected_idx:
                    donor_dyn = t_chunk[:, :dyn_width].reshape(end - start, window, dyn)
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
                    p_chunk[:, shared_tail_cols],
                    rtol,
                    atol,
                )
                if mask == zero_mask():
                    assert_close(
                        f"audit {mask}/{split} source fog compatibility rows {start}:{end}",
                        np.asarray(x[start:end, fog_start : fog_start + fog_dim]),
                        p_chunk[:, fog_start : fog_start + fog_dim],
                        rtol,
                        endpoint_fog_atol,
                    )
                if mask == one_mask():
                    assert_close(
                        f"audit {mask}/{split} source fog compatibility rows {start}:{end}",
                        np.asarray(x[start:end, fog_start : fog_start + fog_dim]),
                        t_chunk[:, fog_start : fog_start + fog_dim],
                        rtol,
                        endpoint_fog_atol,
                    )
            rows.append({"mask": mask, "split": split, "rows": n, "status": "passed"})
    return {
        "status": "passed",
        "masks": list(masks),
        "splits": list(splits),
        "checks": rows,
        "group_order": list(GROUP_ORDER),
        "group_profile": GROUP_PROFILE,
        "dataset_prefix": DATASET_PREFIX,
        "group_definitions": {key: list(value) for key, value in GROUPS.items()},
        "endpoint_source_fog_compatibility_atol": endpoint_fog_atol,
    }


def main() -> None:
    args = parse_args()
    pangu_dir = Path(args.pangu_dir).expanduser().resolve()
    tianji_dir = Path(args.tianji_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    set_group_profile(args.group_profile, args.dataset_prefix or None)
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
            pangu_dir,
            tianji_dir,
            out_root,
            masks,
            splits,
            args.chunk_rows,
            args.rtol,
            args.atol,
            args.endpoint_fog_atol,
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
    alignments: Dict[str, PairAlignment] = {}
    split_alignment: Dict[str, Dict[str, int]] = {}
    for split in splits:
        alignment = verify_source_pair(
            pangu_dir,
            tianji_dir,
            split,
            p_cfg,
            args.limit_rows,
            args.chunk_rows,
            args.rtol,
            args.atol,
        )
        alignments[split] = alignment
        split_rows[split] = alignment.n
        split_hashes[split] = alignment.digest
        split_alignment[split] = {
            "pangu_source_rows": alignment.pangu_source_rows,
            "tianji_source_rows": alignment.tianji_source_rows,
            "common_rows": alignment.n,
            "pangu_excluded_rows": alignment.pangu_source_rows - alignment.n,
            "tianji_excluded_rows": alignment.tianji_source_rows - alignment.n,
        }

    build_records: List[Dict[str, object]] = []
    for mask in masks:
        out_dir = out_root / dataset_dir_name(mask)
        out_dir.mkdir(parents=True, exist_ok=False)
        cfg = config_for_mask(
            p_cfg,
            pangu_dir,
            tianji_dir,
            mask,
            split_hashes,
            split_alignment,
            args.limit_rows,
            args.endpoint_fog_atol,
        )
        for split in splits:
            build_records.append(
                build_one_split(
                    pangu_dir,
                    tianji_dir,
                    out_dir,
                    split,
                    mask,
                    p_cfg,
                    alignments[split],
                    args.chunk_rows,
                    args.rtol,
                    args.atol,
                    args.endpoint_fog_atol,
                )
            )
        with (out_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    audit = audit_outputs(
        pangu_dir,
        tianji_dir,
        out_root,
        masks,
        splits,
        args.chunk_rows,
        args.rtol,
        args.atol,
        args.endpoint_fog_atol,
    )
    manifest = {
        "status": "passed",
        "pangu_dir": str(pangu_dir),
        "tianji_dir": str(tianji_dir),
        "out_root": str(out_root),
        "masks": masks,
        "group_profile": GROUP_PROFILE,
        "dataset_prefix": DATASET_PREFIX,
        "group_order": list(GROUP_ORDER),
        "group_definitions": {key: list(value) for key, value in GROUPS.items()},
        "split_rows": split_rows,
        "row_alignment_policy": "ordered source-key intersection in Pangu row order",
        "source_pair_coverage": split_alignment,
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
