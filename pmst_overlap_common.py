#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for Tianji-vs-IFS overlap baseline datasets compatible with
train/PMST_net_test_11_s2_pm10.py (27 dyn vars = 24 PMST met + zenith + pm10 + pm2p5, FE = 32 + 4).

Dynamic layout matches s2_data_monthtail_v2.ipynb (pm10+pm2p5 cell):
  FINAL_FEATURE_ORDER (24) + zenith (24) + pm10 (25) + pm2p5 (26).
Only overlap NWP fields are filled; other PMST slots stay 0 so IFS/Tianji differ only in those channels.
"""

from __future__ import annotations

import gc
import glob
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from tqdm import tqdm


FINAL_FEATURE_ORDER: List[str] = [
    "RH2M",
    "T2M",
    "PRECIP",
    "MSLP",
    "SW_RAD",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "CAPE",
    "LCC",
    "T_925",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "W_925",
    "W_1000",
    "DPD",
    "INVERSION",
]

PMST_MET_DIM = len(FINAL_FEATURE_ORDER)  # 24
ZENITH_IDX = 24
PM10_IDX = 25
PM25_IDX = 26
TOTAL_DYN = 27

PMST_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FINAL_FEATURE_ORDER)}

OVERLAP_CANONICAL: List[str] = [
    "RH2M",
    "T2M",
    "PRECIP",
    "MSLP",
    "SW_RAD",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "LCC",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "W_925",
    "W_1000",
    "DPD",
]

OVERLAP_AUXILIARY_FIELDS: List[str] = ["D2M", "T_1000", "RH_1000"]
OVERLAP_SOURCE_FIELDS: List[str] = list(
    dict.fromkeys([*OVERLAP_CANONICAL, *OVERLAP_AUXILIARY_FIELDS])
)
COMMON_CORE_PMST_FEATURES: List[str] = [
    "RH2M",
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
    "DPD",
]
COMPACT_COMMON_CORE_PMST_FEATURES: List[str] = [
    name for name in COMMON_CORE_PMST_FEATURES if name != "RH2M"
]
Q_CORE_NO_RH2M_PMST_FEATURES: List[str] = [
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
]
COMPACT_COMMON_CORE_DYN_FEATURES: List[str] = [
    *COMPACT_COMMON_CORE_PMST_FEATURES,
    "ZENITH",
    "PM10_ugm3",
    "PM25_ugm3",
]
Q_CORE_NO_RH2M_DYN_FEATURES: List[str] = [
    *Q_CORE_NO_RH2M_PMST_FEATURES,
    "ZENITH",
    "PM10_ugm3",
    "PM25_ugm3",
]
COMPACT_COMMON_CORE_DYN_INDICES: List[int] = [
    *[PMST_INDEX[name] for name in COMPACT_COMMON_CORE_PMST_FEATURES],
    ZENITH_IDX,
    PM10_IDX,
    PM25_IDX,
]
COMPACT_TOTAL_DYN = len(COMPACT_COMMON_CORE_DYN_FEATURES)
COMPACT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COMPACT_COMMON_CORE_DYN_FEATURES)}
PMST_SOURCE_FIELDS: List[str] = list(dict.fromkeys([*FINAL_FEATURE_ORDER, *OVERLAP_AUXILIARY_FIELDS]))
PANGU2021_SOURCE_FULL_PMST_FEATURES: List[str] = [
    name
    for name in FINAL_FEATURE_ORDER
    if name not in {"PRECIP", "SW_RAD", "CAPE", "LCC", "W_925", "W_1000"}
]
PANGU2025_SOURCE_FULL_PMST_FEATURES: List[str] = [
    name
    for name in FINAL_FEATURE_ORDER
    if name not in {"RH2M", "PRECIP", "SW_RAD", "CAPE", "LCC", "W_925", "W_1000", "DPD"}
]
SOURCE_FULL_PROFILE_PMST_FEATURES: Dict[str, List[str]] = {
    "tianji": list(FINAL_FEATURE_ORDER),
    "t2nd_rh2m": list(FINAL_FEATURE_ORDER),
    "era5": list(FINAL_FEATURE_ORDER),
    "era5_2025": list(FINAL_FEATURE_ORDER),
    "ifs": list(OVERLAP_CANONICAL),
    "pangu": list(PANGU2021_SOURCE_FULL_PMST_FEATURES),
    "pangu2021": list(PANGU2021_SOURCE_FULL_PMST_FEATURES),
    "pangu2025": list(PANGU2025_SOURCE_FULL_PMST_FEATURES),
}
FEATURE_SET_CHOICES: Tuple[str, ...] = (
    "common_core",
    "compact_common_core",
    "compact_common_core_no_rh2m",
    "q_core_no_rh2m",
    "overlap_full",
    "source_full",
)

CANONICAL_UNIT_POLICY_VERSION = "pmst_canonical_units_v2_20260630"
PM_QC_POLICY_VERSION = "pm_explicit_legacy_scale_then_train_median_qc_v2_20260701"
PM_CONCENTRATION_MAX_UGM3 = 10000.0
LEGACY_PM_1E12_UNITS = "legacy_kgm3_times_1e12"
CANONICAL_DYNAMIC_UNITS: Dict[str, str] = {
    "T2M": "K",
    "MSLP": "Pa",
    "RH2M": "%",
    "RH_925": "%",
    "RH_1000": "%",
    "DP_1000": "K",
    "DP_925": "K",
    "Q_1000": "kg kg-1",
    "Q_925": "kg kg-1",
    "PM10_ugm3": "ug m-3",
    "PM25_ugm3": "ug m-3",
}


def _finite_median_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.abs(arr[np.isfinite(arr)])
    return float(np.median(finite)) if finite.size else math.nan


def canonicalize_pm_concentration(values: np.ndarray, declared_units: str = "") -> np.ndarray:
    """Return PM concentration in ug m-3, repairing the legacy CAMS scale.

    Historical station files label raw CAMS kg m-3 values as ``ug m-3`` and
    historical dataset builders multiplied them by 1e12 (ng m-3).  Magnitude
    therefore participates in the decision instead of trusting that bad label.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(arr >= 0.0, arr, np.nan)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    median_abs = float(np.median(positive)) if positive.size else math.nan
    units = re.sub(r"[\s_µμ]+", "", str(declared_units).strip().lower())
    if units == "legacykgm3times1e12":
        # The historical S1 notebook used kg m-3 * 1e12.  Its provenance is
        # known, so never infer this scale independently for each chunk.
        scale = 1.0e-3
    elif not math.isfinite(median_abs) or median_abs == 0.0:
        scale = 1.0
    elif median_abs < 1.0e-3:
        # Raw CAMS mass concentration in kg m-3, including legacy files whose
        # attrs incorrectly claim ug m-3.
        scale = 1.0e9
    elif median_abs > 1.0e4:
        # Legacy builders used kg m-3 * 1e12, which is numerically ng m-3.
        scale = 1.0e-3
    elif "kgm-3" in units or "kg/m3" in units:
        scale = 1.0e9
    else:
        scale = 1.0
    return (arr * scale).astype(np.float32)


def sanitize_pm_concentration(
    values: np.ndarray,
    declared_units: str = "",
    *,
    fill_value: float | None = None,
    max_valid_ugm3: float = PM_CONCENTRATION_MAX_UGM3,
) -> np.ndarray:
    """Canonicalize PM, reject impossible/fill values, and return finite data.

    Values outside ``[0, max_valid_ugm3]`` are missing data, not atmospheric
    extremes.  Dataset builders may pass a training-only median so validation
    and test data never determine their own imputation value.
    """
    arr = canonicalize_pm_concentration(values, declared_units)
    valid = np.isfinite(arr) & (arr >= 0.0) & (arr <= float(max_valid_ugm3))
    if fill_value is None:
        fill = float(np.median(arr[valid])) if valid.any() else 0.0
    else:
        fill = float(fill_value)
        if not math.isfinite(fill) or not (0.0 <= fill <= float(max_valid_ugm3)):
            raise ValueError(f"invalid PM fill_value={fill_value!r}")
    return np.where(valid, arr, fill).astype(np.float32)


def canonicalize_pmst_field(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize unit families used by every S1/S2 forecast-source builder."""
    canon = normalize_var_coord(name)
    arr = np.asarray(values, dtype=np.float32)
    median_abs = _finite_median_abs(arr)
    if not math.isfinite(median_abs):
        return arr
    if canon == "MSLP":
        if 500.0 <= median_abs <= 2000.0:
            return (arr * 100.0).astype(np.float32)
        if 20000.0 <= median_abs <= 120000.0:
            return arr
        raise ValueError(f"MSLP scale is neither hPa nor Pa: median_abs={median_abs:g}")
    if canon in {"T2M", "T_925", "T_1000", "DP_925", "DP_1000", "D2M"}:
        if median_abs < 150.0:
            return (arr + 273.15).astype(np.float32)
        return arr
    if canon in {"RH2M", "RH_925", "RH_1000"}:
        if median_abs <= 1.5:
            return (arr * 100.0).astype(np.float32)
        return arr
    if canon in {"Q_925", "Q_1000"}:
        if median_abs > 0.2:
            return (arr / 1000.0).astype(np.float32)
        return arr
    return arr


def source_full_profile_features(profile: str) -> List[str]:
    key = str(profile or "").strip().lower().replace("-", "_")
    if not key:
        return list(FINAL_FEATURE_ORDER)
    if key not in SOURCE_FULL_PROFILE_PMST_FEATURES:
        choices = ",".join(sorted(SOURCE_FULL_PROFILE_PMST_FEATURES))
        raise ValueError(f"Unknown source_full_profile={profile!r}; expected one of {choices}")
    return list(SOURCE_FULL_PROFILE_PMST_FEATURES[key])

CANONICAL_VAR_ALIASES: Dict[str, str] = {
    "D2M": "D2M",
    "2D": "D2M",
    "DEWPOINT2M": "D2M",
    "DEW_POINT_2M": "D2M",
    "RH925": "RH_925",
    "R925": "RH_925",
    "RH1000": "RH_1000",
    "R1000": "RH_1000",
    "T925": "T_925",
    "T1000": "T_1000",
    "U925": "U_925",
    "V925": "V_925",
    "Q1000": "Q_1000",
    "Q925": "Q_925",
    "W925": "W_925",
    "W1000": "W_1000",
    "DP1000": "DP_1000",
    "DP925": "DP_925",
    "WSPD925": "WSPD925",
    "ZENITH": "ZENITH",
    "PM10": "PM10",
    "PM10_UGM3": "PM10",
    "PM10UGM3": "PM10",
    "PM10_UG_M3": "PM10",
    "PM25": "PM25",
    "PM25_UGM3": "PM25",
    "PM25UGM3": "PM25",
    "PM25_UG_M3": "PM25",
    "PM2P5": "PM25",
    "PM2_5": "PM25",
}

TIANJI_INPUT_TIME_SHIFT_HOURS = float(os.environ.get("TIANJI_INPUT_TIME_SHIFT_HOURS", "0"))
TIANJI_TIME_ALIGNMENT = (
    "bjt_minus_8_to_utc"
    if TIANJI_INPUT_TIME_SHIFT_HOURS == -8
    else ("raw_utc_no_shift" if TIANJI_INPUT_TIME_SHIFT_HOURS == 0 else "custom_shift_to_utc")
)


def normalize_tianji_times(raw_times) -> pd.DatetimeIndex:
    """Return UTC-naive Tianji valid times before feature building and splitting."""
    times = pd.DatetimeIndex(pd.to_datetime(raw_times))
    if TIANJI_INPUT_TIME_SHIFT_HOURS:
        times = times + pd.to_timedelta(TIANJI_INPUT_TIME_SHIFT_HOURS, unit="h")
    return times


def summarize_time_axis(raw_times, expected_step_hours: float = 1.0) -> Dict[str, object]:
    """Summarize valid-time cadence and whether every adjacent step is regular."""
    times = pd.DatetimeIndex(pd.to_datetime(raw_times))
    if times.hasnans:
        raise ValueError("time axis contains NaT values")
    if times.has_duplicates:
        raise ValueError(f"time axis contains {int(times.duplicated(keep=False).sum())} duplicate entries")
    if not times.is_monotonic_increasing:
        raise ValueError("time axis is not monotonically increasing")
    if len(times) < 2:
        return {
            "count": int(len(times)),
            "start": str(times.min()) if len(times) else "",
            "end": str(times.max()) if len(times) else "",
            "expected_step_hours": float(expected_step_hours),
            "regular": True,
            "delta_hours_min": None,
            "delta_hours_max": None,
            "irregular_transition_count": 0,
        }
    delta_hours = np.asarray(np.diff(times.values) / np.timedelta64(1, "h"), dtype=np.float64)
    regular = np.isclose(delta_hours, float(expected_step_hours), rtol=0.0, atol=1.0e-6)
    unique, counts = np.unique(np.round(delta_hours, 6), return_counts=True)
    histogram = {str(float(k)): int(v) for k, v in zip(unique, counts)}
    return {
        "count": int(len(times)),
        "start": str(times[0]),
        "end": str(times[-1]),
        "expected_step_hours": float(expected_step_hours),
        "regular": bool(regular.all()),
        "delta_hours_min": float(np.min(delta_hours)),
        "delta_hours_max": float(np.max(delta_hours)),
        "irregular_transition_count": int((~regular).sum()),
        "delta_hours_histogram": histogram,
    }


def require_regular_time_axis(raw_times, expected_step_hours: float, source_label: str) -> Dict[str, object]:
    summary = summarize_time_axis(raw_times, expected_step_hours)
    if not bool(summary["regular"]):
        raise ValueError(
            f"{source_label} time axis is not a continuous {expected_step_hours:g}-hour sequence: "
            f"irregular_transitions={summary['irregular_transition_count']}, "
            f"delta_range={summary['delta_hours_min']}..{summary['delta_hours_max']} h, "
            f"histogram={summary.get('delta_hours_histogram', {})}. "
            "A 12-step window cannot be interpreted as 12 hours."
        )
    return summary

# Where overlap variables sit in the 24-dim PMST met block
OVERLAP_PMST_INDICES: Dict[str, int] = {
    name: PMST_INDEX[name] for name in OVERLAP_CANONICAL if name in PMST_INDEX
}
RH2M_IDX = PMST_INDEX["RH2M"]
T2M_IDX = PMST_INDEX["T2M"]
D2M_AUX = "D2M"
WSPD10_IDX = PMST_INDEX["WSPD10"]
U10_IDX = PMST_INDEX["U10"]
V10_IDX = PMST_INDEX["V10"]
WDIR10_IDX = PMST_INDEX["WDIR10"]
U925_IDX = PMST_INDEX["U_925"]
V925_IDX = PMST_INDEX["V_925"]
WSPD925_IDX = PMST_INDEX["WSPD925"]
DP1000_IDX = PMST_INDEX["DP_1000"]
DP925_IDX = PMST_INDEX["DP_925"]
Q1000_IDX = PMST_INDEX["Q_1000"]
Q925_IDX = PMST_INDEX["Q_925"]
DPD_IDX = PMST_INDEX["DPD"]


def calculate_zenith_angle(latitudes, longitudes, times):
    try:
        times_pd = pd.DatetimeIndex(times)
        if times_pd.tz is None:
            times_pd = times_pd.tz_localize("UTC")
    except Exception:
        times_pd = pd.DatetimeIndex(times).tz_localize("UTC")
    n_times, n_stations = len(times_pd), len(latitudes)
    b_times = np.repeat(times_pd, n_stations)
    b_lats = np.tile(latitudes, n_times)
    b_lons = np.tile(longitudes, n_times)
    try:
        sp = pvlib.solarposition.get_solarposition(b_times, b_lats, b_lons)
        return sp["apparent_zenith"].values.reshape(n_times, n_stations, 1).astype(np.float32)
    except Exception:
        return np.zeros((n_times, n_stations, 1), dtype=np.float32)


def get_nearest_veg(latitudes, longitudes, veg_data):
    return veg_data.sel(
        latitude=xr.DataArray(latitudes, dims="station"),
        longitude=xr.DataArray(longitudes, dims="station"),
        method="nearest",
    )["htcc"].values


def extract_terrain(latitudes, longitudes, oro_ds, r=2):
    h = oro_ds["h"].values
    lats_idx = np.abs(oro_ds.latitude.values[:, None] - latitudes).argmin(0)
    lons_idx = np.abs(oro_ds.longitude.values[:, None] - (longitudes % 360)).argmin(0)
    max_r, max_c = h.shape
    feats = []
    for r_idx, c_idx in zip(lats_idx, lons_idx):
        w = h[
            max(0, r_idx - r) : min(max_r, r_idx + r + 1),
            max(0, c_idx - r) : min(max_c, c_idx + r + 1),
        ]
        feats.append([h[r_idx, c_idx], h[r_idx, c_idx] - np.mean(w), np.std(w)])
    return np.array(feats, dtype=np.float32)


def build_static_features(
    lats: np.ndarray,
    lons: np.ndarray,
    data_veg: xr.Dataset,
    data_oro: xr.Dataset,
    unique_veg_ids: np.ndarray,
) -> np.ndarray:
    veg_raw = get_nearest_veg(lats, lons, data_veg)
    type_to_idx = {v: i for i, v in enumerate(unique_veg_ids)}
    feat_veg = np.array([type_to_idx.get(v, 0) for v in veg_raw])[:, None].astype(np.float32)
    feat_oro = extract_terrain(lats, lons, data_oro)
    return np.concatenate(
        [
            (lats[:, None] / 90.0).astype(np.float32),
            (lons[:, None] / 180.0).astype(np.float32),
            feat_oro,
            feat_veg,
        ],
        axis=1,
    ).astype(np.float32)


def calculate_rh_from_dewpoint(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    """Return relative humidity (%) from 2 m temperature and dew point in kelvin."""
    t_c = np.asarray(t2m, dtype=np.float32) - 273.15
    td_c = np.asarray(d2m, dtype=np.float32) - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip((e / np.maximum(es, 1e-6)) * 100.0, 0.0, 100.0).astype(np.float32)


def calculate_dewpoint_from_rh(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Return dew-point temperature (K) from temperature (K) and RH (%)."""
    t_c = np.asarray(t, dtype=np.float32) - 273.15
    rh_frac = np.clip(np.asarray(rh, dtype=np.float32) / 100.0, 1e-4, 1.0)
    b, c = 17.67, 243.5
    gamma = np.log(rh_frac) + (b * t_c) / (c + t_c)
    return ((c * gamma) / np.maximum(b - gamma, 1e-6) + 273.15).astype(np.float32)


def calculate_dewpoint_from_specific_humidity(q: np.ndarray, pressure_hpa: float) -> np.ndarray:
    """Return dew-point temperature (K) from specific humidity (kg kg-1) and pressure."""
    q_arr = np.clip(np.asarray(q, dtype=np.float32), 1e-8, 0.08)
    eps = 0.622
    e_hpa = (q_arr * float(pressure_hpa)) / np.maximum(eps + (1.0 - eps) * q_arr, 1e-8)
    ln_ratio = np.log(np.maximum(e_hpa, 1e-6) / 6.112)
    td_c = (243.5 * ln_ratio) / np.maximum(17.67 - ln_ratio, 1e-6)
    return (td_c + 273.15).astype(np.float32)


def calculate_specific_humidity(t: np.ndarray, rh: np.ndarray, pressure_hpa: float) -> np.ndarray:
    """Return specific humidity (kg kg-1) from temperature (K), RH (%), and pressure."""
    t_c = np.asarray(t, dtype=np.float32) - 273.15
    rh_frac = np.clip(np.asarray(rh, dtype=np.float32) / 100.0, 0.0, 1.0)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = es * rh_frac
    r = 0.622 * e / np.maximum(float(pressure_hpa) - e, 1e-6)
    return np.clip(r / np.maximum(1.0 + r, 1e-8), 0.0, 0.08).astype(np.float32)


def calculate_wind_speed_dir(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    speed = np.sqrt(u_arr * u_arr + v_arr * v_arr).astype(np.float32)
    direction = ((270.0 - np.degrees(np.arctan2(v_arr, u_arr))) % 360.0).astype(np.float32)
    return speed, direction


def precip_accum_to_hourly(precip: np.ndarray) -> np.ndarray:
    """Convert accumulated precipitation to non-negative hourly increments."""
    arr = np.asarray(precip, dtype=np.float32)
    if arr.ndim < 1:
        return np.maximum(arr, 0.0).astype(np.float32)
    first = arr[:1]
    diff = np.diff(arr, axis=0, prepend=np.zeros_like(first))
    hourly = np.where(diff >= 0.0, diff, arr)
    return np.maximum(hourly, 0.0).astype(np.float32)


def apply_wspd10_from_uv(x_met: np.ndarray) -> None:
    """In-place: WSPD10 = hypot(U10, V10). x_met shape (..., 24)."""
    speed, _ = calculate_wind_speed_dir(x_met[..., U10_IDX], x_met[..., V10_IDX])
    x_met[..., WSPD10_IDX] = speed


def apply_wind_derivations(x_met: np.ndarray) -> None:
    """In-place: derive wind speed/direction channels from U/V components."""
    speed10, dir10 = calculate_wind_speed_dir(x_met[..., U10_IDX], x_met[..., V10_IDX])
    x_met[..., WSPD10_IDX] = speed10
    x_met[..., WDIR10_IDX] = dir10
    speed925, _ = calculate_wind_speed_dir(x_met[..., U925_IDX], x_met[..., V925_IDX])
    x_met[..., WSPD925_IDX] = speed925


def _field_array(
    fields: Dict[str, np.ndarray],
    name: str,
    nt: int,
    ns: int,
) -> Optional[np.ndarray]:
    if name not in fields:
        return None
    arr = np.asarray(fields[name], dtype=np.float32)
    if arr.shape != (nt, ns):
        raise ValueError(f"Field {name} shape {arr.shape}, expected {(nt, ns)}")
    return arr


def describe_available_pmst_features(field_names: Iterable[str]) -> Set[str]:
    names = {normalize_var_coord(v) for v in field_names}
    available = {name for name in names if name in PMST_INDEX}
    if {"U10", "V10"}.issubset(names):
        available.update({"WSPD10", "WDIR10"})
    if {"U_925", "V_925"}.issubset(names):
        available.add("WSPD925")
    if {"T2M", "D2M"}.issubset(names):
        available.update({"RH2M", "DPD"})
    if {"T2M", "RH2M"}.issubset(names):
        available.add("DPD")
    if "Q_1000" in names:
        available.add("DP_1000")
    if {"T_1000", "RH_1000"}.issubset(names):
        available.update({"Q_1000", "DP_1000"})
    if "Q_925" in names:
        available.add("DP_925")
    if {"T_925", "RH_925"}.issubset(names):
        available.update({"Q_925", "DP_925"})
    if {"T_925", "T2M"}.issubset(names):
        available.add("INVERSION")
    return available & set(FINAL_FEATURE_ORDER)


def describe_available_overlap_features(field_names: Iterable[str]) -> Set[str]:
    return describe_available_pmst_features(field_names) & set(OVERLAP_CANONICAL)


def resolve_pmst_feature_set(feature_set: str, available_features: Optional[Iterable[str]] = None) -> List[str]:
    key = str(feature_set or "overlap_full").strip().lower().replace("-", "_")
    if key in {"common", "core", "pangu_core", "common_overlap"}:
        key = "common_core"
    if key in {"compact", "compact_common", "compact_core"}:
        key = "compact_common_core"
    if key in {"compact_no_rh2m", "compact_common_no_rh2m", "compact_core_no_rh2m"}:
        key = "compact_common_core_no_rh2m"
    if key in {"q_core", "q1000_core", "q1000_core_no_rh2m", "q_core_common", "q_core_no_rh"}:
        key = "q_core_no_rh2m"
    if key in {"full", "overlap", "overlap_canonical"}:
        key = "overlap_full"
    if key in {"all", "all_available", "source"}:
        key = "source_full"
    if key == "common_core":
        return list(COMMON_CORE_PMST_FEATURES)
    if key in {"compact_common_core", "compact_common_core_no_rh2m"}:
        return list(COMPACT_COMMON_CORE_PMST_FEATURES)
    if key == "q_core_no_rh2m":
        return list(Q_CORE_NO_RH2M_PMST_FEATURES)
    if key == "overlap_full":
        return list(OVERLAP_CANONICAL)
    if key == "source_full":
        if available_features is None:
            return list(FINAL_FEATURE_ORDER)
        available = {normalize_var_coord(v) for v in available_features}
        return [name for name in FINAL_FEATURE_ORDER if name in available]
    raise ValueError(f"Unknown feature_set={feature_set!r}; expected one of {FEATURE_SET_CHOICES}")


def is_compact_common_core(feature_set: str) -> bool:
    key = str(feature_set or "").strip().lower().replace("-", "_")
    return key in {
        "compact",
        "compact_common",
        "compact_core",
        "compact_common_core",
        "compact_common_core_no_rh2m",
        "compact_no_rh2m",
        "compact_common_no_rh2m",
        "compact_core_no_rh2m",
    }


def _feature_set_key(feature_set: str) -> str:
    key = str(feature_set or "overlap_full").strip().lower().replace("-", "_")
    if key in {"common", "core", "pangu_core", "common_overlap"}:
        return "common_core"
    if key in {"compact", "compact_common", "compact_core", "compact_no_rh2m", "compact_common_no_rh2m", "compact_core_no_rh2m"}:
        return "compact_common_core_no_rh2m"
    if key in {"q_core", "q1000_core", "q1000_core_no_rh2m", "q_core_common", "q_core_no_rh"}:
        return "q_core_no_rh2m"
    if key in {"full", "overlap", "overlap_canonical"}:
        return "overlap_full"
    if key in {"all", "all_available", "source"}:
        return "source_full"
    return key


def _normalise_pmst_feature_list(feature_names: Optional[Iterable[str]], fallback: List[str]) -> List[str]:
    if feature_names is None:
        names = fallback
    else:
        names = [normalize_var_coord(v) for v in feature_names]
    seen: Set[str] = set()
    out: List[str] = []
    for name in names:
        if name in PMST_INDEX and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def dynamic_feature_order_for_feature_set(
    feature_set: str,
    feature_names: Optional[Iterable[str]] = None,
) -> List[str]:
    key = _feature_set_key(feature_set)
    if key == "compact_common_core_no_rh2m":
        met = list(COMPACT_COMMON_CORE_PMST_FEATURES)
    elif key == "q_core_no_rh2m":
        met = list(Q_CORE_NO_RH2M_PMST_FEATURES)
    elif key == "common_core":
        met = list(COMMON_CORE_PMST_FEATURES)
    elif key == "overlap_full":
        met = _normalise_pmst_feature_list(feature_names, list(OVERLAP_CANONICAL))
    elif key == "source_full":
        met = _normalise_pmst_feature_list(feature_names, list(FINAL_FEATURE_ORDER))
    else:
        met = _normalise_pmst_feature_list(feature_names, list(FINAL_FEATURE_ORDER))
    return [*met, "ZENITH", "PM10_ugm3", "PM25_ugm3"]


def dyn_vars_for_feature_set(feature_set: str, feature_names: Optional[Iterable[str]] = None) -> int:
    return len(dynamic_feature_order_for_feature_set(feature_set, feature_names))


def dynamic_layout_name(feature_set: str, feature_names: Optional[Iterable[str]] = None) -> str:
    order = dynamic_feature_order_for_feature_set(feature_set, feature_names)
    met_n = max(0, len(order) - 3)
    return f"{met_n}_native_pmst_met + zenith + pm10 + pm2p5"


def select_dynamic_layout(
    x_dyn: np.ndarray,
    feature_set: str,
    feature_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Return PMST-27 dynamics in the requested native per-time-step layout."""
    arr = np.asarray(x_dyn)
    if arr.shape[-1] != TOTAL_DYN:
        raise ValueError(
            f"native feature selection expects PMST-27 source dynamics before slicing; got last dim={arr.shape[-1]}"
        )
    indices: List[int] = []
    for name in dynamic_feature_order_for_feature_set(feature_set, feature_names):
        canon = normalize_var_coord(name)
        if canon in PMST_INDEX:
            indices.append(PMST_INDEX[canon])
        elif canon == "ZENITH":
            indices.append(ZENITH_IDX)
        elif canon in {"PM10", "PM10_UGM3", "PM10_UG_M3"}:
            indices.append(PM10_IDX)
        elif canon in {"PM25", "PM25_UGM3", "PM25_UG_M3", "PM2P5"}:
            indices.append(PM25_IDX)
        else:
            raise KeyError(f"Unknown dynamic feature name {name!r}")
    return arr[..., indices].astype(np.float32, copy=False)


def scatter_overlap_fields(
    nt: int,
    ns: int,
    fields: Dict[str, np.ndarray],
    feature_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    fields: canonical/source name -> (nt, ns) float32 array.
    Returns X_met (nt, ns, 24) with shared PMST slots filled and derived fields added.
    """
    fields = {
        normalize_var_coord(k): canonicalize_pmst_field(k, v)
        for k, v in fields.items()
    }
    enabled = {normalize_var_coord(v) for v in (feature_names if feature_names is not None else OVERLAP_CANONICAL)}
    enabled = {v for v in enabled if v in PMST_INDEX}
    x = np.zeros((nt, ns, PMST_MET_DIM), dtype=np.float32)
    derived = {"WSPD10", "WDIR10", "WSPD925", "RH2M", "Q_1000", "Q_925", "DP_1000", "DP_925", "DPD", "INVERSION"}
    for name in enabled:
        if name in derived:
            continue
        if name not in fields:
            continue
        arr = _field_array(fields, name, nt, ns)
        if arr is not None:
            x[:, :, PMST_INDEX[name]] = arr

    t2m = _field_array(fields, "T2M", nt, ns)
    rh2m = _field_array(fields, "RH2M", nt, ns)
    d2m = _field_array(fields, "D2M", nt, ns)
    if "RH2M" in enabled and rh2m is not None:
        x[:, :, RH2M_IDX] = rh2m
    elif "RH2M" in enabled and t2m is not None and d2m is not None:
        x[:, :, RH2M_IDX] = calculate_rh_from_dewpoint(t2m, d2m)
    if d2m is None and t2m is not None and rh2m is not None:
        d2m = calculate_dewpoint_from_rh(t2m, rh2m)
    if "DPD" in enabled and t2m is not None and d2m is not None:
        x[:, :, DPD_IDX] = (t2m - d2m).astype(np.float32)

    q1000 = _field_array(fields, "Q_1000", nt, ns)
    if q1000 is None:
        t1000 = _field_array(fields, "T_1000", nt, ns)
        rh1000 = _field_array(fields, "RH_1000", nt, ns)
        if t1000 is not None and rh1000 is not None:
            q1000 = calculate_specific_humidity(t1000, rh1000, 1000.0)
    if "Q_1000" in enabled and q1000 is not None:
        x[:, :, Q1000_IDX] = q1000
    if "DP_1000" in enabled and "DP_1000" in fields:
        x[:, :, DP1000_IDX] = _field_array(fields, "DP_1000", nt, ns)
    elif "DP_1000" in enabled and q1000 is not None:
        x[:, :, DP1000_IDX] = calculate_dewpoint_from_specific_humidity(q1000, 1000.0)

    q925 = _field_array(fields, "Q_925", nt, ns)
    if q925 is None:
        t925 = _field_array(fields, "T_925", nt, ns)
        rh925 = _field_array(fields, "RH_925", nt, ns)
        if t925 is not None and rh925 is not None:
            q925 = calculate_specific_humidity(t925, rh925, 925.0)
    if "Q_925" in enabled and q925 is not None:
        x[:, :, Q925_IDX] = q925
    if "DP_925" in enabled and "DP_925" in fields:
        x[:, :, DP925_IDX] = _field_array(fields, "DP_925", nt, ns)
    elif "DP_925" in enabled and q925 is not None:
        x[:, :, DP925_IDX] = calculate_dewpoint_from_specific_humidity(q925, 925.0)

    u10 = _field_array(fields, "U10", nt, ns)
    v10 = _field_array(fields, "V10", nt, ns)
    if ("WSPD10" in enabled or "WDIR10" in enabled) and u10 is not None and v10 is not None:
        speed10, dir10 = calculate_wind_speed_dir(u10, v10)
        if "WSPD10" in enabled:
            x[:, :, WSPD10_IDX] = speed10
        if "WDIR10" in enabled:
            x[:, :, WDIR10_IDX] = dir10
    else:
        wspd10 = _field_array(fields, "WSPD10", nt, ns)
        wdir10 = _field_array(fields, "WDIR10", nt, ns)
        if "WSPD10" in enabled and wspd10 is not None:
            x[:, :, WSPD10_IDX] = wspd10
        if "WDIR10" in enabled and wdir10 is not None:
            x[:, :, WDIR10_IDX] = wdir10

    u925 = _field_array(fields, "U_925", nt, ns)
    v925 = _field_array(fields, "V_925", nt, ns)
    if "WSPD925" in enabled and u925 is not None and v925 is not None:
        speed925, _ = calculate_wind_speed_dir(u925, v925)
        x[:, :, WSPD925_IDX] = speed925
    else:
        wspd925 = _field_array(fields, "WSPD925", nt, ns)
        if "WSPD925" in enabled and wspd925 is not None:
            x[:, :, WSPD925_IDX] = wspd925
    if "INVERSION" in enabled:
        inv = _field_array(fields, "INVERSION", nt, ns)
        if inv is not None:
            x[:, :, PMST_INDEX["INVERSION"]] = inv
        else:
            t925 = _field_array(fields, "T_925", nt, ns)
            if t925 is not None and t2m is not None:
                x[:, :, PMST_INDEX["INVERSION"]] = (t925 - t2m).astype(np.float32)
    return x


def get_monthly_split_mask_last_days(
    sample_times: pd.DatetimeIndex,
    gap_hours: int,
    val_last_days: int,
    test_last_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same semantics as s2_data_monthtail_v2.ipynb (month-tail split)."""
    times = pd.DatetimeIndex(sample_times)
    train_mask = np.zeros(len(times), dtype=bool)
    val_mask = np.zeros(len(times), dtype=bool)
    test_mask = np.zeros(len(times), dtype=bool)

    if gap_hours % 24 != 0:
        raise ValueError(f"gap_hours must be a multiple of 24, got {gap_hours}")
    gap_days = gap_hours // 24

    for month_period in times.to_period("M").unique():
        start = month_period.start_time
        end = month_period.end_time
        dim = month_period.days_in_month

        d_test0 = dim - test_last_days + 1
        d_val1 = d_test0 - gap_days - 1
        d_val0 = d_val1 - val_last_days + 1
        d_train_end = d_val0 - gap_days - 1

        if d_train_end < 1 or d_val0 < 1 or d_val1 < d_val0:
            print(
                f"  [WARN] split skip month {month_period}: "
                f"d_train_end={d_train_end}, d_val0={d_val0}, d_val1={d_val1}, d_test0={d_test0}",
                flush=True,
            )
            continue

        t_train_end = pd.Timestamp(month_period.year, month_period.month, d_train_end)
        t_val0 = pd.Timestamp(month_period.year, month_period.month, d_val0)
        t_val1 = pd.Timestamp(month_period.year, month_period.month, d_val1)
        t_test0 = pd.Timestamp(month_period.year, month_period.month, d_test0)

        msub = (times >= start) & (times <= end)
        train_mask |= msub & (times <= t_train_end + pd.Timedelta(hours=23, minutes=59, seconds=59))
        val_mask |= msub & (times >= t_val0) & (times <= t_val1 + pd.Timedelta(hours=23, minutes=59, seconds=59))
        test_mask |= msub & (times >= t_test0)

    return train_mask, val_mask, test_mask


def load_pm10_dataarray(pm10_file: str, pm10_dir: str) -> Optional[xr.DataArray]:
    """Return (time, station_id) pm10 DataArray or None."""
    das: List[xr.DataArray] = []
    if os.path.isfile(pm10_file):
        ds = xr.open_dataset(pm10_file, engine="h5netcdf")
        v = "pm10" if "pm10" in ds else list(ds.data_vars)[0]
        das.append(ds[v])
        ds.close()
    elif os.path.isdir(pm10_dir):
        files = sorted(glob.glob(os.path.join(pm10_dir, "*.nc")))
        for fp in files:
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
                if "station_id" not in ds.coords and "station_id" not in ds.data_vars:
                    for alias in ["num_station", "id", "station"]:
                        if alias in ds.coords or alias in ds.data_vars:
                            ds = ds.rename({alias: "station_id"})
                            break
                vn = list(ds.data_vars)[0]
                das.append(ds[vn])
                ds.close()
            except Exception as e:
                print(f"[WARN] skip pm10 file {fp}: {e}", flush=True)
    if not das:
        print("[WARN] No PM10 files found; pm10 channel will be 0.", flush=True)
        return None
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if "station_id" not in da.dims:
        raise ValueError("pm10 DataArray must have station_id dimension")
    return da.transpose("time", "station_id")


def append_pm10_channel(
    x_dyn_25: np.ndarray,
    pm10_da: Optional[xr.DataArray],
    times: pd.DatetimeIndex,
    station_ids: np.ndarray,
) -> np.ndarray:
    """
    x_dyn_25: (nt, ns, 25) =24 met + zenith (last channel).
    Returns (nt, ns, 26) with pm10 as last channel (ug/m^3), same conversion as monthtail notebook.
    """
    nt, ns, nv = x_dyn_25.shape
    if nv != 25:
        raise ValueError(f"Expected 25 channels before pm10 (24 met + zenith), got {nv}")
    if pm10_da is None:
        pm10_ug = np.zeros((nt, ns), dtype=np.float32)
    else:
        pm10_da = pm10_da.load()
        time_vals = pm10_da["time"].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_index = pd.DatetimeIndex(time_vals)
        else:
            time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
        sid_index = pd.Index(pm10_da["station_id"].values)
        time_pos = time_index.get_indexer(times, method="nearest", tolerance=pd.Timedelta(minutes=90))
        sids = station_ids.astype(pm10_da["station_id"].dtype)
        sid_pos = sid_index.get_indexer(sids)
        nt_pm10, ns_pm10 = pm10_da.shape
        pm10_grid = np.full((nt, ns), np.nan, dtype=np.float32)
        # (nt,) & (ns,) must broadcast to (nt, ns), same as linear_idx_grid
        ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
        base = np.asarray(pm10_da.values).reshape(-1)
        linear_idx_grid = time_pos[:, None] * ns_pm10 + sid_pos[None, :]
        if ok_mask.any():
            pm10_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
        pm10_ug = sanitize_pm_concentration(
            pm10_grid,
            str(pm10_da.attrs.get("units", "")),
        )
    return np.concatenate([x_dyn_25, pm10_ug[..., None]], axis=-1).astype(np.float32)


def load_pm25_dataarray(pm25_file: str, pm25_dir: str) -> Optional[xr.DataArray]:
    """Return (time, station_id) PM2.5 DataArray or None. Same layout as PM10 station files."""
    das: List[xr.DataArray] = []
    if os.path.isfile(pm25_file):
        ds = xr.open_dataset(pm25_file, engine="h5netcdf")
        v = None
        for cand in ("pm2p5", "pm25", "pm2_5", "PM2_5"):
            if cand in ds.data_vars:
                v = cand
                break
        if v is None:
            v = list(ds.data_vars)[0]
        das.append(ds[v])
        ds.close()
    elif os.path.isdir(pm25_dir):
        files = sorted(glob.glob(os.path.join(pm25_dir, "*.nc")))
        for fp in files:
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
                if "station_id" not in ds.coords and "station_id" not in ds.data_vars:
                    for alias in ["num_station", "id", "station"]:
                        if alias in ds.coords or alias in ds.data_vars:
                            ds = ds.rename({alias: "station_id"})
                            break
                vn = None
                for cand in ("pm2p5", "pm25", "pm2_5", "PM2_5"):
                    if cand in ds.data_vars:
                        vn = cand
                        break
                if vn is None:
                    vn = list(ds.data_vars)[0]
                das.append(ds[vn])
                ds.close()
            except Exception as e:
                print(f"[WARN] skip pm2.5 file {fp}: {e}", flush=True)
    if not das:
        print("[WARN] No PM2.5 files found; pm2p5 channel will be 0.", flush=True)
        return None
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if "station_id" not in da.dims:
        raise ValueError("pm2.5 DataArray must have station_id dimension")
    return da.transpose("time", "station_id")


def append_pm25_channel(
    x_dyn_26: np.ndarray,
    pm25_da: Optional[xr.DataArray],
    times: pd.DatetimeIndex,
    station_ids: np.ndarray,
) -> np.ndarray:
    """
    x_dyn_26: (nt, ns, 26) = 24 met + zenith + pm10.
    Returns (nt, ns, 27) with pm2p5 as last channel (µg/m³), same scaling as append_pm10_channel.
    """
    nt, ns, nv = x_dyn_26.shape
    if nv != 26:
        raise ValueError(f"Expected 26 channels before pm2p5 (24 met + zenith + pm10), got {nv}")
    if pm25_da is None:
        pm25_ug = np.zeros((nt, ns), dtype=np.float32)
    else:
        pm25_da = pm25_da.load()
        time_vals = pm25_da["time"].values
        if np.issubdtype(time_vals.dtype, np.datetime64):
            time_index = pd.DatetimeIndex(time_vals)
        else:
            time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
        sid_index = pd.Index(pm25_da["station_id"].values)
        time_pos = time_index.get_indexer(times, method="nearest", tolerance=pd.Timedelta(minutes=90))
        sids = station_ids.astype(pm25_da["station_id"].dtype)
        sid_pos = sid_index.get_indexer(sids)
        nt_pm25, ns_pm25 = pm25_da.shape
        pm25_grid = np.full((nt, ns), np.nan, dtype=np.float32)
        ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
        base = np.asarray(pm25_da.values).reshape(-1)
        linear_idx_grid = time_pos[:, None] * ns_pm25 + sid_pos[None, :]
        if ok_mask.any():
            pm25_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
        pm25_ug = sanitize_pm_concentration(
            pm25_grid,
            str(pm25_da.attrs.get("units", "")),
        )
    return np.concatenate([x_dyn_26, pm25_ug[..., None]], axis=-1).astype(np.float32)


def _compute_compact_common_core_fog_features(X_dyn_window: np.ndarray) -> np.ndarray:
    """Compact common-core FE using only variables present in the no-RH2M layout."""
    idx = COMPACT_INDEX
    x_cur = X_dyn_window[:, -1, :]
    t2m = x_cur[:, idx["T2M"]]
    wspd = x_cur[:, idx["WSPD10"]]
    rh925 = x_cur[:, idx["RH_925"]]
    dpd = x_cur[:, idx["DPD"]]
    q1000 = x_cur[:, idx["Q_1000"]]
    q925 = x_cur[:, idx["Q_925"]]
    dp1000 = x_cur[:, idx["DP_1000"]]
    dp925 = x_cur[:, idx["DP_925"]]
    u10 = x_cur[:, idx["U10"]]
    v10 = x_cur[:, idx["V10"]]
    u925 = x_cur[:, idx["U_925"]]
    v925 = x_cur[:, idx["V_925"]]
    mslp = x_cur[:, idx["MSLP"]]
    zenith = x_cur[:, idx["ZENITH"]]
    t2m_c = t2m - 273.15 if np.nanmean(t2m) > 200 else t2m

    dpd_weight = 1.0 / (1.0 + np.exp(dpd / 2.0))
    rh925_norm = np.clip(rh925 / 100.0, 0.0, 1.0)
    wind_fav = np.exp(-0.5 * ((wspd - 3.5) / 2.5) ** 2)
    is_night = (zenith > 90.0).astype(np.float32)
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    theta10 = np.arctan2(v10, u10)
    theta925 = np.arctan2(v925, u925)
    dir_turning = 0.5 * (1.0 - np.cos(theta925 - theta10))
    ventilation = np.tanh((wspd * (1.0 + shear_mag)) / 12.0)
    moisture_strat = np.tanh((q1000 - q925) * 1500.0)
    dewpoint_contrast = np.tanh((dp1000 - dp925) / 8.0)
    pressure_anom = np.tanh((mslp - np.nanmedian(mslp)) / 2000.0)
    fog_pot = dpd_weight * 0.45 + rh925_norm * 0.25 + wind_fav * 0.15 + is_night * 0.15

    feats = [
        dpd_weight.reshape(-1, 1),
        wind_fav.reshape(-1, 1),
        rh925_norm.reshape(-1, 1),
        fog_pot.reshape(-1, 1),
    ]
    for name in ("T2M", "WSPD10", "DPD"):
        seq = X_dyn_window[:, :, idx[name]]
        feats.append((seq[:, -1] - seq[:, -4]).reshape(-1, 1))
        feats.append((seq[:, -1] - seq[:, -7]).reshape(-1, 1))
        feats.append(np.std(seq, axis=1).reshape(-1, 1))
        feats.append((np.max(seq, axis=1) - np.min(seq, axis=1)).reshape(-1, 1))
    dpd_seq = X_dyn_window[:, :, idx["DPD"]]
    dpd_accel = (dpd_seq[:, -1] - dpd_seq[:, -4]) - (dpd_seq[:, -4] - dpd_seq[:, -7])
    feats.extend(
        [
            dpd_accel.reshape(-1, 1),
            (dpd_weight * np.exp(-t2m_c / 10.0)).reshape(-1, 1),
            ((is_night > 0) & (dpd < 2.0)).astype(np.float32).reshape(-1, 1),
            ((dpd < 2.0) & (t2m_c < 10.0) & (wspd < 4.0)).astype(np.float32).reshape(-1, 1),
            (dpd_weight ** 2).reshape(-1, 1),
            np.tanh(shear_mag / 8.0).reshape(-1, 1),
            dir_turning.reshape(-1, 1),
            ventilation.reshape(-1, 1),
            moisture_strat.reshape(-1, 1),
            dewpoint_contrast.reshape(-1, 1),
            pressure_anom.reshape(-1, 1),
        ]
    )
    out = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)


def _dynamic_order_lookup(feature_order: Iterable[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for i, raw in enumerate(feature_order):
        name = normalize_var_coord(raw)
        aliases = {name}
        if name in {"PM10_UGM3", "PM10_UG_M3"}:
            aliases.add("PM10")
        if name in {"PM25_UGM3", "PM25_UG_M3", "PM2P5"}:
            aliases.update({"PM25", "PM2P5"})
        for alias in aliases:
            lookup[alias] = i
    return lookup


def _compute_native_fog_features(X_dyn_window: np.ndarray, feature_order: Iterable[str]) -> np.ndarray:
    idx = _dynamic_order_lookup(feature_order)
    x_cur = X_dyn_window[:, -1, :]
    feats: List[np.ndarray] = []

    def has(*names: str) -> bool:
        return all(normalize_var_coord(n) in idx for n in names)

    def cur(name: str) -> np.ndarray:
        return x_cur[:, idx[normalize_var_coord(name)]].astype(np.float32)

    def seq(name: str) -> np.ndarray:
        return X_dyn_window[:, :, idx[normalize_var_coord(name)]].astype(np.float32)

    rh2m = cur("RH2M") if has("RH2M") else None
    t2m = cur("T2M") if has("T2M") else None
    wspd = cur("WSPD10") if has("WSPD10") else None
    dpd = cur("DPD") if has("DPD") else None
    rh925 = cur("RH_925") if has("RH_925") else None
    zenith = cur("ZENITH") if has("ZENITH") else None
    lcc = cur("LCC") if has("LCC") else None
    sw_rad = cur("SW_RAD") if has("SW_RAD") else None
    inversion = cur("INVERSION") if has("INVERSION") else None
    t2m_c = (t2m - 273.15 if t2m is not None and np.nanmean(t2m) > 200 else t2m)

    fog_terms: List[Tuple[float, np.ndarray]] = []
    if rh2m is not None and dpd is not None:
        rh_norm = np.clip(rh2m / 100.0, 0.0, 1.0)
        dpd_weight = 1.0 / (1.0 + np.exp(dpd / 2.0))
        feats.append((rh_norm * dpd_weight).reshape(-1, 1))
        fog_terms.append((0.40, rh_norm))
        fog_terms.append((0.25, dpd_weight))
    elif dpd is not None:
        dpd_weight = 1.0 / (1.0 + np.exp(dpd / 2.0))
        feats.append(dpd_weight.reshape(-1, 1))
        fog_terms.append((0.45, dpd_weight))
    if wspd is not None:
        wind_fav = np.exp(-0.5 * ((wspd - 3.5) / 2.5) ** 2)
        feats.append(wind_fav.reshape(-1, 1))
        fog_terms.append((0.20, wind_fav))
    if inversion is not None and wspd is not None:
        stability = np.tanh(inversion / (wspd ** 2 + 0.1) / 2.0)
        feats.append(stability.reshape(-1, 1))
        fog_terms.append((0.15, np.clip(stability, 0.0, 1.0)))
    if zenith is not None:
        is_night = (zenith > 90.0).astype(np.float32)
        if lcc is not None and sw_rad is not None:
            clear_sky = np.clip(1.0 - lcc / 0.3, 0.0, 1.0)
            rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0.0) / 800.0, 0.0, 1.0)
            night_rad = is_night * clear_sky * rad_intensity
        else:
            night_rad = is_night
        feats.append(night_rad.reshape(-1, 1))
        fog_terms.append((0.15, night_rad))
    if rh2m is not None and rh925 is not None:
        feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))
    if fog_terms:
        total_w = sum(w for w, _ in fog_terms)
        fog_pot = sum(w * v for w, v in fog_terms) / max(total_w, 1e-6)
        feats.append(fog_pot.reshape(-1, 1))

    for name in ("RH2M", "T2M", "WSPD10", "DPD", "RH_925", "Q_1000", "Q_925"):
        if has(name):
            s = seq(name)
            feats.append((s[:, -1] - s[:, -4]).reshape(-1, 1))
            feats.append((s[:, -1] - s[:, -7]).reshape(-1, 1))
            feats.append(np.std(s, axis=1).reshape(-1, 1))
            feats.append((np.max(s, axis=1) - np.min(s, axis=1)).reshape(-1, 1))

    if has("U10", "V10", "U_925", "V_925"):
        u10, v10, u925, v925 = cur("U10"), cur("V10"), cur("U_925"), cur("V_925")
        shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
        theta10 = np.arctan2(v10, u10)
        theta925 = np.arctan2(v925, u925)
        feats.append(np.tanh(shear_mag / 8.0).reshape(-1, 1))
        feats.append((0.5 * (1.0 - np.cos(theta925 - theta10))).reshape(-1, 1))
        if wspd is not None:
            feats.append(np.tanh((wspd * (1.0 + shear_mag)) / 12.0).reshape(-1, 1))

    if has("CAPE", "PRECIP"):
        cape = np.maximum(cur("CAPE"), 0.0)
        precip = np.maximum(cur("PRECIP"), 0.0)
        convective_wet = (1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))) * (
            1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5))
        )
        feats.append(convective_wet.reshape(-1, 1))
    if sw_rad is not None and wspd is not None and inversion is not None:
        daytime_mixing = (1.0 / (1.0 + np.exp(-(np.maximum(sw_rad, 0.0) - 150.0) / 75.0))) * (
            1.0 / (1.0 + np.exp(-(wspd - 4.0) / 1.5))
        ) * (1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2)))
        feats.append(daytime_mixing.reshape(-1, 1))
    if has("Q_1000", "Q_925"):
        feats.append(np.tanh((cur("Q_1000") - cur("Q_925")) * 1500.0).reshape(-1, 1))
    if has("W_925", "W_1000"):
        feats.append(np.tanh((cur("W_925") - cur("W_1000")) / 0.25).reshape(-1, 1))
    if inversion is not None and t2m_c is not None:
        feats.append(np.tanh((-inversion + np.maximum(t2m_c - 18.0, 0.0) * 0.25) / 3.0).reshape(-1, 1))
    if rh2m is not None and t2m_c is not None:
        feats.append((rh2m * np.exp(-t2m_c / 10.0)).reshape(-1, 1))
    if rh2m is not None and lcc is not None:
        feats.append((rh2m / (lcc * 100.0 + 1.0)).reshape(-1, 1))

    if not feats:
        feats.append(np.zeros((X_dyn_window.shape[0], 1), dtype=np.float32))
    out = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)


def compute_fog_features_pmst(
    X_dyn_window: np.ndarray,
    window_size: int = 12,
    dyn_vars: int = 27,
    feature_order: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    32-dim fog FE + caller appends 4 cyclical => 36 total extra (matches monthtail pm10 notebook).
    X_dyn_window: (N, window, dyn_vars) with dyn_vars=27 (24 met + zenith + pm10 + pm2p5); FE uses met+zenith only.
    """
    if feature_order is not None:
        return _compute_native_fog_features(X_dyn_window, feature_order)
    if dyn_vars == COMPACT_TOTAL_DYN:
        return _compute_compact_common_core_fog_features(X_dyn_window)
    _ = window_size, dyn_vars
    idx = {
        "rh2m": 0,
        "t2m": 1,
        "precip": 2,
        "sw_rad": 4,
        "wspd10": 6,
        "cape": 9,
        "lcc": 10,
        "t925": 11,
        "rh925": 12,
        "dpd": 22,
        "inversion": 23,
        "zenith": 24,
    }
    params = {
        "optimal_wspd": 3.5,
        "wspd_sigma": 2.5,
        "dpd_threshold": 2.0,
        "stability_scale": 2.0,
        "lcc_threshold": 0.3,
        "rad_threshold": 800.0,
    }
    X_current = X_dyn_window[:, -1, :]
    rh2m, rh925 = X_current[:, idx["rh2m"]], X_current[:, idx["rh925"]]
    dpd, wspd = X_current[:, idx["dpd"]], X_current[:, idx["wspd10"]]
    inversion, sw_rad = X_current[:, idx["inversion"]], X_current[:, idx["sw_rad"]]
    lcc, zenith = X_current[:, idx["lcc"]], X_current[:, idx["zenith"]]
    t2m = X_current[:, idx["t2m"]]
    t2m_c = t2m - 273.15 if np.nanmean(t2m) > 200 else t2m
    feats = []
    rh_norm = np.clip(rh2m / 100.0, 0, 1)
    dpd_weight = 1.0 / (1.0 + np.exp(dpd / params["dpd_threshold"]))
    feats.append((rh_norm * dpd_weight).reshape(-1, 1))
    wind_fav = np.exp(-0.5 * ((wspd - params["optimal_wspd"]) / params["wspd_sigma"]) ** 2)
    feats.append(wind_fav.reshape(-1, 1))
    ri = inversion / (wspd**2 + 0.1)
    stability = np.tanh(ri / params["stability_scale"])
    feats.append(stability.reshape(-1, 1))
    is_night = (zenith > 90.0).astype(float)
    clear_sky = np.clip(1.0 - lcc / params["lcc_threshold"], 0, 1)
    rad_intensity = 1.0 - np.clip(np.maximum(sw_rad, 0) / params["rad_threshold"], 0, 1)
    feats.append((is_night * clear_sky * rad_intensity).reshape(-1, 1))
    feats.append(np.tanh((rh2m - rh925) / 50.0).reshape(-1, 1))
    fog_pot = (
        rh_norm * 0.4
        + wind_fav * 0.25
        + np.clip(stability, 0, 1) * 0.2
        + (is_night * clear_sky * rad_intensity) * 0.15
    )
    feats.append(fog_pot.reshape(-1, 1))
    for var_idx in [idx["rh2m"], idx["t2m"], idx["wspd10"]]:
        var_seq = X_dyn_window[:, :, var_idx]
        feats.append((var_seq[:, -1] - var_seq[:, -4]).reshape(-1, 1))
        feats.append((var_seq[:, -1] - var_seq[:, -7]).reshape(-1, 1))
        feats.append(np.std(var_seq, axis=1).reshape(-1, 1))
        feats.append((np.max(var_seq, axis=1) - np.min(var_seq, axis=1)).reshape(-1, 1))
    rh_seq = X_dyn_window[:, :, idx["rh2m"]]
    rh_accel = (rh_seq[:, -1] - rh_seq[:, -4]) - (rh_seq[:, -4] - rh_seq[:, -7])
    feats.append(rh_accel.reshape(-1, 1))
    feats.append((rh2m * np.exp(-t2m_c / 10.0)).reshape(-1, 1))
    feats.append((is_night * (1 - lcc)).reshape(-1, 1))
    feats.append(((rh2m > 90) & (t2m_c < 10) & (wspd < 4)).astype(float).reshape(-1, 1))
    feats.append((rh2m / (lcc * 100 + 1)).reshape(-1, 1))
    feats.append(((rh2m / 100.0) ** 2).reshape(-1, 1))
    u10 = X_current[:, 5]
    v10 = X_current[:, 7]
    u925 = X_current[:, 13]
    v925 = X_current[:, 15]
    precip = np.maximum(X_current[:, idx["precip"]], 0.0)
    cape = np.maximum(X_current[:, idx["cape"]], 0.0)
    q1000 = X_current[:, 18]
    q925 = X_current[:, 19]
    w925 = X_current[:, 20]
    w1000 = X_current[:, 21]
    shear_mag = np.sqrt((u925 - u10) ** 2 + (v925 - v10) ** 2)
    theta10 = np.arctan2(v10, u10)
    theta925 = np.arctan2(v925, u925)
    dir_turning = 0.5 * (1.0 - np.cos(theta925 - theta10))
    convective_wet = (1.0 / (1.0 + np.exp(-(np.log1p(cape) - np.log(200.0)) * 1.6))) * (
        1.0 / (1.0 + np.exp(-(np.log1p(precip) - np.log(0.1)) * 2.5))
    )
    daytime_mixing = (1.0 / (1.0 + np.exp(-(np.maximum(sw_rad, 0.0) - 150.0) / 75.0))) * (
        1.0 / (1.0 + np.exp(-(wspd - 4.0) / 1.5))
    ) * (1.0 / (1.0 + np.exp(-(-inversion + 0.5) / 1.2)))
    ventilation = np.tanh((wspd * (1.0 + shear_mag)) / 12.0)
    moisture_strat = np.tanh((q1000 - q925) * 1500.0)
    omega_contrast = np.tanh((w925 - w1000) / 0.25)
    warm_instability = np.tanh((-inversion + np.maximum(t2m_c - 18.0, 0.0) * 0.25) / 3.0)
    feats.append(np.tanh(shear_mag / 8.0).reshape(-1, 1))
    feats.append(dir_turning.reshape(-1, 1))
    feats.append(convective_wet.reshape(-1, 1))
    feats.append(daytime_mixing.reshape(-1, 1))
    feats.append(ventilation.reshape(-1, 1))
    feats.append(moisture_strat.reshape(-1, 1))
    feats.append(omega_contrast.reshape(-1, 1))
    feats.append(warm_instability.reshape(-1, 1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def normalize_var_coord(v) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    raw = str(v)
    compact = raw.strip().upper().replace("-", "_").replace(" ", "_")
    compact_no_underscore = compact.replace("_", "")
    return CANONICAL_VAR_ALIASES.get(compact, CANONICAL_VAR_ALIASES.get(compact_no_underscore, raw.strip()))


def build_station_reindex_map(
    station_ids_tianji: np.ndarray,
    station_ids_ifs: np.ndarray,
) -> np.ndarray:
    """For each Tianji station, index into IFS station axis. Raises if any missing."""
    sid_ifs = [normalize_var_coord(s) for s in np.asarray(station_ids_ifs).tolist()]
    sid_tj = [normalize_var_coord(s) for s in np.asarray(station_ids_tianji).tolist()]
    lookup = {s: i for i, s in enumerate(sid_ifs)}
    out = np.empty(len(sid_tj), dtype=np.int64)
    for i, s in enumerate(sid_tj):
        if s not in lookup:
            raise KeyError(f"Tianji station_id {s!r} not present in IFS file stations")
        out[i] = lookup[s]
    return out


def cyclical_time_features(sample_times: pd.DatetimeIndex) -> np.ndarray:
    months = sample_times.month.values.astype(np.float32)
    hours = sample_times.hour.values.astype(np.float32)
    return np.column_stack(
        [
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
        ]
    ).astype(np.float32)


def save_chunked_monthtail(
    X_dyn_flat: np.ndarray,
    X_stat_flat: np.ndarray,
    fe_flat: np.ndarray,
    y_flat: np.ndarray,
    mask: np.ndarray,
    meta: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_dir: str,
    gap_hours: int,
    val_last_days: int,
    test_last_days: int,
) -> None:
    times_all, stats_all, lats_all, lons_all = meta
    valid_idxs = np.where(mask)[0]
    n = len(valid_idxs)
    print(f"  Valid Samples: {n} ({n / max(len(mask), 1):.1%})", flush=True)
    valid_times = pd.DatetimeIndex(times_all[valid_idxs])
    tr_m, val_m, test_m = get_monthly_split_mask_last_days(
        valid_times, gap_hours, val_last_days, test_last_days
    )
    splits = {"train": valid_idxs[tr_m], "val": valid_idxs[val_m], "test": valid_idxs[test_m]}
    dims = [X_dyn_flat.shape[1], X_stat_flat.shape[1], fe_flat.shape[1]]
    for tag, ix in splits.items():
        if len(ix) == 0:
            continue
        print(f"    Saving {tag} (N={len(ix)})", flush=True)
        np.save(os.path.join(out_dir, f"y_{tag}.npy"), y_flat[ix])
        pd.DataFrame(
            {"time": times_all[ix], "station_id": stats_all[ix], "lat": lats_all[ix], "lon": lons_all[ix]}
        ).to_csv(os.path.join(out_dir, f"meta_{tag}.csv"), index=False)
        fp = np.lib.format.open_memmap(
            os.path.join(out_dir, f"X_{tag}.npy"),
            mode="w+",
            dtype="float32",
            shape=(len(ix), sum(dims)),
        )
        bs = 100000
        for i in tqdm(range(0, len(ix), bs), desc=tag):
            bi = ix[i : i + bs]
            n_bi = len(bi)
            fp[i : i + n_bi, : dims[0]] = X_dyn_flat[bi]
            fp[i : i + n_bi, dims[0] : dims[0] + dims[1]] = X_stat_flat[bi]
            fp[i : i + n_bi, dims[0] + dims[1] :] = fe_flat[bi]
        del fp
        gc.collect()
