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
import os
from typing import Dict, List, Optional, Tuple

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

OVERLAP_CANONICAL: List[str] = ["T2M", "PRECIP", "MSLP", "SW_RAD", "U10", "V10"]

PMST_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FINAL_FEATURE_ORDER)}

# Where overlap variables sit in the 24-dim PMST met block
OVERLAP_PMST_INDICES: Dict[str, int] = {
    "T2M": PMST_INDEX["T2M"],
    "PRECIP": PMST_INDEX["PRECIP"],
    "MSLP": PMST_INDEX["MSLP"],
    "SW_RAD": PMST_INDEX["SW_RAD"],
    "U10": PMST_INDEX["U10"],
    "V10": PMST_INDEX["V10"],
}
WSPD10_IDX = PMST_INDEX["WSPD10"]
U10_IDX = PMST_INDEX["U10"]
V10_IDX = PMST_INDEX["V10"]


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


def apply_wspd10_from_uv(x_met: np.ndarray) -> None:
    """In-place: WSPD10 = hypot(U10, V10). x_met shape (..., 24)."""
    u = x_met[..., U10_IDX]
    v = x_met[..., V10_IDX]
    x_met[..., WSPD10_IDX] = np.sqrt(u * u + v * v).astype(np.float32)


def scatter_overlap_fields(
    nt: int,
    ns: int,
    fields: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    fields: canonical name -> (nt, ns) float32 array for each OVERLAP_CANONICAL key present.
    Returns X_met (nt, ns, 24) with overlap slots filled, WSPD10 derived, rest 0.
    """
    x = np.zeros((nt, ns, PMST_MET_DIM), dtype=np.float32)
    for name in OVERLAP_CANONICAL:
        if name not in fields:
            continue
        arr = np.asarray(fields[name], dtype=np.float32)
        if arr.shape != (nt, ns):
            raise ValueError(f"Field {name} shape {arr.shape}, expected {(nt, ns)}")
        j = OVERLAP_PMST_INDICES[name]
        x[:, :, j] = arr
    apply_wspd10_from_uv(x)
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
        pm10_grid = np.maximum(pm10_grid, 0.0)
        pm10_ug = pm10_grid * 1e12
        med = np.nanmedian(pm10_ug)
        if not np.isfinite(med):
            med = 0.0
        pm10_ug = np.where(np.isfinite(pm10_ug), pm10_ug, med).astype(np.float32)
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
        pm25_grid = np.maximum(pm25_grid, 0.0)
        pm25_ug = pm25_grid * 1e12
        med = np.nanmedian(pm25_ug)
        if not np.isfinite(med):
            med = 0.0
        pm25_ug = np.where(np.isfinite(pm25_ug), pm25_ug, med).astype(np.float32)
    return np.concatenate([x_dyn_26, pm25_ug[..., None]], axis=-1).astype(np.float32)


def compute_fog_features_pmst(X_dyn_window: np.ndarray, window_size: int = 12, dyn_vars: int = 27) -> np.ndarray:
    """
    32-dim fog FE + caller appends 4 cyclical => 36 total extra (matches monthtail pm10 notebook).
    X_dyn_window: (N, window, dyn_vars) with dyn_vars=27 (24 met + zenith + pm10 + pm2p5); FE uses met+zenith only.
    """
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
    return str(v)


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
