#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFS overlap baseline aligned to Tianji: PMST 27-dyn + 36-FE, month-tail split, PM10 + PM2.5 from station files.

Primary input: interpolated NetCDF from remote_batch/interpolate_ifs_specs_12_24h.py:
  - variable ``ifs_interp`` dims (time, station_id, variable), coord ``variable`` = canonical names.
  - Multiple outputs (e.g. ifs_interp_core_surface_2025.nc + ifs_interp_rad_precip_cloud_2025.nc)
    are merged on the variable axis via ``--ifs_interp_nc`` (comma-separated) or ``--ifs_interp_glob``.

Fallback: gridded multi-file layout under --ifs_root (per-variable folders), nearest gridpoint sampling.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from pmst_overlap_common import (
    OVERLAP_CANONICAL,
    OVERLAP_SOURCE_FIELDS,
    TOTAL_DYN,
    append_pm10_channel,
    append_pm25_channel,
    build_static_features,
    build_station_reindex_map,
    compute_fog_features_pmst,
    cyclical_time_features,
    load_pm10_dataarray,
    load_pm25_dataarray,
    normalize_tianji_times,
    TIANJI_INPUT_TIME_SHIFT_HOURS,
    TIANJI_TIME_ALIGNMENT,
    describe_available_overlap_features,
    normalize_var_coord,
    scatter_overlap_fields,
    save_chunked_monthtail,
    calculate_zenith_angle,
)

IFS_ROOT_DEFAULT = "/public/home/sd3team/sd3_database/src_data/IFS/nc_0p1"
VIS_MLP_ROOT = "/public/home/putianshu/vis_mlp"
IFS_BASELINE_ROOT = os.path.join(VIS_MLP_ROOT, "ifs_baseline")
BASE_PATH = VIS_MLP_ROOT
TIANJI_FILE_DEFAULT = os.path.join(BASE_PATH, "tianji_auto_station", "merged_final_all_vars.nc")
VEG_FILE_DEFAULT = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE_DEFAULT = "/public/home/putianshu/vis_cnn/data_orography.nc"
PM10_S2_FILE_DEFAULT = os.path.join(BASE_PATH, "pm10_station", "pm10_station_s2_2025.nc")
PM10_DIR_DEFAULT = os.path.join(BASE_PATH, "pm10_station")
PM25_S2_FILE_DEFAULT = os.path.join(BASE_PATH, "pm2.5_station", "pm2p5_station_s2_2025.nc")
PM25_DIR_DEFAULT = os.path.join(BASE_PATH, "pm2.5_station")

WINDOW_SIZE_DEFAULT = 12
STEP_SIZE_DEFAULT = 1
MAX_VIS_THRESHOLD = 30000
UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])

VAL_LAST_DAYS_DEFAULT = 3
TEST_LAST_DAYS_DEFAULT = 3
GAP_HOURS_DEFAULT = 24


class IfsVarSpec:
    __slots__ = ("folder", "var_name", "level_hpa")

    def __init__(self, folder: str, var_name: str, level_hpa: Optional[float] = None):
        self.folder = folder
        self.var_name = var_name
        self.level_hpa = level_hpa


IFS_MAP: Dict[str, IfsVarSpec] = {
    "T2M": IfsVarSpec("2t", "t2m"),
    "D2M": IfsVarSpec("2d", "d2m"),
    "PRECIP": IfsVarSpec("tp", "tp"),
    "MSLP": IfsVarSpec("msl", "msl"),
    "SW_RAD": IfsVarSpec("ssrd", "ssrd"),
    "U10": IfsVarSpec("10u", "u10"),
    "V10": IfsVarSpec("10v", "v10"),
    "LCC": IfsVarSpec("lcc", "lcc"),
    "RH_925": IfsVarSpec("r", "r", 925.0),
    "U_925": IfsVarSpec("u", "u", 925.0),
    "V_925": IfsVarSpec("v", "v", 925.0),
    "Q_1000": IfsVarSpec("q", "q", 1000.0),
    "Q_925": IfsVarSpec("q", "q", 925.0),
    "W_925": IfsVarSpec("w", "w", 925.0),
    "W_1000": IfsVarSpec("w", "w", 1000.0),
}

def _real_exists(path: str) -> bool:
    try:
        return os.path.exists(os.path.realpath(path))
    except OSError:
        return False


def _list_real_files(folder: str, ifs_root: str, year: int) -> List[str]:
    d = os.path.join(ifs_root, folder)
    files = sorted(glob.glob(os.path.join(d, f"*{year}*.nc")))
    return [fp for fp in files if _real_exists(fp)]


def _open_ifs_concat(folder: str, var_name: str, ifs_root: str, year: int) -> xr.Dataset:
    files = _list_real_files(folder, ifs_root, year)
    if not files:
        raise FileNotFoundError(f"No readable IFS files for {folder} {year}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    if var_name not in ds.data_vars:
        raise KeyError(f"IFS dataset {folder} missing var {var_name}. got={list(ds.data_vars)}")
    return ds


def _select_level(da: xr.DataArray, level_hpa: Optional[float]) -> xr.DataArray:
    if level_hpa is None:
        return da
    level_dim = next(
        (n for n in ["level", "isobaricInhPa", "pressure_level", "lev", "plev"] if n in da.dims),
        None,
    )
    if level_dim is None:
        raise ValueError(f"{da.name}: no pressure-level dimension for level_hpa={level_hpa}")
    levels = np.asarray(da[level_dim].values, dtype=float)
    idx = int(np.argmin(np.abs(levels - float(level_hpa))))
    return da.isel({level_dim: idx})


def _nearest_index(grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    pts = np.asarray(pts)
    return np.abs(grid[:, None] - pts[None, :]).argmin(axis=0).astype(np.int64)


def _parse_ifs_interp_input_paths(ifs_interp_nc: str, ifs_interp_glob: str) -> List[str]:
    """Comma-separated nc paths plus optional glob; dedupe, preserve order."""
    raw: List[str] = []
    for part in (ifs_interp_nc or "").split(","):
        p = part.strip()
        if p:
            raw.append(os.path.realpath(p))
    g = (ifs_interp_glob or "").strip()
    if g:
        raw.extend(sorted(glob.glob(os.path.realpath(g))))
    seen = set()
    out: List[str] = []
    for p in raw:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _fill_from_interp_nc_multi(
    paths_nc: Sequence[str],
    tj_times: pd.DatetimeIndex,
    station_ids_tj: np.ndarray,
) -> np.ndarray:
    """
    Merge one or more ``ifs_interp`` NetCDFs along the variable dimension (same time × station grid).
    Return X_met (nt, ns, 24) PMST scatter layout.
    """
    if not paths_nc:
        raise ValueError("paths_nc is empty")

    merged_var_names: List[str] = []
    chunks: List[np.ndarray] = []
    ref_shape_ts: tuple[int, int] | None = None
    ref_times: pd.DatetimeIndex | None = None
    ref_stations: np.ndarray | None = None

    for path_nc in paths_nc:
        ds = xr.open_dataset(path_nc)
        if "ifs_interp" not in ds.data_vars:
            ds.close()
            raise KeyError(f"{path_nc}: expected data var 'ifs_interp', got {list(ds.data_vars)}")
        arr = np.asarray(ds["ifs_interp"].values, dtype=np.float32)
        ifs_times = pd.DatetimeIndex(ds["time"].values)
        ifs_stations = np.asarray(ds["station_id"].values)
        var_coord = [normalize_var_coord(v) for v in ds["variable"].values]
        ds.close()

        if arr.ndim != 3:
            raise ValueError(f"{path_nc}: ifs_interp must be 3D (time, station, var), got {arr.shape}")
        t_sz, s_sz, v_sz = arr.shape
        if v_sz != len(var_coord):
            raise ValueError(f"{path_nc}: variable coord len {len(var_coord)} != array last dim {v_sz}")

        if ref_shape_ts is None:
            ref_shape_ts = (t_sz, s_sz)
            ref_times = ifs_times
            ref_stations = ifs_stations
        else:
            if (t_sz, s_sz) != ref_shape_ts:
                raise ValueError(
                    f"{path_nc}: shape (time,station) {(t_sz, s_sz)} != first file {ref_shape_ts}"
                )
            if len(ifs_times) != len(ref_times) or not np.array_equal(ifs_times.values, ref_times.values):
                raise ValueError(f"{path_nc}: time coordinate does not match first interp file")
            if ifs_stations.shape != ref_stations.shape or not np.array_equal(ifs_stations, ref_stations):
                raise ValueError(f"{path_nc}: station_id coordinate does not match first interp file")

        for vn in var_coord:
            if vn in merged_var_names:
                raise ValueError(
                    f"Duplicate canonical variable {vn!r} in {path_nc}; remove duplicate outputs."
                )
        merged_var_names.extend(var_coord)
        chunks.append(arr)

    assert ref_times is not None and ref_stations is not None
    big = np.concatenate(chunks, axis=2)
    st_cols = build_station_reindex_map(station_ids_tj, ref_stations)
    time_pos = ref_times.get_indexer(tj_times, method="nearest", tolerance=pd.Timedelta(minutes=90))

    available = describe_available_overlap_features(merged_var_names)
    missing = [name for name in OVERLAP_CANONICAL if name not in available]
    if missing:
        raise KeyError(
            f"Merged IFS interp cannot populate overlap variable(s) {missing!r}. "
            f"Have source variables {merged_var_names!r} from {len(paths_nc)} file(s)."
        )

    nt, ns = len(tj_times), len(station_ids_tj)
    fields: Dict[str, np.ndarray] = {}
    for name in OVERLAP_SOURCE_FIELDS:
        if name not in merged_var_names:
            continue
        vi = merged_var_names.index(name)
        slot = np.full((nt, ns), np.nan, dtype=np.float32)
        for t in range(nt):
            if time_pos[t] < 0:
                continue
            slot[t, :] = big[time_pos[t], st_cols, vi]
        fields[name] = slot
    return scatter_overlap_fields(nt, ns, fields)


def _fill_from_gridded(
    ifs_root: str,
    year: int,
    tj_times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    ifs_datasets = {}
    for canon, spec in IFS_MAP.items():
        ifs_datasets[canon] = _open_ifs_concat(spec.folder, spec.var_name, ifs_root, year)

    first = next(iter(ifs_datasets.values()))
    lat_name = "latitude" if "latitude" in first.coords or "latitude" in first.dims else "lat"
    lon_name = "longitude" if "longitude" in first.coords or "longitude" in first.dims else "lon"
    lat_grid = first[lat_name].values
    lon_grid = first[lon_name].values
    lons_360 = (lons % 360).astype(np.float64)
    lat_idx = _nearest_index(lat_grid, lats.astype(np.float64))
    lon_idx = _nearest_index(lon_grid, lons_360)

    ifs_time = pd.DatetimeIndex(first["time"].values)
    time_pos = ifs_time.get_indexer(tj_times, method="nearest", tolerance=pd.Timedelta(minutes=90))

    nt, ns = len(tj_times), len(lats)
    fields: Dict[str, np.ndarray] = {}
    try:
        for canon, spec in IFS_MAP.items():
            ds = ifs_datasets[canon]
            da = _select_level(ds[spec.var_name], spec.level_hpa)
            ok_t = time_pos >= 0
            if not ok_t.any():
                raise RuntimeError("No overlapping times between Tianji and IFS (after tolerance).")
            da_t = da.isel(time=xr.DataArray(time_pos[ok_t], dims="time_sel"))
            lat_dim = "latitude" if "latitude" in da_t.dims else "lat"
            lon_dim = "longitude" if "longitude" in da_t.dims else "lon"
            da_s = da_t.isel(
                {
                    lat_dim: xr.DataArray(lat_idx, dims="station"),
                    lon_dim: xr.DataArray(lon_idx, dims="station"),
                }
            )
            slot = np.full((nt, ns), np.nan, dtype=np.float32)
            arr = np.asarray(da_s.values, dtype=np.float32)
            if arr.ndim != 2:
                raise RuntimeError(f"Unexpected sampled array shape for {canon}: {arr.shape}")
            slot[ok_t, :] = arr
            fields[canon] = slot
    finally:
        for ds in ifs_datasets.values():
            ds.close()
    available = describe_available_overlap_features(fields.keys())
    missing = [name for name in OVERLAP_CANONICAL if name not in available]
    if missing:
        raise KeyError(f"Gridded IFS inputs cannot populate overlap variable(s): {missing!r}")
    return scatter_overlap_fields(nt, ns, fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ifs_interp_nc",
        default="",
        help="Interpolated IFS NetCDF(s): single path, or comma-separated list (ifs_interp var each). "
        "If empty and --ifs_interp_glob empty, use gridded --ifs_root.",
    )
    ap.add_argument(
        "--ifs_interp_glob",
        default="",
        help="Optional glob for extra ifs_interp files (e.g. .../ifs_interp_out/ifs_interp_*_2025.nc). "
        "Merged with --ifs_interp_nc on the variable axis; time/station must match.",
    )
    ap.add_argument("--ifs_root", default=IFS_ROOT_DEFAULT)
    ap.add_argument("--tianji_file", default=TIANJI_FILE_DEFAULT)
    ap.add_argument("--veg_file", default=VEG_FILE_DEFAULT)
    ap.add_argument("--oro_file", default=ORO_FILE_DEFAULT)
    ap.add_argument("--pm10_file", default=PM10_S2_FILE_DEFAULT)
    ap.add_argument("--pm10_dir", default=PM10_DIR_DEFAULT)
    ap.add_argument("--pm25_file", default=PM25_S2_FILE_DEFAULT)
    ap.add_argument("--pm25_dir", default=PM25_DIR_DEFAULT)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--out_dir",
        default=os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"),
    )
    ap.add_argument("--window", type=int, default=WINDOW_SIZE_DEFAULT)
    ap.add_argument("--step", type=int, default=STEP_SIZE_DEFAULT)
    ap.add_argument("--val_last_days", type=int, default=VAL_LAST_DAYS_DEFAULT)
    ap.add_argument("--test_last_days", type=int, default=TEST_LAST_DAYS_DEFAULT)
    ap.add_argument("--gap_hours", type=int, default=GAP_HOURS_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds_tj = xr.open_dataset(args.tianji_file, engine="h5netcdf")
    print(
        "[Time Alignment] merged_final_all_vars.nc time alignment: "
        f"{TIANJI_TIME_ALIGNMENT} (shift={TIANJI_INPUT_TIME_SHIFT_HOURS:+g} h before split).",
        flush=True,
    )

    if "lat" in ds_tj:
        lats, lons = ds_tj["lat"].values, ds_tj["lon"].values
    else:
        lats, lons = ds_tj["latitude"].values, ds_tj["longitude"].values
    times = normalize_tianji_times(ds_tj.time.values)
    stations = ds_tj.station_id.values

    data_veg = xr.open_dataset(args.veg_file, engine="h5netcdf")
    data_oro = xr.open_dataset(args.oro_file, engine="h5netcdf")
    X_stat = build_static_features(lats, lons, data_veg, data_oro, UNIQUE_VEG_IDS)

    interp_paths = _parse_ifs_interp_input_paths(args.ifs_interp_nc, args.ifs_interp_glob)
    if interp_paths:
        for p in interp_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
        X_met = _fill_from_interp_nc_multi(interp_paths, times, stations)
        source_tag = "ifs_interp_nc"
    else:
        X_met = _fill_from_gridded(args.ifs_root, args.year, times, lats, lons)
        source_tag = "ifs_gridded"

    zenith = calculate_zenith_angle(lats, lons, times)
    X_dyn_25 = np.concatenate([X_met, zenith], axis=-1).astype(np.float32)
    del X_met, zenith
    gc.collect()

    pm10_da = load_pm10_dataarray(args.pm10_file, args.pm10_dir)
    X_dyn_26 = append_pm10_channel(X_dyn_25, pm10_da, times, stations)
    del X_dyn_25
    gc.collect()
    pm25_da = load_pm25_dataarray(args.pm25_file, args.pm25_dir)
    X_dyn = append_pm25_channel(X_dyn_26, pm25_da, times, stations)
    del X_dyn_26
    gc.collect()

    vis_key = "vis" if "vis" in ds_tj.data_vars else "visibility"
    y = ds_tj[vis_key].values.astype(np.float32)
    y = np.where(y <= MAX_VIS_THRESHOLD, y, np.nan)
    y = y[args.window - 1 :: args.step].reshape(-1)

    nt = len(times)
    ns = len(stations)
    n_wins = (nt - args.window) // args.step + 1
    m_t = np.repeat(times[args.window - 1 :: args.step].values, ns)
    m_s = np.tile(stations, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)

    X_wins = sliding_window_view(X_dyn, args.window, axis=0)[:: args.step]
    X_wins = X_wins.transpose(0, 1, 3, 2)
    X_samples = X_wins.reshape(-1, args.window, TOTAL_DYN).astype(np.float32)
    del X_wins, X_dyn
    gc.collect()

    fe_flat = compute_fog_features_pmst(X_samples, args.window, TOTAL_DYN)
    fe_flat = np.concatenate([fe_flat, cyclical_time_features(pd.DatetimeIndex(m_t))], axis=1).astype(
        np.float32
    )

    X_dyn_flat = X_samples.reshape(X_samples.shape[0], -1).astype(np.float32)
    del X_samples
    gc.collect()
    X_stat_flat = np.tile(X_stat, (n_wins, 1)).astype(np.float32)

    mask = ~np.isnan(y) & (y >= 0) & (y <= MAX_VIS_THRESHOLD)
    save_chunked_monthtail(
        X_dyn_flat,
        X_stat_flat,
        fe_flat,
        y,
        mask,
        (m_t, m_s, m_la, m_lo),
        args.out_dir,
        args.gap_hours,
        args.val_last_days,
        args.test_last_days,
    )

    cfg = {
        "dataset": "ifs_overlap_pmst27_monthtail",
        "ifs_source": source_tag,
        "ifs_interp_nc": args.ifs_interp_nc or None,
        "ifs_interp_glob": args.ifs_interp_glob or None,
        "ifs_interp_nc_list": interp_paths if interp_paths else None,
        "ifs_root": args.ifs_root,
        "year": args.year,
        "overlap_vars": OVERLAP_CANONICAL,
        "ifs_map_gridded": {
            k: {"folder": v.folder, "var_name": v.var_name, "level_hpa": v.level_hpa}
            for k, v in IFS_MAP.items()
        },
        "dyn_layout": "24_pmst_met + zenith + pm10 + pm2p5",
        "derived_overlap_vars": {
            "RH2M": "computed from IFS T2M and D2M when no direct RH2M is present",
            "DP_1000": "computed from IFS Q_1000 and 1000 hPa pressure",
            "DP_925": "computed from IFS Q_925 and 925 hPa pressure",
            "WSPD10/WDIR10/WSPD925": "computed from U/V wind components",
            "DPD": "computed from T2M and D2M",
        },
        "precipitation_transform": "IFS PRECIP is treated as hourly amount/rate and is not differenced.",
        "fe_dim": int(fe_flat.shape[1]),
        "window": args.window,
        "step": args.step,
        "split": "month_tail",
        "tianji_raw_time_alignment": TIANJI_TIME_ALIGNMENT,
        "tianji_input_time_shift_hours": TIANJI_INPUT_TIME_SHIFT_HOURS,
        "time_coordinate": "UTC",
        "ifs_time_match": "nearest_90min_utc",
        "pm_time_match": "nearest_90min_utc",
        "val_last_days": args.val_last_days,
        "test_last_days": args.test_last_days,
        "gap_hours": args.gap_hours,
        "max_vis_threshold": MAX_VIS_THRESHOLD,
        "time_align": "nearest_90min",
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote dataset to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
