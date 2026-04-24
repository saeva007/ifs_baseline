#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFS overlap baseline aligned to Tianji: PMST 27-dyn + 36-FE, month-tail split.

Fast/robust version:
  - Skips broken IFS NetCDF files before xarray opens the multi-file dataset.
  - Opens gridded IFS variables sequentially to avoid h5netcdf/HDF5 parallel-open failures.
  - Keeps the canonical 27-channel PMST dynamic layout.
  - Streams sliding-window samples directly into split .npy memmaps instead of materializing
    the full flattened dataset in RAM.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ================== 架构底线：必须扼杀底层 C 库的多线程暴增 ==================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

from pmst_overlap_common import (
    OVERLAP_CANONICAL,
    OVERLAP_PMST_INDICES,
    PM10_IDX,
    PM25_IDX,
    PMST_MET_DIM,
    TOTAL_DYN,
    U10_IDX,
    V10_IDX,
    WSPD10_IDX,
    build_static_features,
    build_station_reindex_map,
    compute_fog_features_pmst,
    cyclical_time_features,
    get_monthly_split_mask_last_days,
    load_pm10_dataarray,
    load_pm25_dataarray,
    normalize_var_coord,
    scatter_overlap_fields,
    calculate_zenith_angle,
)

IFS_ROOT_DEFAULT = "/public/home/sd3team/sd3_database/src_data/IFS/nc_0p1"
VIS_MLP_ROOT = "/public/home/putianshu/vis_mlp"
IFS_BASELINE_ROOT = os.path.join(VIS_MLP_ROOT, "ifs_baseline")
IFS_INTERP_DIR_DEFAULT = os.path.join(IFS_BASELINE_ROOT, "ifs_interp_out")
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
WINDOW_CHUNK_DEFAULT = 64


class IfsVarSpec:
    __slots__ = ("folder", "var_name")
    def __init__(self, folder: str, var_name: str):
        self.folder = folder
        self.var_name = var_name

IFS_MAP: Dict[str, IfsVarSpec] = {
    "SW_RAD": IfsVarSpec("DSWRF", "ssrd"),
    "MSLP": IfsVarSpec("SLP", "msl"),
    "T2M": IfsVarSpec("T2M", "t2m"),
    "U10": IfsVarSpec("U10M", "u10"),
    "V10": IfsVarSpec("V10M", "v10"),
    "PRECIP": IfsVarSpec("TP", "tp"),
}

def _timer(func):
    """极客级性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"[Timing] {func.__name__} took {time.perf_counter() - start:.2f}s", flush=True)
        return res
    return wrapper

# ================== 基础功能保留区 ==================
def _real_exists(path: str) -> bool:
    try: return os.path.exists(os.path.realpath(path))
    except OSError: return False

def _list_real_files(folder: str, ifs_root: str, year: int) -> List[str]:
    d = os.path.join(ifs_root, folder)
    return[fp for fp in sorted(glob.glob(os.path.join(d, f"*{year}*.nc"))) if _real_exists(fp)]

def _netcdf_kind(path: str) -> str:
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
    except OSError:
        return "missing"
    if sig == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if sig[:3] == b"CDF":
        return "cdf"
    return "unknown"

def _filter_ifs_files_by_signature(files: Sequence[str], folder: str) -> Tuple[List[str], str]:
    good_by_kind: Dict[str, List[str]] = {"hdf5": [], "cdf": []}
    bad: List[Tuple[str, str]] = []
    for fp in files:
        kind = _netcdf_kind(fp)
        if kind not in good_by_kind:
            bad.append((fp, f"not a NetCDF file (signature={kind})"))
            continue
        good_by_kind[kind].append(fp)

    kind = "hdf5" if len(good_by_kind["hdf5"]) >= len(good_by_kind["cdf"]) else "cdf"
    good = good_by_kind[kind]
    skipped_other = good_by_kind["cdf" if kind == "hdf5" else "hdf5"]
    if skipped_other:
        print(
            f"[WARN] Skipping {len(skipped_other)} {folder} files with mixed NetCDF storage; "
            f"using {kind} files only.",
            flush=True,
        )
    for fp, reason in bad[:20]:
        print(f"[WARN] Skipping bad IFS file for {folder}: {fp} ({reason})", flush=True)
    if len(bad) > 20:
        print(f"[WARN] ... skipped {len(bad) - 20} more bad files for {folder}", flush=True)

    engine = "h5netcdf" if kind == "hdf5" else "scipy"
    print(
        f"[INFO] {folder}: {len(good)} candidate {kind} NetCDF files after fast signature filter.",
        flush=True,
    )
    return good, engine

def _filter_openable_ifs_files(files: Sequence[str], var_name: str, folder: str, engine: str) -> List[str]:
    good: List[str] = []
    bad: List[Tuple[str, str]] = []
    total = len(files)
    print(f"[INFO] {folder}: deep-checking {total} files after open failure...", flush=True)
    for i, fp in enumerate(files, 1):
        try:
            with xr.open_dataset(fp, engine=engine) as ds:
                if var_name not in ds.data_vars:
                    bad.append((fp, f"missing var {var_name!r}; vars={list(ds.data_vars)}"))
                    continue
                if "time" not in ds.coords and "time" not in ds.dims:
                    bad.append((fp, "missing time coordinate"))
                    continue
            good.append(fp)
        except Exception as e:
            bad.append((fp, f"{type(e).__name__}: {e}"))
        if i == 1 or i % 50 == 0 or i == total:
            print(f"[INFO] {folder}: checked {i}/{total}, good={len(good)}, bad={len(bad)}", flush=True)
    for fp, reason in bad[:20]:
        print(f"[WARN] Skipping bad IFS file for {folder}: {fp} ({reason})", flush=True)
    if len(bad) > 20:
        print(f"[WARN] ... skipped {len(bad) - 20} more bad files for {folder}", flush=True)
    return good

def _zero_memmap_in_chunks(arr: np.ndarray, station_chunk: int = 256) -> None:
    ns = arr.shape[0]
    print(f"[INFO] Initializing dynamic memmap to zeros by station chunks (ns={ns}).", flush=True)
    for st0 in range(0, ns, station_chunk):
        st1 = min(st0 + station_chunk, ns)
        arr[st0:st1, :, :] = 0.0
        if st0 == 0 or st1 == ns or (st1 // station_chunk) % 10 == 0:
            print(f"[INFO] Initialized stations {st0}:{st1} / {ns}", flush=True)

def _open_mfdataset_checked(files: Sequence[str], engine: str) -> xr.Dataset:
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        engine=engine,
        chunks={"time": 24},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

def _open_ifs_concat(folder: str, var_name: str, ifs_root: str, year: int) -> xr.Dataset:
    files = _list_real_files(folder, ifs_root, year)
    if not files: raise FileNotFoundError(f"No files for {folder} {year}")
    files, engine = _filter_ifs_files_by_signature(files, folder)
    if not files:
        raise FileNotFoundError(f"No openable NetCDF files for {folder} {year}")
    try:
        ds = _open_mfdataset_checked(files, engine)
    except ValueError as e:
        if "monotonic" in str(e) or "time" in str(e):
            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="time",
                coords="minimal",
                compat="override",
                parallel=False,
                engine=engine,
                chunks={"time": 24},
            )
            _, unique_indices = np.unique(ds["time"].values, return_index=True)
            ds = ds.isel(time=np.sort(unique_indices)).sortby("time")
        else:
            raise e
    except OSError as e:
        print(f"[WARN] {folder}: open_mfdataset failed ({e}); isolating bad files.", flush=True)
        files = _filter_openable_ifs_files(files, var_name, folder, engine)
        if not files:
            raise FileNotFoundError(f"No openable NetCDF files for {folder} {year}") from e
        ds = _open_mfdataset_checked(files, engine)
    if var_name not in ds.data_vars:
        ds.close()
        raise KeyError(f"IFS dataset {folder} missing var {var_name}. got={list(ds.data_vars)}")
    return ds

def _nearest_index(grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    grid, pts = np.asarray(grid).astype(np.float64), np.asarray(pts).astype(np.float64)
    tree = cKDTree(grid.reshape(-1, 1))
    _, idx = tree.query(pts.reshape(-1, 1), workers=-1)
    return idx.astype(np.int64)

def _parse_ifs_interp_input_paths(ifs_interp_nc: str, ifs_interp_glob: str) -> List[str]:
    raw =[os.path.realpath(p.strip()) for p in (ifs_interp_nc or "").split(",") if p.strip()]
    if (ifs_interp_glob or "").strip():
        raw.extend(sorted(glob.glob(os.path.realpath(ifs_interp_glob.strip()))))
    return [p for p in dict.fromkeys(raw) if os.path.isfile(p)]

def _discover_default_ifs_interp_paths(year: int, interp_dir: str) -> List[str]:
    """Auto-discover station-interpolated IFS files, preferring them over raw grids."""
    if not interp_dir or not os.path.isdir(interp_dir):
        return []
    patterns = [
        os.path.join(interp_dir, f"ifs_interp_*_{year}.nc"),
        os.path.join(interp_dir, "**", f"ifs_interp_*_{year}.nc"),
        os.path.join(interp_dir, f"interpolated_ifs_{year}.nc"),
        os.path.join(interp_dir, "**", f"interpolated_ifs_{year}.nc"),
    ]
    raw: List[str] = []
    for pat in patterns:
        raw.extend(sorted(glob.glob(pat, recursive=True)))
    return [os.path.realpath(p) for p in dict.fromkeys(raw) if os.path.isfile(p)]

def _load_single_interp(path_nc: str):
    with xr.open_dataset(path_nc) as ds:
        if "ifs_interp" not in ds.data_vars:
            raise KeyError(f"{path_nc}: expected data var 'ifs_interp', got {list(ds.data_vars)}")
        return ds["ifs_interp"].values.astype(np.float32), pd.DatetimeIndex(ds["time"].values), ds["station_id"].values,[normalize_var_coord(v) for v in ds["variable"].values], path_nc

def _fill_from_interp_nc_multi(paths_nc: Sequence[str], tj_times: pd.DatetimeIndex, station_ids_tj: np.ndarray) -> np.ndarray:
    print(f"[INFO] Merging {len(paths_nc)} interp files...", flush=True)
    merged_var_names, chunks = [],[]
    results = [_load_single_interp(p) for p in paths_nc]

    ref_shape_ts, ref_times, ref_stations = None, None, None
    for arr, ifs_times, ifs_stations, var_coord, path_nc in results:
        if ref_shape_ts is None: ref_shape_ts, ref_times, ref_stations = (arr.shape[0], arr.shape[1]), ifs_times, ifs_stations
        elif (arr.shape[0], arr.shape[1]) != ref_shape_ts:
            raise ValueError(f"{path_nc}: shape {(arr.shape[0], arr.shape[1])} != first interp file {ref_shape_ts}")
        for vn in var_coord:
            if vn in merged_var_names:
                raise ValueError(f"Duplicate interpolated IFS variable {vn!r} in {path_nc}")
        merged_var_names.extend(var_coord)
        chunks.append(arr)

    total_vars = sum(c.shape[2] for c in chunks)
    big = np.empty((ref_shape_ts[0], ref_shape_ts[1], total_vars), dtype=np.float32)
    offset = 0
    for c in chunks:
        v_size = c.shape[2]
        big[:, :, offset:offset+v_size] = c
        offset += v_size
    del chunks, results; gc.collect()

    st_cols = build_station_reindex_map(station_ids_tj, ref_stations)
    time_pos = ref_times.get_indexer(tj_times, method="nearest", tolerance=pd.Timedelta(minutes=90))
    valid_t_mask = time_pos >= 0
    valid_time_idx = time_pos[valid_t_mask]

    nt, ns = len(tj_times), len(station_ids_tj)
    fields = {}
    for name in OVERLAP_CANONICAL:
        if name not in merged_var_names:
            raise KeyError(f"Interpolated IFS files missing {name!r}; have {merged_var_names}")
        vi = merged_var_names.index(name)
        slot = np.full((nt, ns), np.nan, dtype=np.float32)
        if np.any(valid_t_mask): slot[valid_t_mask, :] = big[valid_time_idx[:, None], st_cols[None, :], vi]
        fields[name] = slot
    del big; gc.collect()
    return scatter_overlap_fields(nt, ns, fields)

# ================== 核心优化组件 ==================
def _extract_pm_channel(pm_da: Optional[xr.DataArray], times: pd.DatetimeIndex, station_ids: np.ndarray) -> np.ndarray:
    """Return PM channel as (nt, ns) ug/m^3 without concatenating dynamic arrays."""
    nt, ns = len(times), len(station_ids)
    if pm_da is None:
        return np.zeros((nt, ns), dtype=np.float32)

    pm_da = pm_da.load()
    time_vals = pm_da["time"].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        time_index = pd.DatetimeIndex(time_vals)
    else:
        time_index = pd.to_datetime(time_vals, unit="s", origin="unix")
    sid_index = pd.Index(pm_da["station_id"].values)
    sids = station_ids.astype(pm_da["station_id"].dtype)
    time_pos = time_index.get_indexer(times, method="nearest", tolerance=pd.Timedelta(minutes=90))
    sid_pos = sid_index.get_indexer(sids)

    nt_pm, ns_pm = pm_da.shape
    pm_grid = np.full((nt, ns), np.nan, dtype=np.float32)
    ok_mask = (time_pos[:, None] >= 0) & (sid_pos[None, :] >= 0)
    if ok_mask.any():
        base = np.asarray(pm_da.values).reshape(-1)
        linear_idx_grid = time_pos[:, None] * ns_pm + sid_pos[None, :]
        pm_grid[ok_mask] = base[linear_idx_grid[ok_mask]].astype(np.float32)
    pm_grid = np.maximum(pm_grid, 0.0)
    pm_ug = pm_grid * 1e12
    med = np.nanmedian(pm_ug)
    if not np.isfinite(med):
        med = 0.0
    return np.where(np.isfinite(pm_ug), pm_ug, med).astype(np.float32)

def _fill_wspd10_in_place(buffer_array: np.ndarray, station_chunk: int = 512) -> None:
    for st0 in range(0, buffer_array.shape[0], station_chunk):
        st1 = min(st0 + station_chunk, buffer_array.shape[0])
        u = buffer_array[st0:st1, :, U10_IDX]
        v = buffer_array[st0:st1, :, V10_IDX]
        buffer_array[st0:st1, :, WSPD10_IDX] = np.sqrt(u * u + v * v).astype(np.float32)

@_timer
def fetch_ifs_data_in_place(ifs_root: str, year: int, tj_times: pd.DatetimeIndex, 
                            lats: np.ndarray, lons: np.ndarray, 
                            buffer_array: np.ndarray, start_idx: int):
    """Xarray -> station-first memmap injection in PMST 27-channel layout."""
    _ = start_idx
    lat_idx = lon_idx = None

    for canon in OVERLAP_CANONICAL:
        spec = IFS_MAP[canon]
        print(f"[INFO] Loading IFS variable {canon} from {spec.folder}/{spec.var_name}...", flush=True)
        ds = _open_ifs_concat(spec.folder, spec.var_name, ifs_root, year)
        try:
            lat_name = "latitude" if "latitude" in ds.coords or "latitude" in ds.dims else "lat"
            lon_name = "longitude" if "longitude" in ds.coords or "longitude" in ds.dims else "lon"
            if lat_idx is None or lon_idx is None:
                lat_grid = ds[lat_name].values
                lon_grid = ds[lon_name].values
                lat_idx = _nearest_index(lat_grid, lats)
                lon_idx = _nearest_index(lon_grid, lons % 360)

            ifs_time = pd.DatetimeIndex(ds["time"].values)
            time_pos = ifs_time.get_indexer(tj_times, method="nearest", tolerance=pd.Timedelta(minutes=90))
            valid_t_mask = time_pos >= 0
            if not valid_t_mask.any():
                raise RuntimeError(f"No overlapping times between Tianji and IFS for {canon}")

            da = ds[spec.var_name]
            lat_dim = "latitude" if "latitude" in da.dims else "lat"
            lon_dim = "longitude" if "longitude" in da.dims else "lon"
            isel_dict = {
                "time": xr.DataArray(time_pos[valid_t_mask], dims=["time"]),
                lat_dim: xr.DataArray(lat_idx, dims=["station"]),
                lon_dim: xr.DataArray(lon_idx, dims=["station"]),
            }
            for dim in da.dims:
                if dim not in isel_dict:
                    isel_dict[dim] = 0

            print(f"[INFO] {canon}: sampling {valid_t_mask.sum()} times x {len(lats)} stations.", flush=True)
            extracted = da.isel(**isel_dict).load().values.astype(np.float32, copy=False)
            pmst_idx = OVERLAP_PMST_INDICES[canon]
            buffer_array[:, :, pmst_idx] = np.nan
            buffer_array[:, valid_t_mask, pmst_idx] = extracted.T
            print(f"  -> {canon} mapped into PMST channel {pmst_idx}.", flush=True)
        finally:
            ds.close()

    _fill_wspd10_in_place(buffer_array)
    return PMST_MET_DIM

def _write_split_metadata(
    out_dir: str,
    splits: Dict[str, np.ndarray],
    y_flat: np.ndarray,
    times_all: np.ndarray,
    stations_all: np.ndarray,
    lats_all: np.ndarray,
    lons_all: np.ndarray,
) -> None:
    for tag, ix in splits.items():
        if len(ix) == 0:
            continue
        np.save(os.path.join(out_dir, f"y_{tag}.npy"), y_flat[ix])
        pd.DataFrame(
            {"time": times_all[ix], "station_id": stations_all[ix], "lat": lats_all[ix], "lon": lons_all[ix]}
        ).to_csv(os.path.join(out_dir, f"meta_{tag}.csv"), index=False)

def _prepare_split_memmaps(
    out_dir: str,
    splits: Dict[str, np.ndarray],
    total_dim: int,
) -> Dict[str, np.memmap]:
    out: Dict[str, np.memmap] = {}
    for tag, ix in splits.items():
        if len(ix) == 0:
            continue
        print(f"    Creating {tag} memmap (N={len(ix)})", flush=True)
        out[tag] = np.lib.format.open_memmap(
            os.path.join(out_dir, f"X_{tag}.npy"),
            mode="w+",
            dtype="float32",
            shape=(len(ix), total_dim),
        )
    return out

def save_streamed_monthtail(
    X_dyn: np.ndarray,
    X_stat: np.ndarray,
    y_full: np.ndarray,
    times: pd.DatetimeIndex,
    stations: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    out_dir: str,
    window: int,
    step: int,
    gap_hours: int,
    val_last_days: int,
    test_last_days: int,
    window_chunk: int,
) -> Tuple[int, int, int]:
    """Write train/val/test arrays in time-major order without materializing all windows."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if window < 7:
        raise ValueError("window must be >= 7 because fog features use -4 and -7 lags")
    nt, ns = len(times), len(stations)
    if nt < window:
        raise ValueError(f"Not enough timesteps ({nt}) for window={window}")

    n_wins = (nt - window) // step + 1
    sample_times = times[window - 1 :: step][:n_wins]
    total_rows = n_wins * ns

    y_flat = np.where(y_full <= MAX_VIS_THRESHOLD, y_full, np.nan)[window - 1 :: step].reshape(-1)
    mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)
    valid_idxs = np.where(mask)[0]
    print(f"  Valid Samples: {len(valid_idxs)} ({len(valid_idxs) / max(total_rows, 1):.1%})", flush=True)

    times_all = np.repeat(sample_times.values, ns)
    stations_all = np.tile(stations, n_wins)
    lats_all = np.tile(lats, n_wins)
    lons_all = np.tile(lons, n_wins)
    valid_times = pd.DatetimeIndex(times_all[valid_idxs])
    tr_m, val_m, test_m = get_monthly_split_mask_last_days(valid_times, gap_hours, val_last_days, test_last_days)
    splits = {"train": valid_idxs[tr_m], "val": valid_idxs[val_m], "test": valid_idxs[test_m]}

    dummy_fe_dim = compute_fog_features_pmst(np.zeros((1, window, TOTAL_DYN), dtype=np.float32), window, TOTAL_DYN).shape[1]
    dyn_dim = window * TOTAL_DYN
    stat_dim = X_stat.shape[1]
    fe_dim = dummy_fe_dim + 4
    total_dim = dyn_dim + stat_dim + fe_dim

    _write_split_metadata(out_dir, splits, y_flat, times_all, stations_all, lats_all, lons_all)
    split_fps = _prepare_split_memmaps(out_dir, splits, total_dim)

    try:
        for w0 in range(0, n_wins, window_chunk):
            w1 = min(w0 + window_chunk, n_wins)
            t0 = w0 * step
            t1 = (w1 - 1) * step + window
            block = X_dyn[:, t0:t1, :]
            win_view = sliding_window_view(block, window_shape=window, axis=1)[:, ::step, :, :]
            win_view = win_view[:, : (w1 - w0), :, :]
            samples = np.ascontiguousarray(win_view.transpose(1, 0, 3, 2).reshape(-1, window, TOTAL_DYN))
            fe_base = compute_fog_features_pmst(samples, window, TOTAL_DYN)
            chunk_times = np.repeat(sample_times[w0:w1].values, ns)
            time_fe = cyclical_time_features(pd.DatetimeIndex(chunk_times))
            global_start = w0 * ns
            global_end = w1 * ns

            for tag, ix in splits.items():
                if len(ix) == 0:
                    continue
                lo = int(np.searchsorted(ix, global_start, side="left"))
                hi = int(np.searchsorted(ix, global_end, side="left"))
                if hi <= lo:
                    continue
                g_ix = ix[lo:hi]
                local_ix = g_ix - global_start
                fp = split_fps[tag]
                fp[lo:hi, :dyn_dim] = samples[local_ix].reshape(len(local_ix), dyn_dim)
                fp[lo:hi, dyn_dim : dyn_dim + stat_dim] = X_stat[g_ix % ns]
                fp[lo:hi, dyn_dim + stat_dim : dyn_dim + stat_dim + dummy_fe_dim] = fe_base[local_ix]
                fp[lo:hi, dyn_dim + stat_dim + dummy_fe_dim :] = time_fe[local_ix]

            del samples, fe_base, time_fe, win_view
            gc.collect()
            print(f"    Streamed windows {w0}:{w1} / {n_wins}", flush=True)
    finally:
        for fp in split_fps.values():
            fp.flush()
            del fp
        gc.collect()

    return n_wins, dummy_fe_dim, fe_dim


@_timer
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifs_interp_nc", default="")
    ap.add_argument("--ifs_interp_glob", default="")
    ap.add_argument("--ifs_interp_dir", default=IFS_INTERP_DIR_DEFAULT)
    ap.add_argument(
        "--no_auto_ifs_interp",
        action="store_true",
        help="Disable auto-discovery of station-interpolated IFS files and use gridded --ifs_root fallback.",
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
    ap.add_argument("--out_dir", default=os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"))
    ap.add_argument("--window", type=int, default=WINDOW_SIZE_DEFAULT)
    ap.add_argument("--step", type=int, default=STEP_SIZE_DEFAULT)
    ap.add_argument("--val_last_days", type=int, default=VAL_LAST_DAYS_DEFAULT)
    ap.add_argument("--test_last_days", type=int, default=TEST_LAST_DAYS_DEFAULT)
    ap.add_argument("--gap_hours", type=int, default=GAP_HOURS_DEFAULT)
    ap.add_argument("--window_chunk", type=int, default=WINDOW_CHUNK_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Booting streaming builder with window_chunk={args.window_chunk}.", flush=True)

    with xr.open_dataset(args.tianji_file, engine="h5netcdf") as ds_tj:
        ds_tj = ds_tj.assign_coords(time=ds_tj.time - pd.Timedelta(hours=8))
        lats = ds_tj["lat"].values if "lat" in ds_tj else ds_tj["latitude"].values
        lons = ds_tj["lon"].values if "lon" in ds_tj else ds_tj["longitude"].values
        times = pd.to_datetime(ds_tj.time.values)
        stations = ds_tj.station_id.values
        vis_key = "vis" if "vis" in ds_tj.data_vars else "visibility"
        y = ds_tj[vis_key].values.astype(np.float32)

    data_veg = xr.open_dataset(args.veg_file, engine="h5netcdf")
    data_oro = xr.open_dataset(args.oro_file, engine="h5netcdf")
    X_stat = build_static_features(lats, lons, data_veg, data_oro, UNIQUE_VEG_IDS)
    data_veg.close()
    data_oro.close()

    # 确定特征数量与内存布局: (NS, NT, 27)
    nt, ns = len(times), len(stations)
    total_features = TOTAL_DYN
    scratch_path = os.path.join(args.out_dir, "_tmp_X_dyn_pmst27.npy")
    bytes_size = ns * nt * total_features * 4
    win, step = args.window, args.step
    n_wins_preview = (nt - win) // step + 1 if nt >= win and step >= 1 else 0
    print(
        "[mem] nt={} ns={} n_wins={} window_chunk={} dynamic_staging ~{:.2f} GiB".format(
            nt,
            ns,
            n_wins_preview,
            args.window_chunk,
            bytes_size / (1024 ** 3),
        ),
        flush=True,
    )
    print(f"[INFO] Creating dynamic memmap {scratch_path} ({bytes_size / 1e9:.2f} GB logical)...", flush=True)
    
    try:
        X_dyn = np.lib.format.open_memmap(
            scratch_path,
            mode="w+",
            dtype="float32",
            shape=(ns, nt, total_features),
        )
        _zero_memmap_in_chunks(X_dyn)
        
        # 1. 填充气象场
        interp_paths = _parse_ifs_interp_input_paths(args.ifs_interp_nc, args.ifs_interp_glob)
        if not interp_paths and not args.no_auto_ifs_interp:
            interp_paths = _discover_default_ifs_interp_paths(args.year, args.ifs_interp_dir)
            if interp_paths:
                print(
                    f"[INFO] Auto-discovered {len(interp_paths)} interpolated IFS files under {args.ifs_interp_dir}.",
                    flush=True,
                )
        if interp_paths:
            # 兼容旧的高维读取逻辑，读取后转置进共享内存
            print(f"[INFO] Using interpolated IFS files: {len(interp_paths)}", flush=True)
            for p in interp_paths:
                print(f"  -> {p}", flush=True)
            X_met = _fill_from_interp_nc_multi(interp_paths, times, stations)
            X_dyn[:, :, :PMST_MET_DIM] = X_met.transpose(1, 0, 2)
            del X_met; gc.collect()
            source_tag = "ifs_interp_nc"
        else:
            print(f"[INFO] Using gridded IFS root: {args.ifs_root}", flush=True)
            fetch_ifs_data_in_place(args.ifs_root, args.year, times, lats, lons, X_dyn, 0)
            source_tag = "ifs_gridded"

        # 2. 填充 Zenith 与 PM
        print("[INFO] Computing and writing Zenith & PM to dynamic memmap...", flush=True)
        zenith = calculate_zenith_angle(lats, lons, times)  # (nt, ns, 1)
        X_dyn[:, :, PMST_MET_DIM] = zenith.squeeze(-1).T     # 转置写入
        del zenith; gc.collect()

        pm10_da = load_pm10_dataarray(args.pm10_file, args.pm10_dir)
        pm10_arr = _extract_pm_channel(pm10_da, times, stations) # (nt, ns)
        X_dyn[:, :, PM10_IDX] = pm10_arr.T
        del pm10_arr, pm10_da; gc.collect()
        
        pm25_da = load_pm25_dataarray(args.pm25_file, args.pm25_dir)
        pm25_arr = _extract_pm_channel(pm25_da, times, stations) # (nt, ns)
        X_dyn[:, :, PM25_IDX] = pm25_arr.T
        del pm25_arr, pm25_da; gc.collect()

        X_dyn.flush()

        print("[INFO] Streaming windows directly into split dataset files...", flush=True)
        n_wins, fog_fe_dim, fe_dim = save_streamed_monthtail(
            X_dyn,
            X_stat,
            y,
            times,
            stations,
            lats,
            lons,
            args.out_dir,
            args.window,
            args.step,
            args.gap_hours,
            args.val_last_days,
            args.test_last_days,
            max(1, args.window_chunk),
        )
    finally:
        try:
            del X_dyn
        except UnboundLocalError:
            pass
        gc.collect()
        if os.path.exists(scratch_path):
            os.remove(scratch_path)
        print("[INFO] Temporary dynamic memmap cleaned.", flush=True)

    cfg = {
        "dataset": "ifs_overlap_pmst27_monthtail",
        "ifs_source": source_tag,
        "ifs_interp_nc": args.ifs_interp_nc or None,
        "ifs_interp_glob": args.ifs_interp_glob or None,
        "ifs_interp_dir": args.ifs_interp_dir,
        "no_auto_ifs_interp": bool(args.no_auto_ifs_interp),
        "ifs_interp_nc_list": interp_paths if interp_paths else None,
        "ifs_root": args.ifs_root,
        "year": args.year,
        "overlap_vars": OVERLAP_CANONICAL,
        "ifs_map_gridded": {k: {"folder": v.folder, "var_name": v.var_name} for k, v in IFS_MAP.items()},
        "dyn_layout": "24_pmst_met + zenith + pm10 + pm2p5",
        "fe_dim": int(fe_dim),
        "fog_fe_dim": int(fog_fe_dim),
        "window": args.window,
        "step": args.step,
        "n_windows": int(n_wins),
        "window_chunk": args.window_chunk,
        "val_last_days": args.val_last_days,
        "test_last_days": args.test_last_days,
        "architect_note": "Time-major aligned, PMST-27 layout, streamed window writer"
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Extreme Optimized Baseline strictly written to {args.out_dir}", flush=True)

if __name__ == "__main__":
    main()
