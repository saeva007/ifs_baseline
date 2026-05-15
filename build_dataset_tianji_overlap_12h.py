#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tianji overlap-only baseline: PMST 27-dyn + 36-FE (24 met + zenith + pm10 + pm2p5), month-tail split,
only overlap NWP channels filled (IFS-reproducible subset); station PM10 + PM2.5 appended.

Memory: avoids xr.Dataset.copy() and never materializes full (n_wins*ns, window, 27) tensor.
Use --chunk_wins to tune peak RAM (smaller => less RAM, more passes).
"""

import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from pmst_overlap_common import (
    OVERLAP_CANONICAL,
    TOTAL_DYN,
    append_pm10_channel,
    append_pm25_channel,
    build_static_features,
    compute_fog_features_pmst,
    cyclical_time_features,
    load_pm10_dataarray,
    load_pm25_dataarray,
    scatter_overlap_fields,
    save_chunked_monthtail,
    calculate_zenith_angle,
)

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

VAR_MAPPING = {
    "rh2m": "RH2M",
    "TMP2m": "T2M",
    "PRATEsfc": "PRECIP",
    "slp": "MSLP",
    "DSWRFsfc": "SW_RAD",
    "UGRD10m": "U10",
    "VGRD10m": "V10",
    "wind_speed": "WSPD10",
    "wd10m": "WDIR10",
    "cape": "CAPE",
    "cldl": "LCC",
    "t925": "T_925",
    "rh925": "RH_925",
    "u925": "U_925",
    "v925": "V_925",
    "wind_speed_925": "WSPD925",
    "dp1000": "DP_1000",
    "dp925": "DP_925",
    "q1000": "Q_1000",
    "q925": "Q_925",
    "omg925": "W_925",
    "omg1000": "W_1000",
    "t1000": "T_1000",
}

WINDOW_SIZE_DEFAULT = 12
STEP_SIZE_DEFAULT = 1
MAX_VIS_THRESHOLD = 30000
UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])

VAL_LAST_DAYS_DEFAULT = 3
TEST_LAST_DAYS_DEFAULT = 3
GAP_HOURS_DEFAULT = 24
# Default window-starts per chunk; lower if the job is OOM-killed.
CHUNK_WINS_DEFAULT = int(os.environ.get("TIANJI_OVERLAP_CHUNK_WINS", "256"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", default=TIANJI_FILE_DEFAULT)
    ap.add_argument("--veg_file", default=VEG_FILE_DEFAULT)
    ap.add_argument("--oro_file", default=ORO_FILE_DEFAULT)
    ap.add_argument("--pm10_file", default=PM10_S2_FILE_DEFAULT)
    ap.add_argument("--pm10_dir", default=PM10_DIR_DEFAULT)
    ap.add_argument("--pm25_file", default=PM25_S2_FILE_DEFAULT)
    ap.add_argument("--pm25_dir", default=PM25_DIR_DEFAULT)
    ap.add_argument(
        "--out_dir",
        default=os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_baseline"),
    )
    ap.add_argument("--window", type=int, default=WINDOW_SIZE_DEFAULT)
    ap.add_argument("--step", type=int, default=STEP_SIZE_DEFAULT)
    ap.add_argument("--val_last_days", type=int, default=VAL_LAST_DAYS_DEFAULT)
    ap.add_argument("--test_last_days", type=int, default=TEST_LAST_DAYS_DEFAULT)
    ap.add_argument("--gap_hours", type=int, default=GAP_HOURS_DEFAULT)
    ap.add_argument(
        "--chunk_wins",
        type=int,
        default=CHUNK_WINS_DEFAULT,
        help="Number of sliding-window starts per chunk (smaller uses less RAM).",
    )
    ap.add_argument(
        "--keep_staging",
        action="store_true",
        help="Keep _staging_*.npy under out_dir after build (for debug).",
    )
    args = ap.parse_args()

    if args.chunk_wins < 1:
        raise ValueError("chunk_wins must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)

    ds_in = None
    data_veg = xr.open_dataset(args.veg_file, engine="h5netcdf")
    data_oro = xr.open_dataset(args.oro_file, engine="h5netcdf")
    ds_in = xr.open_dataset(args.input_file, engine="h5netcdf")
    print("[Time Alignment] merged_final_all_vars.nc raw time is UTC; no shift applied.", flush=True)

    if "vis" in ds_in.data_vars:
        ds_in = ds_in.rename({"vis": "visibility"})
    rename_map = {k: v for k, v in VAR_MAPPING.items() if k in ds_in.data_vars and v not in ds_in.data_vars}
    if rename_map:
        ds_in = ds_in.rename(rename_map)

    missing = [v for v in OVERLAP_CANONICAL if v not in ds_in.data_vars]
    if missing:
        raise KeyError("Tianji file missing overlap variables: {!r}".format(missing))

    if "lat" in ds_in:
        lats, lons = ds_in["lat"].values, ds_in["lon"].values
    elif "latitude" in ds_in:
        lats, lons = ds_in["latitude"].values, ds_in["longitude"].values
    else:
        raise AttributeError("Latitude/Longitude coordinates not found.")

    times = pd.to_datetime(ds_in.time.values)
    stations = ds_in.station_id.values
    nt, ns = len(times), len(stations)

    win, step = args.window, args.step
    n_wins = (nt - win) // step + 1
    if n_wins <= 0:
        raise ValueError("Time series too short: nt={} window={}".format(nt, win))

    n_samples = n_wins * ns
    dyn_flat_dim = win * TOTAL_DYN

    X_stat = build_static_features(lats, lons, data_veg, data_oro, UNIQUE_VEG_IDS)
    stat_dim = int(X_stat.shape[1])
    data_veg.close()
    data_oro.close()

    vis_key = "visibility" if "visibility" in ds_in.data_vars else "vis"
    y_arr = ds_in[vis_key].values.astype(np.float32)
    y_arr = np.where(y_arr <= MAX_VIS_THRESHOLD, y_arr, np.nan)
    y_flat = y_arr[win - 1 :: step].reshape(-1).astype(np.float32)
    del y_arr
    gc.collect()

    m_t = np.repeat(times[win - 1 :: step].values, ns)
    m_s = np.tile(stations, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)

    mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)

    pm10_da = load_pm10_dataarray(args.pm10_file, args.pm10_dir)
    pm25_da = load_pm25_dataarray(args.pm25_file, args.pm25_dir)
    if pm10_da is not None:
        pm10_da.load()
    if pm25_da is not None:
        pm25_da.load()

    st_dyn = os.path.join(args.out_dir, "_staging_X_dyn_flat.npy")
    st_stat = os.path.join(args.out_dir, "_staging_X_stat_flat.npy")
    st_fe = os.path.join(args.out_dir, "_staging_fe_flat.npy")
    fe_dim = 32 + 4

    print(
        "[mem] nt={} ns={} n_wins={} n_samples={} chunk_wins={} staging ~{:.1f} GiB on disk".format(
            nt,
            ns,
            n_wins,
            n_samples,
            args.chunk_wins,
            n_samples * (dyn_flat_dim + stat_dim + fe_dim) * 4 / (1024 ** 3),
        ),
        flush=True,
    )

    mm_dyn = np.lib.format.open_memmap(
        st_dyn, mode="w+", dtype=np.float32, shape=(n_samples, dyn_flat_dim)
    )
    mm_stat = np.lib.format.open_memmap(
        st_stat, mode="w+", dtype=np.float32, shape=(n_samples, stat_dim)
    )
    mm_fe = np.lib.format.open_memmap(st_fe, mode="w+", dtype=np.float32, shape=(n_samples, fe_dim))

    try:
        for w0 in tqdm(
            range(0, n_wins, args.chunk_wins),
            desc="time_chunks",
            unit="chunk",
        ):
            w1 = min(w0 + args.chunk_wins, n_wins)
            t0 = w0 * step
            t1 = (w1 - 1) * step + win
            tlen = t1 - t0

            fields = {
                name: ds_in[name].isel(time=slice(t0, t1)).values.astype(np.float32)
                for name in OVERLAP_CANONICAL
            }
            X_met = scatter_overlap_fields(tlen, ns, fields)
            del fields
            gc.collect()

            times_chunk = pd.DatetimeIndex(times[t0:t1])
            zenith = calculate_zenith_angle(lats, lons, times_chunk.values)
            X_dyn_25 = np.concatenate([X_met, zenith], axis=-1).astype(np.float32)
            del X_met, zenith

            X_dyn_26 = append_pm10_channel(X_dyn_25, pm10_da, times_chunk, stations)
            del X_dyn_25
            X_chunk = append_pm25_channel(X_dyn_26, pm25_da, times_chunk, stations)
            del X_dyn_26
            gc.collect()

            raw = sliding_window_view(X_chunk, win, axis=0)[::step]
            raw = raw.transpose(0, 1, 3, 2)
            X_samples = raw.reshape(-1, win, TOTAL_DYN).astype(np.float32)
            del raw, X_chunk
            gc.collect()

            n_loc = (w1 - w0) * ns
            if X_samples.shape[0] != n_loc:
                raise RuntimeError(
                    "Chunk sample count mismatch: got {} expected {}".format(X_samples.shape[0], n_loc)
                )

            row_lo = w0 * ns
            row_hi = w1 * ns
            mm_dyn[row_lo:row_hi] = X_samples.reshape(n_loc, dyn_flat_dim)

            fe_part = compute_fog_features_pmst(X_samples, win, TOTAL_DYN)
            cyc = cyclical_time_features(pd.DatetimeIndex(m_t[row_lo:row_hi]))
            mm_fe[row_lo:row_hi] = np.concatenate([fe_part, cyc], axis=1).astype(np.float32)
            del fe_part, cyc

            mm_stat[row_lo:row_hi] = np.tile(X_stat, (w1 - w0, 1)).astype(np.float32)
            del X_samples
            gc.collect()

        for _mm in (mm_dyn, mm_stat, mm_fe):
            if hasattr(_mm, "flush"):
                _mm.flush()
        del mm_dyn, mm_stat, mm_fe
        gc.collect()

        mm_dyn = np.load(st_dyn, mmap_mode="r")
        mm_stat = np.load(st_stat, mmap_mode="r")
        mm_fe = np.load(st_fe, mmap_mode="r")

        save_chunked_monthtail(
            mm_dyn,
            mm_stat,
            mm_fe,
            y_flat,
            mask,
            (m_t, m_s, m_la, m_lo),
            args.out_dir,
            args.gap_hours,
            args.val_last_days,
            args.test_last_days,
        )
        del mm_dyn, mm_stat, mm_fe
        gc.collect()
    finally:
        if ds_in is not None:
            ds_in.close()
        if not args.keep_staging:
            for p in (st_dyn, st_stat, st_fe):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    cfg = {
        "dataset": "tianji_overlap_pmst27_monthtail",
        "overlap_vars": OVERLAP_CANONICAL,
        "dyn_layout": "24_pmst_met + zenith + pm10 + pm2p5",
        "fe_dim": fe_dim,
        "window": args.window,
        "step": args.step,
        "split": "month_tail",
        "tianji_raw_time_alignment": "raw_utc_no_shift",
        "time_coordinate": "UTC",
        "pm_time_match": "nearest_90min_utc",
        "val_last_days": args.val_last_days,
        "test_last_days": args.test_last_days,
        "gap_hours": args.gap_hours,
        "max_vis_threshold": MAX_VIS_THRESHOLD,
        "chunk_wins": args.chunk_wins,
        "low_mem_pipeline": True,
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[OK] wrote dataset to {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
