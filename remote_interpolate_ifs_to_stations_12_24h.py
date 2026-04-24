#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolate remote IFS grid fields to station points with accelerated KDTree-IDW,
and assemble Tianji-style valid-time series using dual init cycles:
  - t00z: use lead 12..24
  - t12z: use lead 12..24

Target valid-time selection rule (hourly):
  - valid hour 00..11: from previous day t12z (lead 12..23)
  - valid hour 12..23: from same day t00z (lead 12..23)
This gives continuous 24h/day coverage and avoids duplicate lead-24 collisions.

Input requirements:
1) IFS root, e.g. /sharedata/dataset/GroupData/GD001-EC_Forcasting
2) station csv with columns: station_id, lat, lon
3) variable mapping json (can be edited after running remote_check_ifs_catalog.py)

Output:
  - interpolated_ifs_2025.nc  (time, station_id, variable)
  - interp_build_report.json
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


LEAD_RE = re.compile(r"^[A-Z0-9]{3}(\d{8})(\d{8})1_([A-Za-z0-9]+)\.nc$")

DEFAULT_MAPPING = {
    # canonical_name: {"group":"Single_level|Pressure_levels", "folder":"...", "nc_var":"...", "level_hpa": null|925|1000}
    "T2M": {"group": "Single_level", "folder": "2t", "nc_var": "t2m", "level_hpa": None},
    "PRECIP": {"group": "Single_level", "folder": "tp", "nc_var": "tp", "level_hpa": None},
    "MSLP": {"group": "Single_level", "folder": "msl", "nc_var": "msl", "level_hpa": None},
    "SW_RAD": {"group": "Single_level", "folder": "ssrd", "nc_var": "ssrd", "level_hpa": None},
    "U10": {"group": "Single_level", "folder": "10u", "nc_var": "u10", "level_hpa": None},
    "V10": {"group": "Single_level", "folder": "10v", "nc_var": "v10", "level_hpa": None},
    # examples for RH importance:
    "RH2M": {"group": "Single_level", "folder": "2r", "nc_var": "r2", "level_hpa": None},
    "RH_925": {"group": "Pressure_levels", "folder": "r", "nc_var": "r", "level_hpa": 925},
}


def idw_interpolation_fast(grid_data, grid_lats, grid_lons, station_lats, station_lons, power=2.0, max_distance=5.0):
    """
    Adapted from xiahang_forecast_system.py:
    - KDTree over valid grid points
    - batch nearest-neighbor query
    - IDW weighted average
    grid_data: (time, lat, lon)
    returns: (time, station)
    """
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)
    grid_points = np.column_stack((grid_lat_2d.ravel(), grid_lon_2d.ravel()))
    station_points = np.column_stack((station_lats, station_lons))
    n_neighbors = min(10, len(grid_points))
    out = []

    for t in range(grid_data.shape[0]):
        vals = grid_data[t].ravel()
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() < 4:
            fill = np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0
            out.append(np.full(len(station_lats), fill, dtype=np.float32))
            continue

        valid_points = grid_points[finite_mask]
        valid_vals = vals[finite_mask]
        tree = cKDTree(valid_points)
        k = min(n_neighbors, len(valid_points))
        dists, idx = tree.query(station_points, k=k, distance_upper_bound=max_distance, workers=-1)

        # normalize shape when k=1
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        st_vals = np.zeros(len(station_lats), dtype=np.float32)
        for i in range(len(station_lats)):
            ok = idx[i] < len(valid_points)
            if not np.any(ok):
                st_vals[i] = float(np.nanmean(valid_vals))
                continue
            di = dists[i][ok]
            ii = idx[i][ok]
            very_close = di < 0.01
            if np.any(very_close):
                st_vals[i] = float(np.mean(valid_vals[ii[very_close]]))
                continue
            di = np.maximum(di, 1e-10)
            w = 1.0 / (di ** power)
            w /= np.sum(w)
            st_vals[i] = float(np.sum(w * valid_vals[ii]))
        out.append(st_vals)
    return np.asarray(out, dtype=np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="IFS root, e.g. /sharedata/dataset/GroupData/GD001-EC_Forcasting")
    ap.add_argument("--station_csv", required=True, help="CSV with station_id,lat,lon")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--mapping_json", default="", help="Optional mapping json. If empty, use built-in DEFAULT_MAPPING")
    ap.add_argument("--vars", default="T2M,PRECIP,MSLP,SW_RAD,U10,V10", help="comma-separated canonical vars to export")
    ap.add_argument("--out_nc", default="./interpolated_ifs_2025.nc")
    ap.add_argument("--out_report", default="./interp_build_report.json")
    ap.add_argument("--max_distance", type=float, default=5.0)
    ap.add_argument("--idw_power", type=float, default=2.0)
    return ap.parse_args()


def load_mapping(mapping_json: str) -> Dict[str, dict]:
    if mapping_json and os.path.exists(mapping_json):
        with open(mapping_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_MAPPING


def build_file_map(root: str, year: int, group: str, folder: str) -> Dict[Tuple[pd.Timestamp, pd.Timestamp], str]:
    """
    map[(init_time, valid_time)] = filepath
    """
    pat = os.path.join(root, "0P125", f"{year}*", "t*z", group, folder, "*.nc")
    files = sorted(glob.glob(pat))
    out = {}
    for fp in files:
        m = LEAD_RE.match(os.path.basename(fp))
        if not m:
            continue
        init_s, valid_s = m.group(1), m.group(2)
        try:
            init_t = pd.Timestamp(datetime.strptime(init_s, "%m%d%H%M").replace(year=year))
            valid_t = pd.Timestamp(datetime.strptime(valid_s, "%m%d%H%M").replace(year=year))
            # year wrap handling (Dec->Jan)
            if valid_t < init_t - pd.Timedelta(hours=6):
                valid_t = valid_t + pd.DateOffset(years=1)
            if init_t.year != year and valid_t.year != year:
                continue
            out[(init_t, valid_t)] = fp
        except Exception:
            continue
    return out


def pick_source_for_valid_time(vt: pd.Timestamp) -> Tuple[pd.Timestamp, str]:
    """
    Tianji-style composition:
    - hour 00..11 => previous day t12z
    - hour 12..23 => same day t00z
    """
    if vt.hour < 12:
        init = (vt.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(hours=12)
        cyc = "t12z"
    else:
        init = vt.normalize()
        cyc = "t00z"
    return init, cyc


def open_slice_and_interp(fp: str, nc_var: str, station_lats, station_lons, level_hpa: Optional[float], max_distance: float, idw_power: float) -> np.ndarray:
    ds = xr.open_dataset(fp)
    if nc_var not in ds.data_vars:
        # fallback first var
        nc_var = list(ds.data_vars)[0]
    da = ds[nc_var]

    # select pressure level if requested
    if level_hpa is not None:
        level_dim = next((n for n in ["level", "isobaricInhPa", "pressure_level", "lev", "plev"] if n in da.dims), None)
        if level_dim is None:
            raise ValueError(f"{fp}: no level dim but level_hpa requested")
        levels = ds[level_dim].values.astype(float)
        idx = int(np.argmin(np.abs(levels - float(level_hpa))))
        da = da.isel({level_dim: idx})

    # detect lat/lon dims
    lat_dim = next((n for n in ["latitude", "lat", "y", "grid_yt"] if n in da.dims), None)
    lon_dim = next((n for n in ["longitude", "lon", "x", "grid_xt"] if n in da.dims), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"{fp}: cannot detect lat/lon dims from {da.dims}")

    # take first time slice if time exists
    if da.ndim >= 3:
        time_dim = next((n for n in ["time", "valid_time", "t"] if n in da.dims), None)
        if time_dim is not None:
            da = da.isel({time_dim: 0})

    # reorder to (lat, lon)
    if da.dims != (lat_dim, lon_dim):
        da = da.transpose(lat_dim, lon_dim)

    grid_lats = ds[lat_dim].values
    grid_lons = ds[lon_dim].values
    vals2d = da.values.astype(np.float32)
    ds.close()

    # convert lon to [-180, 180] and sort
    if np.any(grid_lons > 180):
        grid_lons = np.where(grid_lons > 180, grid_lons - 360, grid_lons)
        idx = np.argsort(grid_lons)
        grid_lons = grid_lons[idx]
        vals2d = vals2d[:, idx]

    out = idw_interpolation_fast(
        vals2d[np.newaxis, :, :], grid_lats, grid_lons, station_lats, station_lons,
        power=idw_power, max_distance=max_distance
    )
    return out[0]


def main():
    args = parse_args()
    mapping = load_mapping(args.mapping_json)
    use_vars = [v.strip() for v in args.vars.split(",") if v.strip()]

    st = pd.read_csv(args.station_csv)
    for c in ["station_id", "lat", "lon"]:
        if c not in st.columns:
            raise ValueError(f"station_csv missing column: {c}")
    station_ids = st["station_id"].values
    station_lats = st["lat"].values.astype(float)
    station_lons = st["lon"].values.astype(float)

    # target valid times for whole year hourly
    valid_times = pd.date_range(f"{args.year}-01-01 00:00:00", f"{args.year}-12-31 23:00:00", freq="1H")

    data = np.full((len(valid_times), len(station_ids), len(use_vars)), np.nan, dtype=np.float32)
    report = {"year": args.year, "variables": {}, "n_valid_times": len(valid_times)}

    for vi, canon in enumerate(use_vars):
        if canon not in mapping:
            raise KeyError(f"{canon} not found in mapping")
        cfg = mapping[canon]
        group = cfg["group"]
        folder = cfg["folder"]
        nc_var = cfg["nc_var"]
        level_hpa = cfg.get("level_hpa", None)

        file_map = build_file_map(args.root, args.year, group, folder)
        ok_count = 0
        miss_count = 0

        for ti, vt in enumerate(valid_times):
            init_t, _ = pick_source_for_valid_time(vt)
            key = (init_t, vt)
            fp = file_map.get(key, None)
            if fp is None or (not os.path.exists(fp)):
                miss_count += 1
                continue
            try:
                vals = open_slice_and_interp(
                    fp, nc_var, station_lats, station_lons, level_hpa,
                    max_distance=args.max_distance, idw_power=args.idw_power
                )
                data[ti, :, vi] = vals
                ok_count += 1
            except Exception:
                miss_count += 1
                continue

        report["variables"][canon] = {
            "group": group,
            "folder": folder,
            "nc_var": nc_var,
            "level_hpa": level_hpa,
            "ok_times": ok_count,
            "missing_or_failed_times": miss_count,
        }
        print(f"[{canon}] ok={ok_count}, miss={miss_count}")

    ds_out = xr.Dataset(
        {
            "ifs_interp": (["time", "station_id", "variable"], data),
        },
        coords={
            "time": valid_times,
            "station_id": station_ids,
            "lat": ("station_id", station_lats),
            "lon": ("station_id", station_lons),
            "variable": use_vars,
        },
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_nc)) or ".", exist_ok=True)
    ds_out.to_netcdf(args.out_nc)

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {args.out_nc}")
    print(f"[OK] wrote {args.out_report}")


if __name__ == "__main__":
    main()

