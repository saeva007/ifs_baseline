#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpolate Pangu China gridded monthly files to CMA stations."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


VIS_MLP_ROOT = "/public/home/putianshu/vis_mlp"
PANGU_DIR_DEFAULT = os.path.join(VIS_MLP_ROOT, "pangu_2021_china_monthly")
TARGET_FILE_DEFAULT = os.path.join(VIS_MLP_ROOT, "CMA_visibility_2021_2023_GeoCoords_1.nc")
OUT_FILE_DEFAULT = os.path.join(VIS_MLP_ROOT, "ifs_baseline", "pangu_station", "pangu_station_2021_lead24h.nc")


def _open_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def _coord_name(obj, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    low = {str(k).lower(): str(k) for k in list(obj.coords) + list(obj.dims)}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    raise KeyError(f"Cannot infer coordinate from {candidates}; available={list(obj.coords) + list(obj.dims)}")


def _load_stations(target_file: str, year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = _open_dataset(target_file)
    try:
        station_name = _coord_name(ds, ("station_id", "num_station", "station", "id"))
        lat_name = _coord_name(ds, ("lat", "latitude"))
        lon_name = _coord_name(ds, ("lon", "longitude"))
        if "time" in ds:
            subset = ds.sel(time=ds.time.dt.year == int(year))
            if subset.sizes.get("time", 0) > 0:
                ds = subset
        station_ids = np.asarray(ds[station_name].values)
        lats = np.asarray(ds[lat_name].values, dtype=np.float64)
        lons = np.asarray(ds[lon_name].values, dtype=np.float64)
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("Station lat/lon must be 1D arrays.")
        return station_ids, lats, lons
    finally:
        ds.close()


def _grid_points(ds: xr.Dataset) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    lat_name = _coord_name(ds, ("latitude", "lat", "grid_yt", "y"))
    lon_name = _coord_name(ds, ("longitude", "lon", "grid_xt", "x"))
    lat_vals = np.asarray(ds[lat_name].values, dtype=np.float64)
    lon_vals = np.asarray(ds[lon_name].values, dtype=np.float64)
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon2, lat2 = np.meshgrid(lon_vals, lat_vals)
    else:
        lat2, lon2 = lat_vals, lon_vals
    points = np.column_stack([lat2.reshape(-1), lon2.reshape(-1)])
    return lat_name, lon_name, lat2, lon2, points


def _station_neighbors(points: np.ndarray, station_lats: np.ndarray, station_lons: np.ndarray, k: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    dist, idx = tree.query(np.column_stack([station_lats, station_lons]), k=max(1, int(k)))
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]
    weights = 1.0 / np.maximum(dist, float(eps)) ** 2
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return idx, weights.astype(np.float32)


def _interp_var(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = da.transpose(time_name, lat_name, lon_name).values.astype(np.float32)
    nt = arr.shape[0]
    flat = arr.reshape(nt, -1)
    vals = flat[:, idx]
    return np.sum(vals * weights[None, :, :], axis=2).astype(np.float32)


def _input_files(args: argparse.Namespace) -> List[str]:
    if args.input_files:
        files = [p.strip() for p in args.input_files.split(",") if p.strip()]
    else:
        pattern = args.input_glob or os.path.join(args.input_dir, f"pangu_china_{args.year}*_lead*.nc")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No Pangu monthly NetCDF files found.")
    return [os.path.realpath(p) for p in files]


def main() -> None:
    ap = argparse.ArgumentParser(description="Interpolate Pangu China monthly NetCDF files to station_id axis.")
    ap.add_argument("--input_dir", default=PANGU_DIR_DEFAULT)
    ap.add_argument("--input_glob", default="")
    ap.add_argument("--input_files", default="")
    ap.add_argument("--target_file", default=TARGET_FILE_DEFAULT)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--out_file", default=OUT_FILE_DEFAULT)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--compress_level", type=int, default=1)
    args = ap.parse_args()

    files = _input_files(args)
    station_ids, station_lats, station_lons = _load_stations(args.target_file, args.year)
    out_datasets = []
    sources = []
    for fp in files:
        print(f"[pangu-idw] {fp}", flush=True)
        ds = _open_dataset(fp)
        try:
            time_name = _coord_name(ds, ("time", "valid_time"))
            lat_name, lon_name, _, _, points = _grid_points(ds)
            idx, weights = _station_neighbors(points, station_lats, station_lons, args.k, args.eps)
            data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
            for name in ds.data_vars:
                da = ds[name]
                if not {time_name, lat_name, lon_name}.issubset(set(da.dims)):
                    print(f"  [skip] {name}: dims={da.dims}", flush=True)
                    continue
                data_vars[str(name)] = (("time", "station_id"), _interp_var(da, time_name, lat_name, lon_name, idx, weights))
            part = xr.Dataset(
                data_vars,
                coords={
                    "time": pd.DatetimeIndex(pd.to_datetime(ds[time_name].values)),
                    "station_id": station_ids,
                    "lat": ("station_id", station_lats.astype(np.float32)),
                    "lon": ("station_id", station_lons.astype(np.float32)),
                },
                attrs={
                    "source": "Pangu WeatherBench2 China lead24h",
                    "interpolation": f"IDW k={int(args.k)} in latitude/longitude degrees",
                    "source_file": fp,
                },
            )
            out_datasets.append(part)
            sources.append(fp)
        finally:
            ds.close()
    out = xr.concat(out_datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
    out = out.sortby("time")
    _, unique_idx = np.unique(pd.DatetimeIndex(out["time"].values).values, return_index=True)
    if len(unique_idx) != out.sizes["time"]:
        out = out.isel(time=np.sort(unique_idx))
    out.attrs.update(
        {
            "target_file": args.target_file,
            "year": int(args.year),
            "source_files_json": json.dumps(sources, ensure_ascii=False),
        }
    )
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    if int(args.compress_level) > 0:
        encoding = {
            name: {"zlib": True, "complevel": int(args.compress_level), "shuffle": True, "dtype": "float32"}
            for name in out.data_vars
        }
    print(f"[pangu-idw] writing {out_path}", flush=True)
    out.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    print(f"[OK] {out_path}", flush=True)


if __name__ == "__main__":
    main()
