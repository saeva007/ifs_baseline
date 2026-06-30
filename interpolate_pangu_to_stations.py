#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpolate Pangu China gridded files to CMA station locations.

The default station source follows ``era5_interp_test.py``:
``China_national_station_info_without_polar.csv`` with ``id/lat/lon`` columns.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


WORK_DIR_DEFAULT = "/public/home/putianshu/vis_mlp"
PANGU_DIR_DEFAULT = os.path.join(WORK_DIR_DEFAULT, "pangu_2025_china_chunks")
STATION_FILE_DEFAULT = ""
TARGET_FILE_DEFAULT = os.path.join(WORK_DIR_DEFAULT, "tianji_auto_station", "merged_final_all_vars.nc")
OUT_FILE_DEFAULT = os.path.join(
    WORK_DIR_DEFAULT,
    "ifs_baseline",
    "pangu_station",
    "pangu_station_2025_lead24h.nc",
)


def _open_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def _choose_netcdf_engine() -> str:
    for module_name, engine_name in (("h5netcdf", "h5netcdf"), ("netCDF4", "netcdf4"), ("scipy", "scipy")):
        if importlib.util.find_spec(module_name) is not None:
            return engine_name
    raise RuntimeError("No NetCDF writer is available. Install one of h5netcdf, netCDF4, or scipy.")


def _encoding(ds: xr.Dataset, compress_level: int, engine: str) -> Dict[str, Dict[str, object]]:
    encoding: Dict[str, Dict[str, object]] = {}
    for name in ds.data_vars:
        encoding[name] = {}
        if engine in {"h5netcdf", "netcdf4"}:
            encoding[name]["dtype"] = "float32"
        if engine in {"h5netcdf", "netcdf4"} and int(compress_level) > 0:
            encoding[name].update({"zlib": True, "complevel": int(compress_level), "shuffle": True})
    return encoding


def _coord_name(obj, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    low = {str(k).lower(): str(k) for k in list(obj.coords) + list(obj.dims)}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    raise KeyError(f"Cannot infer coordinate from {candidates}; available={list(obj.coords) + list(obj.dims)}")


def _load_stations_from_csv(station_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(station_file):
        raise FileNotFoundError(f"Station CSV does not exist: {station_file}")
    df = pd.read_csv(station_file)
    required = {"id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Station CSV must contain columns {sorted(required)}; missing={sorted(missing)}")
    df = df[(df["lon"] >= 65) & (df["lon"] <= 145) & (df["lat"] >= 10) & (df["lat"] <= 60)].copy()
    finite = np.isfinite(df["lat"].to_numpy(dtype=np.float64)) & np.isfinite(df["lon"].to_numpy(dtype=np.float64))
    if not finite.all():
        n_bad = int((~finite).sum())
        print(f"[WARN] drop {n_bad} stations with non-finite lat/lon from {station_file}", flush=True)
        df = df.loc[finite].copy()
    if len(df) == 0:
        raise ValueError(f"No usable station rows found in {station_file}.")
    return (
        df["id"].to_numpy(),
        df["lat"].to_numpy(dtype=np.float64),
        df["lon"].to_numpy(dtype=np.float64),
    )


def _load_stations_from_netcdf(target_file: str, year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        finite = np.isfinite(lats) & np.isfinite(lons)
        if not finite.all():
            n_bad = int((~finite).sum())
            print(f"[WARN] drop {n_bad} stations with non-finite lat/lon from {target_file}", flush=True)
            station_ids = station_ids[finite]
            lats = lats[finite]
            lons = lons[finite]
        if len(station_ids) == 0:
            raise ValueError(f"No finite station lat/lon pairs found in {target_file}.")
        return station_ids, lats, lons
    finally:
        ds.close()


def _load_stations(station_file: str, target_file: str, year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if station_file:
        print(f"[station] CSV={station_file}", flush=True)
        return _load_stations_from_csv(station_file)
    print(f"[station] NetCDF={target_file}", flush=True)
    return _load_stations_from_netcdf(target_file, year)


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
    points = np.asarray(points, dtype=np.float64)
    station_xy = np.column_stack([station_lats, station_lons]).astype(np.float64)
    if not np.isfinite(station_xy).all():
        raise ValueError("Station lat/lon passed to KDTree query still contain nan or inf values.")
    finite_points = np.isfinite(points).all(axis=1)
    point_idx = np.flatnonzero(finite_points)
    if len(point_idx) == 0:
        raise ValueError("Pangu grid has no finite lat/lon points for station interpolation.")
    if len(point_idx) != len(points):
        print(f"[WARN] drop {len(points) - len(point_idx)} non-finite Pangu grid points before KDTree.", flush=True)
    tree = cKDTree(points[point_idx])
    k_eff = min(max(1, int(k)), len(point_idx))
    dist, idx = tree.query(station_xy, k=k_eff)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]
    idx = point_idx[idx]
    weights = 1.0 / np.maximum(dist, float(eps)) ** 2
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return idx, weights.astype(np.float32)


def _interp_var(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = da.transpose(time_name, lat_name, lon_name).values.astype(np.float32)
    nt = arr.shape[0]
    flat = arr.reshape(nt, -1)
    vals = flat[:, idx]
    return np.sum(vals * weights[None, :, :], axis=2).astype(np.float32)


def _lead_coordinates(
    ds: xr.Dataset,
    time_name: str,
    allow_missing: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return per-valid-time initialization and lead coordinates.

    Pangu grid outputs carry ``init_time`` and ``forecast_lead_hours``.  These
    coordinates must survive station interpolation because a filename is not
    evidence of forecast lead.
    """
    valid = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
    init_values: Optional[np.ndarray] = None
    if "init_time" in ds.coords or "init_time" in ds.data_vars:
        raw = np.asarray(ds["init_time"].values).reshape(-1)
        if raw.size == 1 and len(valid) > 1:
            raw = np.repeat(raw, len(valid))
        if raw.size != len(valid):
            raise ValueError(
                f"init_time length={raw.size} does not match valid-time length={len(valid)}"
            )
        init_values = pd.DatetimeIndex(pd.to_datetime(raw)).values

    declared = ds.attrs.get("forecast_lead_hours")
    lead_values: Optional[np.ndarray] = None
    if init_values is not None:
        lead_values = (
            (valid.values - init_values) / np.timedelta64(1, "h")
        ).astype(np.float32)
        if declared is not None and not np.allclose(
            lead_values, float(declared), rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(
                f"forecast_lead_hours attr={declared} disagrees with valid_time-init_time "
                f"range={float(np.min(lead_values))}..{float(np.max(lead_values))} h"
            )
    elif declared is not None:
        lead_values = np.full(len(valid), float(declared), dtype=np.float32)
        init_values = (
            valid.values - pd.to_timedelta(lead_values, unit="h").to_numpy()
        )

    if (init_values is None or lead_values is None) and not allow_missing:
        raise ValueError(
            "Pangu source lacks init_time/forecast_lead_hours metadata. "
            "Regenerate it with run_pangu_onnx_2025.py, or pass "
            "--allow_missing_lead_metadata for diagnostics only."
        )
    return init_values, lead_values


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
    ap = argparse.ArgumentParser(description="Interpolate Pangu China NetCDF files to station_id axis.")
    ap.add_argument("--input_dir", default=PANGU_DIR_DEFAULT)
    ap.add_argument("--input_glob", default="")
    ap.add_argument("--input_files", default="")
    ap.add_argument(
        "--station_file",
        default=STATION_FILE_DEFAULT,
        help="CSV station table with id/lat/lon columns. If empty, --target_file is used as station source.",
    )
    ap.add_argument("--target_file", default=TARGET_FILE_DEFAULT)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out_file", default=OUT_FILE_DEFAULT)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--compress_level", type=int, default=1)
    ap.add_argument(
        "--allow_missing_lead_metadata",
        action="store_true",
        help="Permit legacy inputs without init_time/lead metadata; diagnostics only.",
    )
    args = ap.parse_args()

    files = _input_files(args)
    station_ids, station_lats, station_lons = _load_stations(args.station_file, args.target_file, args.year)
    print(f"[station] n={len(station_ids)}", flush=True)
    out_datasets = []
    sources = []
    for fp in files:
        print(f"[pangu-idw] {fp}", flush=True)
        ds = _open_dataset(fp)
        try:
            time_name = _coord_name(ds, ("time", "valid_time"))
            init_values, lead_values = _lead_coordinates(
                ds, time_name, args.allow_missing_lead_metadata
            )
            lat_name, lon_name, _, _, points = _grid_points(ds)
            idx, weights = _station_neighbors(points, station_lats, station_lons, args.k, args.eps)
            data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
            variable_attrs: Dict[str, Dict[str, object]] = {}
            for name in ds.data_vars:
                da = ds[name]
                if not {time_name, lat_name, lon_name}.issubset(set(da.dims)):
                    print(f"  [skip] {name}: dims={da.dims}", flush=True)
                    continue
                data_vars[str(name)] = (("time", "station_id"), _interp_var(da, time_name, lat_name, lon_name, idx, weights))
                variable_attrs[str(name)] = dict(da.attrs)
            coords = {
                "time": pd.DatetimeIndex(pd.to_datetime(ds[time_name].values)),
                "station_id": station_ids,
                "lat": ("station_id", station_lats.astype(np.float32)),
                "lon": ("station_id", station_lons.astype(np.float32)),
            }
            if init_values is not None and lead_values is not None:
                coords["init_time"] = ("time", init_values)
                coords["forecast_lead_hours"] = ("time", lead_values)
            part = xr.Dataset(
                data_vars,
                coords=coords,
                attrs={
                    "source": f"Pangu China lead files for year {int(args.year)}",
                    "interpolation": f"IDW k={int(args.k)} in latitude/longitude degrees",
                    "source_file": fp,
                },
            )
            for name, attrs in variable_attrs.items():
                part[name].attrs.update(attrs)
            out_datasets.append(part)
            sources.append(fp)
        finally:
            ds.close()
    out = xr.concat(out_datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
    out = out.sortby("time")
    _, unique_idx = np.unique(pd.DatetimeIndex(out["time"].values).values, return_index=True)
    if len(unique_idx) != out.sizes["time"]:
        raise ValueError(
            f"Pangu inputs contain {int(out.sizes['time'] - len(unique_idx))} duplicate valid times; "
            "do not silently choose one initialization/lead."
        )
    if "forecast_lead_hours" in out.coords:
        lead = np.asarray(out["forecast_lead_hours"].values, dtype=np.float64)
        out.attrs["forecast_lead_hours_min"] = float(np.nanmin(lead))
        out.attrs["forecast_lead_hours_max"] = float(np.nanmax(lead))
        out.attrs["lead_provenance"] = "per-time valid_time minus init_time"
    out.attrs.update(
        {
            "target_file": args.target_file,
            "year": int(args.year),
            "source_files_json": json.dumps(sources, ensure_ascii=False),
        }
    )
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _choose_netcdf_engine()
    encoding = _encoding(out, args.compress_level, engine)
    print(f"[pangu-idw] writing {out_path} engine={engine}", flush=True)
    out.to_netcdf(out_path, engine=engine, encoding=encoding)
    print(f"[OK] {out_path}", flush=True)


if __name__ == "__main__":
    main()
