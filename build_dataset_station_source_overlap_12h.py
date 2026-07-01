#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Static-RNN overlap S2 datasets from a station-level meteorological source.

This builder is for source-family experiments where the source is already on
station x time axes, such as station-interpolated Pangu or station ERA5 files.
It writes the same PMST-27 row layout as the Tianji/IFS overlap builders.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from pmst_overlap_common import (
    CANONICAL_DYNAMIC_UNITS,
    CANONICAL_UNIT_POLICY_VERSION,
    FEATURE_SET_CHOICES,
    FINAL_FEATURE_ORDER,
    PMST_SOURCE_FIELDS,
    TOTAL_DYN,
    append_pm10_channel,
    append_pm25_channel,
    build_static_features,
    calculate_dewpoint_from_rh,
    calculate_rh_from_dewpoint,
    calculate_specific_humidity,
    calculate_wind_speed_dir,
    compute_fog_features_pmst,
    cyclical_time_features,
    describe_available_pmst_features,
    dynamic_feature_order_for_feature_set,
    dynamic_layout_name,
    dyn_vars_for_feature_set,
    load_pm10_dataarray,
    load_pm25_dataarray,
    normalize_var_coord,
    resolve_pmst_feature_set,
    require_regular_time_axis,
    summarize_time_axis,
    scatter_overlap_fields,
    select_dynamic_layout,
    save_chunked_monthtail,
    calculate_zenith_angle,
)


VIS_MLP_ROOT = "/public/home/putianshu/vis_mlp"
IFS_BASELINE_ROOT = os.path.join(VIS_MLP_ROOT, "ifs_baseline")
FEATURE_DIR_DEFAULT = os.path.join(VIS_MLP_ROOT, "station_data", "station_data_merged")
TARGET_FILE_DEFAULT = os.path.join(VIS_MLP_ROOT, "CMA_visibility_2021_2023_GeoCoords_1.nc")
VEG_FILE_DEFAULT = "/public/home/putianshu/vis_cnn/data_vegtype.nc"
ORO_FILE_DEFAULT = "/public/home/putianshu/vis_cnn/data_orography.nc"
PM10_DIR_DEFAULT = os.path.join(VIS_MLP_ROOT, "pm10_station")
PM25_DIR_DEFAULT = os.path.join(VIS_MLP_ROOT, "pm2.5_station")

WINDOW_SIZE_DEFAULT = 12
STEP_SIZE_DEFAULT = 1
VAL_LAST_DAYS_DEFAULT = 3
TEST_LAST_DAYS_DEFAULT = 3
GAP_HOURS_DEFAULT = 24
CHUNK_WINS_DEFAULT = int(os.environ.get("STATION_SOURCE_CHUNK_WINS", "128"))
MAX_VIS_THRESHOLD = 30000
UNIQUE_VEG_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20])


ERA5_VAR_CONFIG = {
    "T2M": {"name": "2m_temperature", "level": None},
    "D2M": {"name": "2m_dewpoint_temperature", "level": None},
    "PRECIP": {"name": "total_precipitation", "level": None},
    "MSLP": {"name": "mean_sea_level_pressure", "level": None},
    "SW_RAD": {"name": "mean_surface_downward_short_wave_radiation_flux", "level": None},
    "U10": {"name": "10m_u_component_of_wind", "level": None},
    "V10": {"name": "10m_v_component_of_wind", "level": None},
    "CAPE": {"name": "convective_available_potential_energy", "level": None},
    "LCC": {"name": "low_cloud_cover", "level": None},
    "T_925": {"name": "temperature", "level": 925},
    "T_1000": {"name": "temperature", "level": 1000},
    "RH_925": {"name": "relative_humidity", "level": 925},
    "RH_1000": {"name": "relative_humidity", "level": 1000},
    "Q_925": {"name": "specific_humidity", "level": 925},
    "Q_1000": {"name": "specific_humidity", "level": 1000},
    "U_925": {"name": "u_component_of_wind", "level": 925},
    "V_925": {"name": "v_component_of_wind", "level": 925},
    "W_925": {"name": "vertical_velocity", "level": 925},
    "W_1000": {"name": "vertical_velocity", "level": 1000},
}

RAW_ALIASES = {
    "2M_TEMPERATURE": "T2M",
    "2M_DEWPOINT_TEMPERATURE": "D2M",
    "TOTAL_PRECIPITATION": "PRECIP",
    "MEAN_SEA_LEVEL_PRESSURE": "MSLP",
    "MEAN_SURFACE_DOWNWARD_SHORT_WAVE_RADIATION_FLUX": "SW_RAD",
    "10M_U_COMPONENT_OF_WIND": "U10",
    "10M_V_COMPONENT_OF_WIND": "V10",
    "CONVECTIVE_AVAILABLE_POTENTIAL_ENERGY": "CAPE",
    "LOW_CLOUD_COVER": "LCC",
    "RELATIVE_HUMIDITY": "RH",
    "TEMPERATURE": "T",
    "SPECIFIC_HUMIDITY": "Q",
    "U_COMPONENT_OF_WIND": "U",
    "V_COMPONENT_OF_WIND": "V",
    "VERTICAL_VELOCITY": "W",
    "RH925": "RH_925",
    "RH1000": "RH_1000",
    "T925": "T_925",
    "T1000": "T_1000",
}


def _open_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def _coord_name(ds_or_da, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds_or_da.coords or name in ds_or_da.dims:
            return name
    return None


def _normalize_station_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    time_name = _coord_name(ds, ("time", "Time", "valid_time"))
    station_name = _coord_name(ds, ("station_id", "num_station", "station", "id"))
    lat_name = _coord_name(ds, ("lat", "latitude"))
    lon_name = _coord_name(ds, ("lon", "longitude"))
    if time_name and time_name != "time":
        rename[time_name] = "time"
    if station_name and station_name != "station_id":
        rename[station_name] = "station_id"
    if lat_name and lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name and lon_name != "lon":
        rename[lon_name] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def _rename_canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for name in list(ds.data_vars):
        key = name.upper().replace("-", "_").replace(" ", "_")
        canon = RAW_ALIASES.get(key, normalize_var_coord(name))
        if canon != name and canon not in ds.data_vars:
            rename[name] = canon
    if rename:
        ds = ds.rename(rename)
    return ds


def _load_station_nc(paths: Sequence[str]) -> xr.Dataset:
    datasets = []
    for path in paths:
        ds = _normalize_station_dims(_open_dataset(path))
        ds = _rename_canonical(ds)
        datasets.append(ds)
    if not datasets:
        raise FileNotFoundError("No station source NetCDF files were found.")
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
        for extra in datasets:
            if extra is not ds:
                extra.close()
    ds = _normalize_station_dims(ds)
    if "time" not in ds.dims or "station_id" not in ds.dims:
        raise ValueError(f"station_nc source must have time and station_id dimensions; got dims={dict(ds.sizes)}")
    _, unique_idx = np.unique(pd.DatetimeIndex(ds["time"].values).values, return_index=True)
    if len(unique_idx) != ds.sizes["time"]:
        raise ValueError(
            f"station source contains {int(ds.sizes['time'] - len(unique_idx))} duplicate valid times; "
            "lead selection must be explicit before dataset construction"
        )
    return ds.sortby("time")


def _load_era5_feature_dir(feature_dir: str, year: int) -> xr.Dataset:
    arrays = []
    for canon, conf in ERA5_VAR_CONFIG.items():
        folder = str(conf["name"])
        level = conf["level"]
        fname = f"{folder}_{year}_{level}hPa_merged.nc" if level else f"{folder}_{year}_merged.nc"
        path = os.path.join(feature_dir, fname)
        if not os.path.isfile(path):
            print(f"[era5] skip missing {path}", flush=True)
            continue
        ds_var = _normalize_station_dims(_open_dataset(path))
        try:
            raw_name = next(iter(ds_var.data_vars))
            da = ds_var[raw_name].rename(canon)
            for coord in ("level", "number", "expver", "metpy_crs"):
                if coord in da.coords:
                    da = da.drop_vars(coord)
            arrays.append(da.load())
        finally:
            ds_var.close()
    if not arrays:
        raise FileNotFoundError(f"No ERA5 station files for year={year} under {feature_dir}")
    ds = xr.merge(arrays, compat="override")
    return _normalize_station_dims(ds).sortby("time")


def _ensure_derived(ds: xr.Dataset) -> xr.Dataset:
    def assign_like(name: str, template: str, values) -> None:
        ds[name] = xr.DataArray(np.asarray(values, dtype=np.float32), dims=ds[template].dims, coords=ds[template].coords)

    if "RH2M" not in ds and "T2M" in ds and "D2M" in ds:
        assign_like("RH2M", "T2M", calculate_rh_from_dewpoint(ds["T2M"], ds["D2M"]))
    if "WSPD10" not in ds and "U10" in ds and "V10" in ds:
        speed, direction = calculate_wind_speed_dir(ds["U10"], ds["V10"])
        assign_like("WSPD10", "U10", speed)
        assign_like("WDIR10", "U10", direction)
    if "WSPD925" not in ds and "U_925" in ds and "V_925" in ds:
        speed, _ = calculate_wind_speed_dir(ds["U_925"], ds["V_925"])
        assign_like("WSPD925", "U_925", speed)
    if "Q_1000" not in ds and "T_1000" in ds and "RH_1000" in ds:
        assign_like("Q_1000", "T_1000", calculate_specific_humidity(ds["T_1000"], ds["RH_1000"], 1000.0))
    if "Q_925" not in ds and "T_925" in ds and "RH_925" in ds:
        assign_like("Q_925", "T_925", calculate_specific_humidity(ds["T_925"], ds["RH_925"], 925.0))
    if "DP_1000" not in ds:
        if "Q_1000" in ds:
            pass
        elif "T_1000" in ds and "RH_1000" in ds:
            assign_like("DP_1000", "T_1000", calculate_dewpoint_from_rh(ds["T_1000"], ds["RH_1000"]))
    if "DP_925" not in ds:
        if "Q_925" in ds:
            pass
        elif "T_925" in ds and "RH_925" in ds:
            assign_like("DP_925", "T_925", calculate_dewpoint_from_rh(ds["T_925"], ds["RH_925"]))
    if "DPD" not in ds and "T2M" in ds:
        if "D2M" in ds:
            ds["DPD"] = ds["T2M"] - ds["D2M"]
        elif "RH2M" in ds:
            assign_like("DPD", "T2M", np.asarray(ds["T2M"].values, dtype=np.float32) - calculate_dewpoint_from_rh(ds["T2M"], ds["RH2M"]))
    if "INVERSION" not in ds and "T_925" in ds and "T2M" in ds:
        ds["INVERSION"] = ds["T_925"] - ds["T2M"]
    return ds


def _paths_from_args(source_file: str, source_glob: str) -> List[str]:
    paths = []
    if source_file:
        paths.extend([p.strip() for p in source_file.split(",") if p.strip()])
    if source_glob:
        paths.extend(sorted(glob.glob(source_glob)))
    paths = [os.path.realpath(p) for p in paths]
    if not paths:
        raise FileNotFoundError("Pass --source_file or --source_glob for station_nc mode.")
    return paths


def _load_target(path: str) -> xr.Dataset:
    ds = _normalize_station_dims(_open_dataset(path))
    if "vis" in ds.data_vars and "visibility" not in ds.data_vars:
        ds = ds.rename({"vis": "visibility"})
    if "visibility" not in ds.data_vars:
        raise KeyError(f"target_file must contain visibility/vis; got {list(ds.data_vars)}")
    if "time" not in ds.dims or "station_id" not in ds.dims:
        raise ValueError(f"target_file must have time and station_id dimensions; got {dict(ds.sizes)}")
    return ds.sortby("time")


def _station_indexer(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    source_index = pd.Index(source_ids)
    pos = source_index.get_indexer(target_ids)
    if (pos < 0).any():
        source_index = pd.Index(np.asarray(source_ids).astype(str))
        pos = source_index.get_indexer(np.asarray(target_ids).astype(str))
    return pos


def _aligned_target_visibility(
    target: xr.Dataset,
    times: pd.DatetimeIndex,
    stations: np.ndarray,
    tolerance_minutes: int,
) -> np.ndarray:
    target_time = pd.DatetimeIndex(pd.to_datetime(target["time"].values))
    target_station = target["station_id"].values
    time_pos = target_time.get_indexer(times, method="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))
    station_pos = _station_indexer(target_station, stations)
    out = np.full((len(times), len(stations)), np.nan, dtype=np.float32)
    ok = (time_pos[:, None] >= 0) & (station_pos[None, :] >= 0)
    if ok.any():
        arr = np.asarray(target["visibility"].values, dtype=np.float32).reshape(-1)
        ns_t = int(target.sizes["station_id"])
        linear = time_pos[:, None] * ns_t + station_pos[None, :]
        out[ok] = arr[linear[ok]]
    out = np.where(out <= MAX_VIS_THRESHOLD, out, np.nan)
    return out.astype(np.float32)


def _coords_for_source(source: xr.Dataset, target: xr.Dataset, stations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Shared station/static/solar features must use the canonical observation
    # coordinates for every forecast source. Source-file coordinates are only
    # a fallback for targets without station coordinates.
    for ds in (target, source):
        if "lat" in ds:
            lat_da = ds["lat"]
            lon_da = ds["lon"]
            if "station_id" in lat_da.dims:
                try:
                    lat_da = lat_da.sel(station_id=stations)
                    lon_da = lon_da.sel(station_id=stations)
                except Exception:
                    pos = _station_indexer(ds["station_id"].values, stations)
                    if (pos < 0).any():
                        continue
                    lat_da = lat_da.isel(station_id=pos)
                    lon_da = lon_da.isel(station_id=pos)
            lat = np.asarray(lat_da.values)
            lon = np.asarray(lon_da.values)
            if lat.ndim == 1 and len(lat) == len(stations):
                return lat.astype(np.float64), lon.astype(np.float64)
    raise AttributeError("Could not find 1D station lat/lon coordinates in source or target dataset.")


def _extract_fields(ds: xr.Dataset, t0: int, t1: int) -> Dict[str, np.ndarray]:
    fields = {}
    for name in PMST_SOURCE_FIELDS:
        if name in ds.data_vars:
            fields[name] = ds[name].isel(time=slice(t0, t1)).values.astype(np.float32)
    return fields


def _forecast_lead_summary(
    ds: xr.Dataset,
    infer_pangu_lead12_23_from_valid_time: bool = False,
) -> Dict[str, object]:
    valid = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    lead = None
    provenance = ""
    if "forecast_lead_hours" in ds.coords or "forecast_lead_hours" in ds.data_vars:
        raw = np.asarray(ds["forecast_lead_hours"].values, dtype=np.float64).reshape(-1)
        if raw.size == 1 and len(valid) > 1:
            raw = np.repeat(raw, len(valid))
        if raw.size != len(valid):
            raise ValueError(
                f"forecast_lead_hours length={raw.size} does not match time length={len(valid)}"
            )
        lead = raw
        provenance = "forecast_lead_hours coordinate"
    if "init_time" in ds.coords or "init_time" in ds.data_vars:
        init = np.asarray(ds["init_time"].values).reshape(-1)
        if init.size == 1 and len(valid) > 1:
            init = np.repeat(init, len(valid))
        if init.size != len(valid):
            raise ValueError(f"init_time length={init.size} does not match time length={len(valid)}")
        derived = np.asarray(
            (valid.values - pd.DatetimeIndex(pd.to_datetime(init)).values) / np.timedelta64(1, "h"),
            dtype=np.float64,
        )
        if lead is not None and not np.allclose(lead, derived, rtol=0.0, atol=1.0e-6):
            raise ValueError("forecast_lead_hours disagrees with valid_time-init_time")
        lead = derived
        provenance = "valid_time minus init_time"
    if lead is None and "forecast_lead_hours" in ds.attrs:
        lead = np.full(len(valid), float(ds.attrs["forecast_lead_hours"]), dtype=np.float64)
        provenance = "global forecast_lead_hours attribute"
    if lead is None and infer_pangu_lead12_23_from_valid_time:
        hours = valid.hour.to_numpy(dtype=np.float64)
        lead = np.where(hours < 12.0, hours + 12.0, hours)
        provenance = "explicit stitched 00/12 UTC schedule reconstructed from valid_time hour"
    if lead is None:
        return {"available": False, "provenance": "missing"}
    finite = np.isfinite(lead)
    if not finite.all():
        raise ValueError(f"forecast lead contains {int((~finite).sum())} non-finite values")
    unique, counts = np.unique(np.round(lead, 6), return_counts=True)
    return {
        "available": True,
        "provenance": provenance,
        "min_hours": float(np.min(lead)),
        "max_hours": float(np.max(lead)),
        "unique_hours": [float(v) for v in unique.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PMST-27 overlap dataset from station-level source fields.")
    ap.add_argument("--source_kind", choices=["station_nc", "era5_feature_dir"], default=os.environ.get("SOURCE_KIND", "station_nc"))
    ap.add_argument("--source_file", default=os.environ.get("SOURCE_FILE", ""))
    ap.add_argument("--source_glob", default=os.environ.get("SOURCE_GLOB", ""))
    ap.add_argument("--feature_dir", default=os.environ.get("FEATURE_DIR", FEATURE_DIR_DEFAULT))
    ap.add_argument("--year", type=int, default=int(os.environ.get("YEAR", "2021")))
    ap.add_argument("--source_tag", default=os.environ.get("SOURCE_TAG", "station_source"))
    ap.add_argument("--target_file", default=os.environ.get("TARGET_FILE", TARGET_FILE_DEFAULT))
    ap.add_argument("--target_time_tolerance_minutes", type=int, default=int(os.environ.get("TARGET_TIME_TOLERANCE_MINUTES", "90")))
    ap.add_argument("--veg_file", default=VEG_FILE_DEFAULT)
    ap.add_argument("--oro_file", default=ORO_FILE_DEFAULT)
    ap.add_argument("--pm10_file", default=os.environ.get("PM10_FILE", ""))
    ap.add_argument("--pm10_dir", default=os.environ.get("PM10_DIR", PM10_DIR_DEFAULT))
    ap.add_argument("--pm25_file", default=os.environ.get("PM25_FILE", ""))
    ap.add_argument("--pm25_dir", default=os.environ.get("PM25_DIR", PM25_DIR_DEFAULT))
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", ""))
    ap.add_argument("--feature_set", choices=FEATURE_SET_CHOICES, default=os.environ.get("FEATURE_SET", "common_core"))
    ap.add_argument("--window", type=int, default=WINDOW_SIZE_DEFAULT)
    ap.add_argument("--step", type=int, default=STEP_SIZE_DEFAULT)
    ap.add_argument("--expected_time_step_hours", type=float, default=1.0)
    ap.add_argument("--allow_irregular_time_axis", action="store_true")
    ap.add_argument("--expected_lead_min_hours", type=float, default=None)
    ap.add_argument("--expected_lead_max_hours", type=float, default=None)
    ap.add_argument("--allow_missing_forecast_lead", action="store_true")
    ap.add_argument(
        "--infer_pangu_lead12_23_from_valid_time",
        action="store_true",
        help=(
            "For the legacy hourly stitched product only: reconstruct lead 12..23 h from "
            "valid hour using 00/12 UTC initialization cycles."
        ),
    )
    ap.add_argument("--val_last_days", type=int, default=VAL_LAST_DAYS_DEFAULT)
    ap.add_argument("--test_last_days", type=int, default=TEST_LAST_DAYS_DEFAULT)
    ap.add_argument("--gap_hours", type=int, default=GAP_HOURS_DEFAULT)
    ap.add_argument("--chunk_wins", type=int, default=CHUNK_WINS_DEFAULT)
    ap.add_argument("--staging_dir", default=os.environ.get("STAGING_DIR", ""))
    ap.add_argument("--keep_staging", action="store_true")
    args = ap.parse_args()

    if not args.out_dir:
        args.out_dir = os.path.join(
            IFS_BASELINE_ROOT,
            f"ml_dataset_overlap_{args.source_tag}_12h_pm10_pm25_{args.feature_set}",
        )
    os.makedirs(args.out_dir, exist_ok=True)
    staging_dir = args.staging_dir or args.out_dir
    os.makedirs(staging_dir, exist_ok=True)

    if args.source_kind == "era5_feature_dir":
        ds_source = _load_era5_feature_dir(args.feature_dir, args.year)
        source_inputs = args.feature_dir
    else:
        paths = _paths_from_args(args.source_file, args.source_glob)
        ds_source = _load_station_nc(paths)
        source_inputs = paths
    ds_source = _rename_canonical(_normalize_station_dims(ds_source))
    native_source_features = sorted(str(v) for v in ds_source.data_vars)
    ds_source = _ensure_derived(ds_source)
    derived_source_features = sorted(set(str(v) for v in ds_source.data_vars) - set(native_source_features))
    target = _load_target(args.target_file)

    source_stations = ds_source["station_id"].values
    target_station_strings = pd.Index(np.asarray(target["station_id"].values).astype(str))
    source_station_strings = pd.Index(np.asarray(source_stations).astype(str))
    keep_source_pos = np.flatnonzero(source_station_strings.isin(target_station_strings))
    if len(keep_source_pos) == 0:
        raise RuntimeError("No common station_id between source and target visibility.")
    ds_source = ds_source.isel(station_id=keep_source_pos)
    source_stations = ds_source["station_id"].values

    times = pd.DatetimeIndex(pd.to_datetime(ds_source["time"].values))
    if args.allow_irregular_time_axis:
        time_axis_summary = summarize_time_axis(times, args.expected_time_step_hours)
    else:
        time_axis_summary = require_regular_time_axis(
            times, args.expected_time_step_hours, args.source_tag
        )
    require_lead = args.source_tag.lower().replace("-", "") in {"pangu2025", "pangu_2025"}
    if args.infer_pangu_lead12_23_from_valid_time:
        if not require_lead:
            raise ValueError("--infer_pangu_lead12_23_from_valid_time is valid only for Pangu-2025")
        if args.expected_lead_min_hours is None:
            args.expected_lead_min_hours = 12.0
        if args.expected_lead_max_hours is None:
            args.expected_lead_max_hours = 23.0
        if not (
            math.isclose(args.expected_lead_min_hours, 12.0, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(args.expected_lead_max_hours, 23.0, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "Pangu stitched valid-time inference is fixed to the documented 12..23 h range"
            )
    lead_summary = _forecast_lead_summary(
        ds_source,
        infer_pangu_lead12_23_from_valid_time=args.infer_pangu_lead12_23_from_valid_time,
    )
    if require_lead and not bool(lead_summary["available"]) and not args.allow_missing_forecast_lead:
        raise ValueError(
            "Pangu-2025 source has no per-time forecast lead metadata. "
            "Regenerate station interpolation with the current interpolator; filenames are not provenance."
        )
    if bool(lead_summary["available"]):
        lead_min = float(lead_summary["min_hours"])
        lead_max = float(lead_summary["max_hours"])
        if args.expected_lead_min_hours is not None and lead_min < args.expected_lead_min_hours - 1.0e-6:
            raise ValueError(
                f"{args.source_tag} lead minimum {lead_min:g} h is below required "
                f"{args.expected_lead_min_hours:g} h"
            )
        if args.expected_lead_max_hours is not None and lead_max > args.expected_lead_max_hours + 1.0e-6:
            raise ValueError(
                f"{args.source_tag} lead maximum {lead_max:g} h exceeds required "
                f"{args.expected_lead_max_hours:g} h"
            )
    lats, lons = _coords_for_source(ds_source, target, source_stations)
    y_arr = _aligned_target_visibility(target, times, source_stations, args.target_time_tolerance_minutes)

    available = describe_available_pmst_features(ds_source.data_vars)
    feature_vars = resolve_pmst_feature_set(args.feature_set, available)
    missing = [v for v in feature_vars if v not in available]
    if args.feature_set != "source_full" and missing:
        raise KeyError(
            f"{args.source_tag} cannot populate feature_set={args.feature_set}: missing {missing}; "
            f"available={sorted(available)}"
        )
    print(
        "[feature-set] {} populates PMST slots: {}".format(args.feature_set, ",".join(feature_vars)),
        flush=True,
    )

    data_veg = _open_dataset(args.veg_file)
    data_oro = _open_dataset(args.oro_file)
    X_stat = build_static_features(lats, lons, data_veg, data_oro, UNIQUE_VEG_IDS)
    data_veg.close()
    data_oro.close()

    pm10_da = load_pm10_dataarray(args.pm10_file, args.pm10_dir)
    pm25_da = load_pm25_dataarray(args.pm25_file, args.pm25_dir)
    if pm10_da is not None:
        pm10_da.load()
    if pm25_da is not None:
        pm25_da.load()

    nt, ns = len(times), len(source_stations)
    win, step = args.window, args.step
    n_wins = (nt - win) // step + 1
    if n_wins <= 0:
        raise ValueError(f"Time series too short: nt={nt} window={win}")
    n_samples = n_wins * ns
    dynamic_order = dynamic_feature_order_for_feature_set(args.feature_set, feature_vars)
    dyn_vars_count = dyn_vars_for_feature_set(args.feature_set, feature_vars)
    dyn_flat_dim = win * dyn_vars_count
    stat_dim = int(X_stat.shape[1])
    fog_fe_dim = compute_fog_features_pmst(
        np.zeros((1, win, dyn_vars_count), dtype=np.float32), win, dyn_vars_count, dynamic_order
    ).shape[1]
    fe_dim = fog_fe_dim + 4

    y_flat = y_arr[win - 1 :: step].reshape(-1).astype(np.float32)
    mask = ~np.isnan(y_flat) & (y_flat >= 0) & (y_flat <= MAX_VIS_THRESHOLD)
    m_t = np.repeat(times[win - 1 :: step].values, ns)
    m_s = np.tile(source_stations, n_wins)
    m_la = np.tile(lats, n_wins)
    m_lo = np.tile(lons, n_wins)

    st_dyn = os.path.join(staging_dir, "_staging_X_dyn_flat.npy")
    st_stat = os.path.join(staging_dir, "_staging_X_stat_flat.npy")
    st_fe = os.path.join(staging_dir, "_staging_fe_flat.npy")
    print(
        "[mem] source={} nt={} ns={} n_wins={} n_samples={} chunk_wins={} staging ~{:.1f} GiB".format(
            args.source_tag,
            nt,
            ns,
            n_wins,
            n_samples,
            args.chunk_wins,
            n_samples * (dyn_flat_dim + stat_dim + fe_dim) * 4 / (1024 ** 3),
        ),
        flush=True,
    )

    mm_dyn = np.lib.format.open_memmap(st_dyn, mode="w+", dtype=np.float32, shape=(n_samples, dyn_flat_dim))
    mm_stat = np.lib.format.open_memmap(st_stat, mode="w+", dtype=np.float32, shape=(n_samples, stat_dim))
    mm_fe = np.lib.format.open_memmap(st_fe, mode="w+", dtype=np.float32, shape=(n_samples, fe_dim))

    try:
        for w0 in tqdm(range(0, n_wins, args.chunk_wins), desc="time_chunks", unit="chunk"):
            w1 = min(w0 + args.chunk_wins, n_wins)
            t0 = w0 * step
            t1 = (w1 - 1) * step + win
            tlen = t1 - t0

            fields = _extract_fields(ds_source, t0, t1)
            X_met = scatter_overlap_fields(tlen, ns, fields, feature_vars)
            del fields
            gc.collect()

            times_chunk = pd.DatetimeIndex(times[t0:t1])
            zenith = calculate_zenith_angle(lats, lons, times_chunk.values)
            X_dyn_25 = np.concatenate([X_met, zenith], axis=-1).astype(np.float32)
            del X_met, zenith
            X_dyn_26 = append_pm10_channel(X_dyn_25, pm10_da, times_chunk, source_stations)
            del X_dyn_25
            X_chunk = append_pm25_channel(X_dyn_26, pm25_da, times_chunk, source_stations)
            X_chunk = select_dynamic_layout(X_chunk, args.feature_set, feature_vars)
            del X_dyn_26
            gc.collect()

            raw = sliding_window_view(X_chunk, win, axis=0)[::step]
            raw = raw.transpose(0, 1, 3, 2)
            X_samples = raw.reshape(-1, win, dyn_vars_count).astype(np.float32)
            del raw, X_chunk
            gc.collect()

            row_lo = w0 * ns
            row_hi = w1 * ns
            n_loc = row_hi - row_lo
            mm_dyn[row_lo:row_hi] = X_samples.reshape(n_loc, dyn_flat_dim)
            fe_part = compute_fog_features_pmst(X_samples, win, dyn_vars_count, dynamic_order)
            cyc = cyclical_time_features(pd.DatetimeIndex(m_t[row_lo:row_hi]))
            mm_fe[row_lo:row_hi] = np.concatenate([fe_part, cyc], axis=1).astype(np.float32)
            mm_stat[row_lo:row_hi] = np.tile(X_stat, (w1 - w0, 1)).astype(np.float32)
            del X_samples, fe_part, cyc
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
    finally:
        ds_source.close()
        target.close()
        if not args.keep_staging:
            for p in (st_dyn, st_stat, st_fe):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    cfg = {
        "dataset": f"station_source_overlap_{args.feature_set}_native_monthtail",
        "source_tag": args.source_tag,
        "source_kind": args.source_kind,
        "source_inputs": source_inputs,
        "native_source_features": native_source_features,
        "derived_source_features": derived_source_features,
        "year": args.year,
        "target_file": args.target_file,
        "target_time_tolerance_minutes": args.target_time_tolerance_minutes,
        "feature_set": args.feature_set,
        "overlap_vars": feature_vars,
        "available_pmst_features": [name for name in resolve_pmst_feature_set("source_full", available)],
        "zero_filled_pmst_features": [],
        "excluded_pmst_features": [name for name in FINAL_FEATURE_ORDER if name not in feature_vars],
        "dyn_layout": dynamic_layout_name(args.feature_set, feature_vars),
        "dynamic_feature_order": dynamic_order,
        "dyn_vars": int(dyn_vars_count),
        "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
        "canonical_dynamic_units": CANONICAL_DYNAMIC_UNITS,
        "fe_dim": fe_dim,
        "fog_fe_dim": int(fog_fe_dim),
        "window": args.window,
        "step": args.step,
        "split": "month_tail",
        "time_coordinate": "UTC",
        "source_time_axis": time_axis_summary,
        "source_forecast_lead": lead_summary,
        "pm_time_match": "nearest_90min_utc",
        "val_last_days": args.val_last_days,
        "test_last_days": args.test_last_days,
        "gap_hours": args.gap_hours,
        "max_vis_threshold": MAX_VIS_THRESHOLD,
        "chunk_wins": args.chunk_wins,
        "staging_dir": staging_dir,
        "low_mem_pipeline": True,
    }
    with open(os.path.join(args.out_dir, "dataset_build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote dataset to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
