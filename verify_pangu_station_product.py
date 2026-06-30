#!/usr/bin/env python3
"""Fail-fast verification for a corrected Pangu station product.

The check is intentionally independent of model training.  It proves that the
new interpolation uses the canonical target stations, differs from the legacy
product, has a regular valid-time axis, and retains forecast-lead metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def open_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def coord_name(obj, candidates: Sequence[str]) -> str:
    names = list(obj.coords) + list(obj.dims)
    lower = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Cannot infer coordinate {tuple(candidates)}; available={names}")


def station_keys(values: Iterable[object]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 2 and raw.dtype.kind in {"S", "U"}:
        rows = []
        for row in raw:
            parts = [
                item.decode("utf-8", errors="strict") if isinstance(item, bytes) else str(item)
                for item in row
            ]
            rows.append("".join(parts).strip())
        raw = np.asarray(rows, dtype=object)
    else:
        raw = raw.reshape(-1)
    decoded = [
        value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        for value in raw.tolist()
    ]
    series = pd.Series(decoded, dtype="object")
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .to_numpy(dtype=str)
    )


def station_table(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    station = coord_name(ds, ("station_id", "num_station", "station", "id"))
    lat = coord_name(ds, ("lat", "latitude"))
    lon = coord_name(ds, ("lon", "longitude"))
    keys = station_keys(ds[station].values)
    lats = np.asarray(ds[lat].values, dtype=np.float64).reshape(-1)
    lons = np.asarray(ds[lon].values, dtype=np.float64).reshape(-1)
    if len(keys) != len(lats) or len(keys) != len(lons):
        raise ValueError(
            f"station/lat/lon lengths disagree: {len(keys)}/{len(lats)}/{len(lons)}"
        )
    if pd.Index(keys).duplicated().any():
        dup = pd.Index(keys)[pd.Index(keys).duplicated()].unique()[:5].tolist()
        raise ValueError(f"duplicate station identifiers: {dup}")
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("station latitude/longitude contains non-finite values")
    return keys, lats, lons, station


def aligned_positions(reference: np.ndarray, candidate: np.ndarray, label: str) -> np.ndarray:
    ref = pd.Index(reference)
    cand = pd.Index(candidate)
    missing = ref.difference(cand)
    extra = cand.difference(ref)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} station set differs: missing={missing[:5].tolist()} "
            f"(n={len(missing)}), extra={extra[:5].tolist()} (n={len(extra)})"
        )
    return cand.get_indexer(ref)


def common_station_positions(
    reference: np.ndarray, candidate: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reference-ordered common keys/positions plus set differences."""
    ref = pd.Index(reference)
    cand = pd.Index(candidate)
    common_mask = ref.isin(cand)
    common = ref[common_mask]
    if len(common) == 0:
        raise ValueError("old and canonical products have no common station identifiers")
    return (
        common.to_numpy(dtype=str),
        np.flatnonzero(common_mask),
        cand.get_indexer(common),
        ref.difference(cand).to_numpy(dtype=str),
        cand.difference(ref).to_numpy(dtype=str),
    )


def lead_summary(
    ds: xr.Dataset,
    time_name: str,
    infer_stitched_lead12_23: bool,
) -> Dict[str, object]:
    valid = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
    lead = None
    provenance = ""
    if "forecast_lead_hours" in ds.coords or "forecast_lead_hours" in ds.data_vars:
        lead = np.asarray(ds["forecast_lead_hours"].values, dtype=np.float64).reshape(-1)
        if lead.size == 1 and len(valid) > 1:
            lead = np.repeat(lead, len(valid))
        if lead.size != len(valid):
            raise ValueError(
                f"forecast_lead_hours length={lead.size}, expected time length={len(valid)}"
            )
        provenance = "forecast_lead_hours coordinate"
    if "init_time" in ds.coords or "init_time" in ds.data_vars:
        init = np.asarray(ds["init_time"].values).reshape(-1)
        if init.size == 1 and len(valid) > 1:
            init = np.repeat(init, len(valid))
        if init.size != len(valid):
            raise ValueError(f"init_time length={init.size}, expected time length={len(valid)}")
        derived = np.asarray(
            (valid.values - pd.DatetimeIndex(pd.to_datetime(init)).values)
            / np.timedelta64(1, "h"),
            dtype=np.float64,
        )
        if lead is not None and not np.allclose(lead, derived, rtol=0.0, atol=1e-6):
            raise ValueError("forecast_lead_hours disagrees with valid_time-init_time")
        lead = derived
        provenance = "valid_time minus init_time"
    if lead is None and "forecast_lead_hours" in ds.attrs:
        lead = np.full(len(valid), float(ds.attrs["forecast_lead_hours"]), dtype=np.float64)
        provenance = "global forecast_lead_hours attribute"
    if lead is None and infer_stitched_lead12_23:
        # The legacy hourly product stitches 00/12 UTC initializations: valid
        # hours 12..23 use the same-day 00 UTC cycle, while 00..11 use the
        # previous-day 12 UTC cycle.  This reconstructs leads 12..23 exactly.
        hours = valid.hour.to_numpy(dtype=np.float64)
        lead = np.where(hours < 12.0, hours + 12.0, hours)
        provenance = "explicit stitched 00/12 UTC schedule reconstructed from valid_time hour"
    if lead is None:
        raise ValueError(
            "new product lacks init_time/forecast_lead_hours; either regenerate it with the "
            "current interpolator or explicitly enable the legacy lead12_23 stitched schedule"
        )
    if not np.isfinite(lead).all():
        raise ValueError("forecast lead contains non-finite values")
    unique, counts = np.unique(np.round(lead, 6), return_counts=True)
    return {
        "provenance": provenance,
        "min_hours": float(np.min(lead)),
        "max_hours": float(np.max(lead)),
        "unique_hours": [float(v) for v in unique],
        "counts": [int(v) for v in counts],
    }


def time_summary(ds: xr.Dataset, year: int) -> Tuple[str, Dict[str, object]]:
    name = coord_name(ds, ("time", "valid_time", "Time"))
    values = pd.DatetimeIndex(pd.to_datetime(ds[name].values))
    if len(values) == 0:
        raise ValueError("new product has no valid times")
    if values.has_duplicates:
        raise ValueError(f"new product has {int(values.duplicated().sum())} duplicate valid times")
    ordered = values.sort_values()
    delta = np.diff(ordered.values) / np.timedelta64(1, "h")
    if len(delta) and not np.allclose(delta, 1.0, rtol=0.0, atol=1e-6):
        unique, counts = np.unique(delta, return_counts=True)
        raise ValueError(
            "new product is not hourly; delta histogram="
            + str({float(k): int(v) for k, v in zip(unique, counts)})
        )
    years = sorted({int(v) for v in ordered.year})
    allowed = {int(year), int(year) + 1}
    if not set(years).issubset(allowed):
        raise ValueError(f"unexpected valid-time years={years}; expected subset of {sorted(allowed)}")
    return name, {
        "count": int(len(ordered)),
        "min": str(ordered.min()),
        "max": str(ordered.max()),
        "years": years,
        "regular_hourly": True,
    }


def sampled_data_differences(
    old: xr.Dataset,
    new: xr.Dataset,
    old_station_name: str,
    new_station_name: str,
    old_station_pos: np.ndarray,
    new_station_pos: np.ndarray,
    max_times: int,
    max_stations: int,
) -> Dict[str, object]:
    old_time = coord_name(old, ("time", "valid_time", "Time"))
    new_time = coord_name(new, ("time", "valid_time", "Time"))
    old_times = pd.DatetimeIndex(pd.to_datetime(old[old_time].values))
    new_times = pd.DatetimeIndex(pd.to_datetime(new[new_time].values))
    common = old_times.intersection(new_times).sort_values()
    if len(common) == 0:
        raise ValueError("old and new products have no common valid times")
    time_pick = np.linspace(0, len(common) - 1, min(max_times, len(common)), dtype=int)
    chosen_times = common[time_pick]
    old_t = old_times.get_indexer(chosen_times)
    new_t = new_times.get_indexer(chosen_times)
    station_pick = np.linspace(
        0, len(old_station_pos) - 1, min(max_stations, len(old_station_pos)), dtype=int
    )
    old_s = old_station_pos[station_pick]
    new_s = new_station_pos[station_pick]

    rows = []
    for name in sorted(set(old.data_vars).intersection(new.data_vars)):
        oda = old[name]
        nda = new[name]
        if old_time not in oda.dims or old_station_name not in oda.dims:
            continue
        if new_time not in nda.dims or new_station_name not in nda.dims:
            continue
        if not np.issubdtype(oda.dtype, np.number) or not np.issubdtype(nda.dtype, np.number):
            continue
        oa = np.asarray(
            oda.isel({old_time: old_t, old_station_name: old_s})
            .transpose(old_time, old_station_name)
            .values,
            dtype=np.float64,
        )
        na = np.asarray(
            nda.isel({new_time: new_t, new_station_name: new_s})
            .transpose(new_time, new_station_name)
            .values,
            dtype=np.float64,
        )
        finite = np.isfinite(oa) & np.isfinite(na)
        if not finite.any():
            continue
        diff = np.abs(oa[finite] - na[finite])
        rows.append(
            {
                "variable": name,
                "values_compared": int(diff.size),
                "changed_values": int((diff > 1e-7).sum()),
                "max_abs_diff": float(np.max(diff)),
                "mean_abs_diff": float(np.mean(diff)),
            }
        )
    if not rows:
        raise ValueError("no common numeric time-by-station variables could be compared")
    return {
        "sampled_times": int(len(chosen_times)),
        "sampled_stations": int(len(station_pick)),
        "variables": rows,
        "variables_with_changes": int(sum(row["changed_values"] > 0 for row in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-file", required=True)
    ap.add_argument("--new-file", required=True)
    ap.add_argument("--target-file", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--expected-lead-min-hours", type=float, required=True)
    ap.add_argument("--expected-lead-max-hours", type=float, required=True)
    ap.add_argument("--infer-stitched-lead12-23", action="store_true")
    ap.add_argument("--coord-atol-degrees", type=float, default=1e-6)
    ap.add_argument("--max-sample-times", type=int, default=24)
    ap.add_argument("--max-sample-stations", type=int, default=128)
    args = ap.parse_args()

    for label, value in (("old", args.old_file), ("new", args.new_file), ("target", args.target_file)):
        path = Path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label} file is missing or empty: {path}")
    if args.expected_lead_min_hours > args.expected_lead_max_hours:
        raise ValueError("expected lead minimum exceeds expected lead maximum")

    old = open_dataset(args.old_file)
    new = open_dataset(args.new_file)
    target = open_dataset(args.target_file)
    try:
        old_keys, old_lat, old_lon, old_station = station_table(old)
        new_keys, new_lat, new_lon, new_station = station_table(new)
        target_keys, target_lat, target_lon, _ = station_table(target)

        new_to_target = aligned_positions(target_keys, new_keys, "new versus canonical target")
        (
            common_keys,
            target_common_pos,
            old_common_pos,
            old_missing_from_target,
            old_extra_vs_target,
        ) = common_station_positions(target_keys, old_keys)
        new_index = pd.Index(new_keys)
        new_common_pos = new_index.get_indexer(common_keys)
        new_lat_aligned = new_lat[new_to_target]
        new_lon_aligned = new_lon[new_to_target]
        old_lat_common = old_lat[old_common_pos]
        old_lon_common = old_lon[old_common_pos]
        new_lat_common = new_lat[new_common_pos]
        new_lon_common = new_lon[new_common_pos]

        canonical_lat_error = np.abs(new_lat_aligned - target_lat)
        canonical_lon_error = np.abs(new_lon_aligned - target_lon)
        if (
            float(np.max(canonical_lat_error)) > args.coord_atol_degrees
            or float(np.max(canonical_lon_error)) > args.coord_atol_degrees
        ):
            raise ValueError(
                "new product does not use canonical station coordinates: "
                f"max_lat_error={float(np.max(canonical_lat_error)):.9g}, "
                f"max_lon_error={float(np.max(canonical_lon_error)):.9g} degrees"
            )

        old_new_distance = np.hypot(
            old_lat_common - new_lat_common, old_lon_common - new_lon_common
        )
        changed_coords = old_new_distance > args.coord_atol_degrees
        if not changed_coords.any():
            raise ValueError(
                "old and new station coordinates are identical within tolerance; the uploaded "
                "file does not demonstrate the canonical-coordinate correction"
            )

        new_time, time_info = time_summary(new, args.year)
        lead = lead_summary(new, new_time, args.infer_stitched_lead12_23)
        if not math.isclose(
            float(lead["min_hours"]), args.expected_lead_min_hours, rel_tol=0.0, abs_tol=1e-6
        ) or not math.isclose(
            float(lead["max_hours"]), args.expected_lead_max_hours, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"new product lead range={lead['min_hours']}..{lead['max_hours']} h; expected "
                f"{args.expected_lead_min_hours}..{args.expected_lead_max_hours} h"
            )

        changed_common_pos = np.flatnonzero(changed_coords)
        differences = sampled_data_differences(
            old,
            new,
            old_station,
            new_station,
            old_common_pos[changed_common_pos],
            new_common_pos[changed_common_pos],
            args.max_sample_times,
            args.max_sample_stations,
        )
        if int(differences["variables_with_changes"]) == 0:
            raise ValueError(
                "sampled meteorological values are identical despite changed station coordinates"
            )

        report = {
            "status": "passed",
            "old_file": str(Path(args.old_file).resolve()),
            "new_file": str(Path(args.new_file).resolve()),
            "target_file": str(Path(args.target_file).resolve()),
            "station_count": int(len(target_keys)),
            "old_station_count": int(len(old_keys)),
            "old_new_common_station_count": int(len(common_keys)),
            "old_missing_canonical_station_count": int(len(old_missing_from_target)),
            "old_missing_canonical_stations": old_missing_from_target.tolist(),
            "old_extra_station_count": int(len(old_extra_vs_target)),
            "old_extra_stations": old_extra_vs_target.tolist(),
            "canonical_coordinate_tolerance_degrees": args.coord_atol_degrees,
            "new_vs_target_max_lat_error_degrees": float(np.max(canonical_lat_error)),
            "new_vs_target_max_lon_error_degrees": float(np.max(canonical_lon_error)),
            "old_vs_new_changed_station_count": int(changed_coords.sum()),
            "old_vs_new_max_coordinate_distance_degrees": float(np.max(old_new_distance)),
            "time_axis": time_info,
            "forecast_lead": lead,
            "sampled_data_difference": differences,
        }
    finally:
        old.close()
        new.close()
        target.close()

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] verification report: {out}", flush=True)


if __name__ == "__main__":
    main()
