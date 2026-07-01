#!/usr/bin/env python3
"""Inspect raw/CF-decoded PM grid metadata and sampled value distributions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PM_NAMES = ("pm10", "pm2p5", "pm25", "pm2_5", "PM2_5")


def pick_pm(ds: xr.Dataset) -> str:
    for name in PM_NAMES:
        if name in ds.data_vars:
            return name
    candidates = [name for name, da in ds.data_vars.items() if da.ndim >= 3]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"cannot identify PM variable; data_vars={list(ds.data_vars)}")


def sampled(da: xr.DataArray, target_values: int) -> np.ndarray:
    indexers = {}
    total = int(np.prod(da.shape, dtype=np.int64))
    stride = max(1, int(round((total / max(target_values, 1)) ** (1.0 / max(da.ndim, 1)))))
    for dim, size in da.sizes.items():
        indexers[dim] = slice(0, int(size), stride)
    values = np.asarray(da.isel(indexers).values, dtype=np.float64).reshape(-1)
    if values.size > target_values:
        values = values[:: max(1, values.size // target_values)][:target_values]
    return values


def stats(values: np.ndarray) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "finite": 0}
    q = np.quantile(finite, [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    return {
        "n": int(values.size),
        "finite": int(finite.size),
        "finite_fraction": float(finite.size / values.size),
        "q000_q001_q01_q50_q99_q999_q100": [float(x) for x in q],
        "negative_fraction": float(np.mean(finite < 0.0)),
    }


def time_summary(ds: xr.Dataset) -> dict[str, object]:
    for name in ("valid_time", "time", "forecast_reference_time"):
        if name not in ds:
            continue
        try:
            values = pd.to_datetime(np.asarray(ds[name].values).reshape(-1), errors="coerce")
            values = values[~pd.isna(values)]
            if len(values):
                return {"coordinate": name, "min": str(values.min()), "max": str(values.max())}
        except Exception:
            pass
    return {}


def inspect(path: Path, target_values: int) -> dict[str, object]:
    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as raw_ds:
        name = pick_pm(raw_ds)
        raw_da = raw_ds[name]
        attrs = {key: repr(value) for key, value in raw_da.attrs.items()}
        raw_values = sampled(raw_da, target_values)
        raw_info = {
            "variable": name,
            "dtype": str(raw_da.dtype),
            "dims": list(raw_da.dims),
            "shape": list(raw_da.shape),
            "attrs": attrs,
            "sample": stats(raw_values),
        }
    with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as decoded_ds:
        decoded_da = decoded_ds[name]
        decoded_values = sampled(decoded_da, target_values)
        units = re.sub(r"[\s_*]+", "", str(decoded_da.attrs.get("units", "")).lower())
        if "kg" in units and ("m-3" in units or "/m3" in units):
            ug_values = decoded_values * 1.0e9
            conversion = "decoded kg m-3 * 1e9"
        else:
            ug_values = decoded_values
            conversion = "no automatic conversion (units not recognized as kg m-3)"
        decoded_info = {
            "dtype": str(decoded_da.dtype),
            "attrs": {key: repr(value) for key, value in decoded_da.attrs.items()},
            "encoding": {
                key: repr(decoded_da.encoding.get(key))
                for key in ("dtype", "_FillValue", "missing_value", "scale_factor", "add_offset")
                if key in decoded_da.encoding
            },
            "time": time_summary(decoded_ds),
            "sample_native": stats(decoded_values),
            "ugm3_conversion": conversion,
            "sample_ugm3": stats(ug_values),
            "outside_0_10000_fraction_of_finite": float(
                np.mean((ug_values[np.isfinite(ug_values)] < 0.0) | (ug_values[np.isfinite(ug_values)] > 10000.0))
            )
            if np.isfinite(ug_values).any()
            else None,
        }
    return {"path": str(path), "raw": raw_info, "decoded": decoded_info}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample-values", type=int, default=250000)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(inspect(path, args.sample_values), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
