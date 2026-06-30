#!/usr/bin/env python3
"""Audit Q1000 provenance from Pangu grids through station/model datasets.

This is a lineage audit, not another forecast-score script.  It verifies actual
valid time, initialization time, forecast lead, cadence, Q units, DP derivation,
and the terrain sensitivity of 1000-hPa fields.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from pmst_overlap_common import summarize_time_axis


DEFAULT_BASE = "/public/home/putianshu/vis_mlp"
QUANTILES = (0.5, 0.9, 0.95, 0.99)


def open_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)


def jsonable_attrs(attrs: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def find_name(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    available = {str(v).upper().replace("-", "_"): str(v) for v in ds.data_vars}
    for candidate in candidates:
        key = candidate.upper().replace("-", "_")
        if key in available:
            return available[key]
    return None


def sample_dataarray(da: xr.DataArray, max_values: int) -> np.ndarray:
    work = da
    for dim in da.dims:
        size = int(da.sizes[dim])
        if dim.lower() in {"time", "valid_time"} and size > 12:
            work = work.isel({dim: np.linspace(0, size - 1, 12, dtype=np.int64)})
    values = np.asarray(work.values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) > max_values:
        values = values[np.linspace(0, len(values) - 1, max_values, dtype=np.int64)]
    return values


def q_to_gkg(values: np.ndarray) -> Tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return values, "unknown"
    p99 = float(np.nanpercentile(np.abs(finite), 99))
    if p99 < 0.2:
        return values * 1000.0, "kg kg-1 inferred; converted to g kg-1"
    return values, "g kg-1 inferred"


def dewpoint_from_q_k(q_gkg: np.ndarray, pressure_hpa: float) -> np.ndarray:
    q = np.clip(np.asarray(q_gkg, dtype=np.float64) / 1000.0, 1.0e-8, 0.08)
    e = q * float(pressure_hpa) / np.maximum(0.622 + 0.378 * q, 1.0e-8)
    ratio = np.log(np.maximum(e, 1.0e-6) / 6.112)
    return 243.5 * ratio / np.maximum(17.67 - ratio, 1.0e-6) + 273.15


def quantile_rows(label: str, stage: str, feature: str, values: np.ndarray, unit: str) -> List[Dict[str, object]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rows: List[Dict[str, object]] = []
    if not len(finite):
        return rows
    for quantile in QUANTILES:
        rows.append(
            {
                "source": label,
                "stage": stage,
                "feature": feature,
                "quantile": quantile,
                "value": float(np.quantile(finite, quantile)),
                "unit": unit,
                "n_sampled": int(len(finite)),
            }
        )
    return rows


def lead_evidence(
    ds: xr.Dataset,
    time_name: str,
    infer_pangu_lead12_23_from_valid_time: bool = False,
) -> Dict[str, object]:
    valid = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
    lead: Optional[np.ndarray] = None
    provenance = "missing"
    if "forecast_lead_hours" in ds.coords or "forecast_lead_hours" in ds.data_vars:
        raw = np.asarray(ds["forecast_lead_hours"].values, dtype=np.float64).reshape(-1)
        if raw.size == 1 and len(valid) > 1:
            raw = np.repeat(raw, len(valid))
        if raw.size == len(valid):
            lead = raw
            provenance = "forecast_lead_hours coordinate"
    if "init_time" in ds.coords or "init_time" in ds.data_vars:
        raw_init = np.asarray(ds["init_time"].values).reshape(-1)
        if raw_init.size == 1 and len(valid) > 1:
            raw_init = np.repeat(raw_init, len(valid))
        if raw_init.size == len(valid):
            derived = np.asarray(
                (valid.values - pd.DatetimeIndex(pd.to_datetime(raw_init)).values) / np.timedelta64(1, "h"),
                dtype=np.float64,
            )
            if lead is not None and not np.allclose(lead, derived, rtol=0.0, atol=1.0e-6):
                return {"available": True, "consistent": False, "reason": "lead coordinate disagrees with init_time"}
            lead = derived
            provenance = "valid_time minus init_time"
    if lead is None and "forecast_lead_hours" in ds.attrs:
        lead = np.full(len(valid), float(ds.attrs["forecast_lead_hours"]), dtype=np.float64)
        provenance = "global attribute"
    if lead is None and infer_pangu_lead12_23_from_valid_time:
        hours = valid.hour.to_numpy(dtype=np.float64)
        lead = np.where(hours < 12.0, hours + 12.0, hours)
        provenance = "explicit stitched 00/12 UTC schedule reconstructed from valid_time hour"
    if lead is None:
        return {"available": False, "consistent": False, "provenance": provenance}
    finite = lead[np.isfinite(lead)]
    return {
        "available": bool(len(finite)),
        "consistent": bool(len(finite) == len(lead)),
        "provenance": provenance,
        "min_hours": float(np.min(finite)) if len(finite) else math.nan,
        "max_hours": float(np.max(finite)) if len(finite) else math.nan,
        "unique_hours": [float(v) for v in np.unique(np.round(finite, 6)).tolist()],
    }


def inspect_nc(
    label: str,
    stage: str,
    path: str,
    max_values: int,
    expected_hourly: bool,
    expected_lead_min: Optional[float],
    expected_lead_max: Optional[float],
    infer_pangu_lead12_23_from_valid_time: bool,
    issues: List[Dict[str, str]],
    quantiles: List[Dict[str, object]],
) -> Dict[str, object]:
    ds = open_dataset(path)
    try:
        time_name = "time" if "time" in ds.coords else "valid_time" if "valid_time" in ds.coords else ""
        report: Dict[str, object] = {
            "path": str(Path(path).resolve()),
            "sizes": {str(k): int(v) for k, v in ds.sizes.items()},
            "data_vars": [str(v) for v in ds.data_vars],
            "attrs": jsonable_attrs(ds.attrs),
        }
        if time_name:
            time_summary = summarize_time_axis(ds[time_name].values, 1.0)
            report["time_axis"] = time_summary
            if expected_hourly and not bool(time_summary["regular"]):
                issues.append({"severity": "error", "stage": stage, "message": "valid-time axis is not hourly"})
            lead = lead_evidence(
                ds,
                time_name,
                infer_pangu_lead12_23_from_valid_time=infer_pangu_lead12_23_from_valid_time,
            )
            report["forecast_lead"] = lead
            if expected_lead_min is not None or expected_lead_max is not None:
                if expected_lead_min is None or expected_lead_max is None:
                    raise ValueError("both expected lead bounds are required")
                if not bool(lead.get("available")):
                    issues.append({"severity": "error", "stage": stage, "message": "forecast lead metadata is missing"})
                elif not (
                    math.isclose(float(lead.get("min_hours", math.nan)), expected_lead_min, abs_tol=1.0e-6)
                    and math.isclose(float(lead.get("max_hours", math.nan)), expected_lead_max, abs_tol=1.0e-6)
                ):
                    issues.append(
                        {
                            "severity": "error",
                            "stage": stage,
                            "message": (
                                f"actual lead range is not {expected_lead_min:g}.."
                                f"{expected_lead_max:g} h: {lead}"
                            ),
                        }
                    )

        q_name = find_name(ds, ("Q_1000", "q1000"))
        dp_name = find_name(ds, ("DP_1000", "dp1000"))
        if q_name:
            q_raw = sample_dataarray(ds[q_name], max_values)
            q_gkg, inferred = q_to_gkg(q_raw)
            report["q1000"] = {
                "variable": q_name,
                "declared_units": str(ds[q_name].attrs.get("units", "")),
                "unit_inference": inferred,
                "sample_min_gkg": float(np.min(q_gkg)) if len(q_gkg) else math.nan,
                "sample_max_gkg": float(np.max(q_gkg)) if len(q_gkg) else math.nan,
            }
            quantiles.extend(quantile_rows(label, stage, "Q_1000", q_gkg, "g kg-1"))
            if len(q_gkg) and (float(np.min(q_gkg)) < -0.01 or float(np.max(q_gkg)) > 80.0):
                issues.append({"severity": "error", "stage": stage, "message": "Q1000 sample is outside broad physical bounds"})
            if dp_name:
                dp = sample_dataarray(ds[dp_name], max_values)
                if len(dp) and np.nanmedian(dp) < 100.0:
                    dp = dp + 273.15
                n = min(len(q_gkg), len(dp))
                if n:
                    expected_dp = dewpoint_from_q_k(q_gkg[:n], 1000.0)
                    report["dp1000_vs_q1000_sample_mae_k"] = float(np.mean(np.abs(dp[:n] - expected_dp)))
                quantiles.extend(quantile_rows(label, stage, "DP_1000", dp, "K"))
        else:
            issues.append({"severity": "error", "stage": stage, "message": "Q1000 variable is missing"})
        return report
    finally:
        ds.close()


def inspect_pangu_grids(
    pattern: str,
    max_files: int,
    max_values: int,
    expected_lead_min: float,
    expected_lead_max: float,
    infer_pangu_lead12_23_from_valid_time: bool,
    issues: List[Dict[str, str]],
    quantiles: List[Dict[str, object]],
) -> Dict[str, object]:
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        issues.append({"severity": "error", "stage": "pangu_grid", "message": f"no files match {pattern}"})
        return {"glob": pattern, "files": []}
    files = list(all_files)
    if max_files > 0 and len(files) > max_files:
        chosen = np.linspace(0, len(files) - 1, max_files, dtype=np.int64)
        files = [files[int(i)] for i in chosen]
    reports = []
    all_times: List[np.ndarray] = []
    for path in files:
        report = inspect_nc(
            "Pangu",
            "pangu_grid",
            path,
            max_values,
            False,
            expected_lead_min,
            expected_lead_max,
            infer_pangu_lead12_23_from_valid_time,
            issues,
            quantiles,
        )
        reports.append(report)
        ds = open_dataset(path)
        try:
            time_name = "time" if "time" in ds.coords else "valid_time"
            token = re.search(r"lead(\d+)h", Path(path).name)
            lead = report.get("forecast_lead", {})
            if (
                token
                and bool(lead.get("available"))
                and math.isclose(float(lead["min_hours"]), float(lead["max_hours"]), abs_tol=1.0e-6)
            ):
                named = float(token.group(1))
                if not math.isclose(named, float(lead["min_hours"]), abs_tol=1.0e-6):
                    issues.append(
                        {"severity": "error", "stage": "pangu_grid", "message": f"filename lead {named:g} h disagrees with metadata in {path}"}
                    )
        finally:
            ds.close()
    for path in all_files:
        ds = open_dataset(path)
        try:
            time_name = "time" if "time" in ds.coords else "valid_time"
            all_times.append(pd.DatetimeIndex(pd.to_datetime(ds[time_name].values)).values)
        finally:
            ds.close()
    concatenated = np.concatenate(all_times)
    duplicate_count = int(len(concatenated) - len(np.unique(concatenated)))
    if duplicate_count:
        issues.append(
            {"severity": "error", "stage": "pangu_grid", "message": f"{duplicate_count} duplicate valid times occur across Pangu files"}
        )
    combined = np.unique(concatenated)
    cadence = summarize_time_axis(combined, 1.0)
    if not bool(cadence["regular"]):
        issues.append(
            {"severity": "error", "stage": "pangu_grid", "message": f"combined Pangu valid times are not hourly: {cadence}"}
        )
    return {
        "glob": pattern,
        "file_count": int(len(all_files)),
        "duplicate_valid_time_count": duplicate_count,
        "files_inspected": reports,
        "combined_time_axis": cadence,
    }


def inspect_model_dataset(
    label: str,
    path: str,
    max_rows: int,
    issues: List[Dict[str, str]],
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    root = Path(path)
    cfg_path = root / "dataset_build_config.json"
    if not cfg_path.is_file():
        issues.append({"severity": "error", "stage": f"{label}_dataset", "message": f"missing {cfg_path}"})
        return {"path": str(root), "available": False}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    order = [str(v) for v in cfg.get("dynamic_feature_order", [])]
    if "Q_1000" not in order:
        issues.append({"severity": "error", "stage": f"{label}_dataset", "message": "Q_1000 is absent from dynamic order"})
        return {"path": str(root), "available": True, "config": cfg}
    x_path = root / "X_test.npy"
    if not x_path.is_file():
        issues.append({"severity": "error", "stage": f"{label}_dataset", "message": f"missing {x_path}"})
        return {"path": str(root), "available": True, "config": cfg}
    x = np.load(x_path, mmap_mode="r")
    n = int(x.shape[0])
    idx = np.arange(n, dtype=np.int64)
    if max_rows > 0 and n > max_rows:
        idx = np.linspace(0, n - 1, max_rows, dtype=np.int64)
    window = int(cfg.get("window", 12))
    dyn = len(order)
    q_col = (window - 1) * dyn + order.index("Q_1000")
    q_gkg, inferred = q_to_gkg(np.asarray(x[idx, q_col], dtype=np.float64))
    elevation_col = window * dyn + 2
    elevation = np.asarray(x[idx, elevation_col], dtype=np.float64) if x.shape[1] > elevation_col else np.full(len(idx), np.nan)
    bands = (("elev_lt100m", -np.inf, 100.0), ("elev_100_500m", 100.0, 500.0), ("elev_ge500m", 500.0, np.inf))
    for band, lo, hi in bands:
        mask = np.isfinite(q_gkg) & (elevation >= lo) & (elevation < hi)
        if not mask.any():
            continue
        for quantile in QUANTILES:
            rows.append(
                {
                    "source": label,
                    "elevation_band": band,
                    "quantile": quantile,
                    "q1000_gkg": float(np.quantile(q_gkg[mask], quantile)),
                    "n_sampled": int(mask.sum()),
                }
            )
    return {
        "path": str(root.resolve()),
        "available": True,
        "shape": [int(v) for v in x.shape],
        "q1000_unit_inference": inferred,
        "source_time_axis": cfg.get("source_time_axis"),
        "source_forecast_lead": cfg.get("source_forecast_lead"),
        "native_source_features": cfg.get("native_source_features"),
        "derived_source_features": cfg.get("derived_source_features"),
    }


def load_dataset_q_frame(label: str, path: str) -> pd.DataFrame:
    root = Path(path)
    cfg = json.loads((root / "dataset_build_config.json").read_text(encoding="utf-8"))
    order = [str(v) for v in cfg["dynamic_feature_order"]]
    window = int(cfg.get("window", 12))
    dyn = len(order)
    x = np.load(root / "X_test.npy", mmap_mode="r")
    meta = pd.read_csv(root / "meta_test.csv")
    if len(meta) != len(x):
        raise ValueError(f"{label}: meta_test rows={len(meta)} but X_test rows={len(x)}")
    q_col = (window - 1) * dyn + order.index("Q_1000")
    q_gkg, _ = q_to_gkg(np.asarray(x[:, q_col], dtype=np.float64))
    elevation_col = window * dyn + 2
    elevation = np.asarray(x[:, elevation_col], dtype=np.float64)
    time = pd.to_datetime(meta["time"], errors="raise", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    station = meta["station_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    frame = pd.DataFrame({"time": time, "station": station, "q_gkg": q_gkg, "elevation_m": elevation})
    if frame[["time", "station"]].duplicated().any():
        raise ValueError(f"{label}: duplicate (valid_time, station_id) rows in test metadata")
    frame.index = pd.MultiIndex.from_frame(frame[["time", "station"]])
    return frame


def paired_complete_case_metrics(
    dataset_specs: Mapping[str, str],
    max_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frames = {label: load_dataset_q_frame(label, path) for label, path in dataset_specs.items() if path}
    if "ERA5" not in frames or len(frames) < 2:
        return pd.DataFrame(), {"available": False, "reason": "ERA5 and at least one forecast dataset are required"}
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index, sort=False)
    assert common is not None
    common = common.sort_values()
    if max_rows > 0 and len(common) > max_rows:
        common = common[np.linspace(0, len(common) - 1, max_rows, dtype=np.int64)]
    aligned = {label: frame.reindex(common) for label, frame in frames.items()}
    finite = np.ones(len(common), dtype=bool)
    for frame in aligned.values():
        finite &= np.isfinite(frame["q_gkg"].to_numpy(dtype=np.float64))
    common_finite = common[finite]
    aligned = {label: frame.reindex(common_finite) for label, frame in frames.items()}
    ref = aligned["ERA5"]["q_gkg"].to_numpy(dtype=np.float64)
    elevation = aligned["ERA5"]["elevation_m"].to_numpy(dtype=np.float64)
    bands = (
        ("all", -np.inf, np.inf),
        ("elev_lt100m", -np.inf, 100.0),
        ("elev_100_500m", 100.0, 500.0),
        ("elev_ge500m", 500.0, np.inf),
    )
    rows: List[Dict[str, object]] = []
    for label, frame in aligned.items():
        if label == "ERA5":
            continue
        source = frame["q_gkg"].to_numpy(dtype=np.float64)
        for band, lo, hi in bands:
            mask = (elevation >= lo) & (elevation < hi)
            if not mask.any():
                continue
            error = source[mask] - ref[mask]
            rows.append(
                {
                    "source": label,
                    "reference": "ERA5 reference analysis",
                    "elevation_band": band,
                    "n_complete_case": int(mask.sum()),
                    "bias_gkg": float(np.mean(error)),
                    "mae_gkg": float(np.mean(np.abs(error))),
                    "rmse_gkg": float(np.sqrt(np.mean(error * error))),
                    "corr": float(np.corrcoef(source[mask], ref[mask])[0, 1]),
                    "source_p95_gkg": float(np.quantile(source[mask], 0.95)),
                    "reference_p95_gkg": float(np.quantile(ref[mask], 0.95)),
                    "p95_delta_gkg": float(np.quantile(source[mask], 0.95) - np.quantile(ref[mask], 0.95)),
                }
            )
    coverage = {
        "available": True,
        "indexed_common_rows": int(len(common)),
        "finite_complete_case_rows": int(len(common_finite)),
        "finite_complete_case_fraction": float(len(common_finite) / max(len(common), 1)),
        "sources": {label: int(len(frame)) for label, frame in frames.items()},
    }
    return pd.DataFrame(rows), coverage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-grid-glob", default=f"{DEFAULT_BASE}/pangu_2025_china_chunks/pangu_china_2025*_lead*.nc")
    ap.add_argument(
        "--pangu-station-file",
        default=f"{DEFAULT_BASE}/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h_canonical.nc",
    )
    ap.add_argument("--tianji-file", default=f"{DEFAULT_BASE}/tianji_auto_station/merged_final_all_vars.nc")
    ap.add_argument("--pangu-dataset-dir", default="")
    ap.add_argument("--tianji-dataset-dir", default="")
    ap.add_argument("--ifs-dataset-dir", default="")
    ap.add_argument("--era5-dataset-dir", default="")
    ap.add_argument("--out-dir", default=f"{DEFAULT_BASE}/paper_eval_results_pm10_pm25_journal/q1000_lineage_audit")
    ap.add_argument("--max-grid-files", type=int, default=24)
    ap.add_argument("--max-values", type=int, default=200000)
    ap.add_argument("--max-dataset-rows", type=int, default=300000)
    ap.add_argument("--paired-max-rows", type=int, default=0, help="0 uses every common test row.")
    ap.add_argument("--expected-pangu-lead-min-hours", type=float, default=12.0)
    ap.add_argument("--expected-pangu-lead-max-hours", type=float, default=23.0)
    ap.add_argument("--infer-pangu-lead12-23-from-valid-time", action="store_true")
    ap.add_argument("--skip-pangu-grid", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when any error-level issue is found.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: List[Dict[str, str]] = []
    quantiles: List[Dict[str, object]] = []
    elevation_rows: List[Dict[str, object]] = []
    if args.expected_pangu_lead_min_hours > args.expected_pangu_lead_max_hours:
        raise ValueError("expected Pangu lead minimum exceeds maximum")
    report: Dict[str, object] = {"method": "metadata-backed Q1000 lineage audit"}
    if not args.skip_pangu_grid:
        report["pangu_grid"] = inspect_pangu_grids(
            args.pangu_grid_glob,
            args.max_grid_files,
            args.max_values,
            args.expected_pangu_lead_min_hours,
            args.expected_pangu_lead_max_hours,
            args.infer_pangu_lead12_23_from_valid_time,
            issues,
            quantiles,
        )
    else:
        report["pangu_grid"] = {"skipped": True}
    if os.path.isfile(args.pangu_station_file):
        report["pangu_station"] = inspect_nc(
            "Pangu",
            "pangu_station",
            args.pangu_station_file,
            args.max_values,
            True,
            args.expected_pangu_lead_min_hours,
            args.expected_pangu_lead_max_hours,
            args.infer_pangu_lead12_23_from_valid_time,
            issues,
            quantiles,
        )
    else:
        issues.append({"severity": "error", "stage": "pangu_station", "message": f"missing {args.pangu_station_file}"})
    if os.path.isfile(args.tianji_file):
        report["tianji_station"] = inspect_nc(
            "Tianji",
            "tianji_station",
            args.tianji_file,
            args.max_values,
            True,
            None,
            None,
            False,
            issues,
            quantiles,
        )
    else:
        issues.append({"severity": "error", "stage": "tianji_station", "message": f"missing {args.tianji_file}"})

    dataset_specs = {
        "Pangu": args.pangu_dataset_dir,
        "Tianji": args.tianji_dataset_dir,
        "IFS": args.ifs_dataset_dir,
        "ERA5": args.era5_dataset_dir,
    }
    dataset_report = {}
    for label, path in dataset_specs.items():
        if path:
            dataset_report[label] = inspect_model_dataset(
                label, path, args.max_dataset_rows, issues, elevation_rows
            )
            if label == "Pangu":
                lead = dataset_report[label].get("source_forecast_lead")
                if not isinstance(lead, Mapping) or not bool(lead.get("available")):
                    issues.append(
                        {"severity": "error", "stage": "Pangu_dataset", "message": "dataset lead evidence is missing"}
                    )
                elif not (
                    math.isclose(
                        float(lead.get("min_hours", math.nan)),
                        args.expected_pangu_lead_min_hours,
                        abs_tol=1.0e-6,
                    )
                    and math.isclose(
                        float(lead.get("max_hours", math.nan)),
                        args.expected_pangu_lead_max_hours,
                        abs_tol=1.0e-6,
                    )
                ):
                    issues.append(
                        {
                            "severity": "error",
                            "stage": "Pangu_dataset",
                            "message": f"dataset lead range does not match expected bounds: {lead}",
                        }
                    )
    report["model_datasets"] = dataset_report
    paired_df, paired_coverage = paired_complete_case_metrics(dataset_specs, args.paired_max_rows)
    paired_df.to_csv(out_dir / "q1000_paired_complete_case_elevation_metrics.csv", index=False)
    report["paired_complete_case"] = paired_coverage
    issues.append(
        {
            "severity": "warning",
            "stage": "physical_interpretation",
            "message": "1000 hPa is below ground at many elevated stations; use elevation-band results and prefer terrain-following near-surface moisture where possible.",
        }
    )
    report["issues"] = issues
    report["status"] = "failed" if any(v["severity"] == "error" for v in issues) else "passed"
    (out_dir / "q1000_lineage_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    pd.DataFrame(quantiles).to_csv(out_dir / "q1000_lineage_quantiles.csv", index=False)
    pd.DataFrame(elevation_rows).to_csv(out_dir / "q1000_dataset_elevation_sensitivity.csv", index=False)
    print(json.dumps({"status": report["status"], "issues": issues}, indent=2, ensure_ascii=False))
    print(f"[OK] audit outputs: {out_dir}")
    if args.strict and report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
