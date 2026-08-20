#!/usr/bin/env python3
"""Leadtime-resolved paired RMSE (12-24 h) for the Fig. 3 c/d panels.

The overlap q-core datasets are built by stitching each initialization's
12-24 h forecast leads into an hourly valid-time series.  With 00/12Z
initializations this maps valid hour ``h`` to

    lead = h + 12 if h < 12 else h
    init = valid_time - lead

which is the same rule used by audit_pangu_tianji_q1000_lineage.py
(``INFER_PANGU_LEAD12_23_FROM_VALID_TIME``).  This script computes pointwise
RMSE per lead hour for the station-observation surface features and the ERA5
pressure features, reusing the value conversion and common-finite-row policy
of analyze_q_core_paired_source_quality.py without changing that analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import analyze_q_core_paired_source_quality as base


STATION_FEATURES = ("T2M", "WSPD10", "MSLP")
ERA5_FEATURES = ("T_925", "Q_1000", "Q_925", "UV_925_VECTOR")
FORECAST_SOURCES = ("pangu", "tianji")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pangu-dir", required=True)
    ap.add_argument("--tianji-dir", required=True)
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--obs-root", required=True)
    ap.add_argument("--paper-eval-dir", default="/public/home/putianshu/vis_mlp/paper_eval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-max-rows", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=20260715)
    ap.add_argument("--low-vis-threshold-m", type=float, default=1000.0)
    return ap.parse_args()


def derive_lead_hours(keys: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Map valid UTC hours to 12-24 h leads under the 00/12Z stitching rule."""

    hours = keys["time_utc"].dt.hour.to_numpy(dtype=np.int64)
    leads = np.where(hours < 12, hours + 12, hours).astype(np.int64)
    inits = keys["time_utc"] - pd.to_timedelta(leads, unit="h")
    return leads, inits.to_numpy()


def rmse_by_lead(
    diff: np.ndarray,
    leads: np.ndarray,
    dates: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for lead in range(12, 24):
        mask = leads == lead
        n = int(mask.sum())
        if n == 0:
            continue
        selected = diff[mask]
        rows.append(
            {
                "lead_hour": int(lead),
                "n": n,
                "n_dates": int(np.unique(dates[mask]).size),
                "rmse": float(np.sqrt(np.mean(selected * selected))),
            }
        )
    return rows


def pressure_rows(
    split: base.QualitySplit,
    leads: np.ndarray,
    dates: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature in ("T_925", "Q_1000", "Q_925"):
        converted = {
            source: base.convert_feature(feature, split.values[source][feature])
            for source in base.SOURCES
        }
        complete = np.logical_and.reduce(
            [np.isfinite(converted[source]) for source in base.SOURCES]
        )
        reference = converted["era5_reference_analysis"]
        for source in FORECAST_SOURCES:
            diff = converted[source] - reference
            for row in rmse_by_lead(diff[complete], leads[complete], dates[complete]):
                rows.append({"feature": feature, "source": source, **row})

    wind = {
        source: {
            component: base.convert_feature(component, split.values[source][component])
            for component in ("U_925", "V_925")
        }
        for source in base.SOURCES
    }
    complete_wind = np.logical_and.reduce(
        [
            np.isfinite(wind[source][component])
            for source in base.SOURCES
            for component in ("U_925", "V_925")
        ]
    )
    for source in FORECAST_SOURCES:
        diff2 = (
            (wind[source]["U_925"] - wind["era5_reference_analysis"]["U_925"]) ** 2
            + (wind[source]["V_925"] - wind["era5_reference_analysis"]["V_925"]) ** 2
        )
        for row in rmse_by_lead(
            diff2[complete_wind], leads[complete_wind], dates[complete_wind]
        ):
            rows.append(
                {
                    "feature": "UV_925_VECTOR",
                    "source": source,
                    **row,
                }
            )
    return rows


def surface_rows(
    split: base.QualitySplit,
    leads: np.ndarray,
    dates: np.ndarray,
    obs_root: Path,
    paper_eval_dir: Path,
) -> List[Dict[str, object]]:
    frame = split.keys[["time_utc", "station_key"]].copy()
    frame["_aligned_row"] = np.arange(len(frame), dtype=np.int64)
    for source in base.SOURCES:
        for feature in base.SURFACE_OBS_FEATURES:
            frame[f"{feature}_{source}"] = split.values[source][feature]
    frame, obs_info = base.joint.hybrid_analysis.attach_observations(
        frame, obs_root, paper_eval_dir
    )
    if len(frame) != len(split.keys) or frame["_aligned_row"].duplicated().any():
        raise RuntimeError("Observation attachment changed common test-row cardinality")
    frame = frame.sort_values("_aligned_row", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        frame["_aligned_row"].to_numpy(dtype=np.int64), np.arange(len(frame))
    ):
        raise RuntimeError("Observation attachment changed common test-row order")

    rows: List[Dict[str, object]] = []
    for feature in base.SURFACE_OBS_FEATURES:
        obs_column = str(base.SURFACE_INFO[feature]["observation_column"])
        if obs_column not in frame:
            raise KeyError(f"Observation attachment lacks required column {obs_column}")
        observation = base.joint.hybrid_analysis.clean_observation_values(
            obs_column, frame[obs_column]
        )
        forecasts = {
            source: base.convert_surface(
                feature, frame[f"{feature}_{source}"].to_numpy(dtype=np.float64)
            )
            for source in base.SOURCES
        }
        complete = np.isfinite(observation) & np.logical_and.reduce(
            [np.isfinite(forecasts[source]) for source in base.SOURCES]
        )
        for source in FORECAST_SOURCES:
            diff = forecasts[source] - observation
            for row in rmse_by_lead(diff[complete], leads[complete], dates[complete]):
                rows.append(
                    {
                        "feature": feature,
                        "source": source,
                        "reference": "automatic station observations",
                        **row,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    paths = {
        "pangu": Path(args.pangu_dir).expanduser().resolve(),
        "tianji": Path(args.tianji_dir).expanduser().resolve(),
        "era5_reference_analysis": Path(args.era5_dir).expanduser().resolve(),
    }
    layouts = {
        source: base.joint.load_layout(path, source) for source, path in paths.items()
    }
    for source, layout in layouts.items():
        missing = [
            feature for feature in base.REQUIRED_FEATURES if feature not in layout.order
        ]
        if missing:
            raise ValueError(
                f"{source}: diagnostic dataset lacks {missing}; rebuild q_core_t925_no_rh2m data"
            )

    split = base.align_test(layouts, args.test_max_rows, args.bootstrap_seed)
    leads, inits = derive_lead_hours(split.keys)
    dates = split.keys["time_utc"].dt.strftime("%Y-%m-%d").to_numpy()

    init_hours = pd.DatetimeIndex(inits).hour
    lead_min, lead_max = int(leads.min()), int(leads.max())
    if lead_min < 12 or lead_max > 23:
        raise ValueError(
            f"Derived lead hours fall outside 12-24: min={lead_min} max={lead_max}"
        )
    init_hour_counts = {
        str(int(hour)): int((init_hours == hour).sum())
        for hour in sorted(set(init_hours.tolist()))
    }

    pressure = pd.DataFrame(pressure_rows(split, leads, dates))
    surface = pd.DataFrame(surface_rows(
        split,
        leads,
        dates,
        Path(args.obs_root).expanduser().resolve(),
        Path(args.paper_eval_dir).expanduser().resolve(),
    ))
    pressure.insert(0, "reference", "ERA5 reference analysis")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pressure.to_csv(out_dir / "era5_reference_rmse_by_lead_hour.csv", index=False)
    surface.to_csv(out_dir / "station_reference_rmse_by_lead_hour.csv", index=False)

    rows_per_lead = {
        str(lead): int((leads == lead).sum()) for lead in range(12, 24)
    }
    validation = {
        "status": "completed",
        "analysis_type": "diagnostic_only_no_training",
        "stitching_rule": "00/12Z initializations, each contributing leads 12-23",
        "lead_derivation": "lead = valid_hour + 12 if valid_hour < 12 else valid_hour; init = valid_time - lead",
        "lead_min": lead_min,
        "lead_max": lead_max,
        "init_hour_counts": init_hour_counts,
        "rows_per_lead": rows_per_lead,
        "test_rows": int(len(split.keys)),
        "represented_utc_dates": int(split.keys["time_utc"].dt.strftime("%Y-%m-%d").nunique()),
        "station_features": list(STATION_FEATURES),
        "era5_features": list(ERA5_FEATURES),
        "sources": list(FORECAST_SOURCES),
        "input_datasets": {
            source: str(layout.path) for source, layout in layouts.items()
        },
        "outputs": [
            "era5_reference_rmse_by_lead_hour.csv",
            "station_reference_rmse_by_lead_hour.csv",
        ],
    }
    (out_dir / "leadtime_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[by-leadtime] wrote {out_dir}", flush=True)
    print(
        json.dumps(
            {
                "lead_min": lead_min,
                "lead_max": lead_max,
                "init_hour_counts": init_hour_counts,
                "rows_per_lead": rows_per_lead,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
