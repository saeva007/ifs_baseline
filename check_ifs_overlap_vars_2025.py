#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check which Tianji-used variables have IFS counterparts (2025 coverage).

Inputs
------
- Tianji merged station dataset (for canonical feature list): merged_final_all_vars.nc
- IFS gridded dataset root: /public/home/sd3team/sd3_database/src_data/IFS/nc_0p1

Outputs (under --out_dir)
-------------------------
- ifs_overlap_report_2025.json : full machine-readable report
- ifs_overlap_report_2025.csv  : compact table
- overlap_vars.txt             : canonical variable names present in BOTH (and readable in IFS 2025)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr


IFS_ROOT_DEFAULT = "/public/home/sd3team/sd3_database/src_data/IFS/nc_0p1"
TIANJI_FILE_DEFAULT = "/public/home/putianshu/vis_mlp/tianji_auto_station/merged_final_all_vars.nc"


# Canonical (model-facing) variable names used by current 12h Tianji dataset builder.
# From /public/home/putianshu/vis_mlp/s2_data_monthtail_v2.ipynb "latest valid cell".
TIANJI_CANONICAL_USED = [
    "RH2M",
    "T2M",
    "PRECIP",
    "MSLP",
    "SW_RAD",
    "U10",
    "WSPD10",
    "V10",
    "WDIR10",
    "CAPE",
    "LCC",
    "T_925",
    "RH_925",
    "U_925",
    "WSPD925",
    "V_925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
    "W_925",
    "W_1000",
    "DPD",
    "INVERSION",
]


@dataclass(frozen=True)
class IfsVarSpec:
    folder: str
    var_name: str


# A minimal mapping based on the station-interpolation reports for 2025.
# RH2M, DP_1000, DP_925, DPD and wind-speed/direction slots are populated by
# downstream dataset builders from these source fields rather than read as
# independent IFS variables.
IFS_CANONICAL_MAP: Dict[str, IfsVarSpec] = {
    "T2M": IfsVarSpec(folder="2t", var_name="t2m"),
    "D2M": IfsVarSpec(folder="2d", var_name="d2m"),
    "PRECIP": IfsVarSpec(folder="tp", var_name="tp"),
    "MSLP": IfsVarSpec(folder="msl", var_name="msl"),
    "SW_RAD": IfsVarSpec(folder="ssrd", var_name="ssrd"),
    "U10": IfsVarSpec(folder="10u", var_name="u10"),
    "V10": IfsVarSpec(folder="10v", var_name="v10"),
    "LCC": IfsVarSpec(folder="lcc", var_name="lcc"),
    "RH_925": IfsVarSpec(folder="r", var_name="r"),
    "U_925": IfsVarSpec(folder="u", var_name="u"),
    "V_925": IfsVarSpec(folder="v", var_name="v"),
    "Q_1000": IfsVarSpec(folder="q", var_name="q"),
    "Q_925": IfsVarSpec(folder="q", var_name="q"),
    "W_925": IfsVarSpec(folder="w", var_name="w"),
    "W_1000": IfsVarSpec(folder="w", var_name="w"),
}

DERIVED_IFS_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "RH2M": ("T2M", "D2M"),
    "WSPD10": ("U10", "V10"),
    "WDIR10": ("U10", "V10"),
    "WSPD925": ("U_925", "V_925"),
    "DP_1000": ("Q_1000",),
    "DP_925": ("Q_925",),
    "DPD": ("T2M", "D2M"),
}


FNAME_RE = re.compile(r"^(?P<prefix>[^_]+)_(?P<yyyymmdd>\d{8})_(?P<hh>\d{2})\.nc$")


def _real_exists(path: str) -> bool:
    try:
        return os.path.exists(os.path.realpath(path))
    except OSError:
        return False


def _list_ifs_files(folder: str, ifs_root: str, year: int) -> List[str]:
    d = os.path.join(ifs_root, folder)
    if not os.path.isdir(d):
        return []
    files = sorted(glob.glob(os.path.join(d, f"*{year}*.nc")))
    return files


def _parse_start_from_fname(fp: str) -> Optional[pd.Timestamp]:
    m = FNAME_RE.match(os.path.basename(fp))
    if not m:
        return None
    ymd = m.group("yyyymmdd")
    hh = m.group("hh")
    try:
        return pd.Timestamp(datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H"))
    except Exception:
        return None


def _quick_open_checks(fp: str, var_name: str) -> Tuple[bool, Optional[str], Optional[Tuple[str, ...]]]:
    """
    Returns: (ok, err, dims)
    """
    try:
        ds = xr.open_dataset(fp)
        if var_name not in ds.data_vars:
            return False, f"var_not_found:{var_name}", None
        dims = tuple(ds[var_name].dims)
        return True, None, dims
    except Exception as e:
        return False, f"open_failed:{type(e).__name__}:{e}", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifs_root", default=IFS_ROOT_DEFAULT)
    ap.add_argument("--tianji_file", default=TIANJI_FILE_DEFAULT)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out_dir", default="/public/home/putianshu/vis_mlp/ifs_baseline/out_overlap_check")
    ap.add_argument("--max_open_checks_per_var", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Tianji availability check (we don't need to load full arrays; only metadata + var names).
    ds_tj = xr.open_dataset(args.tianji_file, engine="h5netcdf")
    tj_vars = set(ds_tj.data_vars)
    # Canonical list is from dataset builder; we additionally check what exists in file.
    tj_used_present = [v for v in TIANJI_CANONICAL_USED if v in tj_vars or v in ("DPD", "INVERSION")]
    ds_tj.close()

    report = {
        "year": args.year,
        "ifs_root": args.ifs_root,
        "tianji_file": args.tianji_file,
        "tianji_canonical_used": TIANJI_CANONICAL_USED,
        "tianji_used_present_in_file": tj_used_present,
        "ifs_canonical_map": {k: {"folder": v.folder, "var_name": v.var_name} for k, v in IFS_CANONICAL_MAP.items()},
        "variables": {},
        "overlap_canonical": [],
    }

    overlap = []
    for canon in TIANJI_CANONICAL_USED:
        spec = IFS_CANONICAL_MAP.get(canon)
        entry = {
            "canonical": canon,
            "tianji_in_builder": True,
            "tianji_in_nc": (canon in tj_vars) if "tj_vars" in locals() else None,
            "ifs_mapped": spec is not None,
            "ifs_folder": spec.folder if spec else None,
            "ifs_var_name": spec.var_name if spec else None,
            "ifs_files_2025_total": 0,
            "ifs_files_2025_real_exists": 0,
            "ifs_open_checks": [],
            "ifs_dims_example": None,
        }

        if spec is None:
            report["variables"][canon] = entry
            continue

        files = _list_ifs_files(spec.folder, args.ifs_root, args.year)
        entry["ifs_files_2025_total"] = len(files)
        real_files = [fp for fp in files if _real_exists(fp)]
        entry["ifs_files_2025_real_exists"] = len(real_files)

        # minimal open checks for schema validation (prevent wrong var_name).
        opened_dims = None
        for fp in real_files[: max(0, args.max_open_checks_per_var)]:
            ok, err, dims = _quick_open_checks(fp, spec.var_name)
            entry["ifs_open_checks"].append({"file": fp, "realpath": os.path.realpath(fp), "ok": ok, "err": err, "dims": dims})
            if ok and opened_dims is None:
                opened_dims = dims
        entry["ifs_dims_example"] = opened_dims

        if entry["ifs_files_2025_real_exists"] > 0 and opened_dims is not None:
            overlap.append(canon)

        report["variables"][canon] = entry

    report["overlap_canonical"] = overlap

    # write outputs
    json_path = os.path.join(args.out_dir, "ifs_overlap_report_2025.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, "ifs_overlap_report_2025.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "canonical",
                "ifs_mapped",
                "ifs_folder",
                "ifs_var_name",
                "ifs_files_2025_total",
                "ifs_files_2025_real_exists",
                "ifs_dims_example",
            ]
        )
        for canon in TIANJI_CANONICAL_USED:
            e = report["variables"][canon]
            w.writerow(
                [
                    canon,
                    e["ifs_mapped"],
                    e["ifs_folder"],
                    e["ifs_var_name"],
                    e["ifs_files_2025_total"],
                    e["ifs_files_2025_real_exists"],
                    "" if e["ifs_dims_example"] is None else "|".join(e["ifs_dims_example"]),
                ]
            )

    txt_path = os.path.join(args.out_dir, "overlap_vars.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for v in overlap:
            f.write(v + "\n")

    print(f"[OK] overlap vars ({len(overlap)}): {overlap}")
    print(f"[OK] wrote: {json_path}")
    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {txt_path}")


if __name__ == "__main__":
    main()

