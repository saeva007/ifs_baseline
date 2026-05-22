#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remote IFS catalog checker for the new server layout.

Target layout example:
  /sharedata/dataset/GroupData/GD001-EC_Forcasting/0P125/20250101/t00z/
      Single_level/<var>/*.nc
      Pressure_levels/<var>/*.nc

This script:
1) Scans available Single_level and Pressure_levels variable folders.
2) Samples files to detect actual NetCDF data variable names / dims / coordinates / pressure levels.
3) Reports coverage for 2025 by init-cycle (t00z/t12z) and by lead-hour patterns in filenames.
4) Generates a mapping proposal to your Tianji canonical names.

Output:
  - ifs_remote_catalog_2025.json
  - ifs_remote_catalog_2025.csv
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


DEFAULT_ROOT = "/sharedata/dataset/GroupData/GD001-EC_Forcasting"

# Tianji canonical names we care about for baseline overlap.
TIANJI_CANONICAL = [
    "RH2M", "T2M", "D2M", "PRECIP", "MSLP", "SW_RAD", "U10", "V10",
    "LCC", "RH_925", "Q_925", "Q_1000", "U_925", "V_925", "W_925", "W_1000",
]

# Heuristic aliases for automatic candidate mapping.
CANONICAL_ALIASES = {
    "RH2M": ["2d", "d2m", "2t", "t2m"],
    "D2M": ["2d", "d2m"],
    "T2M": ["2t", "t2m"],
    "PRECIP": ["tp", "tprate", "prate", "precip"],
    "MSLP": ["msl", "slp"],
    "SW_RAD": ["ssrd", "dswrf", "tisr"],
    "U10": ["10u", "u10", "u10n"],
    "V10": ["10v", "v10", "v10n"],
    "LCC": ["lcc", "cl", "cldl"],
    "RH_925": ["r"],
    "Q_925": ["q"],
    "Q_1000": ["q"],
    "U_925": ["u"],
    "V_925": ["v"],
    "W_925": ["w"],
    "W_1000": ["w"],
}

LEAD_RE = re.compile(r"^[A-Z0-9]{3}(\d{8})(\d{8})1_([A-Za-z0-9]+)\.nc$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="IFS root containing 0P125/YYYYMMDD/t00z|t12z")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out_dir", default="./ifs_remote_check_out")
    ap.add_argument("--sample_days", type=int, default=10, help="max sampled init-days per cycle for quick scan")
    ap.add_argument("--sample_files_per_var", type=int, default=3, help="max sampled files per variable folder")
    return ap.parse_args()


def list_init_dirs(root: str, year: int) -> List[str]:
    pat = os.path.join(root, "0P125", f"{year}*", "t*z")
    dirs = [d for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
    return dirs


def pick_sample_dirs(all_dirs: List[str], sample_days: int) -> List[str]:
    by_cycle = {"t00z": [], "t12z": []}
    for d in all_dirs:
        cyc = os.path.basename(d)
        if cyc in by_cycle:
            by_cycle[cyc].append(d)
    out = []
    for cyc in ("t00z", "t12z"):
        arr = by_cycle[cyc]
        if len(arr) <= sample_days:
            out.extend(arr)
        else:
            idx = np.linspace(0, len(arr) - 1, sample_days, dtype=int)
            out.extend([arr[i] for i in idx])
    return sorted(set(out))


def detect_dims_and_var(fp: str) -> Dict[str, object]:
    ds = xr.open_dataset(fp)
    info: Dict[str, object] = {"file": fp, "data_vars": list(ds.data_vars), "dims": dict(ds.sizes)}
    # choose first non-scalar var as primary
    primary = None
    for v in ds.data_vars:
        if ds[v].ndim >= 2:
            primary = v
            break
    if primary is None and ds.data_vars:
        primary = list(ds.data_vars)[0]
    info["primary_var"] = primary
    if primary is not None:
        da = ds[primary]
        info["primary_dims"] = list(da.dims)
        # detect level coords if present
        lvl_name = next((n for n in ["level", "isobaricInhPa", "pressure_level", "lev", "plev"] if n in da.dims or n in ds.coords), None)
        if lvl_name and lvl_name in ds:
            lv = np.asarray(ds[lvl_name].values).astype(float)
            info["level_name"] = lvl_name
            info["levels_head"] = lv[:10].tolist()
    ds.close()
    return info


def guess_mapping(var_folder_name: str, primary_var_name: Optional[str]) -> List[str]:
    keys = set()
    text = (var_folder_name or "").lower()
    if primary_var_name:
        text = text + " " + primary_var_name.lower()
    for canon, aliases in CANONICAL_ALIASES.items():
        if any(a in text for a in aliases):
            keys.add(canon)
    return sorted(keys)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_init_dirs = list_init_dirs(args.root, args.year)
    sampled_dirs = pick_sample_dirs(all_init_dirs, args.sample_days)

    if not sampled_dirs:
        raise SystemExit(f"No init dirs found under {args.root}/0P125/{args.year}*")

    catalog: Dict[str, Dict[str, dict]] = {"Single_level": {}, "Pressure_levels": {}}
    coverage = {"t00z": 0, "t12z": 0}
    for d in all_init_dirs:
        cyc = os.path.basename(d)
        if cyc in coverage:
            coverage[cyc] += 1

    # Collect candidate var folders from sampled dirs
    folder_seen = {"Single_level": set(), "Pressure_levels": set()}
    for d in sampled_dirs:
        for group in ("Single_level", "Pressure_levels"):
            gd = os.path.join(d, group)
            if not os.path.isdir(gd):
                continue
            for sub in sorted(os.listdir(gd)):
                sp = os.path.join(gd, sub)
                if os.path.isdir(sp):
                    folder_seen[group].add(sub)

    # For each var folder, sample files across sampled dirs
    for group in ("Single_level", "Pressure_levels"):
        for var_folder in sorted(folder_seen[group]):
            files = []
            for d in sampled_dirs:
                p = os.path.join(d, group, var_folder, "*.nc")
                files.extend(glob.glob(p))
            files = sorted(files)
            if not files:
                continue

            # sample several files
            if len(files) > args.sample_files_per_var:
                idx = np.linspace(0, len(files) - 1, args.sample_files_per_var, dtype=int)
                sample_files = [files[i] for i in idx]
            else:
                sample_files = files

            detected = []
            lead_hours = []
            for fp in sample_files:
                try:
                    detected.append(detect_dims_and_var(fp))
                except Exception as e:
                    detected.append({"file": fp, "error": f"{type(e).__name__}:{e}"})

            # parse lead hrs from all files (quick stats)
            for fp in files[:2000]:
                m = LEAD_RE.match(os.path.basename(fp))
                if m:
                    init_s = m.group(1)
                    valid_s = m.group(2)
                    try:
                        init_h = np.datetime64(f"{init_s[:4]}-{init_s[4:6]}-{init_s[6:8]}T{init_s[8:10]}:{init_s[10:12]}")
                        valid_h = np.datetime64(f"{valid_s[:4]}-{valid_s[4:6]}-{valid_s[6:8]}T{valid_s[8:10]}:{valid_s[10:12]}")
                        lead = int((valid_h - init_h) / np.timedelta64(1, "h"))
                        lead_hours.append(lead)
                    except Exception:
                        pass

            primary = None
            for dct in detected:
                if "primary_var" in dct and dct["primary_var"] is not None:
                    primary = dct["primary_var"]
                    break

            mapped = guess_mapping(var_folder, primary)
            catalog[group][var_folder] = {
                "n_files_sampled_scope": len(files),
                "sample_files": sample_files,
                "detected": detected,
                "primary_var_guess": primary,
                "lead_hour_min": int(np.min(lead_hours)) if lead_hours else None,
                "lead_hour_max": int(np.max(lead_hours)) if lead_hours else None,
                "lead_hour_unique_head": sorted(set(lead_hours))[:30] if lead_hours else [],
                "mapped_canonical_candidates": mapped,
            }

    # Build merged mapping candidates by canonical
    reverse_map: Dict[str, List[dict]] = defaultdict(list)
    for group in catalog:
        for vf, info in catalog[group].items():
            for canon in info["mapped_canonical_candidates"]:
                reverse_map[canon].append({
                    "group": group,
                    "var_folder": vf,
                    "primary_var_guess": info["primary_var_guess"],
                    "lead_hour_min": info["lead_hour_min"],
                    "lead_hour_max": info["lead_hour_max"],
                })

    report = {
        "root": args.root,
        "year": args.year,
        "n_init_dirs_total": len(all_init_dirs),
        "coverage_by_cycle": coverage,
        "sampled_init_dirs": sampled_dirs,
        "catalog": catalog,
        "canonical_mapping_candidates": {k: reverse_map.get(k, []) for k in TIANJI_CANONICAL},
    }

    out_json = os.path.join(args.out_dir, "ifs_remote_catalog_2025.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(args.out_dir, "ifs_remote_catalog_2025.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "var_folder", "primary_var_guess", "lead_hour_min", "lead_hour_max", "mapped_canonical_candidates"])
        for group in ("Single_level", "Pressure_levels"):
            for vf, info in sorted(catalog[group].items()):
                w.writerow([
                    group, vf, info["primary_var_guess"], info["lead_hour_min"], info["lead_hour_max"],
                    "|".join(info["mapped_canonical_candidates"]),
                ])

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")
    print("Next: send this JSON back, I will generate exact final mapping config for interpolation.")


if __name__ == "__main__":
    main()

