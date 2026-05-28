#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge split overlap/source evaluation metrics and redraw the key source figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from test_PMST_overlap_forecast_source_s2 import plot_key_metrics_figure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge split overlap source evaluation outputs.")
    p.add_argument("--root_dir", default="", help="Directory searched recursively for overall_metrics.csv files.")
    p.add_argument(
        "--input_dirs",
        default="",
        help="Comma/semicolon-separated run directories containing overall_metrics.csv; overrides --root_dir.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_figures", action="store_true")
    return p.parse_args()


def split_dirs(value: str) -> List[Path]:
    out: List[Path] = []
    for raw in str(value or "").replace(";", ",").split(","):
        item = raw.strip()
        if item:
            out.append(Path(item))
    return out


def metric_paths(args: argparse.Namespace) -> List[Path]:
    dirs = split_dirs(args.input_dirs)
    if dirs:
        return [d / "overall_metrics.csv" for d in dirs]
    root = Path(args.root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--root_dir does not exist: {root}")
    out_dir = Path(args.out_dir).resolve()
    paths = []
    for path in sorted(root.rglob("overall_metrics.csv")):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == out_dir / "overall_metrics.csv" or out_dir in resolved.parents:
            continue
        paths.append(path)
    return paths


def read_table(paths: List[Path], name: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        table_path = path.parent / name
        if table_path.is_file():
            df = pd.read_csv(table_path)
            df["run_dir"] = str(path.parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def dedupe_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source" not in df:
        return df
    preferred = ["tianji", "ifs", "T2ND_rh2m_common_core", "pangu2021_common_core", "era5_2025_common_core"]
    df = df.copy()
    df["_order"] = df["source"].astype(str).map({s: i for i, s in enumerate(preferred)}).fillna(len(preferred))
    df = df.sort_values(["_order", "source", "run_dir"], kind="stable")
    return df.drop_duplicates("source", keep="last").drop(columns=["_order"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in metric_paths(args) if p.is_file()]
    if not paths:
        raise FileNotFoundError("No overall_metrics.csv files were found.")

    overall = dedupe_sources(read_table(paths, "overall_metrics.csv"))
    validation = dedupe_sources(read_table(paths, "validation_metrics.csv"))
    overall.to_csv(out_dir / "overall_metrics.csv", index=False)
    if not validation.empty:
        validation.to_csv(out_dir / "validation_metrics.csv", index=False)
    if not args.no_figures:
        plot_key_metrics_figure(overall, out_dir)

    keep = [
        c
        for c in (
            "source",
            "source_label",
            "n",
            "fog_pod",
            "fog_csi",
            "mist_pod",
            "mist_csi",
            "low_vis_recall",
            "low_vis_csi",
            "low_vis_precision",
            "low_vis_fpr",
            "run_dir",
        )
        if c in overall.columns
    ]
    print(f"[OK] merged {len(paths)} metric file(s) into {out_dir}", flush=True)
    print(overall[keep].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
