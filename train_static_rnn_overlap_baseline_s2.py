#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the Static-MLP + RNN model on Tianji/IFS overlap S2 datasets.

This is a thin experiment wrapper around the paper-facing
``train_static_rnn_lowvis.py`` script in the main vis_mlp training checkout.
Keeping the model implementation in one place prevents the IFS data-source
experiment from silently drifting away from the current main architecture.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import List


VIS_MLP_ROOT = os.environ.get("VIS_MLP_ROOT", "/public/home/putianshu/vis_mlp")
IFS_BASELINE_ROOT = os.environ.get(
    "IFS_BASELINE_ROOT", os.path.join(VIS_MLP_ROOT, "ifs_baseline")
)
DEFAULT_CKPT_DIR = os.path.join(IFS_BASELINE_ROOT, "checkpoints")
DEFAULT_STATIC_TRAIN_DIR = os.environ.get(
    "STATIC_RNN_TRAIN_DIR", os.path.join(VIS_MLP_ROOT, "train")
)
DEFAULT_TIANJI_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_baseline"
)
DEFAULT_IFS_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"
)


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Static-MLP + RNN low-vis trainer on the overlap "
            "Tianji/IFS S2 datasets."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--data_source", "--data-source", choices=["tianji", "ifs"], default="tianji")
    parser.add_argument("--s2_data_dir", "--s2-data-dir", default="")
    parser.add_argument("--ckpt_dir", "--ckpt-dir", default=os.environ.get("OVERLAP_CKPT_DIR", DEFAULT_CKPT_DIR))
    parser.add_argument("--static_train_dir", "--static-train-dir", default=DEFAULT_STATIC_TRAIN_DIR)
    parser.add_argument("--static_train_script", "--static-train-script", default="train_static_rnn_lowvis.py")
    parser.add_argument("--mode", choices=["s2", "both"], default="s2")
    parser.add_argument("--run_id", "--run-id", default="")
    parser.add_argument("--base_path", "--base-path", default=VIS_MLP_ROOT)
    parser.add_argument("--pretrained_ckpt", "--pretrained-ckpt", default=os.environ.get("OVERLAP_STATIC_RNN_PRETRAINED_CKPT", ""))
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def resolve_data_dir(args: argparse.Namespace) -> str:
    if args.s2_data_dir:
        return args.s2_data_dir
    if args.data_source == "ifs":
        return DEFAULT_IFS_DIR
    return DEFAULT_TIANJI_DIR


def resolve_static_script(args: argparse.Namespace) -> Path:
    explicit_dir = Path(args.static_train_dir).expanduser()
    candidates = [
        explicit_dir / args.static_train_script,
        Path(__file__).resolve().parent.parent / "train" / args.static_train_script,
        Path(__file__).resolve().parent.parent / "vis_mlp" / args.static_train_script,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    checked = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find Static-RNN training script. Checked:\n  {checked}")


def main() -> None:
    args, passthrough = parse_args()
    static_script = resolve_static_script(args)
    data_dir = resolve_data_dir(args)
    run_id = args.run_id or os.environ.get(
        "LOWVIS_RNN_RUN_ID",
        f"exp_overlap_static_rnn_s2_{args.data_source}_pm10_pm25",
    )

    static_argv = [
        str(static_script),
        "--mode",
        args.mode,
        "--run-id",
        run_id,
        "--base-path",
        args.base_path,
        "--s2-data-dir",
        data_dir,
        "--ckpt-dir",
        args.ckpt_dir,
    ]
    if args.pretrained_ckpt:
        static_argv.extend(["--pretrained-ckpt", args.pretrained_ckpt])
    static_argv.extend(passthrough)

    print("[overlap-static-rnn] delegating to:", static_script, flush=True)
    print("[overlap-static-rnn] data_source:", args.data_source, flush=True)
    print("[overlap-static-rnn] run_id:", run_id, flush=True)
    print("[overlap-static-rnn] s2_data_dir:", data_dir, flush=True)
    print("[overlap-static-rnn] ckpt_dir:", args.ckpt_dir, flush=True)

    old_argv = sys.argv[:]
    sys.path.insert(0, str(static_script.parent))
    try:
        sys.argv = static_argv
        runpy.run_path(str(static_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
