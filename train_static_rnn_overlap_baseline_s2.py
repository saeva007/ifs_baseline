#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the Static-MLP + RNN model on overlap S1/S2 datasets.

This is a thin experiment wrapper around the paper-facing
``train_static_rnn_lowvis.py`` script in the main vis_mlp training checkout.
Keeping the model implementation in one place prevents the IFS data-source
experiment from silently drifting away from the current main architecture.
By default the wrapper ``exec``'s the main trainer, so distributed runs behave
like launching the paper-facing script directly.
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
DEFAULT_S1_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap"
)
DEFAULT_S1_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_common_core"
)
DEFAULT_S1_COMPACT_COMMON_CORE_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_compact_common_core_no_rh2m"
)
DEFAULT_S1_SOURCE_FULL_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full"
)
DEFAULT_S1_SOURCE_FULL_TIANJI_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_tianji"
)
DEFAULT_S1_SOURCE_FULL_IFS_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_ifs"
)
DEFAULT_S1_SOURCE_FULL_PANGU_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_pangu"
)
DEFAULT_S1_SOURCE_FULL_PANGU2025_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_pmst_v5_aligned_12h_pm10_pm25_source_full_pangu2025"
)
DEFAULT_TIANJI_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_baseline"
)
DEFAULT_TIANJI_T2ND_RH2M_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m"
)
DEFAULT_IFS_DIR = os.path.join(
    IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_baseline"
)
DEFAULT_S2_DIRS = {
    "tianji": DEFAULT_TIANJI_DIR,
    "T2ND_rh2m": DEFAULT_TIANJI_T2ND_RH2M_DIR,
    "ifs": DEFAULT_IFS_DIR,
    "tianji_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_common_core"),
    "T2ND_rh2m_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_common_core"),
    "ifs_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_common_core"),
    "pangu2021_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2021_12h_pm10_pm25_common_core"),
    "pangu2025_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2025_12h_pm10_pm25_common_core"),
    "era5_2025_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_era5_2025_12h_pm10_pm25_common_core"),
    "tianji_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_compact_common_core_no_rh2m"),
    "T2ND_rh2m_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_compact_common_core_no_rh2m"),
    "ifs_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_compact_common_core_no_rh2m"),
    "pangu2021_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2021_12h_pm10_pm25_compact_common_core_no_rh2m"),
    "pangu2025_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2025_12h_pm10_pm25_compact_common_core_no_rh2m"),
    "era5_2025_compact_common_core": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_era5_2025_12h_pm10_pm25_compact_common_core_no_rh2m"),
    "tianji_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_source_full"),
    "T2ND_rh2m_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_source_full"),
    "ifs_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_ifs_12h_pm10_pm25_source_full"),
    "pangu2021_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2021_12h_pm10_pm25_source_full"),
    "pangu2025_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full"),
    "era5_2025_source_full": os.path.join(IFS_BASELINE_ROOT, "ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full"),
}
DEFAULT_S1_RUN_ID = "exp_overlap_static_rnn_s1_pm10_pm25"
DEFAULT_S1_COMMON_CORE_RUN_ID = "exp_overlap_static_rnn_s1_common_core_pm10_pm25"
DEFAULT_S1_COMPACT_COMMON_CORE_RUN_ID = "exp_overlap_static_rnn_s1_compact_common_core_no_rh2m_pm10_pm25"
DEFAULT_S1_SOURCE_FULL_RUN_ID = "exp_overlap_static_rnn_s1_source_full_pm10_pm25"
DEFAULT_S1_SOURCE_FULL_TIANJI_RUN_ID = "exp_overlap_static_rnn_s1_source_full_tianji_dyn27_pm10_pm25"
DEFAULT_S1_SOURCE_FULL_IFS_RUN_ID = "exp_overlap_static_rnn_s1_source_full_ifs_dyn24_pm10_pm25"
DEFAULT_S1_SOURCE_FULL_PANGU_RUN_ID = "exp_overlap_static_rnn_s1_source_full_pangu_dyn21_pm10_pm25"
DEFAULT_S1_SOURCE_FULL_PANGU2025_RUN_ID = "exp_overlap_static_rnn_s1_source_full_pangu2025_dyn19_pm10_pm25"
DEFAULT_OVERLAP_S2_A_STEPS = "12000"
DEFAULT_OVERLAP_S2_B_STEPS = "40000"
DEFAULT_OVERLAP_S2_PATIENCE = "18"


def default_s2_run_id(source: str) -> str:
    if "compact_common_core" in source and "no_rh2m" not in source:
        source = source.replace("compact_common_core", "compact_common_core_no_rh2m")
    return f"exp_overlap_static_rnn_s2_{source}_pm10_pm25"


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Static-MLP + RNN low-vis trainer on the overlap "
            "S1 and Tianji/IFS S2 datasets."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data_source",
        "--data-source",
        choices=sorted(DEFAULT_S2_DIRS),
        default="tianji",
    )
    parser.add_argument("--s1_data_dir", "--s1-data-dir", default=os.environ.get("OVERLAP_S1_DATA_DIR", DEFAULT_S1_DIR))
    parser.add_argument("--s2_data_dir", "--s2-data-dir", default="")
    parser.add_argument("--ckpt_dir", "--ckpt-dir", default=os.environ.get("OVERLAP_CKPT_DIR", DEFAULT_CKPT_DIR))
    parser.add_argument("--static_train_dir", "--static-train-dir", default=DEFAULT_STATIC_TRAIN_DIR)
    parser.add_argument("--static_train_script", "--static-train-script", default="train_static_rnn_lowvis.py")
    parser.add_argument("--mode", choices=["s1", "s2", "both"], default="s2")
    parser.add_argument("--run_id", "--run-id", default="")
    parser.add_argument("--base_path", "--base-path", default=VIS_MLP_ROOT)
    parser.add_argument("--pretrained_ckpt", "--pretrained-ckpt", default=os.environ.get("OVERLAP_STATIC_RNN_PRETRAINED_CKPT", ""))
    parser.add_argument(
        "--no_default_s1_pretrained",
        "--no-default-s1-pretrained",
        action="store_true",
        help="For S2 mode, do not automatically use the default Static-RNN overlap S1 checkpoint when it exists.",
    )
    parser.add_argument(
        "--runpy",
        action="store_true",
        help="Debug fallback: run the main trainer in-process instead of exec'ing it.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def resolve_s2_data_dir(args: argparse.Namespace) -> str:
    if args.s2_data_dir:
        return args.s2_data_dir
    return DEFAULT_S2_DIRS[args.data_source]


def resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    env_run_id = os.environ.get("LOWVIS_RNN_RUN_ID", "")
    if env_run_id:
        return env_run_id
    if args.mode == "s1":
        if "compact_common_core" in str(args.s1_data_dir):
            return DEFAULT_S1_COMPACT_COMMON_CORE_RUN_ID
        if "source_full" in str(args.s1_data_dir):
            if "pangu2025" in str(args.s1_data_dir):
                return DEFAULT_S1_SOURCE_FULL_PANGU2025_RUN_ID
            if "pangu" in str(args.s1_data_dir):
                return DEFAULT_S1_SOURCE_FULL_PANGU_RUN_ID
            if "ifs" in str(args.s1_data_dir):
                return DEFAULT_S1_SOURCE_FULL_IFS_RUN_ID
            if "tianji" in str(args.s1_data_dir):
                return DEFAULT_S1_SOURCE_FULL_TIANJI_RUN_ID
            return DEFAULT_S1_SOURCE_FULL_RUN_ID
        if "common_core" in str(args.s1_data_dir):
            return DEFAULT_S1_COMMON_CORE_RUN_ID
        return DEFAULT_S1_RUN_ID
    if args.mode == "both":
        return f"exp_overlap_static_rnn_both_{args.data_source}_pm10_pm25"
    return default_s2_run_id(args.data_source)


def source_full_s1_run_id_for_s2(s2_dir: str, data_source: str) -> str:
    text = f"{s2_dir} {data_source}".lower()
    if "pangu2025" in text:
        return DEFAULT_S1_SOURCE_FULL_PANGU2025_RUN_ID
    if "pangu" in text:
        return DEFAULT_S1_SOURCE_FULL_PANGU_RUN_ID
    if "ifs" in text:
        return DEFAULT_S1_SOURCE_FULL_IFS_RUN_ID
    return DEFAULT_S1_SOURCE_FULL_TIANJI_RUN_ID


def resolve_pretrained_ckpt(args: argparse.Namespace) -> str:
    if args.pretrained_ckpt:
        return args.pretrained_ckpt
    if args.mode != "s2" or args.no_default_s1_pretrained:
        return ""
    s2_dir = resolve_s2_data_dir(args)
    if "compact_common_core" in str(s2_dir):
        run_id = DEFAULT_S1_COMPACT_COMMON_CORE_RUN_ID
    elif "source_full" in str(s2_dir):
        run_id = source_full_s1_run_id_for_s2(s2_dir, args.data_source)
    elif "common_core" in str(s2_dir):
        run_id = DEFAULT_S1_COMMON_CORE_RUN_ID
    else:
        run_id = DEFAULT_S1_RUN_ID
    default_path = os.path.join(args.ckpt_dir, f"{run_id}_S1_best_score.pt")
    return default_path if ("source_full" in str(s2_dir) or os.path.isfile(default_path)) else ""


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


def passthrough_has_option(passthrough: List[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in passthrough)


def add_overlap_s2_training_defaults(args: argparse.Namespace, passthrough: List[str]) -> List[str]:
    if args.mode not in {"s2", "both"}:
        return passthrough
    out = list(passthrough)
    defaults = [
        (
            "--s2-phase-a-steps",
            os.environ.get("LOWVIS_RNN_S2_A_STEPS", DEFAULT_OVERLAP_S2_A_STEPS),
        ),
        (
            "--s2-phase-b-steps",
            os.environ.get("LOWVIS_RNN_S2_B_STEPS", DEFAULT_OVERLAP_S2_B_STEPS),
        ),
        (
            "--patience",
            os.environ.get("LOWVIS_RNN_PATIENCE", DEFAULT_OVERLAP_S2_PATIENCE),
        ),
    ]
    for option, value in defaults:
        if not passthrough_has_option(out, option):
            out.extend([option, str(value)])
    return out


def main() -> None:
    args, passthrough = parse_args()
    static_script = resolve_static_script(args)
    s2_data_dir = resolve_s2_data_dir(args)
    run_id = resolve_run_id(args)
    pretrained_ckpt = resolve_pretrained_ckpt(args)
    passthrough = add_overlap_s2_training_defaults(args, passthrough)

    static_argv = [
        str(static_script),
        "--mode",
        args.mode,
        "--run-id",
        run_id,
        "--base-path",
        args.base_path,
        "--s1-data-dir",
        args.s1_data_dir,
        "--s2-data-dir",
        s2_data_dir,
        "--ckpt-dir",
        args.ckpt_dir,
    ]
    if pretrained_ckpt:
        static_argv.extend(["--pretrained-ckpt", pretrained_ckpt])
    static_argv.extend(passthrough)

    print("[overlap-static-rnn] delegating to:", static_script, flush=True)
    print("[overlap-static-rnn] mode:", args.mode, flush=True)
    print("[overlap-static-rnn] data_source:", args.data_source, flush=True)
    print("[overlap-static-rnn] run_id:", run_id, flush=True)
    print("[overlap-static-rnn] s1_data_dir:", args.s1_data_dir, flush=True)
    print("[overlap-static-rnn] s2_data_dir:", s2_data_dir, flush=True)
    print("[overlap-static-rnn] ckpt_dir:", args.ckpt_dir, flush=True)
    print("[overlap-static-rnn] pretrained_ckpt:", pretrained_ckpt or "none", flush=True)

    if not args.runpy:
        os.execv(sys.executable, [sys.executable, *static_argv])

    old_argv = sys.argv[:]
    sys.path.insert(0, str(static_script.parent))
    try:
        sys.argv = static_argv
        runpy.run_path(str(static_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
