#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard acceptance gate for q-core hybrid training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).replace(",", ":").split(":") if item.strip()]


GROUP_PROFILES: Dict[str, Dict[str, object]] = {
    "mtw": {"width": 3, "dataset_prefix": "mtw", "run_mask_prefix": "mtw"},
    "mt2pw": {"width": 4, "dataset_prefix": "mt2pw", "run_mask_prefix": "mt2pw"},
    "m925b": {"width": 3, "dataset_prefix": "m925b", "run_mask_prefix": "m925b"},
    "mhtpw": {"width": 5, "dataset_prefix": "mhtpw", "run_mask_prefix": "mhtpw"},
}


def all_masks(width: int) -> List[str]:
    return [f"{value:0{width}b}" for value in range(1 << width)]


def parse_masks(value: str, width: int) -> List[str]:
    masks = all_masks(width) if str(value).strip().lower() == "all" else parse_csv(value)
    bad = [mask for mask in masks if len(mask) != width or any(bit not in "01" for bit in mask)]
    if bad:
        raise ValueError(f"Invalid {width}-bit mask(s): {bad}")
    return list(dict.fromkeys(masks))


def coupled_mt2pw_to_mtw(mask: str) -> str | None:
    if len(mask) != 4 or mask[1] != mask[2]:
        return None
    return f"{mask[0]}{mask[1]}{mask[3]}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify q-core factorial checkpoint/scaler/config triplets")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--s1-run-tag", default="", help="Run tag that owns the shared S1 checkpoints; defaults to --run-tag.")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--hybrid-data-root", required=True)
    ap.add_argument("--s1-data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-profile", default="mtw", choices=sorted(GROUP_PROFILES))
    ap.add_argument("--dataset-prefix", default="", help="Hybrid dataset directory prefix; defaults to the group profile.")
    ap.add_argument("--run-mask-prefix", default="", help="S2 run-id mask prefix; defaults to the group profile.")
    ap.add_argument("--seeds", default="42:2025:20260702")
    ap.add_argument("--masks", default="all")
    ap.add_argument("--train-masks", default="", help="Masks trained under --run-tag; default verifies every --masks entry.")
    ap.add_argument("--reuse-coupled-mtw-run-tag", default="", help="Existing MTW run tag used for mt2pw masks where T2M==MSLP.")
    ap.add_argument("--reuse-coupled-mtw-hybrid-root", default="", help="Hybrid data root for --reuse-coupled-mtw-run-tag.")
    ap.add_argument("--s2-a-steps", type=int, default=12000)
    ap.add_argument("--s2-b-steps", type=int, default=40000)
    ap.add_argument("--reused-s2-a-steps", type=int, default=12000)
    ap.add_argument("--reused-s2-b-steps", type=int, default=40000)
    ap.add_argument("--allow-smoke", action="store_true", help="Allow a reduced seed/mask matrix while retaining all file/config checks")
    return ap.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def dataset_dyn(data_dir: Path) -> int:
    cfg = read_json(data_dir / "dataset_build_config.json")
    dyn = int(cfg["dyn_vars"])
    order = cfg.get("dynamic_feature_order")
    if not isinstance(order, list) or len(order) != dyn:
        raise ValueError(f"Invalid dynamic_feature_order in {data_dir}")
    return dyn


def resolve_profile(args: argparse.Namespace) -> Dict[str, object]:
    spec = dict(GROUP_PROFILES[str(args.group_profile)])
    if str(args.dataset_prefix).strip():
        spec["dataset_prefix"] = str(args.dataset_prefix).strip()
    if str(args.run_mask_prefix).strip():
        spec["run_mask_prefix"] = str(args.run_mask_prefix).strip()
    return spec


def inspect_triplet(
    ckpt_dir: Path,
    run_id: str,
    stage: str,
    dyn: int,
    expected_seed: int,
    expected_data_dir: Path,
    s2_a_steps: int,
    s2_b_steps: int,
) -> Dict[str, object]:
    checkpoint_tag = "S1_best_score" if stage == "s1" else "S2_PhaseB_best_score"
    checkpoint = ckpt_dir / f"{run_id}_{checkpoint_tag}.pt"
    scaler = ckpt_dir / f"robust_scaler_{run_id}_{stage}_w12_dyn{dyn}_pm.pkl"
    config_path = ckpt_dir / f"{run_id}_static_rnn_config.json"
    missing = [str(path) for path in (checkpoint, scaler, config_path) if not path.is_file()]
    status = "passed"
    reason = ""
    config: Dict[str, object] = {}
    if missing:
        status = "failed"
        reason = "missing: " + "; ".join(missing)
    else:
        config = read_json(config_path)
        checks = {
            "run_id": str(config.get("run_id")) == run_id,
            "seed": int(config.get("seed", -1)) == expected_seed,
            "window_size": int(config.get("window_size", -1)) == 12,
            "data_dir": Path(str(config.get("s1_data_dir" if stage == "s1" else "s2_data_dir", ""))).resolve() == expected_data_dir.resolve(),
        }
        if stage == "s2":
            checks["s2_phase_a_steps"] = int(config.get("s2_phase_a_steps", -1)) == s2_a_steps
            checks["s2_phase_b_steps"] = int(config.get("s2_phase_b_steps", -1)) == s2_b_steps
            checks["pretrained_ckpt"] = bool(str(config.get("pretrained_ckpt", "")).strip())
        failed = [name for name, okay in checks.items() if not okay]
        if failed:
            status = "failed"
            reason = "config mismatch: " + ", ".join(failed)
    return {
        "run_id": run_id,
        "stage": stage,
        "seed": expected_seed,
        "dyn_vars": dyn,
        "data_dir": str(expected_data_dir),
        "checkpoint": str(checkpoint),
        "scaler": str(scaler),
        "config": str(config_path),
        "status": status,
        "reason": reason,
    }


def main() -> None:
    args = parse_args()
    profile = resolve_profile(args)
    width = int(profile["width"])
    dataset_prefix = str(profile["dataset_prefix"])
    run_mask_prefix = str(profile["run_mask_prefix"])
    s1_run_tag = str(args.s1_run_tag).strip() or str(args.run_tag)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    masks = parse_masks(args.masks, width)
    train_masks = parse_masks(args.train_masks, width) if str(args.train_masks).strip() else list(masks)
    unknown_train = sorted(set(train_masks) - set(masks))
    if unknown_train:
        raise ValueError(f"train masks are not in masks: {unknown_train}")
    formal_matrix = len(seeds) == 3 and len(set(seeds)) == 3 and sorted(masks) == all_masks(width)
    if not formal_matrix and not args.allow_smoke:
        raise ValueError(f"Formal acceptance requires three distinct seeds and all {1 << width} unique masks")
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    hybrid_root = Path(args.hybrid_data_root).expanduser().resolve()
    s1_data = Path(args.s1_data_dir).expanduser().resolve()
    reuse_tag = str(args.reuse_coupled_mtw_run_tag).strip()
    reuse_root = (
        Path(args.reuse_coupled_mtw_hybrid_root).expanduser().resolve()
        if str(args.reuse_coupled_mtw_hybrid_root).strip()
        else None
    )
    if reuse_tag and (str(args.group_profile) != "mt2pw" or reuse_root is None):
        raise ValueError("--reuse-coupled-mtw-run-tag requires --group-profile mt2pw and --reuse-coupled-mtw-hybrid-root")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_dyn = dataset_dyn(s1_data)
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        s1_run = f"exp_qcore_hybrid_{s1_run_tag}_s1_seed{seed}_pm10_pm25"
        rows.append(inspect_triplet(ckpt_dir, s1_run, "s1", s1_dyn, seed, s1_data, args.s2_a_steps, args.s2_b_steps))
        for mask in masks:
            provenance = "trained_here"
            data_dir = hybrid_root / f"{dataset_prefix}_{mask}"
            run_id = f"exp_qcore_hybrid_{args.run_tag}_{run_mask_prefix}{mask}_seed{seed}_pm10_pm25"
            if mask not in train_masks:
                old_mask = coupled_mt2pw_to_mtw(mask) if reuse_tag else None
                if old_mask is None or reuse_root is None:
                    raise ValueError(
                        f"Mask {mask} is not in --train-masks and cannot be resolved through coupled MTW reuse"
                    )
                provenance = "reused_coupled_mtw"
                data_dir = reuse_root / f"mtw_{old_mask}"
                run_id = f"exp_qcore_hybrid_{reuse_tag}_mtw{old_mask}_seed{seed}_pm10_pm25"
                expected_a_steps = args.reused_s2_a_steps
                expected_b_steps = args.reused_s2_b_steps
            else:
                expected_a_steps = args.s2_a_steps
                expected_b_steps = args.s2_b_steps
            dyn = dataset_dyn(data_dir)
            rows.append(
                inspect_triplet(
                    ckpt_dir,
                    run_id,
                    "s2",
                    dyn,
                    seed,
                    data_dir,
                    expected_a_steps,
                    expected_b_steps,
                )
            )
            rows[-1]["mask"] = mask
            rows[-1]["artifact_provenance"] = provenance
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "primary_training_artifact_audit.csv", index=False)
    expected_models = len(seeds) * (1 + len(masks))
    passed = bool(len(frame) == expected_models and frame["status"].eq("passed").all())
    report = {
        "status": "passed" if passed else "failed",
        "formal_matrix": formal_matrix,
        "group_profile": args.group_profile,
        "dataset_prefix": dataset_prefix,
        "run_mask_prefix": run_mask_prefix,
        "s1_run_tag": s1_run_tag,
        "train_masks": train_masks,
        "reuse_coupled_mtw_run_tag": reuse_tag,
        "expected_primary_models": expected_models,
        "found_triplets": int(len(frame)),
        "passed_triplets": int(frame["status"].eq("passed").sum()),
        "required_files_per_model": ["best checkpoint", "stage scaler", "training config"],
        "required_s2_steps": {"phase_a": args.s2_a_steps, "phase_b": args.s2_b_steps},
        "required_reused_s2_steps": {"phase_a": args.reused_s2_a_steps, "phase_b": args.reused_s2_b_steps},
    }
    with (out_dir / "primary_training_artifact_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if not passed:
        failures = frame.loc[frame["status"] != "passed", ["run_id", "reason"]]
        raise RuntimeError("Primary training artifact gate failed:\n" + failures.to_string(index=False))
    print(f"[OK] verified all {expected_models} primary checkpoint/scaler/config triplets in {ckpt_dir}")


if __name__ == "__main__":
    main()
