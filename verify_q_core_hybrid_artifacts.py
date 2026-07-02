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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify 3 S1 + 24 S2 q-core factorial checkpoint/scaler/config triplets")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--hybrid-data-root", required=True)
    ap.add_argument("--s1-data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42:2025:20260702")
    ap.add_argument("--masks", default="000:001:010:011:100:101:110:111")
    ap.add_argument("--s2-a-steps", type=int, default=12000)
    ap.add_argument("--s2-b-steps", type=int, default=40000)
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
    seeds = [int(value) for value in parse_csv(args.seeds)]
    masks = parse_csv(args.masks)
    formal_matrix = len(seeds) == 3 and len(set(seeds)) == 3 and sorted(masks) == [f"{value:03b}" for value in range(8)]
    if not formal_matrix and not args.allow_smoke:
        raise ValueError("Formal acceptance requires three distinct seeds and all eight unique MTW masks")
    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    hybrid_root = Path(args.hybrid_data_root).expanduser().resolve()
    s1_data = Path(args.s1_data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_dyn = dataset_dyn(s1_data)
    hybrid_dyn = {mask: dataset_dyn(hybrid_root / f"mtw_{mask}") for mask in masks}
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        s1_run = f"exp_qcore_hybrid_{args.run_tag}_s1_seed{seed}_pm10_pm25"
        rows.append(inspect_triplet(ckpt_dir, s1_run, "s1", s1_dyn, seed, s1_data, args.s2_a_steps, args.s2_b_steps))
        for mask in masks:
            run_id = f"exp_qcore_hybrid_{args.run_tag}_mtw{mask}_seed{seed}_pm10_pm25"
            rows.append(
                inspect_triplet(
                    ckpt_dir,
                    run_id,
                    "s2",
                    hybrid_dyn[mask],
                    seed,
                    hybrid_root / f"mtw_{mask}",
                    args.s2_a_steps,
                    args.s2_b_steps,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "primary_training_artifact_audit.csv", index=False)
    expected_models = len(seeds) * (1 + len(masks))
    passed = bool(len(frame) == expected_models and frame["status"].eq("passed").all())
    report = {
        "status": "passed" if passed else "failed",
        "formal_matrix": formal_matrix,
        "expected_primary_models": 27 if formal_matrix else expected_models,
        "found_triplets": int(len(frame)),
        "passed_triplets": int(frame["status"].eq("passed").sum()),
        "required_files_per_model": ["best checkpoint", "stage scaler", "training config"],
        "required_s2_steps": {"phase_a": args.s2_a_steps, "phase_b": args.s2_b_steps},
    }
    with (out_dir / "primary_training_artifact_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if not passed:
        failures = frame.loc[frame["status"] != "passed", ["run_id", "reason"]]
        raise RuntimeError("Primary training artifact gate failed:\n" + failures.to_string(index=False))
    print(f"[OK] verified all {expected_models} primary checkpoint/scaler/config triplets in {ckpt_dir}")


if __name__ == "__main__":
    main()
