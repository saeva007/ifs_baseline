#!/usr/bin/env python3
"""Self-contained tests for q-core hybrid construction and attribution math."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_q_core_ale import centered_curve
from analyze_q_core_hybrid_factorial import pair_interactions, shapley_values
from build_q_core_hybrid_factorial import EXPECTED_ORDER, GROUPS
from pmst_overlap_common import (
    CANONICAL_UNIT_POLICY_VERSION,
    PM_QC_POLICY_VERSION,
    compute_fog_features_pmst,
)


WINDOW = 12
STATIC_DIM = 6
CYCLICAL_DIM = 4


@contextmanager
def workspace_temp_dir():
    path = Path(__file__).resolve().parent / f".tmp_qcore_hybrid_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_source(root: Path, name: str, dyn: np.ndarray, y: np.ndarray, canonical_pangu: bool) -> Path:
    data_dir = root / name
    data_dir.mkdir()
    fog = compute_fog_features_pmst(dyn, WINDOW, len(EXPECTED_ORDER), EXPECTED_ORDER)
    rng = np.random.default_rng(991)
    static = rng.normal(size=(len(dyn), STATIC_DIM)).astype(np.float32)
    cyc = rng.normal(size=(len(dyn), CYCLICAL_DIM)).astype(np.float32)
    x = np.concatenate([dyn.reshape(len(dyn), -1), static, fog, cyc], axis=1).astype(np.float32)
    np.save(data_dir / "X_train.npy", x)
    np.save(data_dir / "y_train.npy", y.astype(np.float32))
    pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=len(dyn), freq="h"),
            "station_id": np.arange(len(dyn)) + 50000,
            "lat": 30.0,
            "lon": 110.0,
        }
    ).to_csv(data_dir / "meta_train.csv", index=False)
    cfg = {
        "dataset": name,
        "feature_set": "q_core_no_rh2m",
        "dynamic_feature_order": EXPECTED_ORDER,
        "dyn_vars": len(EXPECTED_ORDER),
        "window": WINDOW,
        "fe_dim": int(fog.shape[1] + CYCLICAL_DIM),
        "zero_filled_pmst_features": [],
        "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
        "pm_qc_policy": PM_QC_POLICY_VERSION,
        "time_coordinate": "UTC",
    }
    if canonical_pangu:
        cfg["source_forecast_lead"] = {"min_hours": 12.0, "max_hours": 23.0, "provenance": "unit test"}
    with (data_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return data_dir


class HybridBuilderTest(unittest.TestCase):
    def test_factorial_endpoints_and_group_isolation(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(17)
            n, dyn_n = 10, len(EXPECTED_ORDER)
            p_dyn = rng.normal(size=(n, WINDOW, dyn_n)).astype(np.float32)
            t_dyn = p_dyn.copy()
            for group_i, group in enumerate(("M", "T", "W"), start=1):
                for feature in GROUPS[group]:
                    t_dyn[:, :, EXPECTED_ORDER.index(feature)] += np.float32(group_i * 0.25)
            y = np.asarray([0, 1, 2, 2, 2, 0, 1, 2, 2, 1], dtype=np.float32)
            pangu = write_source(root, "pangu", p_dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", t_dyn, y, canonical_pangu=False)
            out = root / "hybrids"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(out),
                "--splits",
                "train",
                "--masks",
                "all",
                "--chunk-rows",
                "3",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            p_x = np.load(pangu / "X_train.npy")
            t_x = np.load(tianji / "X_train.npy")
            self.assertTrue(np.array_equal(np.load(out / "mtw_000" / "X_train.npy"), p_x))
            self.assertTrue(np.array_equal(np.load(out / "mtw_111" / "X_train.npy"), t_x))

            moisture = np.load(out / "mtw_100" / "X_train.npy")[:, : WINDOW * dyn_n].reshape(n, WINDOW, dyn_n)
            for idx, feature in enumerate(EXPECTED_ORDER):
                expected = t_dyn[:, :, idx] if feature in GROUPS["M"] else p_dyn[:, :, idx]
                self.assertTrue(np.array_equal(moisture[:, :, idx], expected), feature)
            with (out / "hybrid_factorial_manifest.json").open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["status"], "passed")

            audit_cmd = cmd + ["--audit-only"]
            subprocess.run(audit_cmd, check=True, capture_output=True, text=True)

    def test_metadata_mismatch_is_rejected(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(5)
            dyn = rng.normal(size=(4, WINDOW, len(EXPECTED_ORDER))).astype(np.float32)
            y = np.asarray([0, 1, 2, 2], dtype=np.float32)
            pangu = write_source(root, "pangu", dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", dyn, y, canonical_pangu=False)
            meta = pd.read_csv(tianji / "meta_train.csv")
            meta.loc[0, "station_id"] = 99999
            meta.to_csv(tianji / "meta_train.csv", index=False)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(root / "out"),
                "--splits",
                "train",
                "--masks",
                "000",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("metadata/order differ", proc.stderr + proc.stdout)


class ArtifactAndAleTest(unittest.TestCase):
    def test_centered_ale_has_weighted_zero_mean(self) -> None:
        curve = centered_curve(np.asarray([0.1, -0.02, 0.04]), np.asarray([10, 20, 30]))
        self.assertAlmostEqual(float(np.average(curve, weights=[10, 20, 30])), 0.0, places=12)

    def test_formal_artifact_gate_accepts_27_complete_triplets(self) -> None:
        with workspace_temp_dir() as root:
            ckpt = root / "checkpoints"
            ckpt.mkdir()
            s1_data = root / "s1"
            s1_data.mkdir()
            hybrid = root / "hybrid"
            hybrid.mkdir()
            data_cfg = {"dyn_vars": len(EXPECTED_ORDER), "dynamic_feature_order": EXPECTED_ORDER}
            (s1_data / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            seeds = (42, 2025, 20260702)
            masks = tuple(f"{value:03b}" for value in range(8))
            for mask in masks:
                data_dir = hybrid / f"mtw_{mask}"
                data_dir.mkdir()
                (data_dir / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            for seed in seeds:
                s1_run = f"exp_qcore_hybrid_unit_s1_seed{seed}_pm10_pm25"
                (ckpt / f"{s1_run}_S1_best_score.pt").touch()
                (ckpt / f"robust_scaler_{s1_run}_s1_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                (ckpt / f"{s1_run}_static_rnn_config.json").write_text(
                    json.dumps({"run_id": s1_run, "seed": seed, "window_size": 12, "s1_data_dir": str(s1_data)}),
                    encoding="utf-8",
                )
                for mask in masks:
                    run_id = f"exp_qcore_hybrid_unit_mtw{mask}_seed{seed}_pm10_pm25"
                    (ckpt / f"{run_id}_S2_PhaseB_best_score.pt").touch()
                    (ckpt / f"robust_scaler_{run_id}_s2_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                    (ckpt / f"{run_id}_static_rnn_config.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "window_size": 12,
                                "s2_data_dir": str(hybrid / f"mtw_{mask}"),
                                "s2_phase_a_steps": 12000,
                                "s2_phase_b_steps": 40000,
                                "pretrained_ckpt": "s1.pt",
                            }
                        ),
                        encoding="utf-8",
                    )
            out = root / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "verify_q_core_hybrid_artifacts.py"),
                    "--run-tag",
                    "unit",
                    "--checkpoint-dir",
                    str(ckpt),
                    "--hybrid-data-root",
                    str(hybrid),
                    "--s1-data-dir",
                    str(s1_data),
                    "--out-dir",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((out / "primary_training_artifact_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["passed_triplets"], 27)


class AttributionMathTest(unittest.TestCase):
    def test_exact_shapley_efficiency_and_values(self) -> None:
        values = {}
        for i in range(8):
            mask = f"{i:03b}"
            values[mask] = 1.0 + 0.1 * int(mask[0]) + 0.2 * int(mask[1]) + 0.3 * int(mask[2])
        phi = shapley_values(values)
        self.assertAlmostEqual(phi["M"], 0.1)
        self.assertAlmostEqual(phi["T"], 0.2)
        self.assertAlmostEqual(phi["W"], 0.3)
        self.assertAlmostEqual(sum(phi.values()), values["111"] - values["000"])
        for value in pair_interactions(values).values():
            self.assertAlmostEqual(value, 0.0)

    def test_analysis_cli_smoke(self) -> None:
        with workspace_temp_dir() as root:
            eval_root = root / "eval"
            seed_dir = eval_root / "seed_42"
            seed_dir.mkdir(parents=True)
            data_root = root / "data"
            data_root.mkdir()
            rng = np.random.default_rng(404)
            specs = {}
            test_times = pd.date_range("2025-01-01", periods=120, freq="6h")
            for mask in (f"{i:03b}" for i in range(8)):
                data_dir = data_root / mask
                data_dir.mkdir()
                np.save(data_dir / "X_test.npy", rng.normal(size=(120, WINDOW * len(EXPECTED_ORDER) + 10)).astype(np.float32))
                pd.DataFrame(
                    {"time": test_times, "station_id": 50000 + np.arange(120)}
                ).to_csv(data_dir / "meta_test.csv", index=False)
                with (data_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {"dynamic_feature_order": EXPECTED_ORDER, "dyn_vars": len(EXPECTED_ORDER), "window": WINDOW},
                        f,
                    )
                specs[f"qcore_hybrid_{mask}"] = {"data_dir": str(data_dir)}
            era5_dir = data_root / "era5"
            era5_dir.mkdir()
            np.save(
                era5_dir / "X_test.npy",
                np.load(data_root / "111" / "X_test.npy")[:110],
            )
            pd.DataFrame(
                {"time": test_times[:110], "station_id": 50000 + np.arange(110)}
            ).to_csv(era5_dir / "meta_test.csv", index=False)
            with (era5_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"dynamic_feature_order": EXPECTED_ORDER, "dyn_vars": len(EXPECTED_ORDER), "window": WINDOW},
                    f,
                )
            with (seed_dir / "run_config.json").open("w", encoding="utf-8") as f:
                json.dump({"specs": specs}, f)

            for split, n in (("val", 60), ("test", 120)):
                y = np.asarray(([0, 1, 2, 2, 2, 2] * ((n + 5) // 6))[:n], dtype=np.int64)
                time = pd.date_range("2025-01-01", periods=n, freq="6h")
                noise = rng.normal(0.0, 0.16, size=n)
                for i in range(8):
                    mask = f"{i:03b}"
                    improvement = 0.03 * int(mask[0]) + 0.015 * int(mask[1]) + 0.01 * int(mask[2])
                    score = np.clip(0.16 + 0.46 * (y <= 1) + noise + improvement * (2 * (y <= 1) - 1), 0.001, 0.999)
                    probs = np.column_stack([0.58 * score, 0.42 * score, 1.0 - score])
                    pred = np.argmax(probs, axis=1)
                    frame = pd.DataFrame(
                        {
                            "time": time,
                            "station_id": 50000 + np.arange(n),
                            "y_cls": y,
                            "vis_raw_m": np.where(y == 0, 300.0, np.where(y == 1, 750.0, 5000.0)),
                            "pred": pred,
                            "p_fog": probs[:, 0],
                            "p_mist": probs[:, 1],
                            "p_clear": probs[:, 2],
                        }
                    )
                    name = f"per_sample_val_qcore_hybrid_{mask}.csv" if split == "val" else f"per_sample_qcore_hybrid_{mask}.csv"
                    frame.to_csv(seed_dir / name, index=False)
            out = root / "analysis"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "analyze_q_core_hybrid_factorial.py"),
                "--eval-root",
                str(eval_root),
                "--out-dir",
                str(out),
                "--seeds",
                "42",
                "--bootstrap-iters",
                "5",
                "--era5-data-dir",
                str(era5_dir),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertTrue((out / "hybrid_exact_shapley_effects.csv").is_file())
            self.assertTrue((out / "event_case_control_samples.csv.gz").is_file())
            self.assertTrue((out / "q_reference_analysis_quality_and_extreme_placement.csv").is_file())
            self.assertTrue((out / "fig_q_core_endpoint_reliability.png").is_file())
            with (out / "hybrid_factorial_analysis_report.json").open("r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["moisture_followup_gate"]["formal_three_seed_analysis"])


if __name__ == "__main__":
    unittest.main()
