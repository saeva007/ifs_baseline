#!/usr/bin/env python3
"""Regression tests for the minimal Pangu--Tianji q-core+T925 chain."""

from __future__ import annotations

import os
import importlib.util
import importlib.machinery
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from analyze_q_core_hybrid_factorial import SampleSet
from analyze_q_core_t925_fair import paired_bootstrap, point_metrics

# Feature-order constants are pure metadata, while pmst_overlap_common also
# imports pvlib for unrelated solar calculations.  Keep this unit test light.
if importlib.util.find_spec("pvlib") is None:
    pvlib_stub = ModuleType("pvlib")
    pvlib_stub.__spec__ = importlib.machinery.ModuleSpec("pvlib", loader=None)
    sys.modules["pvlib"] = pvlib_stub
from pmst_overlap_common import (
    Q_CORE_NO_RH2M_DYN_FEATURES,
    Q_CORE_T925_NO_RH2M_DYN_FEATURES,
)


BASH_EXE = shutil.which("bash")
if BASH_EXE is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


def sample(scores: np.ndarray, preds: np.ndarray) -> SampleSet:
    scores = np.asarray(scores, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.int64)
    y = np.tile(np.asarray([0, 1, 2, 2], dtype=np.int64), len(scores) // 4)
    time = pd.date_range("2025-01-01", periods=len(scores), freq="6h", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": time.astype(str),
            "station_id": [f"S{i:04d}" for i in range(len(scores))],
            "time_utc": time,
            "time_key": time.strftime("%Y-%m-%d %H:%M:%S"),
            "station_key": [f"S{i:04d}" for i in range(len(scores))],
        }
    )
    return SampleSet(frame=frame, y=y, score=scores, pred=preds)


class QCoreT925FairTest(unittest.TestCase):
    def test_t925_layout_adds_only_t925_to_qcore(self) -> None:
        base = list(Q_CORE_NO_RH2M_DYN_FEATURES)
        t925 = list(Q_CORE_T925_NO_RH2M_DYN_FEATURES)
        self.assertEqual(len(base), 17)
        self.assertEqual(len(t925), 18)
        self.assertEqual([name for name in t925 if name not in base], ["T_925"])
        self.assertNotIn("RH2M", t925)

    def test_point_and_block_bootstrap_favor_better_tianji_scores(self) -> None:
        seeds = [42, 2025, 20260702]
        # Four classes per day pattern: two low-vis and two clear cases.
        pangu_scores = np.tile(np.asarray([0.60, 0.45, 0.55, 0.35]), 8)
        tianji_scores = np.tile(np.asarray([0.90, 0.80, 0.20, 0.10]), 8)
        pangu_preds = np.where(pangu_scores >= 0.5, 1, 2)
        tianji_preds = np.where(tianji_scores >= 0.5, 1, 2)
        val = {}
        test = {}
        for seed in seeds:
            val[(seed, "pangu")] = sample(pangu_scores, pangu_preds)
            val[(seed, "tianji")] = sample(tianji_scores, tianji_preds)
            test[(seed, "pangu")] = sample(pangu_scores, pangu_preds)
            test[(seed, "tianji")] = sample(tianji_scores, tianji_preds)
        metrics, _, _, thresholds, target = point_metrics(val, test, seeds, ece_bins=5)
        means = metrics.groupby("source")["low_vis_ap"].mean()
        self.assertGreater(means["tianji"], means["pangu"])
        self.assertGreaterEqual(target, 0.0)
        draws, summary, info = paired_bootstrap(
            test,
            thresholds,
            seeds,
            iterations=30,
            rng_seed=7,
            max_rows=0,
            initial_bins=256,
            max_bins=2048,
            max_error=0.01,
        )
        ap = summary[summary["metric"] == "low_vis_ap"].iloc[0]
        self.assertGreater(float(ap["ci_low"]), 0.0)
        self.assertEqual(len(draws), 30 * 3)
        self.assertGreater(info["n_valid_dates"], 1)

    def test_cli_smoke_writes_formal_tables_and_standalone_figures(self) -> None:
        repo = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = root / "eval" / "seed_42"
            seed_dir.mkdir(parents=True)
            n = 32
            y = np.tile(np.asarray([0, 1, 2, 2], dtype=np.int64), n // 4)
            time = pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC")
            station = [f"S{i:04d}" for i in range(n)]
            source_scores = {
                "pangu2025_q_core_t925_no_rh2m": np.tile(
                    np.asarray([0.60, 0.45, 0.55, 0.35]), n // 4
                ),
                "tianji": np.tile(np.asarray([0.90, 0.80, 0.20, 0.10]), n // 4),
            }
            for tag, scores in source_scores.items():
                pred = np.where(scores >= 0.5, 1, 2)
                frame = pd.DataFrame(
                    {
                        "time": time.astype(str),
                        "station_id": station,
                        "y_cls": y,
                        "vis_raw_m": np.where(y <= 1, 500.0, 5000.0),
                        "pred": pred,
                        "p_fog": scores * 0.45,
                        "p_mist": scores * 0.55,
                        "p_clear": 1.0 - scores,
                    }
                )
                frame.to_csv(seed_dir / f"per_sample_{tag}.csv", index=False)
                frame.to_csv(seed_dir / f"per_sample_val_{tag}.csv", index=False)
            out_dir = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "analyze_q_core_t925_fair.py"),
                    "--eval-root",
                    str(root / "eval"),
                    "--out-dir",
                    str(out_dir),
                    "--seeds",
                    "42",
                    "--bootstrap-iters",
                    "30",
                    "--ap-hist-initial-bins",
                    "256",
                    "--ap-hist-max-bins",
                    "2048",
                    "--ap-hist-max-error",
                    "0.01",
                    "--formats",
                    "svg:png",
                    "--dpi",
                    "300",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((out_dir / "qcore_t925_metrics_by_seed.csv").is_file())
            self.assertTrue((out_dir / "qcore_t925_bootstrap_summary.csv").is_file())
            self.assertTrue((out_dir / "01_qcore_t925_lowvis_ap.svg").is_file())
            self.assertTrue((out_dir / "02_qcore_t925_matched_fpr_recall.png").is_file())
            self.assertTrue((out_dir / "03_qcore_t925_argmax_lowvis_overview.svg").is_file())

    @unittest.skipUnless(BASH_EXE, "bash is required for launcher dry-run regression")
    def test_launcher_submits_only_two_source_three_seed_chain(self) -> None:
        repo = Path(__file__).resolve().parent
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": "unit_test_qcore_t925_fair",
                "DRY_RUN": "1",
                "BASELINE_DIR": str(repo),
                "BASE": "/public/home/putianshu/vis_mlp",
            }
        )
        result = subprocess.run(
            [str(BASH_EXE), str(repo / "submit_q_core_t925_fair_experiment.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertEqual(output.count("[DRY-RUN]"), 17)
        self.assertEqual(output.count("EXPERIMENT=s1_q_core_t925_no_rh2m"), 3)
        self.assertEqual(output.count("EXPERIMENT=s2_tianji_q_core_t925_no_rh2m"), 3)
        self.assertEqual(output.count("EXPERIMENT=s2_pangu2025_q_core_t925_no_rh2m"), 3)
        self.assertEqual(output.count("SOURCE_SCOPE=pangu_tianji"), 5)
        self.assertNotIn("sub_ifs_data.slurm", output)
        self.assertNotIn("era5_data", output)
        self.assertNotIn("HYBRID_DATA_ROOT", output)
        self.assertIn("FORMATS=svg:pdf:png:tiff", output)
        self.assertIn("analysis_job=dry_joint_analysis", output)


if __name__ == "__main__":
    unittest.main()
