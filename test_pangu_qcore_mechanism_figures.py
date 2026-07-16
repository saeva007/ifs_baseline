#!/usr/bin/env python3
"""Regression tests for the standalone q-core manuscript figures."""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from plot_pangu_qcore_mechanism_ppt import event_observation_advantage_source


BASH_EXE = shutil.which("bash")
if BASH_EXE is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


def synthetic_event_samples() -> pd.DataFrame:
    rows = []
    for day in range(12):
        for station in range(10):
            rows.append(
                {
                    "time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=day, hours=station),
                    "station_key": f"S{station:03d}",
                    "case_category": "tianji_hit_pangu_miss",
                    "vis_raw_m": 500.0,
                    "T2M_pangu": 285.15,
                    "T2M_tianji": 284.15,
                    "tem": 10.0,
                    "WSPD10_pangu": 2.0,
                    "WSPD10_tianji": 1.5,
                    "win_s_avg_10mi": 1.0,
                    "MSLP_pangu": 101000.0,
                    "MSLP_tianji": 101000.0,
                    "prs_sea": 1000.0,
                }
            )
    return pd.DataFrame(rows)


class EventObservationAdvantageTest(unittest.TestCase):
    def test_unit_conversion_and_utc_date_bootstrap(self) -> None:
        source = event_observation_advantage_source(
            synthetic_event_samples(), iterations=200, seed=17
        ).set_index("feature")
        self.assertEqual(set(source.index), {"T2M", "WSPD10", "MSLP"})
        self.assertTrue(np.all(source["target_category_rows"].to_numpy() == 120))
        self.assertTrue(np.all(source["n_complete"].to_numpy() == 120))
        self.assertTrue(np.all(source["represented_utc_dates"].to_numpy() == 12))
        self.assertAlmostEqual(float(source.loc["T2M", "pangu_rmse"]), 2.0)
        self.assertAlmostEqual(float(source.loc["T2M", "tianji_rmse"]), 1.0)
        self.assertAlmostEqual(
            float(source.loc["T2M", "relative_rmse_reduction_percent"]), 50.0
        )
        self.assertAlmostEqual(
            float(source.loc["WSPD10", "relative_rmse_reduction_percent"]), 50.0
        )
        self.assertAlmostEqual(
            float(source.loc["MSLP", "relative_rmse_reduction_percent"]), 0.0
        )
        self.assertTrue(bool(source.loc["T2M", "ci_excludes_zero"]))
        self.assertFalse(bool(source.loc["MSLP", "ci_excludes_zero"]))

    def test_rejects_non_low_visibility_target_rows(self) -> None:
        samples = synthetic_event_samples()
        samples.loc[0, "vis_raw_m"] = 1000.0
        with self.assertRaisesRegex(ValueError, "visibility <1000 m"):
            event_observation_advantage_source(samples, iterations=200, seed=17)

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter regression")
    def test_reuse_quality_submits_without_empty_dependency_array(self) -> None:
        repo = Path(__file__).resolve().parent
        root = repo / f".tmp_qcore_story_submit_{uuid.uuid4().hex}"
        quality = root / "quality"
        quality.mkdir(parents=True)
        try:
            (quality / "paired_source_quality_report.json").write_text(
                "{}\n", encoding="utf-8"
            )
            fake_sbatch = root / "fake_sbatch.sh"
            fake_sbatch.write_text("#!/bin/bash\nprintf '12345\\n'\n", encoding="utf-8")
            fake_sbatch.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "BASELINE_DIR": repo.as_posix(),
                    "PAIRED_QUALITY_DIR": quality.as_posix(),
                    "OUT_DIR": (root / "out").as_posix(),
                    "REUSE_QUALITY": "auto",
                    "SBATCH_BIN": fake_sbatch.as_posix(),
                }
            )
            result = subprocess.run(
                [str(BASH_EXE), (repo / "submit_pangu_qcore_evidence_story.sh").as_posix()],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout + result.stderr
            self.assertIn("quality_job=reused", output)
            self.assertIn("plot_job=12345", output)
            self.assertNotIn("unbound variable", output)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
