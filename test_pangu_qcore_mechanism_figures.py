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
import matplotlib.pyplot as plt

from plot_pangu_qcore_mechanism_ppt import (
    FIGURE_SIZE_KEYS,
    FIGURE_SPECS,
    event_forecast_state_contrast_source,
    event_observation_advantage_source,
    ordered_formats,
    qcore_argmax_source,
    style_axis,
)


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
                    "RH_925_pangu": 70.0,
                    "RH_925_tianji": 78.0,
                    "MSLP_pangu": 101000.0,
                    "MSLP_tianji": 101100.0,
                    "prs_sea": 1010.5,
                }
            )
            rows.append(
                {
                    "time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=day, hours=station),
                    "station_key": f"R{station:03d}",
                    "case_category": "pangu_hit_tianji_miss",
                    "vis_raw_m": 500.0,
                    "T2M_pangu": 285.15,
                    "T2M_tianji": 284.90,
                    "tem": 10.0,
                    "WSPD10_pangu": 2.0,
                    "WSPD10_tianji": 1.9,
                    "win_s_avg_10mi": 1.0,
                    "RH_925_pangu": 72.0,
                    "RH_925_tianji": 74.0,
                    "MSLP_pangu": 101000.0,
                    "MSLP_tianji": 101020.0,
                    "prs_sea": 1010.0,
                }
            )
    return pd.DataFrame(rows)


class EventObservationAdvantageTest(unittest.TestCase):
    def test_complete_figure_inventory_uses_mainline_style_contract(self) -> None:
        self.assertEqual(set(FIGURE_SPECS), set(FIGURE_SIZE_KEYS))
        self.assertEqual(len(FIGURE_SPECS), 13)
        self.assertEqual(plt.rcParams["font.family"], ["sans-serif"])
        self.assertEqual(plt.rcParams["font.sans-serif"][:2], ["Arial", "Helvetica"])
        fig, ax = plt.subplots()
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            style_axis(ax)
            self.assertTrue(all(ax.spines[side].get_visible() for side in ax.spines))
        finally:
            plt.close(fig)

    def test_colon_delimited_slurm_formats_are_supported(self) -> None:
        self.assertEqual(
            ordered_formats("svg:pdf:png:tiff"),
            ["svg", "pdf", "png", "tiff"],
        )

    def test_argmax_overview_uses_only_fair_endpoints_and_seed_means(self) -> None:
        rows = []
        for mask, base in (("0000", 0.10), ("1111", 0.20)):
            for index, seed in enumerate((42, 2025, 20260702)):
                rows.append(
                    {
                        "mask": mask,
                        "seed": seed,
                        "low_vis_precision_argmax": base + 0.01 * index,
                        "low_vis_recall_argmax": base + 0.10 + 0.01 * index,
                        "low_vis_csi_argmax": base + 0.02 + 0.01 * index,
                        "low_vis_fpr_argmax": base - 0.05 + 0.01 * index,
                    }
                )
        source = qcore_argmax_source(pd.DataFrame(rows))
        self.assertEqual(set(source["source"]), {"pangu", "tianji"})
        self.assertEqual(set(source["metric"]), {"Precision", "Recall", "CSI", "FPR"})
        self.assertEqual(len(source), 32)
        mean = source[
            (source["source"] == "tianji")
            & (source["metric"] == "Recall")
            & (source["seed"].astype(str) == "mean")
        ]
        self.assertAlmostEqual(float(mean.iloc[0]["value"]), 0.31)

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

    def test_disagreement_state_contrast_and_reverse_control(self) -> None:
        source = event_forecast_state_contrast_source(
            synthetic_event_samples(), iterations=200, seed=17
        ).set_index(["feature", "contrast"])
        self.assertEqual(len(source), 12)
        expected = {
            ("T2M", "tianji_hit_pangu_miss"): -1.0,
            ("T2M", "pangu_hit_tianji_miss"): -0.25,
            ("T2M", "tianji_only_minus_pangu_only"): -0.75,
            ("WSPD10", "tianji_hit_pangu_miss"): -0.5,
            ("WSPD10", "pangu_hit_tianji_miss"): -0.1,
            ("WSPD10", "tianji_only_minus_pangu_only"): -0.4,
            ("RH_925", "tianji_hit_pangu_miss"): 8.0,
            ("RH_925", "pangu_hit_tianji_miss"): 2.0,
            ("RH_925", "tianji_only_minus_pangu_only"): 6.0,
            ("MSLP", "tianji_hit_pangu_miss"): 1.0,
            ("MSLP", "pangu_hit_tianji_miss"): 0.2,
            ("MSLP", "tianji_only_minus_pangu_only"): 0.8,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(float(source.loc[key, "estimate"]), value)
            self.assertTrue(bool(source.loc[key, "ci_excludes_zero"]))
            self.assertEqual(int(source.loc[key, "represented_utc_dates"]), 12)
        self.assertEqual(
            source.loc[("T2M", "tianji_only_minus_pangu_only"), "bootstrap_unit"],
            "joint_UTC_valid_date",
        )

    def test_state_contrast_rejects_duplicate_station_time_rows(self) -> None:
        samples = synthetic_event_samples()
        samples = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate station-time"):
            event_forecast_state_contrast_source(samples, iterations=200, seed=17)

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
            fake_sbatch.write_text(
                "#!/bin/bash\n"
                "printf '%s\\n' \"$@\" > \"${SBATCH_CAPTURE:?}\"\n"
                "printf '12345\\n'\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            capture = root / "sbatch_args.txt"
            env = os.environ.copy()
            env.update(
                {
                    "BASELINE_DIR": repo.as_posix(),
                    "PAIRED_QUALITY_DIR": quality.as_posix(),
                    "OUT_DIR": (root / "out").as_posix(),
                    "REUSE_QUALITY": "auto",
                    "SBATCH_BIN": fake_sbatch.as_posix(),
                    "SBATCH_CAPTURE": capture.as_posix(),
                    "FORMATS": "svg,pdf,png,tiff",
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
            submitted_args = capture.read_text(encoding="utf-8")
            self.assertIn("FORMATS=svg:pdf:png:tiff", submitted_args)
            self.assertNotIn("FORMATS=svg,pdf", submitted_args)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
