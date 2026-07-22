#!/usr/bin/env python3
"""Regression tests for the zero-training paired source-quality analysis."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze_q_core_paired_source_quality import (
    PRESSURE_LEVEL_FEATURES,
    QualitySplit,
    pressure_level_quality,
    scope_masks,
    surface_observation_quality,
)
from preflight_q_core_paired_source_quality import REQUIRED_FEATURES, validate_quality_dataset
from preflight_q_core_t925_diagnostic_inputs import EXPECTED_UNIT_POLICY, REQUIRED_ARTIFACTS
from plot_pangu_qcore_t925_evidence_story import (
    pressure_ratio_source,
    pressure_rmse_source,
)


BASH_EXE = shutil.which("bash")
if BASH_EXE is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


def synthetic_split() -> QualitySplit:
    n = 24
    keys = pd.DataFrame(
        {
            "time_utc": pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC"),
            "station_key": [f"S{i:03d}" for i in range(n)],
        }
    )
    reference = {
        "T_925": 280.0,
        "RH_925": 80.0,
        "U_925": 3.0,
        "V_925": 4.0,
        "WSPD925": 5.0,
        "DP_1000": 275.0,
        "DP_925": 270.0,
        "Q_1000": 0.008,
        "Q_925": 0.006,
    }
    values = {}
    for source, error in (
        ("pangu", 2.0),
        ("tianji", 1.0),
        ("era5_reference_analysis", 0.0),
    ):
        current = {}
        for feature in PRESSURE_LEVEL_FEATURES:
            scale = 0.001 if feature in {"Q_1000", "Q_925"} else 1.0
            current[feature] = np.full(n, reference[feature] + error * scale, dtype=np.float64)
        current.update(
            {
                "T2M": np.full(n, 273.15 + error, dtype=np.float64),
                "WSPD10": np.full(n, error, dtype=np.float64),
                "MSLP": np.full(n, 100000.0 + 100.0 * error, dtype=np.float64),
            }
        )
        values[source] = current
    return QualitySplit(
        keys=keys,
        positions={},
        values=values,
        visibility_m=np.where(np.arange(n) % 2 == 0, 500.0, 5000.0),
        orography_m=np.where(np.arange(n) % 2 == 0, 100.0, 800.0),
    )


class PairedSourceQualityTest(unittest.TestCase):
    @staticmethod
    def write_preflight_dataset(root: Path, source: str) -> Path:
        data_dir = root / source
        data_dir.mkdir(parents=True)
        for artifact in REQUIRED_ARTIFACTS:
            (data_dir / artifact).write_bytes(b"test")
        native = [feature for feature in REQUIRED_FEATURES if feature != "RH_925"]
        config = {
            "feature_set": "q_core_t925_no_rh2m",
            "dynamic_feature_order": list(REQUIRED_FEATURES),
            "dyn_vars": len(REQUIRED_FEATURES),
            "window": 12,
            "fe_dim": 8,
            "canonical_unit_policy": EXPECTED_UNIT_POLICY,
            "canonical_dynamic_units": {"T_925": "K", "Q_925": "kg kg-1", "RH_925": "%"},
            "time_coordinate": "UTC",
            "native_source_features": native,
            "derived_source_features": ["RH_925"],
        }
        if source == "pangu":
            config["source_feature_provenance"] = {
                "RH_925": "derived from native T_925 and Q_925"
            }
            config["source_forecast_lead"] = {
                "available": True,
                "min_hours": 12.0,
                "max_hours": 23.0,
            }
        else:
            config["native_source_features"].append("RH_925")
            config["derived_source_features"] = []
        (data_dir / "dataset_build_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return data_dir

    def test_quality_preflight_requires_complete_pressure_layout(self) -> None:
        data_dir = Path(__file__).resolve().parent / f".tmp_paired_quality_{uuid.uuid4().hex}"
        data_dir.mkdir()
        try:
            for artifact in REQUIRED_ARTIFACTS:
                (data_dir / artifact).write_bytes(b"test")
            config = {
                "feature_set": "q_core_t925_no_rh2m",
                "dynamic_feature_order": list(REQUIRED_FEATURES),
                "dyn_vars": len(REQUIRED_FEATURES),
                "window": 12,
                "fe_dim": 8,
                "canonical_unit_policy": EXPECTED_UNIT_POLICY,
                "canonical_dynamic_units": {"T_925": "K", "Q_925": "kg kg-1", "RH_925": "%"},
                "time_coordinate": "UTC",
                "native_source_features": ["T_925", "Q_925"],
                "derived_source_features": ["RH_925"],
                "source_feature_provenance": {"RH_925": "derived from native T_925 and Q_925"},
                "source_forecast_lead": {"available": True, "min_hours": 12.0, "max_hours": 23.0},
            }
            (data_dir / "dataset_build_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            checked = validate_quality_dataset(data_dir, "pangu")
            self.assertEqual(checked["dyn_vars"], len(REQUIRED_FEATURES))
            config["dynamic_feature_order"].remove("U_925")
            config["dyn_vars"] = len(config["dynamic_feature_order"])
            (data_dir / "dataset_build_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(Exception, "U_925"):
                validate_quality_dataset(data_dir, "pangu")
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter dry-run regression")
    def test_submitter_dry_run_submits_exactly_one_no_training_job(self) -> None:
        root = Path(__file__).resolve().parent / f".tmp_paired_submit_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            pangu = self.write_preflight_dataset(root, "pangu")
            tianji = self.write_preflight_dataset(root, "tianji")
            era5 = self.write_preflight_dataset(root, "era5_reference_analysis")
            env = os.environ.copy()
            env.update(
                {
                    "RUN_TAG": "unit_test_qcore_pair_quality",
                    "DRY_RUN": "1",
                    "BASELINE_DIR": str(Path(__file__).resolve().parent),
                    "PREFLIGHT_PYTHON": sys.executable,
                    "PANGU_DATA_DIR": str(pangu),
                    "TIANJI_DATA_DIR": str(tianji),
                    "ERA5_DATA_DIR": str(era5),
                    "OUT_ROOT": str(root / "out"),
                }
            )
            result = subprocess.run(
                [str(BASH_EXE), str(Path(__file__).resolve().parent / "submit_q_core_paired_source_quality.sh")],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout + result.stderr
            self.assertIn("TRAINING_JOBS=0", output)
            self.assertEqual(output.count("[DRY-RUN]"), 1)
            self.assertIn("qcore_pair_qa", output)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_all_pressure_level_fields_use_complete_paired_rows(self) -> None:
        table, qc = pressure_level_quality(
            synthetic_split(), iterations=120, seed=11, low_vis_threshold_m=1000.0
        )
        self.assertEqual(len(table), (len(PRESSURE_LEVEL_FEATURES) + 1) * 3)
        self.assertEqual(len(qc), len(PRESSURE_LEVEL_FEATURES) * 3)
        self.assertEqual(
            set(table["scope"]),
            {"all_paired_test", "true_low_visibility", "elevation_le_500m"},
        )
        self.assertTrue(np.all(table["delta_pangu_minus_tianji"].to_numpy(dtype=float) > 0.0))
        self.assertTrue(np.all(table["delta_ci_low"].to_numpy(dtype=float) > 0.0))
        self.assertTrue(
            np.allclose(
                table["tianji_to_pangu_ratio"].to_numpy(dtype=float),
                0.5,
            )
        )
        self.assertTrue(
            np.all(
                table["tianji_to_pangu_ratio_ci_high"].to_numpy(dtype=float)
                < 1.0
            )
        )
        t925 = table[table["feature"] == "T_925"]
        self.assertTrue(np.all(~t925["original_q_core_input"].astype(bool)))
        wspd = table[table["feature"] == "WSPD925"]
        self.assertTrue(np.all(~wspd["independent_evidence"].astype(bool)))
        self.assertTrue(np.all(~qc["excluded_from_primary_rmse"].astype(bool)))

        ratio = pressure_ratio_source(table)
        absolute = pressure_rmse_source(table)
        self.assertEqual(len(ratio), 4 * 2)
        self.assertEqual(len(absolute), 4 * 2 * 2)
        self.assertEqual(
            set(ratio["feature"]),
            {"T_925", "Q_1000", "Q_925", "UV_925_VECTOR"},
        )
        self.assertEqual(
            set(ratio["scope"]),
            {"all_paired_test", "true_low_visibility"},
        )

    def test_low_visibility_scope_excludes_exact_1000m_boundary(self) -> None:
        split = synthetic_split()
        split.visibility_m[:4] = np.array([499.0, 999.0, 1000.0, 1001.0])
        masks = scope_masks(split, low_vis_threshold_m=1000.0)
        self.assertEqual(masks["true_low_visibility"][:4].tolist(), [True, True, False, False])

    def test_surface_table_includes_era5_on_same_observation_rows(self) -> None:
        split = synthetic_split()

        def fake_attach(frame, obs_root, paper_eval_dir):
            out = frame.copy()
            out["tem"] = 0.0
            out["win_s_avg_10mi"] = 0.0
            out["prs_sea"] = 1000.0
            return out, {"chosen_time_shift_hours": 0, "matched_rows": len(out)}

        with patch(
            "analyze_q_core_paired_source_quality.joint.hybrid_analysis.attach_observations",
            side_effect=fake_attach,
        ):
            source, pairs, info = surface_observation_quality(
                split,
                Path("/synthetic/obs"),
                Path("/synthetic/paper_eval"),
                iterations=120,
                seed=31,
                low_vis_threshold_m=1000.0,
            )
        self.assertEqual(len(source), 3 * 3 * 3)
        self.assertEqual(len(pairs), 3 * 3 * 3)
        self.assertEqual(set(source["source"]), {"pangu", "tianji", "era5_reference_analysis"})
        all_t2m = source[
            (source["feature"] == "T2M") & (source["scope"] == "all_paired_test")
        ].set_index("source")
        self.assertAlmostEqual(float(all_t2m.loc["pangu", "rmse"]), 2.0)
        self.assertAlmostEqual(float(all_t2m.loc["tianji", "rmse"]), 1.0)
        self.assertAlmostEqual(float(all_t2m.loc["era5_reference_analysis", "rmse"]), 0.0)
        p_vs_t = pairs[
            (pairs["feature"] == "T2M")
            & (pairs["scope"] == "all_paired_test")
            & (pairs["left_source"] == "pangu")
            & (pairs["right_source"] == "tianji")
        ].iloc[0]
        self.assertAlmostEqual(float(p_vs_t["delta_left_minus_right"]), 1.0)
        self.assertGreater(float(p_vs_t["delta_ci_low"]), 0.0)
        self.assertIn("not a third forecast", info["era5_role"])


if __name__ == "__main__":
    unittest.main()
