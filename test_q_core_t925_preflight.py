#!/usr/bin/env python3
"""Regression tests for the login-node T925 diagnostic input preflight."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from preflight_q_core_t925_diagnostic_inputs import (
    EXPECTED_UNIT_POLICY,
    PreflightError,
    REQUIRED_ARTIFACTS,
    validate_dataset,
)


BASH_EXE = shutil.which("bash")
if BASH_EXE is None and (Path("C:/Program Files/Git/bin/bash.exe")).is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


class T925PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / f".tmp_t925_preflight_{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def write_dataset(self, source: str, policy: str = EXPECTED_UNIT_POLICY) -> Path:
        data_dir = self.root / source
        data_dir.mkdir()
        for name in REQUIRED_ARTIFACTS:
            (data_dir / name).write_bytes(b"test")
        config = {
            "feature_set": "source_full",
            "dynamic_feature_order": ["T2M", "WSPD10", "T_925", "Q_925", "RH_925"],
            "dyn_vars": 5,
            "window": 12,
            "fe_dim": 8,
            "canonical_unit_policy": policy,
            "canonical_dynamic_units": {
                "T_925": "K",
                "Q_925": "kg kg-1",
                "RH_925": "%",
            },
            "time_coordinate": "UTC",
            "native_source_features": ["T_925", "Q_925", "RH_925"],
            "derived_source_features": [],
        }
        if source == "pangu":
            config["source_forecast_lead"] = {
                "available": True,
                "min_hours": 12.0,
                "max_hours": 23.0,
            }
        (data_dir / "dataset_build_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return data_dir

    def test_valid_layouts_pass_for_all_sources(self) -> None:
        for source in ("pangu", "tianji", "era5_reference_analysis"):
            with self.subTest(source=source):
                config = validate_dataset(self.write_dataset(source), source)
                self.assertEqual(config["canonical_unit_policy"], EXPECTED_UNIT_POLICY)

    def test_old_unit_policy_fails_before_submission(self) -> None:
        data_dir = self.write_dataset("pangu", "pmst_canonical_units_v1")
        with self.assertRaisesRegex(PreflightError, "rebuild required"):
            validate_dataset(data_dir, "pangu")

    def test_missing_analysis_artifact_fails_before_submission(self) -> None:
        data_dir = self.write_dataset("tianji")
        (data_dir / "X_test.npy").unlink()
        with self.assertRaisesRegex(PreflightError, "X_test.npy"):
            validate_dataset(data_dir, "tianji")

    def test_missing_observation_supplement_feature_fails_preflight(self) -> None:
        data_dir = self.write_dataset("tianji")
        cfg_path = data_dir / "dataset_build_config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        config["dynamic_feature_order"].remove("WSPD10")
        config["dyn_vars"] = len(config["dynamic_feature_order"])
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "WSPD10"):
            validate_dataset(data_dir, "tianji")

    def test_noncanonical_pangu_lead_fails(self) -> None:
        data_dir = self.write_dataset("pangu")
        cfg_path = data_dir / "dataset_build_config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        config["source_forecast_lead"]["max_hours"] = 24.0
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "12--23 h"):
            validate_dataset(data_dir, "pangu")

    def test_new_pangu_provenance_metadata_must_mark_rh925_derived(self) -> None:
        data_dir = self.write_dataset("pangu")
        cfg_path = data_dir / "dataset_build_config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        config["source_feature_provenance"] = {"RH_925": "native/read directly from source product"}
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "RH_925 as derived"):
            validate_dataset(data_dir, "pangu")

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter dry-run regression")
    def test_auto_submitter_rebuilds_only_failed_source_and_never_trains(self) -> None:
        pangu = self.write_dataset("pangu", "pmst_canonical_units_v1")
        tianji = self.write_dataset("tianji")
        era5 = self.write_dataset("era5_reference_analysis")
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": "unit_test_t925_auto",
                "DRY_RUN": "1",
                "BUILD_DATA": "auto",
                "BASELINE_DIR": str(Path(__file__).resolve().parent),
                "PREFLIGHT_PYTHON": sys.executable,
                "PANGU_REUSE_DATA_DIR": str(pangu),
                "TIANJI_REUSE_DATA_DIR": str(tianji),
                "ERA5_REUSE_DATA_DIR": str(era5),
                "DATA_ROOT": str(self.root / "rebuilt"),
                "OUT_ROOT": str(self.root / "out"),
            }
        )
        result = subprocess.run(
            [str(BASH_EXE), str(Path(__file__).resolve().parent / "submit_q_core_t925_diagnostics.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertIn("TRAINING_JOBS=0", output)
        self.assertIn("PANGU_ACTION=rebuild", output)
        self.assertIn("TIANJI_ACTION=reuse", output)
        self.assertIn("ERA5_ACTION=reuse", output)
        self.assertIn("OBS_ROOT=", output)
        self.assertIn("PAPER_EVAL_DIR=", output)
        self.assertIn("t925_diag_pangu_data", output)
        self.assertNotIn("t925_diag_tianji_data", output)
        self.assertNotIn("t925_diag_era5_data", output)
        self.assertEqual(output.count("[DRY-RUN]"), 2)

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter dry-run regression")
    def test_submitter_auto_resolves_python3(self) -> None:
        pangu = self.write_dataset("pangu")
        tianji = self.write_dataset("tianji")
        era5 = self.write_dataset("era5_reference_analysis")
        env = os.environ.copy()
        env.pop("PREFLIGHT_PYTHON", None)
        env.update(
            {
                "RUN_TAG": "unit_test_t925_python_resolver",
                "DRY_RUN": "1",
                "BUILD_DATA": "auto",
                "BASELINE_DIR": str(Path(__file__).resolve().parent),
                "PANGU_REUSE_DATA_DIR": str(pangu),
                "TIANJI_REUSE_DATA_DIR": str(tianji),
                "ERA5_REUSE_DATA_DIR": str(era5),
                "DATA_ROOT": str(self.root / "rebuilt"),
                "OUT_ROOT": str(self.root / "out"),
            }
        )
        result = subprocess.run(
            [str(BASH_EXE), str(Path(__file__).resolve().parent / "submit_q_core_t925_diagnostics.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertIn("PREFLIGHT_PYTHON=", output)
        self.assertIn("PANGU_ACTION=reuse", output)
        self.assertIn("TIANJI_ACTION=reuse", output)
        self.assertIn("ERA5_ACTION=reuse", output)
        self.assertEqual(output.count("[DRY-RUN]"), 1)

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter dry-run regression")
    def test_preflight_program_failure_aborts_without_sbatch(self) -> None:
        pangu = self.write_dataset("pangu")
        tianji = self.write_dataset("tianji")
        era5 = self.write_dataset("era5_reference_analysis")
        crashing_preflight = self.root / "crashing_preflight.py"
        crashing_preflight.write_text("raise SystemExit(1)\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": "unit_test_t925_preflight_crash",
                "DRY_RUN": "1",
                "BUILD_DATA": "auto",
                "BASELINE_DIR": str(Path(__file__).resolve().parent),
                "PREFLIGHT_PYTHON": sys.executable,
                "PREFLIGHT_SCRIPT": str(crashing_preflight),
                "PANGU_REUSE_DATA_DIR": str(pangu),
                "TIANJI_REUSE_DATA_DIR": str(tianji),
                "ERA5_REUSE_DATA_DIR": str(era5),
                "DATA_ROOT": str(self.root / "rebuilt"),
                "OUT_ROOT": str(self.root / "out"),
            }
        )
        result = subprocess.run(
            [str(BASH_EXE), str(Path(__file__).resolve().parent / "submit_q_core_t925_diagnostics.sh")],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1)
        self.assertIn("preflight infrastructure failed", output)
        self.assertIn("no jobs were submitted", output)
        self.assertNotIn("[DRY-RUN]", output)


if __name__ == "__main__":
    unittest.main()
