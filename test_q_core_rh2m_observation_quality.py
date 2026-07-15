#!/usr/bin/env python3
"""Regression tests for the strict RH2M observation-quality launcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from preflight_q_core_rh2m_observation_quality import (
    EXPECTED_UNIT_POLICY,
    PreflightError,
    validate_rh2m_dataset,
)


BASH_EXE = shutil.which("bash")
if BASH_EXE is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


class RH2MObservationQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / f".tmp_rh2m_quality_{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def write_dataset(self, role: str, allow_missing: bool = False) -> Path:
        data_dir = self.root / role
        data_dir.mkdir()
        for name in ("X_test.npy", "y_test.npy", "meta_test.csv"):
            (data_dir / name).write_bytes(b"test")
        config = {
            "dataset": "era5_overlap" if role == "era5_reference_analysis" else "tianji_overlap",
            "source_tag": "era5_2025" if role == "era5_reference_analysis" else "tianji",
            "feature_set": "common_core",
            "dynamic_feature_order": ["RH2M"],
            "dyn_vars": 1,
            "window": 12,
            "canonical_unit_policy": EXPECTED_UNIT_POLICY,
            "canonical_dynamic_units": {"RH2M": "%"},
            "time_coordinate": "UTC",
            "rh2m_source": "",
            "rh2m_override_file": "",
            "rh2m_override_allow_missing": False,
        }
        if role == "tianji_product":
            config["rh2m_source"] = "tianji_native"
        elif role == "t2nd_raw":
            config["rh2m_source"] = "T2ND_rh2m"
            override = data_dir / "T2ND_rh2m_station_2025.nc"
            override.write_bytes(b"provenance")
            config["rh2m_override_file"] = str(override)
            config["rh2m_override_allow_missing"] = allow_missing
        (data_dir / "dataset_build_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return data_dir

    def test_three_roles_pass_with_explicit_provenance(self) -> None:
        for role in ("tianji_product", "t2nd_raw", "era5_reference_analysis"):
            with self.subTest(role=role):
                config = validate_rh2m_dataset(self.write_dataset(role), role)
                self.assertIn("RH2M", config["dynamic_feature_order"])

    def test_t2nd_fallback_to_product_is_rejected(self) -> None:
        data_dir = self.write_dataset("t2nd_raw", allow_missing=True)
        with self.assertRaisesRegex(PreflightError, "fallback"):
            validate_rh2m_dataset(data_dir, "t2nd_raw")

    @unittest.skipUnless(BASH_EXE, "bash is required for submitter dry-run regression")
    def test_submitter_queues_one_no_training_job(self) -> None:
        tianji = self.write_dataset("tianji_product")
        t2nd = self.write_dataset("t2nd_raw")
        era5 = self.write_dataset("era5_reference_analysis")
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": "unit_test_rh2m_quality",
                "DRY_RUN": "1",
                "BASELINE_DIR": str(Path(__file__).resolve().parent),
                "PREFLIGHT_PYTHON": sys.executable,
                "TIANJI_RH2M_DATA_DIR": str(tianji),
                "T2ND_RH2M_DATA_DIR": str(t2nd),
                "ERA5_RH2M_DATA_DIR": str(era5),
                "OUT_ROOT": str(self.root / "out"),
            }
        )
        result = subprocess.run(
            [
                str(BASH_EXE),
                str(Path(__file__).resolve().parent / "submit_q_core_rh2m_observation_quality.sh"),
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertIn("TRAINING_JOBS=0", output)
        self.assertIn("PANGU_RH2M_PROXY=excluded", output)
        self.assertEqual(output.count("[DRY-RUN]"), 1)
        self.assertIn("qcore_rh2m_qa", output)


if __name__ == "__main__":
    unittest.main()
