#!/usr/bin/env python3
"""Focused numerical tests for the zero-training T925 event diagnosis."""

from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze_q_core_t925_joint_structure import DatasetLayout
from analyze_q_core_t925_upper_air_disagreement import (
    block_bootstrap_summary,
    extract_source_state,
    restrict_events_to_common_layouts,
)


@contextmanager
def workspace_temp_dir():
    path = Path(__file__).resolve().parent / f".tmp_t925_upper_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class T925UpperAirDisagreementTest(unittest.TestCase):
    def test_cpu_slurm_entrypoints_do_not_bootstrap_torch(self) -> None:
        root = Path(__file__).resolve().parent
        for filename in (
            "sub_q_core_t925_upper_air_disagreement.slurm",
            "sub_pangu_qcore_t925_evidence_story.slurm",
        ):
            text = (root / filename).read_text(encoding="utf-8")
            self.assertNotIn("activate_eval_torch_runtime.sh", text, filename)
            self.assertNotIn("import torch", text, filename)
            self.assertIn('PYTHON_BIN="${PYTHON_BIN:-', text, filename)
            self.assertIn('"${PYTHON_BIN}"', text, filename)

    def test_common_reference_alignment_is_explicit_and_bounded(self) -> None:
        events = pd.DataFrame(
            {
                "time_key": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
                "station_key": ["0001", "0002"],
                "case_category": [
                    "tianji_hit_pangu_miss",
                    "pangu_hit_tianji_miss",
                ],
            }
        )
        layouts = {
            key: SimpleNamespace(path=Path(key))
            for key in ("pangu", "tianji", "era5")
        }
        available_all = pd.DataFrame(
            {
                "time_key": events["time_key"],
                "station_key": events["station_key"],
                "source_row": [0, 1],
            }
        )
        available_one = available_all.iloc[[0]].copy()
        with patch(
            "analyze_q_core_t925_upper_air_disagreement.metadata",
            side_effect=[available_all, available_all, available_one],
        ):
            with self.assertRaisesRegex(ValueError, "coverage"):
                restrict_events_to_common_layouts(events, layouts, 0.98)

    def test_last_step_extraction_uses_canonical_units_and_vector_speed(self) -> None:
        with workspace_temp_dir() as tmp:
            order = (
                "T2M",
                "MSLP",
                "WSPD10",
                "T_925",
                "RH_925",
                "Q_1000",
                "Q_925",
                "U_925",
                "V_925",
            )
            layout = DatasetLayout(
                path=tmp,
                order=order,
                window=2,
                dyn_vars=len(order),
                fe_dim=0,
                feature_set="q_core_t925_no_rh2m",
                config={
                    "canonical_dynamic_units": {
                        "T_925": "K",
                        "Q_1000": "kg kg-1",
                        "Q_925": "kg kg-1",
                    }
                },
            )
            x = np.zeros((2, layout.window * layout.dyn_vars + 6), dtype=np.float32)
            base = layout.dyn_vars
            x[:, base + order.index("T_925")] = [280.0, 281.0]
            x[:, base + order.index("Q_1000")] = [0.006, 0.007]
            x[:, base + order.index("Q_925")] = [0.004, 0.005]
            x[:, base + order.index("U_925")] = [3.0, 5.0]
            x[:, base + order.index("V_925")] = [4.0, 12.0]
            np.save(tmp / "X_test.npy", x)
            values = extract_source_state(layout, np.asarray([0, 1], dtype=np.int64))
            self.assertTrue(np.allclose(values["Q_1000"], [6.0, 7.0]))
            self.assertTrue(np.allclose(values["Q_925"], [4.0, 5.0]))
            self.assertTrue(np.allclose(values["WSPD925"], [5.0, 13.0]))
            self.assertTrue(np.allclose(values["T_925"], [280.0, 281.0]))

    def test_date_block_summary_preserves_paired_source_advantage(self) -> None:
        rows = []
        for category_index, category in enumerate(
            ("tianji_hit_pangu_miss", "pangu_hit_tianji_miss")
        ):
            for day in range(12):
                for station in range(10):
                    row = {
                        "time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(days=day),
                        "case_category": category,
                    }
                    for feature in ("T_925", "Q_1000", "Q_925", "WSPD925"):
                        reference = 10.0 + category_index + 0.01 * station
                        row[f"{feature}_era5"] = reference
                        row[f"{feature}_tianji"] = reference + 0.5
                        row[f"{feature}_pangu"] = reference + 1.5
                    rows.append(row)
        summary = block_bootstrap_summary(pd.DataFrame(rows), iterations=250, seed=7)
        self.assertEqual(len(summary), 16)
        physics = summary[summary["source_role"] == "physics"]
        self.assertTrue(np.allclose(physics["bias_forecast_minus_reference"], 0.5))
        self.assertTrue(
            (physics["paired_mae_difference_physics_minus_ai"] < 0.0).all()
        )
        self.assertTrue((physics["paired_mae_difference_ci_high"] < 0.0).all())


if __name__ == "__main__":
    unittest.main()
