#!/usr/bin/env python3
"""Focused regression tests for the no-training T925 quality supplement."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from analyze_q_core_t925_joint_structure import (
    AlignedSplit,
    bootstrap_binary_pair,
    near_saturation_bootstrap,
    observation_rmse_bootstrap,
)

# The provenance classifier is pure metadata logic, but its builder module also
# imports pvlib for an unrelated solar-feature path.  Keep this focused unit test
# runnable in lightweight developer environments that do not install pvlib.
if importlib.util.find_spec("pvlib") is None:
    sys.modules["pvlib"] = ModuleType("pvlib")
from build_dataset_station_source_overlap_12h import classify_source_feature_provenance


WINDOW = 12


class T925SupplementInferenceTest(unittest.TestCase):
    def test_paired_binary_bootstrap_preserves_day_blocks_and_direction(self) -> None:
        dates = np.repeat(
            np.asarray(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]), 4
        )
        observed = np.tile(np.asarray([True, True, False, False]), 4)
        pangu = np.tile(np.asarray([True, False, True, False]), 4)
        tianji = observed.copy()
        result = bootstrap_binary_pair(observed, pangu, tianji, dates, iterations=40, seed=7)
        self.assertAlmostEqual(result["pod"]["pangu"], 0.5)
        self.assertAlmostEqual(result["pod"]["tianji"], 1.0)
        self.assertAlmostEqual(result["csi"]["pangu"], 1.0 / 3.0)
        self.assertAlmostEqual(result["csi"]["tianji"], 1.0)
        self.assertLess(result["pod"]["delta_ci_high"], 0.0)
        self.assertLess(result["csi"]["delta_ci_high"], 0.0)
        self.assertEqual(result["csi"]["represented_utc_dates"], 4)

    def test_near_saturation_bootstrap_freezes_train_thresholds_for_three_scopes(self) -> None:
        n = 16
        keys = pd.DataFrame(
            {
                "time_utc": pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC"),
                "station_key": [f"S{i:03d}" for i in range(n)],
            }
        )
        reference_q = np.tile(np.asarray([0.006, 0.006, 0.002, 0.002], dtype=np.float32), 4)
        pangu_test_q = np.tile(np.asarray([0.006, 0.002, 0.006, 0.002], dtype=np.float32), 4)

        def source_values(q_current):
            q = np.repeat(np.asarray(q_current, dtype=np.float32)[:, None], WINDOW, axis=1)
            return {
                "T_925": np.full((n, WINDOW), 280.0, dtype=np.float32),
                "Q_925": q,
                "RH_925": np.full((n, WINDOW), 80.0, dtype=np.float32),
            }

        fit = AlignedSplit(
            keys=keys,
            positions={},
            values={
                "pangu": source_values(reference_q),
                "tianji": source_values(reference_q),
                "era5_reference_analysis": source_values(reference_q),
            },
            visibility_m=np.where(np.arange(n) % 2 == 0, 500.0, 5000.0),
            orography_m=np.where(np.arange(n) % 2 == 0, 100.0, 800.0),
        )
        test = AlignedSplit(
            keys=keys,
            positions={},
            values={
                "pangu": source_values(pangu_test_q),
                "tianji": source_values(reference_q),
                "era5_reference_analysis": source_values(reference_q),
            },
            visibility_m=fit.visibility_m.copy(),
            orography_m=fit.orography_m.copy(),
        )
        args = SimpleNamespace(
            low_vis_threshold_m=1000.0,
            near_saturation_quantile=0.5,
            bootstrap_iters=30,
            bootstrap_seed=23,
        )
        table, info = near_saturation_bootstrap(fit, test, args)
        self.assertEqual(len(table), 12)
        self.assertEqual(set(table["scope"]), set(info["scopes"]))
        self.assertTrue(np.all(table["threshold_selection_split"] == "aligned_train"))
        selected = table[
            (table["scope"] == "all_paired_test")
            & (table["method"] == "exact_reference_threshold")
            & (table["metric"] == "csi")
        ].iloc[0]
        self.assertLess(float(selected["delta_ci_high"]), 0.0)

    def test_observation_rmse_bootstrap_uses_three_predeclared_scopes(self) -> None:
        n = 16
        keys = pd.DataFrame(
            {
                "time_utc": pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC"),
                "station_key": [f"S{i:03d}" for i in range(n)],
            }
        )
        values = {}
        for source, t_error, w_error in (("pangu", 2.0, 2.0), ("tianji", 1.0, 1.0)):
            values[source] = {
                "T2M": np.full((n, WINDOW), 273.15 + t_error, dtype=np.float32),
                "WSPD10": np.full((n, WINDOW), w_error, dtype=np.float32),
            }
        split = AlignedSplit(
            keys=keys,
            positions={},
            values=values,
            visibility_m=np.where(np.arange(n) % 2 == 0, 500.0, 5000.0),
            orography_m=np.where(np.arange(n) % 2 == 0, 100.0, 800.0),
        )
        args = SimpleNamespace(
            obs_root="/synthetic/obs",
            paper_eval_dir="/synthetic/paper_eval",
            low_vis_threshold_m=1000.0,
            bootstrap_iters=30,
            bootstrap_seed=19,
        )

        def fake_attach(frame, obs_root, paper_eval_dir):
            out = frame.copy()
            out["tem"] = 0.0
            out["win_s_avg_10mi"] = 0.0
            return out, {
                "matched_event_rows": len(out),
                "available_observation_columns": ["tem", "win_s_avg_10mi"],
            }

        with patch(
            "analyze_q_core_t925_joint_structure.hybrid_analysis.attach_observations",
            side_effect=fake_attach,
        ):
            table, info = observation_rmse_bootstrap(split, args)
        self.assertTrue(info["enabled"])
        self.assertEqual(
            set(table["scope"]),
            {"all_paired_test", "true_low_visibility", "elevation_le_500m"},
        )
        self.assertEqual(set(table["feature"]), {"T2M", "WSPD10"})
        self.assertEqual(len(table), 6)
        self.assertTrue(np.all(table["delta_pangu_minus_tianji"].to_numpy(dtype=float) > 0.0))
        self.assertTrue(np.all(table["delta_ci_low"].to_numpy(dtype=float) > 0.0))

    def test_pangu_product_diagnostics_are_not_mislabelled_native(self) -> None:
        ds = xr.Dataset(
            {
                "T_925": ("row", np.asarray([280.0], dtype=np.float32)),
                "Q_925": ("row", np.asarray([0.006], dtype=np.float32)),
                "RH_925": ("row", np.asarray([80.0], dtype=np.float32)),
                "WSPD10": ("row", np.asarray([2.0], dtype=np.float32)),
            }
        )
        ds["RH_925"].attrs["provenance"] = (
            "derived from native Pangu temperature and specific humidity"
        )
        native, derived, provenance = classify_source_feature_provenance(
            ds, list(ds.data_vars), "pangu2025"
        )
        self.assertIn("T_925", native)
        self.assertIn("Q_925", native)
        self.assertNotIn("RH_925", native)
        self.assertIn("RH_925", derived)
        self.assertIn("WSPD10", derived)
        self.assertIn("derived", provenance["RH_925"].lower())


if __name__ == "__main__":
    unittest.main()
