#!/usr/bin/env python3
"""Regression tests for the standalone q-core manuscript figures."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plot_pangu_qcore_mechanism_ppt import event_observation_advantage_source


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


if __name__ == "__main__":
    unittest.main()
