#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import analyze_q_core_task_tail_fidelity_by_leadtime as target


class TaskTailByLeadtimeTest(unittest.TestCase):
    def test_lead_derivation_uses_00_12z_stitching(self) -> None:
        keys = pd.DataFrame(
            {
                "time_utc": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 11:00:00",
                        "2025-01-01 12:00:00",
                        "2025-01-01 23:00:00",
                    ]
                )
            }
        )
        leads, initializations = target.derive_lead_hours(keys)
        np.testing.assert_array_equal(leads, np.asarray([12, 23, 12, 23]))
        self.assertEqual(set(initializations.hour.tolist()), {0, 12})

    def test_validation_frozen_counts_reaggregate_exactly(self) -> None:
        fit_visibility = np.asarray([100, 200, 300, 400, 2000, 2100, 2200, 2300])
        test_visibility = fit_visibility.copy()
        test_leads = np.asarray([12, 12, 13, 13, 12, 12, 13, 13])
        test_dates = np.asarray(
            ["2025-01-01"] * 4 + ["2025-01-02"] * 4, dtype=object
        )

        t2m_reference = np.asarray([0, 1, 2, 3, 10, 11, 12, 13], dtype=float)
        q1000_reference = np.asarray([13, 12, 11, 10, 3, 2, 1, 0], dtype=float)

        def bundle(reference: np.ndarray):
            return {
                "unit": "unit",
                "reference_name": "reference",
                "reference": reference,
                "pangu": reference + np.asarray([0, 0, 1, 0, 0, 0, 0, 0]),
                "tianji": reference + np.asarray([0, 0, 0, 0, 1, 0, 0, 0]),
            }

        fit_bundle = {
            "T2M": bundle(t2m_reference),
            "Q_1000": bundle(q1000_reference),
        }
        test_bundle = {
            "T2M": bundle(t2m_reference),
            "Q_1000": bundle(q1000_reference),
        }
        result, validation = target.task_tail_csi_by_lead(
            fit_bundle,
            test_bundle,
            fit_visibility,
            test_visibility,
            test_leads,
            test_dates,
            ("T2M", "Q_1000"),
            0.25,
            0.75,
            1000.0,
        )

        self.assertEqual(len(result), 8)
        self.assertEqual(set(result["lead_hour"]), {12, 13})
        self.assertEqual(
            validation["aggregate_confusion_count_max_abs_error"], 0.0
        )
        self.assertEqual(validation["thresholds"]["T2M"]["task_tail"], "lower")
        self.assertEqual(
            validation["thresholds"]["Q_1000"]["task_tail"], "upper"
        )
        reconstructed_n = result[["tp", "fp", "fn", "tn"]].sum(axis=1)
        np.testing.assert_array_equal(reconstructed_n, result["test_n"])


if __name__ == "__main__":
    unittest.main()
