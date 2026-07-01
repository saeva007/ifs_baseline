#!/usr/bin/env python3

import importlib.util
import sys
import types
import unittest

import numpy as np

if importlib.util.find_spec("pvlib") is None:
    pvlib = types.ModuleType("pvlib")
    pvlib.solarposition = types.SimpleNamespace()
    sys.modules["pvlib"] = pvlib

from pmst_overlap_common import (  # noqa: E402
    LEGACY_PM_1E12_UNITS,
    PMST_INDEX,
    canonicalize_pm_concentration,
    canonicalize_pmst_field,
    sanitize_pm_concentration,
    scatter_overlap_fields,
)


class CanonicalUnitPolicyTest(unittest.TestCase):
    def test_raw_cams_kgm3_to_ugm3_even_with_bad_label(self):
        raw = np.array([0.0, 5.0e-8, 1.0e-7, np.nan], dtype=np.float32)
        got = canonicalize_pm_concentration(raw, "ug m-3")
        np.testing.assert_allclose(got[:3], [0.0, 50.0, 100.0], rtol=1e-6)
        self.assertTrue(np.isnan(got[3]))

    def test_legacy_times_1e12_is_repaired(self):
        legacy = np.array([0.0, 50000.0, 100000.0], dtype=np.float32)
        got = canonicalize_pm_concentration(legacy, "legacy_mixed")
        np.testing.assert_allclose(got, [0.0, 50.0, 100.0], rtol=1e-6)

    def test_explicit_legacy_provenance_does_not_depend_on_chunk_median(self):
        legacy_low_pollution_chunk = np.array([1000.0, 5000.0, 9000.0], dtype=np.float32)
        got = canonicalize_pm_concentration(legacy_low_pollution_chunk, LEGACY_PM_1E12_UNITS)
        np.testing.assert_allclose(got, [1.0, 5.0, 9.0], rtol=1e-6)

    def test_existing_ugm3_is_unchanged(self):
        values = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        np.testing.assert_allclose(canonicalize_pm_concentration(values, "ug m-3"), values)

    def test_pm_qc_replaces_impossible_and_missing_values(self):
        values = np.array([20.0, 100.0, 20000.0, -999.0, np.nan], dtype=np.float32)
        got = sanitize_pm_concentration(values, "ug m-3")
        np.testing.assert_allclose(got, [20.0, 100.0, 60.0, 60.0, 60.0])

    def test_pm_qc_accepts_training_only_fill_value(self):
        legacy = np.array([100000.0, 2.0e7], dtype=np.float32)
        got = sanitize_pm_concentration(legacy, "legacy_mixed", fill_value=75.5)
        np.testing.assert_allclose(got, [100.0, 75.5])

    def test_mslp_hpa_and_pa_both_become_pa(self):
        hpa = np.array([1000.0, 1010.0], dtype=np.float32)
        pa = np.array([100000.0, 101000.0], dtype=np.float32)
        np.testing.assert_allclose(canonicalize_pmst_field("MSLP", hpa), pa)
        np.testing.assert_allclose(canonicalize_pmst_field("MSLP", pa), pa)

    def test_q_gkg_becomes_kgkg(self):
        q_gkg = np.array([5.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(
            canonicalize_pmst_field("Q_1000", q_gkg),
            [0.005, 0.01],
            rtol=1e-6,
        )

    def test_scatter_applies_mslp_policy(self):
        out = scatter_overlap_fields(
            1,
            2,
            {"MSLP": np.array([[1000.0, 1010.0]], dtype=np.float32)},
            ["MSLP"],
        )
        np.testing.assert_allclose(out[0, :, PMST_INDEX["MSLP"]], [100000.0, 101000.0])


if __name__ == "__main__":
    unittest.main()
