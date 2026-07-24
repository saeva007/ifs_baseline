#!/usr/bin/env python3
"""Self-contained tests for q-core hybrid construction and attribution math."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_q_core_ale import centered_curve
from analyze_multi_source_feature_importance import add_physical_packages
from analyze_q_core_hybrid_factorial import (
    SampleSet,
    adaptive_ap_histograms,
    make_figures,
    pair_interactions,
    set_group_profile as set_analysis_group_profile,
    shapley_values,
)
from analyze_q_core_t925_joint_structure import (
    AlignedSplit,
    event_linkage,
    qsat_kgkg,
    rff,
    rff_sum,
)
from build_q_core_hybrid_factorial import EXPECTED_ORDER, GROUPS
from pmst_overlap_common import (
    CANONICAL_UNIT_POLICY_VERSION,
    PM_QC_POLICY_VERSION,
    Q_CORE_T925_NO_RH2M_DYN_FEATURES,
    compute_fog_features_pmst,
)


WINDOW = 12
STATIC_DIM = 6
CYCLICAL_DIM = 4
BASH_EXE = shutil.which("bash")
if BASH_EXE is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH_EXE = "C:/Program Files/Git/bin/bash.exe"


@contextmanager
def workspace_temp_dir():
    path = Path(__file__).resolve().parent / f".tmp_qcore_hybrid_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_source(
    root: Path,
    name: str,
    dyn: np.ndarray,
    y: np.ndarray,
    canonical_pangu: bool,
    order: list[str] | None = None,
    feature_set: str = "q_core_no_rh2m",
) -> Path:
    order = list(EXPECTED_ORDER if order is None else order)
    data_dir = root / name
    data_dir.mkdir()
    fog = compute_fog_features_pmst(dyn, WINDOW, len(order), order)
    rng = np.random.default_rng(991)
    static = rng.normal(size=(len(dyn), STATIC_DIM)).astype(np.float32)
    cyc = rng.normal(size=(len(dyn), CYCLICAL_DIM)).astype(np.float32)
    x = np.concatenate([dyn.reshape(len(dyn), -1), static, fog, cyc], axis=1).astype(np.float32)
    np.save(data_dir / "X_train.npy", x)
    np.save(data_dir / "y_train.npy", y.astype(np.float32))
    pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=len(dyn), freq="h"),
            "station_id": np.arange(len(dyn)) + 50000,
            "lat": 30.0,
            "lon": 110.0,
        }
    ).to_csv(data_dir / "meta_train.csv", index=False)
    cfg = {
        "dataset": name,
        "feature_set": feature_set,
        "dynamic_feature_order": order,
        "dyn_vars": len(order),
        "window": WINDOW,
        "fe_dim": int(fog.shape[1] + CYCLICAL_DIM),
        "zero_filled_pmst_features": [],
        "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
        "pm_qc_policy": PM_QC_POLICY_VERSION,
        "time_coordinate": "UTC",
    }
    if canonical_pangu:
        cfg["source_forecast_lead"] = {"min_hours": 12.0, "max_hours": 23.0, "provenance": "unit test"}
    with (data_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return data_dir


class HybridBuilderTest(unittest.TestCase):
    def test_endpoint_fog_roundoff_has_separate_bounded_tolerance(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(71)
            n, dyn_n = 8, len(EXPECTED_ORDER)
            dyn = rng.normal(size=(n, WINDOW, dyn_n)).astype(np.float32)
            y = np.asarray([0, 1, 2, 2, 0, 1, 2, 2], dtype=np.float32)
            pangu = write_source(root, "pangu", dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", dyn, y, canonical_pangu=False)
            fog_start = WINDOW * dyn_n + STATIC_DIM

            source_x = np.load(pangu / "X_train.npy")
            source_x[0, fog_start] += np.float32(6.0e-6)
            np.save(pangu / "X_train.npy", source_x)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(root / "accepted"),
                "--splits",
                "train",
                "--masks",
                "000",
                "--chunk-rows",
                "3",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            with (root / "accepted" / "hybrid_factorial_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            observed = manifest["build_records"][0]["endpoint_recomputed_fog_max_abs_diff"]
            self.assertGreater(observed, 5.0e-6)
            self.assertLess(observed, 5.0e-5)

            source_x[0, fog_start] += np.float32(1.0e-3)
            np.save(pangu / "X_train.npy", source_x)
            rejected = cmd.copy()
            rejected[rejected.index(str(root / "accepted"))] = str(root / "rejected")
            proc = subprocess.run(rejected, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("recomputed/source fog compatibility", proc.stderr + proc.stdout)

    def test_factorial_endpoints_and_group_isolation(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(17)
            n, dyn_n = 10, len(EXPECTED_ORDER)
            p_dyn = rng.normal(size=(n, WINDOW, dyn_n)).astype(np.float32)
            t_dyn = p_dyn.copy()
            for group_i, group in enumerate(("M", "T", "W"), start=1):
                for feature in GROUPS[group]:
                    t_dyn[:, :, EXPECTED_ORDER.index(feature)] += np.float32(group_i * 0.25)
            y = np.asarray([0, 1, 2, 2, 2, 0, 1, 2, 2, 1], dtype=np.float32)
            pangu = write_source(root, "pangu", p_dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", t_dyn, y, canonical_pangu=False)
            out = root / "hybrids"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(out),
                "--splits",
                "train",
                "--masks",
                "all",
                "--chunk-rows",
                "3",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            p_x = np.load(pangu / "X_train.npy")
            t_x = np.load(tianji / "X_train.npy")
            self.assertTrue(np.array_equal(np.load(out / "mtw_000" / "X_train.npy"), p_x))
            self.assertTrue(np.array_equal(np.load(out / "mtw_111" / "X_train.npy"), t_x))

            moisture = np.load(out / "mtw_100" / "X_train.npy")[:, : WINDOW * dyn_n].reshape(n, WINDOW, dyn_n)
            for idx, feature in enumerate(EXPECTED_ORDER):
                expected = t_dyn[:, :, idx] if feature in GROUPS["M"] else p_dyn[:, :, idx]
                self.assertTrue(np.array_equal(moisture[:, :, idx], expected), feature)
            with (out / "hybrid_factorial_manifest.json").open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["status"], "passed")

            audit_cmd = cmd + ["--audit-only"]
            subprocess.run(audit_cmd, check=True, capture_output=True, text=True)

    def test_mt2pw_profile_splits_t2m_and_mslp(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(1702)
            n, dyn_n = 9, len(EXPECTED_ORDER)
            p_dyn = rng.normal(size=(n, WINDOW, dyn_n)).astype(np.float32)
            t_dyn = p_dyn.copy()
            deltas = {
                "Q_1000": np.float32(0.10),
                "DP_1000": np.float32(0.10),
                "Q_925": np.float32(0.10),
                "DP_925": np.float32(0.10),
                "RH_925": np.float32(0.10),
                "T2M": np.float32(0.20),
                "MSLP": np.float32(0.30),
                "U10": np.float32(0.40),
                "V10": np.float32(0.40),
                "WSPD10": np.float32(0.40),
                "WDIR10": np.float32(0.40),
                "U_925": np.float32(0.40),
                "V_925": np.float32(0.40),
                "WSPD925": np.float32(0.40),
            }
            for feature, delta in deltas.items():
                t_dyn[:, :, EXPECTED_ORDER.index(feature)] += delta
            y = np.asarray([0, 1, 2, 2, 1, 0, 2, 1, 2], dtype=np.float32)
            pangu = write_source(root, "pangu", p_dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", t_dyn, y, canonical_pangu=False)
            out = root / "hybrids"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(out),
                "--splits",
                "train",
                "--group-profile",
                "mt2pw",
                "--masks",
                "0000,0100,0010,1111",
                "--chunk-rows",
                "4",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            t2m_only = np.load(out / "mt2pw_0100" / "X_train.npy")[:, : WINDOW * dyn_n].reshape(n, WINDOW, dyn_n)
            mslp_only = np.load(out / "mt2pw_0010" / "X_train.npy")[:, : WINDOW * dyn_n].reshape(n, WINDOW, dyn_n)
            for idx, feature in enumerate(EXPECTED_ORDER):
                expected_t2m = t_dyn[:, :, idx] if feature == "T2M" else p_dyn[:, :, idx]
                expected_mslp = t_dyn[:, :, idx] if feature == "MSLP" else p_dyn[:, :, idx]
                self.assertTrue(np.array_equal(t2m_only[:, :, idx], expected_t2m), feature)
                self.assertTrue(np.array_equal(mslp_only[:, :, idx], expected_mslp), feature)
            with (out / "mt2pw_0100" / "dataset_build_config.json").open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["hybrid_group_profile"], "mt2pw")
            self.assertEqual(cfg["replaced_feature_groups"], ["T2"])
            self.assertEqual(cfg["replaced_features"], ["T2M"])

    def test_m925b_profile_isolates_explicit_t925_from_moisture_state(self) -> None:
        with workspace_temp_dir() as root:
            order = list(Q_CORE_T925_NO_RH2M_DYN_FEATURES)
            rng = np.random.default_rng(925)
            n = 8
            p_dyn = rng.normal(size=(n, WINDOW, len(order))).astype(np.float32)
            p_dyn[:, :, order.index("T_925")] = 275.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("T2M")] = 278.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("Q_925")] = 0.006 + 0.0002 * rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("Q_1000")] = 0.007 + 0.0002 * rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("RH_925")] = 75.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("DP_925")] = 271.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("DP_1000")] = 273.0 + rng.normal(size=(n, WINDOW))
            t_dyn = p_dyn.copy()
            moisture = {"Q_1000", "DP_1000", "Q_925", "DP_925", "RH_925"}
            for feature in moisture:
                t_dyn[:, :, order.index(feature)] += np.float32(0.01)
            t_dyn[:, :, order.index("T_925")] += np.float32(2.0)
            for feature in set(order) - moisture - {"T_925", "ZENITH", "PM10_ugm3", "PM25_ugm3"}:
                t_dyn[:, :, order.index(feature)] += np.float32(0.25)
            y = np.asarray([300, 800, 5000, 5000, 800, 300, 5000, 5000], dtype=np.float32)
            pangu = write_source(
                root, "pangu", p_dyn, y, canonical_pangu=True,
                order=order, feature_set="q_core_t925_no_rh2m",
            )
            tianji = write_source(
                root, "tianji", t_dyn, y, canonical_pangu=False,
                order=order, feature_set="q_core_t925_no_rh2m",
            )
            out = root / "hybrids"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                    "--pangu-dir", str(pangu),
                    "--tianji-dir", str(tianji),
                    "--out-root", str(out),
                    "--splits", "train",
                    "--group-profile", "m925b",
                    "--masks", "000,010,100,111",
                    "--chunk-rows", "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(np.array_equal(np.load(out / "m925b_000" / "X_train.npy"), np.load(pangu / "X_train.npy")))
            self.assertTrue(np.array_equal(np.load(out / "m925b_111" / "X_train.npy"), np.load(tianji / "X_train.npy")))
            t925_only = np.load(out / "m925b_010" / "X_train.npy")[:, : WINDOW * len(order)].reshape(n, WINDOW, len(order))
            moisture_only = np.load(out / "m925b_100" / "X_train.npy")[:, : WINDOW * len(order)].reshape(n, WINDOW, len(order))
            for idx, feature in enumerate(order):
                self.assertTrue(
                    np.array_equal(t925_only[:, :, idx], t_dyn[:, :, idx] if feature == "T_925" else p_dyn[:, :, idx]),
                    feature,
                )
                self.assertTrue(
                    np.array_equal(moisture_only[:, :, idx], t_dyn[:, :, idx] if feature in moisture else p_dyn[:, :, idx]),
                    feature,
                )
            cfg = json.loads((out / "m925b_010" / "dataset_build_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["hybrid_group_profile"], "m925b")
            self.assertEqual(cfg["replaced_feature_groups"], ["H"])
            self.assertFalse(cfg["thermodynamic_source_channels_recomputed"])
            self.assertTrue(cfg["recomputed_fog_features"])

    def test_mhtpw_profile_partitions_t925_qcore_without_cross_layer_wind_split(self) -> None:
        with workspace_temp_dir() as root:
            order = list(Q_CORE_T925_NO_RH2M_DYN_FEATURES)
            rng = np.random.default_rng(9255)
            n = 8
            p_dyn = rng.normal(size=(n, WINDOW, len(order))).astype(np.float32)
            p_dyn[:, :, order.index("T_925")] = 275.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("T2M")] = 278.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("MSLP")] = 1013.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("Q_925")] = 0.006 + 0.0002 * rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("Q_1000")] = 0.007 + 0.0002 * rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("RH_925")] = 75.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("DP_925")] = 271.0 + rng.normal(size=(n, WINDOW))
            p_dyn[:, :, order.index("DP_1000")] = 273.0 + rng.normal(size=(n, WINDOW))
            t_dyn = p_dyn.copy()
            source_features = [name for name in order if name not in {"ZENITH", "PM10_ugm3", "PM25_ugm3"}]
            for offset, feature in enumerate(source_features, start=1):
                t_dyn[:, :, order.index(feature)] += np.float32(offset / 100.0)
            y = np.asarray([300, 800, 5000, 5000, 800, 300, 5000, 5000], dtype=np.float32)
            pangu = write_source(
                root, "pangu", p_dyn, y, canonical_pangu=True,
                order=order, feature_set="q_core_t925_no_rh2m",
            )
            tianji = write_source(
                root, "tianji", t_dyn, y, canonical_pangu=False,
                order=order, feature_set="q_core_t925_no_rh2m",
            )
            out = root / "hybrids"
            masks = "00000,00001,00010,00100,01000,10000,11111"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                    "--pangu-dir", str(pangu),
                    "--tianji-dir", str(tianji),
                    "--out-root", str(out),
                    "--splits", "train",
                    "--group-profile", "mhtpw",
                    "--masks", masks,
                    "--chunk-rows", "3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(np.array_equal(np.load(out / "mhtpw_00000" / "X_train.npy"), np.load(pangu / "X_train.npy")))
            self.assertTrue(np.array_equal(np.load(out / "mhtpw_11111" / "X_train.npy"), np.load(tianji / "X_train.npy")))
            expected_groups = {
                "10000": {"Q_1000", "DP_1000"},
                "01000": {"T_925", "Q_925", "DP_925", "RH_925"},
                "00100": {"T2M"},
                "00010": {"MSLP"},
                "00001": {"U10", "V10", "WSPD10", "WDIR10", "U_925", "V_925", "WSPD925"},
            }
            for mask, replaced in expected_groups.items():
                rows = np.load(out / f"mhtpw_{mask}" / "X_train.npy")[:, : WINDOW * len(order)].reshape(n, WINDOW, len(order))
                for idx, feature in enumerate(order):
                    expected = t_dyn[:, :, idx] if feature in replaced else p_dyn[:, :, idx]
                    self.assertTrue(np.array_equal(rows[:, :, idx], expected), f"{mask}/{feature}")
            cfg = json.loads((out / "mhtpw_01000" / "dataset_build_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["hybrid_group_order"], ["M", "H", "T", "P", "W"])
            self.assertEqual(cfg["replaced_feature_groups"], ["H"])
            self.assertIn("always comes from one source", cfg["thermodynamic_cross_source_policy"])
            self.assertTrue(cfg["recomputed_fog_features"])

    def test_different_order_and_partial_overlap_are_aligned(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(5)
            dyn = rng.normal(size=(6, WINDOW, len(EXPECTED_ORDER))).astype(np.float32)
            y = np.asarray([0, 1, 2, 2, 1, 0], dtype=np.float32)
            pangu = write_source(root, "pangu", dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", dyn, y, canonical_pangu=False)
            permutation = np.asarray([4, 1, 5, 0, 3, 2])
            meta = pd.read_csv(tianji / "meta_train.csv")
            meta = meta.iloc[permutation].reset_index(drop=True)
            meta.loc[0, "station_id"] = 99999
            meta.to_csv(tianji / "meta_train.csv", index=False)
            np.save(tianji / "X_train.npy", np.load(tianji / "X_train.npy")[permutation])
            np.save(tianji / "y_train.npy", np.load(tianji / "y_train.npy")[permutation])
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(root / "out"),
                "--splits",
                "train",
                "--masks",
                "000,111",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            expected_pangu_rows = np.asarray([0, 1, 2, 3, 5])
            expected_tianji_rows = np.asarray([3, 1, 5, 4, 2])
            self.assertTrue(
                np.array_equal(
                    np.load(root / "out" / "mtw_000" / "X_train.npy"),
                    np.load(pangu / "X_train.npy")[expected_pangu_rows],
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.load(root / "out" / "mtw_111" / "X_train.npy"),
                    np.load(tianji / "X_train.npy")[expected_tianji_rows],
                )
            )
            with (root / "out" / "hybrid_factorial_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["source_pair_coverage"]["train"]["common_rows"], 5)
            self.assertEqual(manifest["source_pair_coverage"]["train"]["pangu_excluded_rows"], 1)
            self.assertEqual(manifest["source_pair_coverage"]["train"]["tianji_excluded_rows"], 1)

    def test_common_key_label_mismatch_is_rejected(self) -> None:
        with workspace_temp_dir() as root:
            rng = np.random.default_rng(6)
            dyn = rng.normal(size=(4, WINDOW, len(EXPECTED_ORDER))).astype(np.float32)
            y = np.asarray([0, 1, 2, 2], dtype=np.float32)
            pangu = write_source(root, "pangu", dyn, y, canonical_pangu=True)
            tianji = write_source(root, "tianji", dyn, y, canonical_pangu=False)
            donor_y = np.load(tianji / "y_train.npy")
            donor_y[0] = 999.0
            np.save(tianji / "y_train.npy", donor_y)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_q_core_hybrid_factorial.py"),
                "--pangu-dir",
                str(pangu),
                "--tianji-dir",
                str(tianji),
                "--out-root",
                str(root / "out"),
                "--splits",
                "train",
                "--masks",
                "000",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("labels", proc.stderr + proc.stdout)


class ArtifactAndAleTest(unittest.TestCase):
    def test_wind_layer_packages_are_shared_for_qcore_t925_endpoints(self) -> None:
        order = list(Q_CORE_T925_NO_RH2M_DYN_FEATURES)
        groups = add_physical_packages([], order, WINDOW, order)
        by_name = {str(group["feature"]): group for group in groups}
        surface = by_name["native_surface_wind_ventilation"]
        upper = by_name["native_925_wind"]
        self.assertEqual(surface["members"], ["U10", "V10", "WSPD10", "WDIR10"])
        self.assertEqual(upper["members"], ["U_925", "V_925", "WSPD925"])
        self.assertEqual(surface["analysis_level"], "shared_package")
        self.assertEqual(upper["analysis_level"], "shared_package")
        self.assertEqual(surface["n_columns"], 4 * WINDOW)
        self.assertEqual(upper["n_columns"], 3 * WINDOW)

    @unittest.skipUnless(BASH_EXE, "bash is required for launcher dry-run regression")
    def test_mhtpw_launcher_schedules_exact_formal_matrix(self) -> None:
        repo = Path(__file__).resolve().parent
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": "unit_test_mhtpw_formal",
                "DRY_RUN": "1",
                "BASELINE_DIR": str(repo),
                "BASE": str(repo.parent),
                "TRAIN_EXCLUDE_NODES": "e16r3n05",
            }
        )
        result = subprocess.run(
            [str(BASH_EXE), str(repo / "submit_q_core_t925_mhtpw_factorial.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        self.assertEqual(output.count("[DRY-RUN]"), 114)
        self.assertEqual(
            output.count("sub_q_core_mhtpw_runtime_gate.slurm"), 1
        )
        self.assertEqual(output.count("EXPERIMENT=s1_q_core_t925_no_rh2m"), 3)
        self.assertEqual(output.count("EXPERIMENT=s2_q_core_t925_mhtpw"), 96)
        self.assertEqual(output.count("LOWVIS_RNN_REQUIRE_LOCAL_CACHE=1"), 99)
        self.assertEqual(output.count("LOWVIS_RNN_DCU_PREFLIGHT=1"), 99)
        self.assertEqual(output.count("--exclude=e16r3n05"), 99)
        self.assertEqual(output.count("sub_multi_source_feature_importance.slurm"), 3)
        self.assertIn("GROUP_PROFILE=mhtpw", output)
        self.assertIn("GROUPS=M:Q1000+DP1000;H:T925+Q925+DP925+RH925", output)
        self.assertNotIn("IFS_DATA_DIR", output)
        self.assertNotIn("sub_ifs_data.slurm", output)

    def test_centered_ale_has_weighted_zero_mean(self) -> None:
        curve = centered_curve(np.asarray([0.1, -0.02, 0.04]), np.asarray([10, 20, 30]))
        self.assertAlmostEqual(float(np.average(curve, weights=[10, 20, 30])), 0.0, places=12)

    def test_t925_joint_helpers_are_numerically_stable(self) -> None:
        temperature = np.asarray([260.0, 275.0, 290.0])
        saturation = qsat_kgkg(temperature)
        self.assertTrue(np.all(np.diff(saturation) > 0.0))
        rng = np.random.default_rng(77)
        values = rng.uniform(0.0, 1.0, size=(37, 2))
        omega = rng.normal(size=(2, 32))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=32)
        self.assertTrue(np.allclose(rff_sum(values, omega, phase, chunk_rows=7), rff(values, omega, phase).sum(axis=0)))

    def test_event_linkage_accepts_factorial_case_category_schema(self) -> None:
        with workspace_temp_dir() as root:
            times = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
            keys = pd.DataFrame(
                {
                    "time_utc": times,
                    "time_key": times.strftime("%Y-%m-%d %H:%M:%S"),
                    "station_key": ["50001", "50002", "50003", "50004"],
                    "source_row": np.arange(4),
                }
            )
            values = {}
            for source in ("pangu", "tianji", "era5_reference_analysis"):
                values[source] = {
                    "T_925": np.full((4, WINDOW), 275.0, dtype=np.float32),
                    "Q_925": np.full((4, WINDOW), 0.006, dtype=np.float32),
                    "RH_925": np.full((4, WINDOW), 75.0, dtype=np.float32),
                }
            split = AlignedSplit(
                keys=keys,
                positions={source: np.arange(4) for source in values},
                values=values,
                visibility_m=np.asarray([300.0, 500.0, 700.0, 900.0]),
                orography_m=np.asarray([20.0, 30.0, 40.0, 50.0]),
            )
            pd.DataFrame(
                {
                    "time_utc": times,
                    "station_key": keys["station_key"],
                    "case_category": [
                        "tianji_hit_pangu_miss",
                        "tianji_hit_pangu_miss",
                        "both_hit",
                        "pangu_hit_tianji_miss",
                    ],
                }
            ).to_csv(root / "event_case_control_samples.csv.gz", index=False, compression="gzip")
            losses = {
                "pangu": {
                    "standardized_joint_vector_rmse": np.asarray([4.0, 9.0, 1.0, 1.0]),
                    "saturation_deficit_rmse_gkg": np.asarray([1.0, 4.0, 1.0, 1.0]),
                },
                "tianji": {
                    "standardized_joint_vector_rmse": np.asarray([1.0, 1.0, 1.0, 4.0]),
                    "saturation_deficit_rmse_gkg": np.asarray([0.25, 1.0, 1.0, 4.0]),
                },
            }
            summary, info = event_linkage(
                root,
                split,
                losses,
                SimpleNamespace(low_vis_threshold_m=1000.0, bootstrap_seed=8, bootstrap_iters=20),
            )
            self.assertTrue(info["enabled"])
            self.assertEqual(info["target_category_rows"], 2)
            self.assertGreater(info["target_joint_error_delta_ci"][0], 0.0)
            self.assertIn("tianji_hit_pangu_miss", set(summary["category"]))

    def test_joint_structure_cli_smoke(self) -> None:
        with workspace_temp_dir() as root:
            order = list(Q_CORE_T925_NO_RH2M_DYN_FEATURES)
            rng = np.random.default_rng(92514)
            data_root = root / "data"
            analysis_dir = root / "factorial"
            analysis_dir.mkdir()
            source_dirs = {}

            def write_joint_dataset(source: str, t_error: float, q_error: float) -> Path:
                path = data_root / source
                path.mkdir(parents=True)
                source_order = list(order)
                if source == "tianji":
                    source_order = source_order[5:] + source_order[:5]
                elif source == "era5":
                    source_order = source_order[-3:] + source_order[:-3]
                for split, n, start in (("train", 72, "2025-01-01"), ("test", 96, "2025-02-01")):
                    dyn = np.zeros((n, WINDOW, len(source_order)), dtype=np.float32)
                    base_t = 274.0 + rng.normal(0.0, 1.2, size=(n, WINDOW))
                    base_q = 0.006 + rng.normal(0.0, 0.00025, size=(n, WINDOW))
                    dyn[:, :, source_order.index("T_925")] = base_t + t_error
                    dyn[:, :, source_order.index("Q_925")] = base_q + q_error
                    dyn[:, :, source_order.index("Q_1000")] = base_q + 0.001 + q_error
                    dyn[:, :, source_order.index("T2M")] = base_t + 3.0 + 0.25 * t_error
                    dyn[:, :, source_order.index("RH_925")] = 78.0 - 2.0 * t_error + 1000.0 * q_error
                    dyn[:, :, source_order.index("DP_925")] = base_t - 4.0 + 0.2 * t_error
                    dyn[:, :, source_order.index("DP_1000")] = base_t - 2.0 + 0.2 * t_error
                    dyn[:, :, source_order.index("MSLP")] = 101000.0
                    fog = compute_fog_features_pmst(dyn, WINDOW, len(source_order), source_order)
                    static = np.zeros((n, STATIC_DIM), dtype=np.float32)
                    static[:, 2] = 100.0
                    cyc = np.zeros((n, CYCLICAL_DIM), dtype=np.float32)
                    x = np.concatenate([dyn.reshape(n, -1), static, fog, cyc], axis=1).astype(np.float32)
                    np.save(path / f"X_{split}.npy", x)
                    y = np.where(np.arange(n) % 4 == 0, 500.0, 5000.0).astype(np.float32)
                    np.save(path / f"y_{split}.npy", y)
                    pd.DataFrame(
                        {
                            "time": pd.date_range(start, periods=n, freq="h"),
                            "station_id": 51000 + np.arange(n),
                            "lat": 30.0,
                            "lon": 110.0,
                        }
                    ).to_csv(path / f"meta_{split}.csv", index=False)
                (path / "dataset_build_config.json").write_text(
                    json.dumps(
                        {
                            "feature_set": "source_full",
                            "dynamic_feature_order": source_order,
                            "dyn_vars": len(source_order),
                            "window": WINDOW,
                            "fe_dim": int(fog.shape[1] + CYCLICAL_DIM),
                            "canonical_unit_policy": CANONICAL_UNIT_POLICY_VERSION,
                            "canonical_dynamic_units": {
                                "T_925": "K",
                                "Q_925": "kg kg-1",
                                "RH_925": "%",
                            },
                            "time_coordinate": "UTC",
                            "native_source_features": ["T_925", "Q_925", "RH_925"],
                            "derived_source_features": [],
                            "source_forecast_lead": (
                                {"available": True, "min_hours": 12.0, "max_hours": 23.0}
                                if source == "pangu"
                                else {"available": False}
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                return path

            # Tianji is deliberately closer to the ERA5 reference analysis than Pangu.
            source_dirs["era5"] = write_joint_dataset("era5", 0.0, 0.0)
            source_dirs["tianji"] = write_joint_dataset("tianji", 0.3, 0.00005)
            source_dirs["pangu"] = write_joint_dataset("pangu", 1.5, 0.00030)

            metric_rows = []
            interaction_rows = []
            for seed in (42, 2025, 20260702):
                for value in range(8):
                    mask = f"{value:03b}"
                    m, h, b = (int(bit) for bit in mask)
                    metric_rows.append(
                        {
                            "seed": seed,
                            "mask": mask,
                            "low_vis_ap": 0.40 + 0.01 * m + 0.01 * h + 0.01 * b + 0.02 * m * h,
                        }
                    )
                interaction_rows.append({"seed": seed, "metric": "low_vis_ap", "pair": "M:H", "interaction": 0.02})
            pd.DataFrame(metric_rows).to_csv(analysis_dir / "hybrid_factorial_metrics_by_seed.csv", index=False)
            pd.DataFrame(interaction_rows).to_csv(analysis_dir / "hybrid_second_order_interactions_by_seed.csv", index=False)
            pd.DataFrame(
                [{"metric": "low_vis_ap", "effect": "interaction", "term": "M:H", "ci_low": 0.01, "ci_high": 0.03}]
            ).to_csv(analysis_dir / "hybrid_date_block_bootstrap_ci.csv", index=False)
            pd.DataFrame(
                {"metric": ["low_vis_ap"] * 20, "delta_all1_minus_all0": np.linspace(0.04, 0.08, 20)}
            ).to_csv(analysis_dir / "hybrid_total_gap_bootstrap_draws.csv.gz", index=False, compression="gzip")
            event_times = pd.date_range("2025-02-01", periods=12, freq="h", tz="UTC")
            pd.DataFrame(
                {
                    "time_utc": event_times,
                    "station_key": (51000 + np.arange(12)).astype(str),
                    "case_category": ["tianji_hit_pangu_miss"] * 8 + ["both_hit"] * 4,
                }
            ).to_csv(analysis_dir / "event_case_control_samples.csv.gz", index=False, compression="gzip")

            out = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "analyze_q_core_t925_joint_structure.py"),
                    "--pangu-dir", str(source_dirs["pangu"]),
                    "--tianji-dir", str(source_dirs["tianji"]),
                    "--era5-dir", str(source_dirs["era5"]),
                    "--factorial-analysis-dir", str(analysis_dir),
                    "--analysis-mode", "factorial_confirmatory",
                    "--out-dir", str(out),
                    "--fit-max-rows", "0",
                    "--test-max-rows", "0",
                    "--bootstrap-iters", "20",
                    "--rff-dim", "32",
                    "--rff-bandwidth-sample", "64",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((out / "joint_structure_analysis_report.json").read_text(encoding="utf-8"))
            gate = json.loads((out / "joint_structure_evidence_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["factorial_profile"], "m925b")
            self.assertIn(gate["status"], {"joint_structure_supported", "joint_structure_not_fully_supported"})
            self.assertTrue((out / "t925_q925_empirical_copula_quality.csv").is_file())
            self.assertTrue((out / "fig_joint_saturation_deficit_quality.png").is_file())
            self.assertTrue((out / "fig_m_t925_performance_interaction.svg").is_file())

            diagnostic_out = root / "diagnostic_only"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "analyze_q_core_t925_joint_structure.py"),
                    "--pangu-dir", str(source_dirs["pangu"]),
                    "--tianji-dir", str(source_dirs["tianji"]),
                    "--era5-dir", str(source_dirs["era5"]),
                    "--event-analysis-dir", str(analysis_dir),
                    "--out-dir", str(diagnostic_out),
                    "--fit-max-rows", "0",
                    "--test-max-rows", "0",
                    "--bootstrap-iters", "20",
                    "--rff-dim", "32",
                    "--rff-bandwidth-sample", "64",
                    "--no-figures",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            diagnostic_report = json.loads(
                (diagnostic_out / "joint_structure_analysis_report.json").read_text(encoding="utf-8")
            )
            diagnostic_gate = json.loads(
                (diagnostic_out / "joint_structure_evidence_gate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostic_report["analysis_mode"], "diagnostic_only")
            self.assertEqual(diagnostic_report["new_training_models_used"], 0)
            self.assertIsNone(diagnostic_gate["predictive_m_t925_interaction_supported"])
            self.assertFalse((diagnostic_out / "m_t925_coherence_contrasts_by_seed.csv").exists())

    def test_ap_histogram_resolution_adapts_without_relaxing_error(self) -> None:
        sample = SampleSet(
            frame=pd.DataFrame(index=np.arange(2)),
            y=np.asarray([0, 2], dtype=np.int64),
            score=np.asarray([0.50023, 0.50013], dtype=np.float64),
            pred=np.asarray([0, 2], dtype=np.int64),
        )
        _precomputed, bins, error = adaptive_ap_histograms(
            {"sample": sample},
            {"sample": 0.5},
            np.asarray([0, 0], dtype=np.int64),
            1,
            4096,
            8192,
            5.0e-4,
        )
        self.assertEqual(bins, 8192)
        self.assertLessEqual(error, 5.0e-4)

    def test_figure_handles_point_estimate_outside_bootstrap_interval(self) -> None:
        with workspace_temp_dir() as out_dir:
            metrics = pd.DataFrame(
                [
                    {
                        "mask": f"{value:03b}",
                        "low_vis_ap": 0.4 + 0.01 * value,
                        "low_vis_csi_matched_fpr": 0.3 + 0.005 * value,
                        "low_vis_recall_matched_fpr": 0.5 + 0.005 * value,
                    }
                    for value in range(8)
                ]
            )
            reliability = pd.DataFrame(
                [
                    {"mask": mask, "bin": idx, "n": 10, "mean_probability": 0.1 + 0.2 * idx, "observed_frequency": 0.12 + 0.18 * idx}
                    for mask in ("000", "111")
                    for idx in range(4)
                ]
            )
            shapley = pd.DataFrame(
                {
                    "metric": ["low_vis_ap"] * 3,
                    "group": ["M", "T", "W"],
                    "group_label": ["Moisture", "Thermal/pressure", "Wind"],
                    "shapley_mean": [0.01, 0.02, -0.01],
                }
            )
            bootstrap = pd.DataFrame(
                {
                    "metric": ["low_vis_ap"] * 3,
                    "effect": ["shapley"] * 3,
                    "term": ["M", "T", "W"],
                    "ci_low": [0.03, 0.01, -0.03],
                    "ci_high": [0.05, 0.04, 0.01],
                }
            )
            events = pd.DataFrame({"case_category": ["both_hit", "both_miss", "tianji_hit_pangu_miss"]})
            make_figures(metrics, reliability, shapley, bootstrap, events, out_dir)
            self.assertTrue((out_dir / "fig_q_core_hybrid_factorial_mechanism.png").is_file())
            self.assertTrue((out_dir / "fig_q_core_hybrid_factorial_mechanism.pdf").is_file())
            self.assertTrue((out_dir / "fig_q_core_hybrid_factorial_mechanism.svg").is_file())

    def test_formal_artifact_gate_accepts_27_complete_triplets(self) -> None:
        with workspace_temp_dir() as root:
            ckpt = root / "checkpoints"
            ckpt.mkdir()
            s1_data = root / "s1"
            s1_data.mkdir()
            hybrid = root / "hybrid"
            hybrid.mkdir()
            data_cfg = {"dyn_vars": len(EXPECTED_ORDER), "dynamic_feature_order": EXPECTED_ORDER}
            (s1_data / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            seeds = (42, 2025, 20260702)
            masks = tuple(f"{value:03b}" for value in range(8))
            for mask in masks:
                data_dir = hybrid / f"mtw_{mask}"
                data_dir.mkdir()
                (data_dir / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            for seed in seeds:
                s1_run = f"exp_qcore_hybrid_unit_s1_seed{seed}_pm10_pm25"
                (ckpt / f"{s1_run}_S1_best_score.pt").touch()
                (ckpt / f"robust_scaler_{s1_run}_s1_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                (ckpt / f"{s1_run}_static_rnn_config.json").write_text(
                    json.dumps({"run_id": s1_run, "seed": seed, "window_size": 12, "s1_data_dir": str(s1_data)}),
                    encoding="utf-8",
                )
                for mask in masks:
                    run_id = f"exp_qcore_hybrid_unit_mtw{mask}_seed{seed}_pm10_pm25"
                    (ckpt / f"{run_id}_S2_PhaseB_best_score.pt").touch()
                    (ckpt / f"robust_scaler_{run_id}_s2_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                    (ckpt / f"{run_id}_static_rnn_config.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "window_size": 12,
                                "s2_data_dir": str(hybrid / f"mtw_{mask}"),
                                "s2_phase_a_steps": 12000,
                                "s2_phase_b_steps": 40000,
                                "pretrained_ckpt": "s1.pt",
                            }
                        ),
                        encoding="utf-8",
                    )
            out = root / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "verify_q_core_hybrid_artifacts.py"),
                    "--run-tag",
                    "unit",
                    "--checkpoint-dir",
                    str(ckpt),
                    "--hybrid-data-root",
                    str(hybrid),
                    "--s1-data-dir",
                    str(s1_data),
                    "--out-dir",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((out / "primary_training_artifact_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["passed_triplets"], 27)

    def test_mt2pw_artifact_gate_accepts_24_new_s2_with_coupled_mtw_reuse(self) -> None:
        with workspace_temp_dir() as root:
            ckpt = root / "checkpoints"
            ckpt.mkdir()
            s1_data = root / "s1"
            s1_data.mkdir()
            split_hybrid = root / "split_hybrid"
            split_hybrid.mkdir()
            old_hybrid = root / "old_hybrid"
            old_hybrid.mkdir()
            data_cfg = {"dyn_vars": len(EXPECTED_ORDER), "dynamic_feature_order": EXPECTED_ORDER}
            (s1_data / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            seeds = (42, 2025, 20260702)
            all_masks = tuple(f"{value:04b}" for value in range(16))
            train_masks = ("0010", "0011", "0100", "0101", "1010", "1011", "1100", "1101")
            coupled = {mask: f"{mask[0]}{mask[1]}{mask[3]}" for mask in all_masks if mask[1] == mask[2]}
            for mask in train_masks:
                data_dir = split_hybrid / f"mt2pw_{mask}"
                data_dir.mkdir()
                (data_dir / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            for old_mask in sorted(set(coupled.values())):
                data_dir = old_hybrid / f"mtw_{old_mask}"
                data_dir.mkdir()
                (data_dir / "dataset_build_config.json").write_text(json.dumps(data_cfg), encoding="utf-8")
            for seed in seeds:
                s1_run = f"exp_qcore_hybrid_old_s1_seed{seed}_pm10_pm25"
                (ckpt / f"{s1_run}_S1_best_score.pt").touch()
                (ckpt / f"robust_scaler_{s1_run}_s1_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                (ckpt / f"{s1_run}_static_rnn_config.json").write_text(
                    json.dumps({"run_id": s1_run, "seed": seed, "window_size": 12, "s1_data_dir": str(s1_data)}),
                    encoding="utf-8",
                )
                for mask in train_masks:
                    run_id = f"exp_qcore_hybrid_split_mt2pw{mask}_seed{seed}_pm10_pm25"
                    (ckpt / f"{run_id}_S2_PhaseB_best_score.pt").touch()
                    (ckpt / f"robust_scaler_{run_id}_s2_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                    (ckpt / f"{run_id}_static_rnn_config.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "window_size": 12,
                                "s2_data_dir": str(split_hybrid / f"mt2pw_{mask}"),
                                "s2_phase_a_steps": 12000,
                                "s2_phase_b_steps": 40000,
                                "pretrained_ckpt": "s1.pt",
                            }
                        ),
                        encoding="utf-8",
                    )
                for mask, old_mask in coupled.items():
                    run_id = f"exp_qcore_hybrid_old_mtw{old_mask}_seed{seed}_pm10_pm25"
                    (ckpt / f"{run_id}_S2_PhaseB_best_score.pt").touch()
                    (ckpt / f"robust_scaler_{run_id}_s2_w12_dyn{len(EXPECTED_ORDER)}_pm.pkl").touch()
                    (ckpt / f"{run_id}_static_rnn_config.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "window_size": 12,
                                "s2_data_dir": str(old_hybrid / f"mtw_{old_mask}"),
                                "s2_phase_a_steps": 12000,
                                "s2_phase_b_steps": 40000,
                                "pretrained_ckpt": "s1.pt",
                            }
                        ),
                        encoding="utf-8",
                    )
            out = root / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "verify_q_core_hybrid_artifacts.py"),
                    "--run-tag",
                    "split",
                    "--s1-run-tag",
                    "old",
                    "--checkpoint-dir",
                    str(ckpt),
                    "--hybrid-data-root",
                    str(split_hybrid),
                    "--s1-data-dir",
                    str(s1_data),
                    "--out-dir",
                    str(out),
                    "--group-profile",
                    "mt2pw",
                    "--masks",
                    ":".join(all_masks),
                    "--train-masks",
                    ":".join(train_masks),
                    "--reuse-coupled-mtw-run-tag",
                    "old",
                    "--reuse-coupled-mtw-hybrid-root",
                    str(old_hybrid),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((out / "primary_training_artifact_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["passed_triplets"], 51)
            frame = pd.read_csv(out / "primary_training_artifact_audit.csv")
            self.assertEqual(int((frame["artifact_provenance"] == "trained_here").sum()), len(train_masks) * len(seeds))
            self.assertEqual(int((frame["artifact_provenance"] == "reused_coupled_mtw").sum()), len(coupled) * len(seeds))


class AttributionMathTest(unittest.TestCase):
    def test_exact_shapley_efficiency_and_values(self) -> None:
        values = {}
        for i in range(8):
            mask = f"{i:03b}"
            values[mask] = 1.0 + 0.1 * int(mask[0]) + 0.2 * int(mask[1]) + 0.3 * int(mask[2])
        phi = shapley_values(values)
        self.assertAlmostEqual(phi["M"], 0.1)
        self.assertAlmostEqual(phi["T"], 0.2)
        self.assertAlmostEqual(phi["W"], 0.3)
        self.assertAlmostEqual(sum(phi.values()), values["111"] - values["000"])
        for value in pair_interactions(values).values():
            self.assertAlmostEqual(value, 0.0)

    def test_exact_four_group_shapley_and_pair_interactions(self) -> None:
        try:
            set_analysis_group_profile("mt2pw")
            values = {}
            for i in range(16):
                mask = f"{i:04b}"
                values[mask] = (
                    1.0
                    + 0.1 * int(mask[0])
                    + 0.2 * int(mask[1])
                    + 0.05 * int(mask[2])
                    + 0.3 * int(mask[3])
                )
            phi = shapley_values(values)
            self.assertAlmostEqual(phi["M"], 0.1)
            self.assertAlmostEqual(phi["T2"], 0.2)
            self.assertAlmostEqual(phi["P"], 0.05)
            self.assertAlmostEqual(phi["W"], 0.3)
            self.assertAlmostEqual(sum(phi.values()), values["1111"] - values["0000"])
            for value in pair_interactions(values).values():
                self.assertAlmostEqual(value, 0.0)
        finally:
            set_analysis_group_profile("mtw")

    def test_exact_five_group_shapley_and_efficiency(self) -> None:
        try:
            set_analysis_group_profile("mhtpw")
            effects = {"M": 0.08, "H": 0.05, "T": 0.03, "P": -0.01, "W": 0.02}
            values = {}
            for i in range(32):
                mask = f"{i:05b}"
                values[mask] = 1.0 + sum(effects[group] * int(mask[j]) for j, group in enumerate(effects))
            phi = shapley_values(values)
            for group, effect in effects.items():
                self.assertAlmostEqual(phi[group], effect)
            self.assertAlmostEqual(sum(phi.values()), values["11111"] - values["00000"])
            for value in pair_interactions(values).values():
                self.assertAlmostEqual(value, 0.0)
        finally:
            set_analysis_group_profile("mtw")

    def test_analysis_cli_smoke(self) -> None:
        with workspace_temp_dir() as root:
            eval_root = root / "eval"
            seed_dir = eval_root / "seed_42"
            seed_dir.mkdir(parents=True)
            data_root = root / "data"
            data_root.mkdir()
            rng = np.random.default_rng(404)
            specs = {}
            test_times = pd.date_range("2025-01-01", periods=120, freq="6h")
            for mask in (f"{i:03b}" for i in range(8)):
                data_dir = data_root / mask
                data_dir.mkdir()
                np.save(data_dir / "X_test.npy", rng.normal(size=(120, WINDOW * len(EXPECTED_ORDER) + 10)).astype(np.float32))
                pd.DataFrame(
                    {"time": test_times, "station_id": 50000 + np.arange(120)}
                ).to_csv(data_dir / "meta_test.csv", index=False)
                with (data_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {"dynamic_feature_order": EXPECTED_ORDER, "dyn_vars": len(EXPECTED_ORDER), "window": WINDOW},
                        f,
                    )
                specs[f"qcore_hybrid_{mask}"] = {"data_dir": str(data_dir)}
            era5_dir = data_root / "era5"
            era5_dir.mkdir()
            np.save(
                era5_dir / "X_test.npy",
                np.load(data_root / "111" / "X_test.npy")[:110],
            )
            pd.DataFrame(
                {"time": test_times[:110], "station_id": 50000 + np.arange(110)}
            ).to_csv(era5_dir / "meta_test.csv", index=False)
            with (era5_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"dynamic_feature_order": EXPECTED_ORDER, "dyn_vars": len(EXPECTED_ORDER), "window": WINDOW},
                    f,
                )
            with (seed_dir / "run_config.json").open("w", encoding="utf-8") as f:
                json.dump({"specs": specs}, f)

            for split, n in (("val", 60), ("test", 120)):
                y = np.asarray(([0, 1, 2, 2, 2, 2] * ((n + 5) // 6))[:n], dtype=np.int64)
                time = pd.date_range("2025-01-01", periods=n, freq="6h")
                noise = rng.normal(0.0, 0.16, size=n)
                for i in range(8):
                    mask = f"{i:03b}"
                    improvement = 0.03 * int(mask[0]) + 0.015 * int(mask[1]) + 0.01 * int(mask[2])
                    score = np.clip(0.16 + 0.46 * (y <= 1) + noise + improvement * (2 * (y <= 1) - 1), 0.001, 0.999)
                    probs = np.column_stack([0.58 * score, 0.42 * score, 1.0 - score])
                    pred = np.argmax(probs, axis=1)
                    frame = pd.DataFrame(
                        {
                            "time": time,
                            "station_id": 50000 + np.arange(n),
                            "y_cls": y,
                            "vis_raw_m": np.where(y == 0, 300.0, np.where(y == 1, 750.0, 5000.0)),
                            "pred": pred,
                            "p_fog": probs[:, 0],
                            "p_mist": probs[:, 1],
                            "p_clear": probs[:, 2],
                        }
                    )
                    name = f"per_sample_val_qcore_hybrid_{mask}.csv" if split == "val" else f"per_sample_qcore_hybrid_{mask}.csv"
                    frame.to_csv(seed_dir / name, index=False)
            out = root / "analysis"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "analyze_q_core_hybrid_factorial.py"),
                "--eval-root",
                str(eval_root),
                "--out-dir",
                str(out),
                "--seeds",
                "42",
                "--bootstrap-iters",
                "5",
                "--era5-data-dir",
                str(era5_dir),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertTrue((out / "hybrid_exact_shapley_effects.csv").is_file())
            self.assertTrue((out / "event_case_control_samples.csv.gz").is_file())
            self.assertTrue((out / "q_reference_analysis_quality_and_extreme_placement.csv").is_file())
            self.assertTrue((out / "fig_q_core_endpoint_reliability.png").is_file())
            with (out / "hybrid_factorial_analysis_report.json").open("r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["moisture_followup_gate"]["formal_three_seed_analysis"])

    def test_analysis_cli_smoke_mt2pw_profile(self) -> None:
        with workspace_temp_dir() as root:
            eval_root = root / "eval"
            seed_dir = eval_root / "seed_42"
            seed_dir.mkdir(parents=True)
            data_root = root / "data"
            data_root.mkdir()
            rng = np.random.default_rng(2402)
            specs = {}
            test_times = pd.date_range("2025-02-01", periods=96, freq="6h")
            for mask in (f"{i:04b}" for i in range(16)):
                data_dir = data_root / mask
                data_dir.mkdir()
                np.save(data_dir / "X_test.npy", rng.normal(size=(96, WINDOW * len(EXPECTED_ORDER) + 10)).astype(np.float32))
                pd.DataFrame(
                    {"time": test_times, "station_id": 60000 + np.arange(96)}
                ).to_csv(data_dir / "meta_test.csv", index=False)
                with (data_dir / "dataset_build_config.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {"dynamic_feature_order": EXPECTED_ORDER, "dyn_vars": len(EXPECTED_ORDER), "window": WINDOW},
                        f,
                    )
                specs[f"qcore_hybrid_mt2pw_{mask}"] = {"data_dir": str(data_dir)}
            with (seed_dir / "run_config.json").open("w", encoding="utf-8") as f:
                json.dump({"specs": specs}, f)

            for split, n in (("val", 48), ("test", 96)):
                y = np.asarray(([0, 1, 2, 2, 2, 1] * ((n + 5) // 6))[:n], dtype=np.int64)
                time = pd.date_range("2025-02-01", periods=n, freq="6h")
                noise = rng.normal(0.0, 0.12, size=n)
                for i in range(16):
                    mask = f"{i:04b}"
                    improvement = (
                        0.02 * int(mask[0])
                        + 0.05 * int(mask[1])
                        + 0.005 * int(mask[2])
                        + 0.015 * int(mask[3])
                    )
                    score = np.clip(0.18 + 0.50 * (y <= 1) + noise + improvement * (2 * (y <= 1) - 1), 0.001, 0.999)
                    probs = np.column_stack([0.55 * score, 0.45 * score, 1.0 - score])
                    pred = np.argmax(probs, axis=1)
                    frame = pd.DataFrame(
                        {
                            "time": time,
                            "station_id": 60000 + np.arange(n),
                            "y_cls": y,
                            "vis_raw_m": np.where(y == 0, 300.0, np.where(y == 1, 750.0, 5000.0)),
                            "pred": pred,
                            "p_fog": probs[:, 0],
                            "p_mist": probs[:, 1],
                            "p_clear": probs[:, 2],
                        }
                    )
                    name = (
                        f"per_sample_val_qcore_hybrid_mt2pw_{mask}.csv"
                        if split == "val"
                        else f"per_sample_qcore_hybrid_mt2pw_{mask}.csv"
                    )
                    frame.to_csv(seed_dir / name, index=False)
            out = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "analyze_q_core_hybrid_factorial.py"),
                    "--eval-root",
                    str(eval_root),
                    "--out-dir",
                    str(out),
                    "--seeds",
                    "42",
                    "--group-profile",
                    "mt2pw",
                    "--bootstrap-iters",
                    "5",
                    "--no-figures",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            shapley = pd.read_csv(out / "hybrid_exact_shapley_effects.csv")
            self.assertEqual(set(shapley["group"]), {"M", "T2", "P", "W"})
            report = json.loads((out / "hybrid_factorial_analysis_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["group_profile"], "mt2pw")
            self.assertEqual(report["moisture_followup_gate"]["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
