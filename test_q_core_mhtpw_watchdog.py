#!/usr/bin/env python3
"""Regression tests for the scoped MHTPW Slurm watchdog."""

from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import watch_q_core_mhtpw_chain as mod


@contextmanager
def workspace_temp_dir():
    path = Path(__file__).resolve().parent / f".tmp_mhtpw_watchdog_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class MHTPWWatchdogTest(unittest.TestCase):
    def test_dcu_runtime_contract_uses_validated_order_and_rank_stagger(self) -> None:
        repo = Path(__file__).resolve().parent
        runtime = (repo / "activate_mhtpw_dcu_runtime.sh").read_text(
            encoding="utf-8"
        )
        compatibility = runtime.index(
            'export LD_LIBRARY_PATH="${OPENSSL_COMPAT_LIB}:'
        )
        torch_prepend = runtime.index(
            'source "${MHTPW_RUNTIME_DIR}/activate_torch_runtime.sh"'
        )
        self.assertLess(compatibility, torch_prepend)
        self.assertIn(
            'MHTPW_RANK_STAGGER_SECONDS="${MHTPW_RANK_STAGGER_SECONDS:-5}"',
            runtime,
        )
        training = (repo / "sub_ifs_overlap_baseline.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("probe_mhtpw_dcu_runtime.py", training)
        self.assertIn("stagger_mhtpw_torch_entrypoint.py", training)

    def test_dispatcher_registers_mhtpw_experiment_alias(self) -> None:
        repo = Path(__file__).resolve().parent
        self.assertTrue(mod.dispatcher_accepts_mhtpw_alias(repo))

    def test_known_dispatcher_failure_requires_exact_job_log_signature(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            path = logs / "123.out"
            path.write_text(
                "Unknown EXPERIMENT=s2_q_core_t925_mhtpw (use ...)\n",
                encoding="utf-8",
            )
            self.assertTrue(
                mod.has_known_dispatcher_failure(tmp, "123", "mhtpw_00000_s42")
            )
            path.write_text("unrelated training failure\n", encoding="utf-8")
            self.assertFalse(
                mod.has_known_dispatcher_failure(tmp, "123", "mhtpw_00000_s42")
            )

    def test_strict_local_cache_failure_requires_exact_job_log_signature(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            path = logs / "321.err"
            path.write_text(
                f"{mod.STRICT_LOCAL_CACHE_FAILURE} host=n1 insufficient space\n",
                encoding="utf-8",
            )
            self.assertTrue(
                mod.has_job_log_signature(
                    tmp,
                    "321",
                    "mhtpw_s1_s42",
                    mod.STRICT_LOCAL_CACHE_FAILURE,
                )
            )

    def test_known_runtime_failure_signature_is_exact(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            (logs / "777.err").write_text(
                "RuntimeError: No HIP GPUs are available\n"
                "slurmstepd: error: Detected 3 oom-kill event(s)\n",
                encoding="utf-8",
            )
            self.assertEqual(
                mod.matching_job_log_signature(
                    tmp,
                    "777",
                    "mhtpw_s1_s20260702",
                    mod.KNOWN_RUNTIME_FAILURE_SIGNATURES,
                ),
                "RuntimeError: No HIP GPUs are available",
            )

    def test_runtime_gate_failure_adopts_dependency_cancelled_matrix(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            (logs / "900.err").write_text(
                "[DCU-Runtime-Preflight] ERROR: "
                "RuntimeError: No HIP GPUs are available\n",
                encoding="utf-8",
            )
            watch = object.__new__(mod.ChainWatch)
            watch.args = SimpleNamespace(
                auto_retry=True,
                adopt_runtime_failure=True,
                adopt_local_cache_failure=False,
                adopt_dispatcher_failure=False,
                checkpoint_grace_minutes=30,
                exclude_failed_nodes=False,
            )
            watch.run_tag = "demo"
            watch.baseline_dir = tmp
            watch.checkpoint_dir = tmp / "checkpoints"
            watch.checkpoint_dir.mkdir()
            watch.force_end_generation = None
            watch.expected = ("s1:42", "s2:42:00000")
            watch.manifest = {"runtime_gate_job": "900"}
            watch.state = {
                "generation": 2,
                "jobs": {"s1:42": "901", "s2:42:00000": "902"},
                "retries": {},
                "watchdog_cancelled_job_ids": [],
                "progress": {},
                "blocked": {
                    "s1:42": "old",
                    "s2:42:00000": "old",
                },
            }
            watch.log_action = lambda *args, **kwargs: None
            watch.save = lambda: None
            watch.missing_artifacts = lambda: ["s1:42", "s2:42:00000"]
            resumed = []
            watch.resume_chain = lambda missing: resumed.extend(missing)
            statuses = {
                "900": mod.JobStatus(
                    "900", "mhtpw_runtime_gate", "FAILED", "n[1-5]", "sacct"
                ),
                "901": mod.JobStatus(
                    "901", "mhtpw_s1_s42", "CANCELLED", "", "sacct"
                ),
                "902": mod.JobStatus(
                    "902", "mhtpw_00000_s42", "CANCELLED", "", "sacct"
                ),
            }
            with patch.object(
                mod, "query_job", side_effect=lambda job_id: statuses[str(job_id)]
            ):
                self.assertEqual(watch.cycle(), (False, True))
            self.assertEqual(watch.state["blocked"], {})
            self.assertTrue(watch.state["needs_resume"])
            self.assertTrue(
                watch.state["runtime_gate_retry_cause"].startswith(
                    "known_runtime_failure:"
                )
            )
            self.assertEqual(resumed, ["s1:42", "s2:42:00000"])
            self.assertTrue(
                watch.state["retry_causes"]["s1:42"].startswith(
                    "dependency_cancelled_after:runtime_gate:"
                )
            )

    def test_job_names_and_dependency_ids_are_exact(self) -> None:
        self.assertEqual(mod.logical_from_job_name("mhtpw_s1_s2025"), "s1:2025")
        self.assertEqual(mod.logical_from_job_name("mhtpw_10110_s42"), "s2:42:10110")
        self.assertEqual(mod.logical_from_job_name("       mhtpw_s1_s42"), "s1:42")
        self.assertIsNone(mod.logical_from_job_name("mhtpw_imp_s42"))
        detail = "JobId=10 JobName=x Dependency=afterok:101:102:103(unfulfilled) WorkDir=/tmp"
        self.assertEqual(mod.dependency_job_ids(detail), ["101", "102", "103"])

    def test_squeue_fixed_width_job_name_is_stripped(self) -> None:
        result = SimpleNamespace(
            returncode=0,
            stdout="                                                                                                                                                                                            mhtpw_s1_s42|RUNNING|e16r3n[05-09]\n",
            stderr="",
        )
        with patch.object(mod, "run_command", return_value=result):
            status = mod.query_job("117602712")
        self.assertEqual(status.name, "mhtpw_s1_s42")
        self.assertEqual(mod.logical_from_job_name(status.name), "s1:42")
        self.assertEqual(status.nodes, "e16r3n[05-09]")

    def test_modern_training_progress_is_semantic(self) -> None:
        with workspace_temp_dir() as tmp:
            path = tmp / "job.out"
            path.write_text(
                "RUN_ID          : exp_qcore_hybrid_demo_mhtpw00000_seed42_pm10_pm25\n"
                "[S2_PhaseA] start steps=12000 trainable=head\n"
                "[S2_PhaseA] step=50/12000 loss=0.123 at=2026-07-22T00:00:00+08:00\n"
                "[S2_PhaseA] validation start step=50/12000 at=2026-07-22T00:01:00+08:00\n",
                encoding="utf-8",
            )
            progress = mod.parse_progress(path)
            self.assertEqual(progress.phase, "validation")
            self.assertIn("validation:S2_PhaseA:50", progress.token)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("[S2_PhaseB] step=100/40000 loss=0.111 at=2026-07-22T00:10:00+08:00\n")
            progress = mod.parse_progress(path)
            self.assertEqual(progress.phase, "training")
            self.assertIn("step:S2_PhaseB:100/40000", progress.token)

    def test_unrelated_startup_output_does_not_fake_progress(self) -> None:
        with workspace_temp_dir() as tmp:
            path = tmp / "job.out"
            path.write_text("WARNING: NUMA setting\n", encoding="utf-8")
            first = mod.parse_progress(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("WARNING: another transport warning\n")
            second = mod.parse_progress(path)
            self.assertEqual(first.phase, "startup")
            self.assertEqual(first.token, second.token)

    def test_artifact_triplet_and_quarantine_are_scoped(self) -> None:
        with workspace_temp_dir() as tmp:
            ckpt = tmp
            run_id = "exp_qcore_hybrid_demo_mhtpw00000_seed42_pm10_pm25"
            files = [
                ckpt / f"{run_id}_S2_PhaseB_best_score.pt",
                ckpt / f"{run_id}_static_rnn_config.json",
                ckpt / f"robust_scaler_{run_id}_s2_w12_dyn18_pm.pkl",
            ]
            for path in files:
                path.write_bytes(b"x")
            self.assertTrue(mod.artifact_complete(ckpt, run_id, "s2"))
            moved = mod.quarantine_artifacts(ckpt, "demo", "s2:42:00000", 1)
            self.assertEqual(len(moved), 3)
            self.assertFalse(mod.artifact_complete(ckpt, run_id, "s2"))
            self.assertTrue(all(Path(path).is_file() for path in moved))

    def test_confirmed_s2_stall_cancels_only_that_job(self) -> None:
        watch = object.__new__(mod.ChainWatch)
        watch.args = SimpleNamespace(
            auto_retry=True,
            max_retries=2,
            recheck_seconds=10,
            cancel_wait_seconds=1,
            exclude_failed_nodes=True,
        )
        watch.run_tag = "demo"
        watch.checkpoint_dir = Path("C:/unused")
        watch.state = {
            "generation": 0,
            "jobs": {"s2:42:00000": "123"},
            "retries": {},
            "watchdog_cancelled_job_ids": [],
            "progress": {},
        }
        watch.log_action = lambda *args, **kwargs: None
        watch.save = lambda: None
        progress = mod.Progress("training", "step:S2_PhaseA:100/12000:best0", "")
        watch.current_progress = lambda logical, job_id: progress
        statuses = [
            mod.JobStatus("123", "mhtpw_00000_s42", "RUNNING", "n1", "squeue"),
            mod.JobStatus("123", "mhtpw_00000_s42", "RUNNING", "n1", "squeue"),
            mod.JobStatus("123", "mhtpw_00000_s42", "CANCELLED", "n1", "sacct"),
        ]
        commands = []

        def fake_command(args, **kwargs):
            commands.append(list(args))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch.object(mod, "query_job", side_effect=statuses),
            patch.object(mod.time, "sleep", return_value=None),
            patch.object(mod, "run_command", side_effect=fake_command),
            patch.object(mod, "quarantine_artifacts", return_value=[]),
        ):
            self.assertTrue(watch.cancel_stalled("s2:42:00000", "123", progress))
        self.assertIn(["scancel", "123"], commands)
        self.assertEqual(watch.state["watchdog_cancelled_job_ids"], ["123"])
        self.assertEqual(watch.state["failed_node_lists"], ["n1"])
        self.assertTrue(watch.state["needs_resume"])

    def test_dead_s1_cleanup_is_limited_to_same_seed_dependents(self) -> None:
        watch = object.__new__(mod.ChainWatch)
        watch.args = SimpleNamespace(auto_retry=True, cancel_wait_seconds=1)
        watch.run_tag = "demo"
        watch.checkpoint_dir = Path("C:/unused")
        watch.state = {
            "generation": 0,
            "jobs": {
                "s1:42": "100",
                "s2:42:00000": "101",
                "s2:42:00001": "102",
                "s2:2025:00000": "201",
            },
            "watchdog_cancelled_job_ids": [],
        }
        watch.log_action = lambda *args, **kwargs: None
        status_by_id = {
            "101": ["PENDING", "CANCELLED"],
            "102": ["PENDING", "CANCELLED"],
            "201": ["PENDING"],
        }

        def fake_query(job_id):
            state = status_by_id[str(job_id)].pop(0)
            return mod.JobStatus(str(job_id), "x", state, "", "squeue" if state == "PENDING" else "sacct")

        commands = []

        def fake_command(args, **kwargs):
            commands.append(list(args))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch.object(mod, "query_job", side_effect=fake_query),
            patch.object(mod, "scontrol_detail", return_value="Dependency=afterok:100"),
            patch.object(mod, "run_command", side_effect=fake_command),
            patch.object(mod, "quarantine_artifacts", return_value=[]),
        ):
            watch.cancel_dead_s1_children("s1:42", "100", "NODE_FAIL")
        self.assertIn(["scancel", "101", "102"], commands)
        self.assertNotIn("201", watch.state["watchdog_cancelled_job_ids"])
        self.assertEqual(watch.state["watchdog_cancelled_job_ids"], ["101", "102"])

    def test_force_end_generation_cancels_only_exact_tracked_active_jobs(self) -> None:
        with workspace_temp_dir() as tmp:
            watch = object.__new__(mod.ChainWatch)
            watch.args = SimpleNamespace(
                auto_retry=True,
                cancel_wait_seconds=1,
                exclude_failed_nodes=True,
            )
            watch.run_tag = "demo"
            watch.checkpoint_dir = tmp / "checkpoints"
            watch.checkpoint_dir.mkdir()
            watch.force_end_generation = 0
            watch.state = {
                "generation": 0,
                "jobs": {
                    "s1:20260702": "100",
                    "s2:20260702:00000": "101",
                    "s2:42:00000": "201",
                },
                "retries": {},
                "watchdog_cancelled_job_ids": [],
                "blocked": {},
            }
            run_id, _ = mod.run_id_for_logical("demo", "s1:20260702")
            for name in (
                f"{run_id}_S1_best_score.pt",
                f"{run_id}_static_rnn_config.json",
                f"robust_scaler_{run_id}_s1_w12_dyn18_pm.pkl",
            ):
                (watch.checkpoint_dir / name).write_bytes(b"x")
            watch.log_action = lambda *args, **kwargs: None
            watch.save = lambda: None

            calls = {"100": 0, "101": 0, "201": 0}

            def fake_query(job_id):
                job_id = str(job_id)
                calls[job_id] += 1
                if job_id == "100":
                    state = "RUNNING" if calls[job_id] == 1 else "CANCELLED"
                    name = "mhtpw_s1_s20260702"
                elif job_id == "101":
                    state = "PENDING" if calls[job_id] == 1 else "CANCELLED"
                    name = "mhtpw_00000_s20260702"
                else:
                    state = "FAILED"
                    name = "mhtpw_00000_s42"
                return mod.JobStatus(
                    job_id,
                    name,
                    state,
                    "n1" if job_id == "100" else "",
                    "squeue" if state in mod.ACTIVE_STATES else "sacct",
                )

            commands = []

            def fake_command(args, **kwargs):
                commands.append(list(args))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(mod, "query_job", side_effect=fake_query),
                patch.object(mod, "run_command", side_effect=fake_command),
            ):
                watch.force_end_generation_now(watch.state["jobs"])

            self.assertIn(["scancel", "100", "101"], commands)
            self.assertNotIn("201", watch.state["watchdog_cancelled_job_ids"])
            self.assertEqual(
                watch.state["watchdog_cancelled_job_ids"],
                ["100", "101"],
            )
            self.assertEqual(
                watch.state["force_adopted_terminal_job_ids"],
                ["201"],
            )
            self.assertTrue(watch.state["needs_resume"])
            self.assertEqual(
                watch.state["force_ended_generations"]["0"]["quarantined_files"],
                3,
            )
            self.assertFalse(mod.artifact_complete(
                watch.checkpoint_dir, run_id, "s1"
            ))

    def test_runtime_failed_s1_adopts_afterok_cancelled_children(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            (logs / "100.err").write_text(
                "RuntimeError: No HIP GPUs are available\n",
                encoding="utf-8",
            )
            watch = object.__new__(mod.ChainWatch)
            watch.args = SimpleNamespace(
                auto_retry=True,
                adopt_runtime_failure=True,
                adopt_local_cache_failure=False,
                adopt_dispatcher_failure=False,
                checkpoint_grace_minutes=30,
                exclude_failed_nodes=True,
            )
            watch.run_tag = "demo"
            watch.baseline_dir = tmp
            watch.checkpoint_dir = tmp / "checkpoints"
            watch.checkpoint_dir.mkdir()
            watch.force_end_generation = None
            watch.expected = ("s1:20260702", "s2:20260702:00000", "s1:42")
            watch.state = {
                "generation": 1,
                "jobs": {
                    "s1:20260702": "100",
                    "s2:20260702:00000": "101",
                    "s1:42": "200",
                },
                "retries": {},
                "watchdog_cancelled_job_ids": [],
                "progress": {},
                "blocked": {
                    "s1:20260702": "old",
                    "s2:20260702:00000": "old",
                },
            }
            watch.log_action = lambda *args, **kwargs: None
            watch.save = lambda: None
            watch.observe_running = lambda *args, **kwargs: None
            watch.cancel_dead_s1_children = lambda *args, **kwargs: None
            watch.missing_artifacts = lambda: [
                "s1:20260702",
                "s2:20260702:00000",
            ]
            statuses = [
                mod.JobStatus("100", "mhtpw_s1_s20260702", "FAILED", "n5", "sacct"),
                mod.JobStatus("101", "mhtpw_00000_s20260702", "CANCELLED", "", "sacct"),
                mod.JobStatus("200", "mhtpw_s1_s42", "PENDING", "", "squeue"),
            ]
            with (
                patch.object(mod, "query_job", side_effect=statuses),
                patch.object(mod, "quarantine_artifacts", return_value=[]),
            ):
                self.assertEqual(watch.cycle(), (False, True))
            self.assertEqual(
                watch.state["retry_causes"]["s1:20260702"],
                "known_runtime_failure:RuntimeError: No HIP GPUs are available",
            )
            self.assertEqual(
                watch.state["retry_causes"]["s2:20260702:00000"],
                "dependency_cancelled_after:s1:20260702",
            )
            self.assertEqual(watch.state["blocked"], {})
            self.assertNotIn("failed_node_lists", watch.state)

    def test_transient_terminal_artifacts_are_quarantined_once(self) -> None:
        watch = object.__new__(mod.ChainWatch)
        watch.args = SimpleNamespace(auto_retry=True, exclude_failed_nodes=True)
        watch.run_tag = "demo"
        watch.checkpoint_dir = Path("C:/unused")
        watch.state = {
            "generation": 0,
            "jobs": {"s2:42:00000": "123"},
            "retries": {},
            "watchdog_cancelled_job_ids": [],
            "progress": {},
        }
        watch.log_action = lambda *args, **kwargs: None
        watch.save = lambda: None
        watch.cancel_dead_s1_children = lambda *args, **kwargs: None
        watch.missing_artifacts = lambda: ["s2:42:00000"]
        status = mod.JobStatus("123", "mhtpw_00000_s42", "NODE_FAIL", "n2", "sacct")
        quarantines = []

        def fake_quarantine(*args):
            quarantines.append(args)
            return ["moved"]

        with (
            patch.object(mod, "query_job", return_value=status),
            patch.object(mod, "quarantine_artifacts", side_effect=fake_quarantine),
        ):
            # Other generation jobs would normally keep the controller from
            # resubmitting in this cycle. Exercise the terminal processing
            # twice and assert the same failed attempt is handled once.
            watch.state["jobs"]["s2:2025:00000"] = "456"
            active = mod.JobStatus("456", "mhtpw_00000_s2025", "PENDING", "", "squeue")
            with patch.object(mod, "query_job", side_effect=[status, active, status, active]):
                self.assertEqual(watch.cycle(), (False, True))
                self.assertEqual(watch.cycle(), (False, True))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(watch.state["failed_node_lists"], ["n2"])

    def test_exact_dispatcher_failure_is_adopted_but_other_failed_jobs_are_not(self) -> None:
        with workspace_temp_dir() as tmp:
            logs = tmp / "logs"
            logs.mkdir()
            (logs / "123.out").write_text(
                mod.KNOWN_MHTPW_DISPATCHER_FAILURE + "\n",
                encoding="utf-8",
            )
            watch = object.__new__(mod.ChainWatch)
            watch.args = SimpleNamespace(
                auto_retry=True,
                adopt_dispatcher_failure=True,
                checkpoint_grace_minutes=30,
            )
            watch.run_tag = "demo"
            watch.baseline_dir = tmp
            watch.checkpoint_dir = tmp / "checkpoints"
            watch.expected = ("s2:42:00000", "s1:2025")
            watch.state = {
                "generation": 0,
                "jobs": {"s2:42:00000": "123", "s1:2025": "456"},
                "retries": {},
                "watchdog_cancelled_job_ids": [],
                "progress": {},
                "blocked": {
                    "s2:42:00000": "job 123 ended in FAILED; automatic blind retry is disabled"
                },
            }
            watch.log_action = lambda *args, **kwargs: None
            watch.save = lambda: None
            watch.observe_running = lambda *args, **kwargs: None
            watch.cancel_dead_s1_children = lambda *args, **kwargs: None
            statuses = [
                mod.JobStatus("123", "mhtpw_00000_s42", "FAILED", "n1", "sacct"),
                mod.JobStatus("456", "mhtpw_s1_s2025", "RUNNING", "n2", "squeue"),
            ]
            with (
                patch.object(mod, "query_job", side_effect=statuses),
                patch.object(mod, "quarantine_artifacts", return_value=[]),
            ):
                self.assertEqual(watch.cycle(), (False, True))
            self.assertNotIn("s2:42:00000", watch.state["blocked"])
            self.assertTrue(watch.state["needs_resume"])
            self.assertEqual(
                watch.state["retry_causes"]["s2:42:00000"],
                "known_dispatcher_alias_failure",
            )

            watch.state["blocked"] = {}
            watch.state["jobs"] = {"s2:42:00000": "124", "s1:2025": "456"}
            statuses = [
                mod.JobStatus("124", "mhtpw_00000_s42", "FAILED", "n1", "sacct"),
                mod.JobStatus("456", "mhtpw_s1_s2025", "RUNNING", "n2", "squeue"),
            ]
            with patch.object(mod, "query_job", side_effect=statuses):
                with self.assertRaisesRegex(RuntimeError, "watchdog blocked"):
                    watch.cycle()


if __name__ == "__main__":
    unittest.main()
