#!/usr/bin/env python3
"""Watch and recover one submitted q-core+T925 MHTPW Slurm chain.

The controller is intentionally scoped to one RUN_TAG and its recorded
submission manifest. It treats a training job as stalled only when Slurm still
reports RUNNING and a parsed semantic progress token is unchanged beyond the
phase-specific limit for repeated polls plus a final recheck. Recovery cancels
only the affected S2, or an affected S1 together with its still-dependent S2
children. After the rest of that generation becomes terminal, the tracked
RESUME_EXISTING_RUN entrypoint resubmits only incomplete artifact triplets and
creates a fresh downstream audit/evaluation/analysis chain.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - local Windows syntax/tests
    fcntl = None  # type: ignore[assignment]


MASKS = tuple(f"{value:05b}" for value in range(32))
ACTIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}
TRANSIENT_TERMINAL_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REQUEUED"}
BLOCKED_TERMINAL_STATES = {"FAILED", "OUT_OF_MEMORY", "OOM", "TIMEOUT", "DEADLINE"}
KNOWN_MHTPW_DISPATCHER_FAILURE = "Unknown EXPERIMENT=s2_q_core_t925_mhtpw"
STRICT_LOCAL_CACHE_FAILURE = "[Local-Cache-Preflight] ERROR:"
KNOWN_RUNTIME_FAILURE_SIGNATURES = (
    "[DCU-Runtime-Preflight] ERROR:",
    "RuntimeError: No HIP GPUs are available",
    "oom-kill event(s)",
    "Out Of Memory",
)
STATE_ALIASES = {
    "PD": "PENDING",
    "CF": "CONFIGURING",
    "R": "RUNNING",
    "CG": "COMPLETING",
    "CD": "COMPLETED",
    "CA": "CANCELLED",
    "F": "FAILED",
    "NF": "NODE_FAIL",
    "OOM": "OUT_OF_MEMORY",
    "TO": "TIMEOUT",
    "PR": "PREEMPTED",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_manifest(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed manifest line in {path}: {raw!r}")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def parse_colon(value: str) -> list[str]:
    return [part.strip() for part in str(value).replace(",", ":").split(":") if part.strip()]


def manifest_generation_record(
    manifest: Mapping[str, str], jobs: Mapping[str, str]
) -> Dict[str, object]:
    """Return the submission-generation fields that must not drift silently."""
    return {
        "artifact_audit_job": str(manifest.get("artifact_audit_job", "")).strip(),
        "runtime_gate_job": str(manifest.get("runtime_gate_job", "")).strip(),
        "training_job_ids": parse_colon(manifest.get("training_job_ids", "")),
        "jobs": {str(key): str(value) for key, value in sorted(jobs.items())},
    }


def normalize_state(value: str) -> str:
    state = str(value).strip().upper().split("+")[0].split()[0] if str(value).strip() else "UNKNOWN"
    return STATE_ALIASES.get(state, state)


def run_command(args: Sequence[str], *, check: bool = False, cwd: Path | None = None, env=None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout={result.stdout.strip()}\nstderr={result.stderr.strip()}"
        )
    return result


@dataclasses.dataclass(frozen=True)
class JobStatus:
    job_id: str
    name: str
    state: str
    nodes: str
    source: str


def query_job(job_id: str) -> JobStatus:
    active = run_command(["squeue", "-h", "-j", str(job_id), "-o", "%.200j|%T|%N"])
    if active.returncode == 0:
        lines = [line for line in active.stdout.splitlines() if line.strip()]
        if len(lines) == 1:
            parts = lines[0].split("|", 2)
            if len(parts) == 3:
                return JobStatus(
                    str(job_id),
                    parts[0].strip(),
                    normalize_state(parts[1]),
                    parts[2].strip(),
                    "squeue",
                )
        if len(lines) > 1:
            raise RuntimeError(f"squeue returned multiple rows for job {job_id}: {lines}")
    accounting = run_command(
        ["sacct", "-X", "-n", "-P", "-j", str(job_id), "--format=JobIDRaw,JobName,State,NodeList"]
    )
    if accounting.returncode != 0:
        return JobStatus(str(job_id), "", "QUERY_ERROR", "", "sacct_error")
    for line in accounting.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[0] == str(job_id):
            return JobStatus(
                str(job_id),
                parts[1].strip(),
                normalize_state(parts[2]),
                parts[3].strip(),
                "sacct",
            )
    return JobStatus(str(job_id), "", "UNKNOWN", "", "not_found")


def scontrol_detail(job_id: str) -> str:
    result = run_command(["scontrol", "show", "job", "-o", str(job_id)])
    return result.stdout.strip() if result.returncode == 0 else ""


def active_jobs_named(names: Iterable[str]) -> Dict[str, str]:
    wanted = set(names)
    if not wanted:
        return {}
    result = run_command(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A|%.200j"])
    found: Dict[str, str] = {}
    duplicates: Dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) != 2:
            continue
        job_id, name = parts[0].strip(), parts[1].strip()
        if name not in wanted:
            continue
        if name in found:
            duplicates.setdefault(name, [found[name]]).append(job_id)
        found[name] = job_id
    if duplicates:
        raise RuntimeError(f"ambiguous active job names: {duplicates}")
    return found


def dependency_job_ids(detail: str) -> list[str]:
    match = re.search(r"(?:^|\s)Dependency=(\S+)", detail)
    if not match or match.group(1) in {"(null)", "None", ""}:
        return []
    return list(dict.fromkeys(re.findall(r"\d+", match.group(1))))


def logical_from_job_name(name: str) -> str | None:
    normalized = str(name).strip()
    s1 = re.fullmatch(r"mhtpw_s1_s(\d+)", normalized)
    if s1:
        return f"s1:{s1.group(1)}"
    s2 = re.fullmatch(r"mhtpw_([01]{5})_s(\d+)", normalized)
    if s2:
        return f"s2:{s2.group(2)}:{s2.group(1)}"
    return None


def expected_logicals(seeds: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        [f"s1:{seed}" for seed in seeds]
        + [f"s2:{seed}:{mask}" for seed in seeds for mask in MASKS]
    )


def job_name_for_logical(logical: str) -> str:
    parts = logical.split(":")
    if parts[0] == "s1":
        return f"mhtpw_s1_s{parts[1]}"
    return f"mhtpw_{parts[2]}_s{parts[1]}"


def run_id_for_logical(run_tag: str, logical: str) -> tuple[str, str]:
    parts = logical.split(":")
    if parts[0] == "s1":
        return f"exp_qcore_hybrid_{run_tag}_s1_seed{parts[1]}_pm10_pm25", "s1"
    return f"exp_qcore_hybrid_{run_tag}_mhtpw{parts[2]}_seed{parts[1]}_pm10_pm25", "s2"


def artifact_paths(checkpoint_dir: Path, run_id: str, stage: str) -> list[Path]:
    checkpoint_tag = "S1_best_score" if stage == "s1" else "S2_PhaseB_best_score"
    scalers = [Path(item) for item in glob.glob(str(checkpoint_dir / f"robust_scaler_{run_id}_{stage}_w12_dyn*_pm.pkl"))]
    paths = [
        checkpoint_dir / f"{run_id}_{checkpoint_tag}.pt",
        checkpoint_dir / f"{run_id}_static_rnn_config.json",
    ]
    if len(scalers) == 1:
        paths.append(scalers[0])
    return paths


def artifact_complete(checkpoint_dir: Path, run_id: str, stage: str) -> bool:
    paths = artifact_paths(checkpoint_dir, run_id, stage)
    return len(paths) == 3 and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def dispatcher_accepts_mhtpw_alias(baseline_dir: Path) -> bool:
    path = baseline_dir / "sub_ifs_overlap_baseline.slurm"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return bool(
        re.search(
            r"(?m)^\s*[^#\n]*s2_q_core_t925_mhtpw[^#\n]*\)\s*$",
            text,
        )
    )


def has_job_log_signature(
    baseline_dir: Path, job_id: str, job_name: str, signature: str
) -> bool:
    safe_name = str(job_name).strip()
    candidates = [
        baseline_dir / "logs" / f"{job_id}.{suffix}"
        for suffix in ("out", "err")
    ]
    if safe_name:
        candidates.extend(
            baseline_dir / "logs" / f"{job_id}_{safe_name}.{suffix}"
            for suffix in ("out", "err")
        )
    for path in candidates:
        if not path.is_file():
            continue
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > 256_000:
                handle.seek(size - 256_000)
            text = handle.read().decode("utf-8", errors="replace")
        if signature in text:
            return True
    return False


def has_known_dispatcher_failure(
    baseline_dir: Path, job_id: str, job_name: str
) -> bool:
    return has_job_log_signature(
        baseline_dir,
        job_id,
        job_name,
        KNOWN_MHTPW_DISPATCHER_FAILURE,
    )


def matching_job_log_signature(
    baseline_dir: Path,
    job_id: str,
    job_name: str,
    signatures: Sequence[str],
) -> str:
    for signature in signatures:
        if has_job_log_signature(
            baseline_dir,
            job_id,
            job_name,
            signature,
        ):
            return signature
    return ""


def quarantine_artifacts(checkpoint_dir: Path, run_tag: str, logical: str, attempt: int) -> list[str]:
    run_id, _ = run_id_for_logical(run_tag, logical)
    candidates = list(checkpoint_dir.glob(f"{run_id}_*"))
    candidates.extend(checkpoint_dir.glob(f"robust_scaler_{run_id}_*"))
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        return []
    target = checkpoint_dir / "watchdog_quarantine" / run_tag / logical.replace(":", "_") / f"attempt_{attempt}"
    target.mkdir(parents=True, exist_ok=True)
    destination_root = target
    if any((target / source.name).exists() for source in candidates):
        # A controller may stop after Slurm cancellation but before its state
        # transaction is committed.  Re-entering the same logical attempt must
        # preserve both the earlier forensic files and the newly produced
        # artifacts instead of raising FileExistsError after the job is gone.
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"collision_{stamp}_pid{os.getpid()}"
        destination_root = target / base_name
        suffix = 0
        while destination_root.exists():
            suffix += 1
            destination_root = target / f"{base_name}_{suffix}"
        destination_root.mkdir(parents=False, exist_ok=False)
    moved: list[str] = []
    for source in candidates:
        destination = destination_root / source.name
        shutil.move(str(source), str(destination))
        moved.append(str(destination))
    return moved


@dataclasses.dataclass(frozen=True)
class Progress:
    phase: str
    token: str
    run_id_seen: str


def parse_progress(path: Path) -> Progress:
    if not path.is_file():
        return Progress("startup", "no_log", "")
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > 4_000_000:
            handle.seek(size - 4_000_000)
        text = handle.read().decode("utf-8", errors="replace")
    run_ids = re.findall(r"(?:^|\n)RUN_ID\s*:\s*(\S+)", text)
    run_id = run_ids[-1] if run_ids else ""
    tag = r"(S1|S2(?:_Phase[A-D]|-[A-D]))"
    steps = list(
        re.finditer(rf"\[{tag}\]\s+(?:Step\s+|step=)(\d+)\s*/\s*(\d+)", text)
    )
    validations = list(
        re.finditer(rf"\[{tag}\]\s+(?:=== Validation at step\s+|validation start step=)(\d+)", text)
    )
    best_count = len(re.findall(r"(?:New best target_achievement|val score=)", text, re.I))
    final = re.search(r"(?:Final Best|Training complete|Finished at:|\[Done\].*finished)", text, re.I)
    if final:
        return Progress("complete", f"complete:{best_count}:{size}", run_id)
    last_step = steps[-1] if steps else None
    last_validation = validations[-1] if validations else None
    if last_validation is not None and (last_step is None or last_validation.start() > last_step.start()):
        return Progress(
            "validation",
            f"validation:{last_validation.group(1)}:{last_validation.group(2)}:best{best_count}",
            run_id,
        )
    if last_step is not None:
        return Progress(
            "training",
            f"step:{last_step.group(1)}:{last_step.group(2)}/{last_step.group(3)}:best{best_count}",
            run_id,
        )
    if run_id or re.search(r"copy|dataset|dataloader|scaler|model|torchrun|RCCL|NCCL", text, re.I):
        markers = (
            bool(run_id),
            bool(re.search(r"copy", text, re.I)),
            bool(re.search(r"dataset|dataloader", text, re.I)),
            bool(re.search(r"scaler", text, re.I)),
            bool(re.search(r"model", text, re.I)),
            bool(re.search(r"Training started", text)),
        )
        return Progress("data", "data:" + "".join("1" if value else "0" for value in markers), run_id)
    return Progress("startup", "startup:no_semantic_marker", run_id)


class ChainWatch:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_tag = args.run_tag
        self.baseline_dir = Path(args.baseline_dir).expanduser().resolve()
        if args.adopt_dispatcher_failure and not dispatcher_accepts_mhtpw_alias(
            self.baseline_dir
        ):
            raise RuntimeError(
                "refusing known-dispatcher-failure recovery because the current "
                "sub_ifs_overlap_baseline.slurm still lacks s2_q_core_t925_mhtpw"
            )
        self.eval_root = Path(args.eval_root).expanduser().resolve()
        self.manifest_path = self.eval_root / f"submission_manifest_{self.run_tag}.txt"
        self.manifest = parse_manifest(self.manifest_path)
        if self.manifest.get("run_tag") != self.run_tag:
            raise ValueError(f"manifest run_tag mismatch: {self.manifest.get('run_tag')} != {self.run_tag}")
        if self.manifest.get("group_profile") != "mhtpw":
            raise ValueError(f"not an mhtpw manifest: {self.manifest_path}")
        self.seeds = parse_colon(self.manifest.get("seeds", ""))
        if not self.seeds:
            raise ValueError("manifest has no seeds")
        self.expected = expected_logicals(self.seeds)
        self.checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        self.watch_dir = self.eval_root / "watchdog"
        self.state_path = self.watch_dir / "mhtpw_watchdog_state.json"
        self.actions_path = self.watch_dir / "mhtpw_watchdog_actions.tsv"
        self.resolved_manifest = self.watch_dir / "resolved_submission_manifest.txt"
        self.lock_path = self.watch_dir / "mhtpw_watchdog.lock"
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.lock_path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("run_tag") != self.run_tag:
                raise ValueError("watchdog state belongs to a different run tag")
        else:
            jobs = self.discover_training_jobs(self.manifest)
            self.state = {
                "run_tag": self.run_tag,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "generation": 0,
                "jobs": jobs,
                "progress": {},
                "retries": {},
                "watchdog_cancelled_job_ids": [],
                "needs_resume": False,
                "blocked": {},
                "current_manifest": str(self.manifest_path),
                "manifest_generation": manifest_generation_record(
                    self.manifest, jobs
                ),
            }
            self.save()
        self.force_end_generation = (
            int(self.state.get("generation", 0))
            if args.force_end_current_generation
            else None
        )
        self.reconcile_inflight()
        self.reconcile_manifest_generation()

    def close(self) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_handle.close()

    def save(self) -> None:
        self.state["updated_at"] = now_iso()
        atomic_json(self.state_path, self.state)

    def reconcile_inflight(self) -> None:
        """Recover conservatively if the controller stopped mid-transaction."""
        changed = False
        cancel_intent = self.state.get("cancel_intent")
        if isinstance(cancel_intent, dict):
            ids = [str(item) for item in cancel_intent.get("job_ids", [])]
            states = {job_id: query_job(job_id).state for job_id in ids}
            if ids and all(state == "CANCELLED" for state in states.values()):
                cancelled = set(self.state.get("watchdog_cancelled_job_ids", []))
                cancelled.update(ids)
                self.state["watchdog_cancelled_job_ids"] = sorted(cancelled)
                self.state["needs_resume"] = True
                self.state.pop("cancel_intent", None)
                changed = True
            elif any(state in ACTIVE_STATES for state in states.values()):
                # The pre-cancel intent was saved, but cancellation was not
                # authoritatively completed. Clear it and require a new full
                # stale confirmation instead of replaying scancel blindly.
                self.state.pop("cancel_intent", None)
                changed = True
            else:
                self.state.setdefault("blocked", {})["cancel_transaction"] = (
                    "uncertain cancel transaction states=" + json.dumps(states, sort_keys=True)
                )
                changed = True

        resume_intent = self.state.get("resume_intent")
        if isinstance(resume_intent, dict):
            old_artifact = str(resume_intent.get("old_artifact_job", ""))
            current_artifact = self.manifest.get("artifact_audit_job", "")
            if current_artifact and current_artifact != old_artifact:
                self.state["generation"] = int(self.state.get("generation", 0)) + 1
                self.state["jobs"] = self.discover_training_jobs(self.manifest)
                self.state["progress"] = {}
                self.state["needs_resume"] = False
                self.state.pop("resume_intent", None)
                shutil.copy2(self.manifest_path, self.resolved_manifest)
                changed = True
            else:
                self.state.setdefault("blocked", {})["resume_transaction"] = (
                    "resume was interrupted before a new complete submission manifest was recorded; "
                    "refusing a duplicate submission"
                )
                changed = True
        if changed:
            self.save()

    def reconcile_manifest_generation(self) -> None:
        """Prevent an overwritten manifest from being paired with stale state.

        A RESUME_EXISTING_RUN submission rewrites the manifest at the same path.
        Comparing only ``current_manifest`` therefore cannot distinguish the old
        and new generations.  The exact logical-to-JobID mapping is authoritative.
        """
        current_jobs = self.discover_training_jobs(self.manifest)
        tracked_jobs = {
            str(key): str(value)
            for key, value in dict(self.state.get("jobs", {})).items()
        }
        current_record = manifest_generation_record(self.manifest, current_jobs)
        if tracked_jobs == current_jobs:
            if self.state.get("manifest_generation") != current_record:
                self.state["manifest_generation"] = current_record
                self.save()
            return

        if not self.args.adopt_current_manifest:
            tracked_ids = sorted(tracked_jobs.values())
            current_ids = sorted(current_jobs.values())
            raise RuntimeError(
                "watchdog state tracks a different submission generation; "
                f"tracked_jobs={len(tracked_jobs)} current_manifest_jobs={len(current_jobs)} "
                f"tracked_sample={tracked_ids[:3]} current_sample={current_ids[:3]}. "
                "No cancellation or submission was performed. Re-run the attachment "
                "with ADOPT_CURRENT_MANIFEST=YES after checking the preflight mapping."
            )

        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        old_generation = int(self.state.get("generation", 0))
        history_dir = (
            self.watch_dir
            / "history"
            / f"manifest_adopt_{timestamp}_generation_{old_generation}"
        )
        history_dir.mkdir(parents=True, exist_ok=False)
        for path in (
            self.state_path,
            self.actions_path,
            self.resolved_manifest,
        ):
            if path.is_file():
                shutil.copy2(path, history_dir / path.name)
        shutil.copy2(
            self.manifest_path,
            history_dir / f"adopted_{self.manifest_path.name}",
        )

        old_state = self.state
        self.state = {
            "run_tag": self.run_tag,
            "created_at": old_state.get("created_at", now_iso()),
            "updated_at": now_iso(),
            "generation": old_generation + 1,
            "jobs": current_jobs,
            "progress": {},
            "retries": dict(old_state.get("retries", {})),
            "watchdog_cancelled_job_ids": list(
                old_state.get("watchdog_cancelled_job_ids", [])
            ),
            "failed_node_lists": list(old_state.get("failed_node_lists", [])),
            "needs_resume": False,
            "blocked": {},
            "current_manifest": str(self.manifest_path),
            "manifest_generation": current_record,
            "adopted_previous_generation": {
                "at": now_iso(),
                "old_generation": old_generation,
                "old_jobs": tracked_jobs,
                "history_dir": str(history_dir),
            },
        }
        shutil.copy2(self.manifest_path, self.resolved_manifest)
        self.save()
        self.log_action(
            "chain",
            "ADOPT_CURRENT_MANIFEST",
            "",
            f"old_jobs={len(tracked_jobs)} current_jobs={len(current_jobs)} "
            f"history={history_dir}",
        )

    def log_action(self, logical: str, action: str, old_job: str = "", detail: str = "") -> None:
        new_file = not self.actions_path.exists()
        with self.actions_path.open("a", encoding="utf-8", newline="") as handle:
            if new_file:
                handle.write("time_utc\tgeneration\tlogical\taction\told_job\tdetail\n")
            safe = str(detail).replace("\t", " ").replace("\n", " ")
            handle.write(
                f"{now_iso()}\t{self.state['generation']}\t{logical}\t{action}\t{old_job}\t{safe}\n"
            )
        print(f"[watchdog] {action} logical={logical} old_job={old_job} {detail}", flush=True)

    def discover_training_jobs(self, manifest: Mapping[str, str]) -> Dict[str, str]:
        ids = parse_colon(manifest.get("training_job_ids", ""))
        if not ids:
            artifact = manifest.get("artifact_audit_job", "")
            if artifact.isdigit():
                ids = dependency_job_ids(scontrol_detail(artifact))
        mapping: Dict[str, str] = {}
        for job_id in ids:
            if not job_id.isdigit():
                continue
            status = query_job(job_id)
            logical = logical_from_job_name(status.name)
            if logical is None:
                raise ValueError(f"artifact dependency {job_id} is not an MHTPW training job: {status}")
            if logical in mapping:
                raise ValueError(f"duplicate active generation job for {logical}: {mapping[logical]}, {job_id}")
            mapping[logical] = job_id
        if not mapping:
            # Fallback is deliberately strict: duplicate job names from a smoke
            # and formal run make attachment ambiguous and stop the watcher.
            result = run_command(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A|%.200j"])
            by_logical: Dict[str, list[str]] = {}
            for line in result.stdout.splitlines():
                parts = line.strip().split("|", 1)
                if len(parts) != 2:
                    continue
                job_id, name = parts[0].strip(), parts[1].strip()
                logical = logical_from_job_name(name)
                if logical in self.expected:
                    by_logical.setdefault(logical, []).append(job_id)
            duplicates = {key: value for key, value in by_logical.items() if len(value) != 1}
            if duplicates:
                raise ValueError(f"ambiguous MHTPW active job names: {duplicates}")
            mapping = {key: values[0] for key, values in by_logical.items()}
        unknown = sorted(set(mapping) - set(self.expected))
        if unknown:
            raise ValueError(f"unexpected training rows in generation: {unknown}")
        return mapping

    def preflight(self) -> Dict[str, object]:
        jobs = dict(self.state.get("jobs", {}))
        rows = []
        for logical, job_id in sorted(jobs.items()):
            status = query_job(job_id)
            rows.append({"logical": logical, **dataclasses.asdict(status)})
        complete = 0
        missing = []
        for logical in self.expected:
            run_id, stage = run_id_for_logical(self.run_tag, logical)
            if artifact_complete(self.checkpoint_dir, run_id, stage):
                complete += 1
            else:
                missing.append(logical)
        report = {
            "run_tag": self.run_tag,
            "manifest": str(self.manifest_path),
            "expected_training_rows": len(self.expected),
            "tracked_generation_jobs": len(rows),
            "complete_artifact_triplets": complete,
            "missing_artifact_triplets": len(missing),
            "jobs": rows,
            "runtime_gate": self.runtime_gate_status_report(),
            "auto_retry": bool(self.args.auto_retry),
            "adopt_dispatcher_failure": bool(self.args.adopt_dispatcher_failure),
            "adopt_local_cache_failure": bool(
                self.args.adopt_local_cache_failure
            ),
            "adopt_runtime_failure": bool(self.args.adopt_runtime_failure),
            "adopt_current_manifest": bool(
                self.args.adopt_current_manifest
            ),
            "force_end_current_generation": bool(
                self.args.force_end_current_generation
            ),
            "force_end_generation": self.force_end_generation,
            "max_retries": self.args.max_retries,
        }
        return report

    def runtime_gate_status_report(self) -> Dict[str, str]:
        manifest = getattr(self, "manifest", {})
        job_id = str(manifest.get("runtime_gate_job", "")).strip()
        if not job_id or not job_id.isdigit():
            return {
                "job_id": job_id,
                "state": "NOT_RECORDED",
                "name": "",
                "nodes": "",
                "source": "manifest",
            }
        return dataclasses.asdict(query_job(job_id))

    def inspect_runtime_gate(
        self, blocked: MutableMapping[str, str]
    ) -> tuple[bool, str, bool]:
        """Return (active, recognized_failure_signature, terminal_failure)."""
        manifest = getattr(self, "manifest", {})
        job_id = str(manifest.get("runtime_gate_job", "")).strip()
        if not job_id or not job_id.isdigit():
            return False, "", False
        status = query_job(job_id)
        if status.state in ACTIVE_STATES or status.state in {
            "UNKNOWN",
            "QUERY_ERROR",
        }:
            return True, "", False
        if status.state == "COMPLETED":
            blocked.pop("runtime_gate", None)
            return False, "", False

        signature = ""
        if getattr(self.args, "adopt_runtime_failure", False):
            signature = matching_job_log_signature(
                self.baseline_dir,
                job_id,
                status.name,
                KNOWN_RUNTIME_FAILURE_SIGNATURES,
            )
        if not signature:
            blocked["runtime_gate"] = (
                f"job {job_id} ended in {status.state} without a recognized "
                "DCU runtime signature; automatic blind retry is disabled"
            )
            return False, "", True

        marker = f"{job_id}:{status.state}:{signature}"
        handled = self.state.setdefault("runtime_gate_failures", {})
        blocked.pop("runtime_gate", None)
        if handled.get("latest") != marker:
            handled["latest"] = marker
            self.log_action(
                "runtime_gate",
                "ADOPT_RUNTIME_GATE_FAILURE",
                job_id,
                f"state={status.state} signature={signature}",
            )
        self.state["needs_resume"] = True
        self.state["runtime_gate_retry_cause"] = (
            f"known_runtime_failure:{signature}"
        )
        return False, signature, True

    def phase_limit_seconds(self, phase: str) -> int:
        minutes = {
            "startup": self.args.startup_stale_minutes,
            "data": self.args.data_stale_minutes,
            "training": self.args.training_stale_minutes,
            "validation": self.args.validation_stale_minutes,
            "complete": self.args.validation_stale_minutes,
        }.get(phase, self.args.training_stale_minutes)
        return int(minutes * 60)

    def record_failed_nodes(self, nodes: str) -> None:
        if not self.args.exclude_failed_nodes:
            return
        value = str(nodes).strip()
        if not value or value in {"(null)", "N/A", "n/a", "None assigned"}:
            return
        failed = set(self.state.get("failed_node_lists", []))
        failed.add(value)
        self.state["failed_node_lists"] = sorted(failed)

    def current_progress(self, logical: str, job_id: str) -> Progress:
        path = self.baseline_dir / "logs" / f"{job_id}.out"
        progress = parse_progress(path)
        expected_run_id, _ = run_id_for_logical(self.run_tag, logical)
        if progress.run_id_seen and progress.run_id_seen != expected_run_id:
            raise RuntimeError(
                f"job/log identity mismatch for {logical}: expected {expected_run_id}, saw {progress.run_id_seen}"
            )
        return progress

    def cancel_stalled(self, logical: str, job_id: str, progress: Progress) -> bool:
        if not self.args.auto_retry:
            self.log_action(logical, "STALE_REPORT_ONLY", job_id, progress.token)
            return False
        retry_count = int(self.state.get("retries", {}).get(logical, 0))
        if retry_count >= self.args.max_retries:
            self.state.setdefault("blocked", {})[logical] = f"retry limit reached at stale {progress.token}"
            self.log_action(logical, "BLOCKED_RETRY_LIMIT", job_id, progress.token)
            return False
        first = query_job(job_id)
        if first.state != "RUNNING" or first.source != "squeue":
            return False
        time.sleep(self.args.recheck_seconds)
        second = query_job(job_id)
        second_progress = self.current_progress(logical, job_id)
        if second.state != "RUNNING" or second.source != "squeue" or second_progress.token != progress.token:
            self.log_action(logical, "STALE_RECHECK_ABORTED", job_id, f"{second.state} {second_progress.token}")
            return False
        self.record_failed_nodes(second.nodes)

        cancel_ids = [job_id]
        if logical.startswith("s1:"):
            seed = logical.split(":")[1]
            for child, child_id in dict(self.state.get("jobs", {})).items():
                if not child.startswith(f"s2:{seed}:"):
                    continue
                child_status = query_job(child_id)
                if child_status.state not in ACTIVE_STATES:
                    continue
                detail = scontrol_detail(child_id)
                if child_status.state in {"PENDING", "CONFIGURING"} and job_id not in dependency_job_ids(detail):
                    raise RuntimeError(
                        f"refusing S1-chain cancellation: child {child_id} no longer depends on S1 {job_id}"
                    )
                cancel_ids.append(child_id)

        self.state["cancel_intent"] = {
            "logical": logical,
            "job_ids": cancel_ids,
            "token": progress.token,
            "created_at": now_iso(),
        }
        self.save()
        run_command(["scancel", *cancel_ids], check=True)
        deadline = time.time() + self.args.cancel_wait_seconds
        remaining = set(cancel_ids)
        while remaining and time.time() < deadline:
            remaining = {item for item in remaining if query_job(item).state in ACTIVE_STATES}
            if remaining:
                time.sleep(3)
        if remaining:
            raise RuntimeError(f"cancel confirmation timed out for jobs {sorted(remaining)}")

        attempt = retry_count + 1
        moved = quarantine_artifacts(self.checkpoint_dir, self.run_tag, logical, attempt)
        if logical.startswith("s1:"):
            for child, child_id in dict(self.state.get("jobs", {})).items():
                if child_id not in cancel_ids or child_id == job_id:
                    continue
                child_attempt = int(self.state.get("retries", {}).get(child, 0)) + 1
                moved.extend(
                    quarantine_artifacts(
                        self.checkpoint_dir, self.run_tag, child, child_attempt
                    )
                )
        cancelled = set(self.state.get("watchdog_cancelled_job_ids", []))
        cancelled.update(cancel_ids)
        self.state["watchdog_cancelled_job_ids"] = sorted(cancelled)
        self.state["needs_resume"] = True
        self.state.pop("cancel_intent", None)
        self.state.setdefault("retry_causes", {})[logical] = f"semantic_stall:{progress.token}"
        self.log_action(
            logical,
            "CANCEL_CONFIRMED_STALL",
            job_id,
            f"cancelled={','.join(cancel_ids)} quarantined={len(moved)} token={progress.token}",
        )
        self.save()
        return True

    def cancel_dead_s1_children(self, logical: str, parent_job_id: str, parent_state: str) -> None:
        if not self.args.auto_retry or not logical.startswith("s1:"):
            return
        seed = logical.split(":")[1]
        child_ids: list[str] = []
        for child, child_id in dict(self.state.get("jobs", {})).items():
            if not child.startswith(f"s2:{seed}:"):
                continue
            status = query_job(child_id)
            if status.state not in ACTIVE_STATES:
                continue
            detail = scontrol_detail(child_id)
            if status.state in {"PENDING", "CONFIGURING"} and parent_job_id not in dependency_job_ids(detail):
                raise RuntimeError(
                    f"refusing dependent cleanup: child {child_id} does not depend on dead S1 {parent_job_id}"
                )
            child_ids.append(child_id)
        if not child_ids:
            return
        run_command(["scancel", *child_ids], check=True)
        deadline = time.time() + self.args.cancel_wait_seconds
        remaining = set(child_ids)
        while remaining and time.time() < deadline:
            remaining = {item for item in remaining if query_job(item).state in ACTIVE_STATES}
            if remaining:
                time.sleep(3)
        if remaining:
            raise RuntimeError(f"dependent S2 cancel confirmation timed out: {sorted(remaining)}")
        cancelled = set(self.state.get("watchdog_cancelled_job_ids", []))
        cancelled.update(child_ids)
        self.state["watchdog_cancelled_job_ids"] = sorted(cancelled)
        moved = 0
        for child, child_id in dict(self.state.get("jobs", {})).items():
            if child_id not in child_ids:
                continue
            attempt = int(self.state.get("retries", {}).get(child, 0)) + 1
            moved += len(
                quarantine_artifacts(self.checkpoint_dir, self.run_tag, child, attempt)
            )
        self.log_action(
            logical,
            "CANCEL_DEAD_S1_DEPENDENTS",
            parent_job_id,
            f"parent_state={parent_state} children={','.join(child_ids)} quarantined={moved}",
        )

    def cancel_failed_runtime_gate_dependents(
        self, jobs: Mapping[str, str]
    ) -> None:
        gate_id = str(self.manifest.get("runtime_gate_job", "")).strip()
        marker = f"{self.state.get('generation', 0)}:{gate_id}"
        if (
            not gate_id.isdigit()
            or self.state.get("runtime_gate_dependents_cancelled") == marker
        ):
            return

        active: list[tuple[str, str]] = []
        for logical, job_id in jobs.items():
            status = query_job(job_id)
            if status.state not in ACTIVE_STATES:
                continue
            if logical_from_job_name(status.name) != logical:
                raise RuntimeError(
                    f"refusing runtime-gate cleanup: job {job_id} identity "
                    f"{status.name!r} does not match {logical}"
                )
            dependencies = dependency_job_ids(scontrol_detail(job_id))
            if gate_id not in dependencies:
                raise RuntimeError(
                    f"refusing runtime-gate cleanup: active job {job_id} "
                    f"for {logical} does not depend on gate {gate_id}"
                )
            active.append((logical, job_id))

        ids = [job_id for _, job_id in active]
        if ids:
            run_command(["scancel", *ids], check=True)
            deadline = time.time() + self.args.cancel_wait_seconds
            remaining = set(ids)
            while remaining and time.time() < deadline:
                remaining = {
                    item
                    for item in remaining
                    if query_job(item).state in ACTIVE_STATES
                }
                if remaining:
                    time.sleep(3)
            if remaining:
                raise RuntimeError(
                    "runtime-gate dependent cancellation timed out: "
                    f"{sorted(remaining)}"
                )
            cancelled = set(
                self.state.get("watchdog_cancelled_job_ids", [])
            )
            cancelled.update(ids)
            self.state["watchdog_cancelled_job_ids"] = sorted(cancelled)
        self.state["runtime_gate_dependents_cancelled"] = marker
        self.state["needs_resume"] = True
        self.log_action(
            "runtime_gate",
            "CANCEL_FAILED_GATE_DEPENDENTS",
            gate_id,
            f"jobs={len(ids)}",
        )
        self.save()

    def force_end_generation_now(self, jobs: Mapping[str, str]) -> None:
        generation = int(self.state.get("generation", 0))
        target = getattr(self, "force_end_generation", None)
        if target is None or generation != target:
            return
        completed = self.state.setdefault("force_ended_generations", {})
        key = str(generation)
        if key in completed:
            return
        if not self.args.auto_retry:
            raise RuntimeError("force-ending a generation requires --auto-retry")

        active: list[tuple[str, str, JobStatus]] = []
        terminal_to_adopt: list[tuple[str, str, JobStatus]] = []
        for logical, job_id in jobs.items():
            status = query_job(job_id)
            if status.state in {"UNKNOWN", "QUERY_ERROR"}:
                raise RuntimeError(
                    f"refusing forced generation end: cannot authoritatively "
                    f"resolve job {job_id} for {logical} ({status.state})"
                )
            observed = logical_from_job_name(status.name)
            if observed != logical:
                raise RuntimeError(
                    f"refusing forced generation end: job {job_id} identity "
                    f"{observed!r} != {logical!r}"
                )
            if status.state == "COMPLETED":
                continue
            if status.state in ACTIVE_STATES:
                active.append((logical, job_id, status))
            else:
                terminal_to_adopt.append((logical, job_id, status))

        if not active and not terminal_to_adopt:
            completed[key] = {
                "at": now_iso(),
                "job_ids": [],
                "detail": "all tracked training jobs already completed",
            }
            self.log_action(
                "chain",
                "FORCE_END_ALL_JOBS_COMPLETED",
                "",
                f"generation={generation}",
            )
            self.save()
            return

        job_ids = [job_id for _, job_id, _ in active]
        self.state["force_end_intent"] = {
            "generation": generation,
            "created_at": now_iso(),
            "active_jobs": {logical: job_id for logical, job_id, _ in active},
            "terminal_jobs": {
                logical: {"job_id": job_id, "state": status.state}
                for logical, job_id, status in terminal_to_adopt
            },
        }
        self.save()
        if job_ids:
            run_command(["scancel", *job_ids], check=True)
            deadline = time.time() + self.args.cancel_wait_seconds
            remaining = set(job_ids)
            while remaining and time.time() < deadline:
                remaining = {
                    item for item in remaining
                    if query_job(item).state in ACTIVE_STATES
                }
                if remaining:
                    time.sleep(3)
            if remaining:
                raise RuntimeError(
                    f"forced generation-end cancellation timed out: {sorted(remaining)}"
                )

        cancelled = set(self.state.get("watchdog_cancelled_job_ids", []))
        force_adopted = set(self.state.get("force_adopted_terminal_job_ids", []))
        moved = 0
        for logical, job_id, status in active:
            cancelled.add(job_id)
            self.record_failed_nodes(status.nodes)
            attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
            moved += len(
                quarantine_artifacts(
                    self.checkpoint_dir, self.run_tag, logical, attempt
                )
            )
            self.state.setdefault("blocked", {}).pop(logical, None)
            self.state.setdefault("retry_causes", {})[logical] = (
                f"forced_generation_end:{generation}:active"
            )
        for logical, job_id, status in terminal_to_adopt:
            force_adopted.add(job_id)
            self.record_failed_nodes(status.nodes)
            attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
            moved += len(
                quarantine_artifacts(
                    self.checkpoint_dir, self.run_tag, logical, attempt
                )
            )
            self.state.setdefault("blocked", {}).pop(logical, None)
            self.state.setdefault("retry_causes", {})[logical] = (
                f"forced_generation_end:{generation}:terminal:{status.state}"
            )
        self.state["watchdog_cancelled_job_ids"] = sorted(cancelled)
        self.state["force_adopted_terminal_job_ids"] = sorted(force_adopted)
        self.state["needs_resume"] = True
        completed[key] = {
            "at": now_iso(),
            "job_ids": job_ids,
            "terminal_job_ids": [
                job_id for _, job_id, _ in terminal_to_adopt
            ],
            "quarantined_files": moved,
        }
        self.state.pop("force_end_intent", None)
        self.log_action(
            "chain",
            "FORCE_END_CURRENT_GENERATION",
            ",".join(job_ids),
            f"generation={generation} active={len(job_ids)} "
            f"terminal_adopted={len(terminal_to_adopt)} quarantined={moved}",
        )
        self.save()

    def observe_running(self, logical: str, job_id: str) -> None:
        progress = self.current_progress(logical, job_id)
        now = time.time()
        entries = self.state.setdefault("progress", {})
        entry = entries.get(logical)
        is_new_job = (
            not isinstance(entry, dict) or entry.get("job_id") != job_id
        )
        token_changed = (
            isinstance(entry, dict) and entry.get("token") != progress.token
        )
        if is_new_job or token_changed:
            last_change_epoch = now
            if is_new_job:
                log_path = self.baseline_dir / "logs" / f"{job_id}.out"
                try:
                    last_change_epoch = min(now, log_path.stat().st_mtime)
                except OSError:
                    pass
            entries[logical] = {
                "job_id": job_id,
                "token": progress.token,
                "phase": progress.phase,
                "last_change_epoch": last_change_epoch,
                "confirmations": 0,
            }
            age_minutes = max(0.0, (now - last_change_epoch) / 60.0)
            self.log_action(
                logical,
                "PROGRESS",
                job_id,
                f"{progress.phase} {progress.token} observed_age_min={age_minutes:.1f}",
            )
            return
        elapsed = now - float(entry.get("last_change_epoch", now))
        if elapsed < self.phase_limit_seconds(progress.phase):
            return
        entry["confirmations"] = int(entry.get("confirmations", 0)) + 1
        self.log_action(
            logical,
            "STALE_CONFIRMATION",
            job_id,
            f"{entry['confirmations']}/{self.args.confirmations} elapsed_min={elapsed/60:.1f} token={progress.token}",
        )
        if int(entry["confirmations"]) >= self.args.confirmations:
            self.cancel_stalled(logical, job_id, progress)

    def missing_artifacts(self) -> list[str]:
        missing = []
        for logical in self.expected:
            run_id, stage = run_id_for_logical(self.run_tag, logical)
            if not artifact_complete(self.checkpoint_dir, run_id, stage):
                missing.append(logical)
        return missing

    def resume_chain(self, missing: Sequence[str]) -> None:
        if not self.args.auto_retry:
            raise RuntimeError(f"training artifacts are missing and auto retry is disabled: {missing[:8]}")
        retries = self.state.setdefault("retries", {})
        exceeded = [logical for logical in missing if int(retries.get(logical, 0)) >= self.args.max_retries]
        if exceeded:
            raise RuntimeError(f"retry limit reached for missing rows: {exceeded}")
        for logical in missing:
            retries[logical] = int(retries.get(logical, 0)) + 1

        current_snapshot = self.watch_dir / f"submission_manifest_generation_{self.state['generation']}.txt"
        shutil.copy2(self.manifest_path, current_snapshot)
        self.state["resume_intent"] = {
            "created_at": now_iso(),
            "old_artifact_job": self.manifest.get("artifact_audit_job", ""),
            "missing": list(missing),
        }
        self.save()
        env = os.environ.copy()
        env.update(
            {
                "RUN_TAG": self.run_tag,
                "SOURCE_DATA_ROOT": self.manifest["source_data_root"],
                "HYBRID_DATA_ROOT": self.manifest["hybrid_data_root"],
                "EVAL_ROOT": str(self.eval_root),
                "ANALYSIS_DIR": self.manifest.get(
                    "analysis_dir", str(self.eval_root / "analysis")
                ),
                "SEEDS": self.manifest["seeds"],
                "RESUME_EXISTING_RUN": "1",
                "REUSE_COMPLETED_AUDITS": "1",
                "DRY_RUN": "0",
                "RUN_IMPORTANCE": self.manifest.get("run_importance", os.environ.get("WATCH_RUN_IMPORTANCE", "1")),
                "BOOTSTRAP_ITERS": self.manifest.get("bootstrap_iters", os.environ.get("WATCH_BOOTSTRAP_ITERS", "1000")),
                "BOOTSTRAP_MAX_ROWS": self.manifest.get("bootstrap_max_rows", os.environ.get("WATCH_BOOTSTRAP_MAX_ROWS", "0")),
                "LIMIT_SAMPLES": self.manifest.get("limit_samples", os.environ.get("WATCH_LIMIT_SAMPLES", "0")),
                "LOWVIS_RNN_S1_STEPS": self.manifest.get("s1_steps", os.environ.get("WATCH_S1_STEPS", "15000")),
                "LOWVIS_RNN_S2_A_STEPS": self.manifest.get("s2_a_steps", os.environ.get("WATCH_S2_A_STEPS", "12000")),
                "LOWVIS_RNN_S2_B_STEPS": self.manifest.get("s2_b_steps", os.environ.get("WATCH_S2_B_STEPS", "40000")),
                "OBS_ROOT": self.manifest.get("obs_root", os.environ.get("WATCH_OBS_ROOT", "")),
                "ERA5_DATA_DIR": self.manifest.get("era5_data_dir", os.environ.get("WATCH_ERA5_DATA_DIR", "")),
            }
        )
        if self.args.exclude_failed_nodes:
            existing_exclude = self.manifest.get("train_exclude_nodes", "").strip()
            exclude_lists = [existing_exclude] if existing_exclude else []
            exclude_lists.extend(
                str(value) for value in self.state.get("failed_node_lists", [])
            )
            env["TRAIN_EXCLUDE_NODES"] = ",".join(
                dict.fromkeys(value for value in exclude_lists if value)
            )
        else:
            # A code-induced failure (for example pre-Torch page-cache OOM)
            # must not progressively blacklist otherwise healthy allocations.
            env["TRAIN_EXCLUDE_NODES"] = ""
        launcher = self.baseline_dir / "submit_q_core_t925_mhtpw_factorial.sh"
        self.log_action("chain", "RESUME_SUBMIT_START", "", f"missing={len(missing)}")
        result = run_command(["bash", str(launcher)], cwd=self.baseline_dir, env=env)
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"resume launcher failed with exit code {result.returncode}")
        self.manifest = parse_manifest(self.manifest_path)
        jobs = self.discover_training_jobs(self.manifest)
        self.state["generation"] = int(self.state["generation"]) + 1
        self.state["jobs"] = jobs
        self.state["progress"] = {}
        causes = dict(self.state.get("retry_causes", {}))
        if causes:
            self.state.setdefault("retry_cause_history", []).append(
                {
                    "generation": int(self.state["generation"]) - 1,
                    "recorded_at": now_iso(),
                    "causes": causes,
                }
            )
        self.state["retry_causes"] = {}
        self.state["needs_resume"] = False
        self.state.pop("resume_intent", None)
        self.state["current_manifest"] = str(self.manifest_path)
        shutil.copy2(self.manifest_path, self.resolved_manifest)
        self.log_action("chain", "RESUME_SUBMIT_COMPLETE", "", f"new_training_jobs={len(jobs)}")
        self.save()

    def post_chain_status(self) -> tuple[bool, bool]:
        ids = []
        for key in ("artifact_audit_job", "eval_job_ids", "importance_job_ids", "factorial_analysis_job"):
            ids.extend(parse_colon(self.manifest.get(key, "")))
        any_active = False
        failures = []
        for job_id in dict.fromkeys(ids):
            if not job_id.isdigit():
                continue
            status = query_job(job_id)
            if status.state in ACTIVE_STATES or status.state in {"UNKNOWN", "QUERY_ERROR"}:
                any_active = True
            elif status.state != "COMPLETED":
                failures.append(f"{job_id}:{status.name}:{status.state}")
        if failures:
            raise RuntimeError("downstream job failure; not blindly retried: " + ", ".join(failures))
        report_path = Path(self.manifest.get("analysis_dir", str(self.eval_root / "analysis"))) / "hybrid_factorial_analysis_report.json"
        report_ok = False
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_ok = report.get("status") == "passed" and report.get("group_profile") == "mhtpw"
        importance_ok = True
        if self.manifest.get("run_importance", os.environ.get("WATCH_RUN_IMPORTANCE", "1")) == "1":
            missing_importance = [
                seed
                for seed in self.seeds
                if not (self.eval_root / "feature_importance" / f"seed_{seed}" / "run_config.json").is_file()
            ]
            importance_ok = not missing_importance
            if missing_importance and not self.manifest.get("importance_job_ids", ""):
                active_importance = active_jobs_named(f"mhtpw_imp_s{seed}" for seed in missing_importance)
                if active_importance:
                    any_active = True
                elif report_ok:
                    raise RuntimeError(
                        "factorial analysis completed but endpoint importance outputs are missing for seeds "
                        + ",".join(missing_importance)
                    )
        done = report_ok and importance_ok and not any_active
        return done, any_active

    def cycle(self) -> tuple[bool, bool]:
        jobs = dict(self.state.get("jobs", {}))
        self.force_end_generation_now(jobs)
        blocked = self.state.setdefault("blocked", {})
        gate_active, gate_signature, gate_terminal = (
            self.inspect_runtime_gate(blocked)
        )
        if gate_signature:
            self.cancel_failed_runtime_gate_dependents(jobs)
        any_active = gate_active
        for logical, job_id in jobs.items():
            status = query_job(job_id)
            if status.state in ACTIVE_STATES:
                blocked.pop(logical, None)
                any_active = True
                if status.state == "RUNNING":
                    self.observe_running(logical, job_id)
                continue
            if status.state == "COMPLETED":
                run_id, stage = run_id_for_logical(self.run_tag, logical)
                if not artifact_complete(self.checkpoint_dir, run_id, stage):
                    seen = self.state.setdefault("completed_without_artifact", {}).setdefault(logical, time.time())
                    if time.time() - float(seen) > self.args.checkpoint_grace_minutes * 60:
                        blocked[logical] = f"job {job_id} completed without a valid {stage} artifact triplet"
                    else:
                        any_active = True
                continue
            if (
                status.state not in ACTIVE_STATES
                and job_id
                in set(self.state.get("force_adopted_terminal_job_ids", []))
            ):
                blocked.pop(logical, None)
                self.state["needs_resume"] = True
                continue
            if status.state in TRANSIENT_TERMINAL_STATES:
                handled = self.state.setdefault("transient_handled", {})
                marker = f"{job_id}:{status.state}"
                if handled.get(logical) != marker:
                    self.record_failed_nodes(status.nodes)
                    self.cancel_dead_s1_children(logical, job_id, status.state)
                    attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
                    moved = quarantine_artifacts(
                        self.checkpoint_dir, self.run_tag, logical, attempt
                    )
                    handled[logical] = marker
                    self.log_action(
                        logical,
                        "TRANSIENT_TERMINAL",
                        job_id,
                        f"state={status.state} quarantined={len(moved)}",
                    )
                self.state["needs_resume"] = True
                self.state.setdefault("retry_causes", {})[logical] = f"transient_terminal:{status.state}"
                continue
            if (
                status.state == "CANCELLED"
                and job_id
                in set(self.state.get("watchdog_cancelled_job_ids", []))
            ):
                self.state["needs_resume"] = True
                continue
            if status.state == "CANCELLED" and gate_terminal:
                blocked.pop(logical, None)
                if gate_signature:
                    self.state["needs_resume"] = True
                    self.state.setdefault("retry_causes", {})[logical] = (
                        f"dependency_cancelled_after:runtime_gate:{gate_signature}"
                    )
                    self.log_action(
                        logical,
                        "ADOPT_RUNTIME_GATE_DEPENDENCY_CANCELLED",
                        job_id,
                        f"runtime_gate_signature={gate_signature}",
                    )
                # If the gate failure is unrecognized, runtime_gate remains the
                # single blocker instead of emitting 99 misleading job errors.
                continue
            if status.state == "CANCELLED" and logical.startswith("s2:"):
                seed = logical.split(":")[1]
                parent = f"s1:{seed}"
                parent_cause = str(
                    self.state.get("retry_causes", {}).get(parent, "")
                )
                allowed_parent_causes = (
                    "known_runtime_failure:",
                    "strict_local_cache_failure",
                    "transient_terminal:",
                    "semantic_stall:",
                    "forced_generation_end:",
                )
                if parent_cause.startswith(allowed_parent_causes):
                    blocked.pop(logical, None)
                    self.state["needs_resume"] = True
                    self.state.setdefault("retry_causes", {})[logical] = (
                        f"dependency_cancelled_after:{parent}"
                    )
                    self.log_action(
                        logical,
                        "ADOPT_DEPENDENCY_CANCELLED",
                        job_id,
                        f"parent={parent} cause={parent_cause}",
                    )
                    continue
            if status.state in {"UNKNOWN", "QUERY_ERROR"}:
                any_active = True
                continue
            runtime_signature = ""
            if (
                status.state
                in {"FAILED", "OUT_OF_MEMORY", "OOM"}
                and getattr(self.args, "adopt_runtime_failure", False)
            ):
                runtime_signature = matching_job_log_signature(
                    self.baseline_dir,
                    job_id,
                    status.name,
                    KNOWN_RUNTIME_FAILURE_SIGNATURES,
                )
            if runtime_signature:
                marker = f"{job_id}:{status.state}:{runtime_signature}"
                handled = self.state.setdefault("known_runtime_failures", {})
                blocked.pop(logical, None)
                if handled.get(logical) != marker:
                    self.cancel_dead_s1_children(
                        logical, job_id, status.state
                    )
                    attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
                    moved = quarantine_artifacts(
                        self.checkpoint_dir, self.run_tag, logical, attempt
                    )
                    handled[logical] = marker
                    self.log_action(
                        logical,
                        "ADOPT_KNOWN_RUNTIME_FAILURE",
                        job_id,
                        f"state={status.state} quarantined={len(moved)} signature={runtime_signature}",
                    )
                self.state["needs_resume"] = True
                self.state.setdefault("retry_causes", {})[logical] = (
                    f"known_runtime_failure:{runtime_signature}"
                )
                continue
            if (
                status.state == "FAILED"
                and getattr(self.args, "adopt_local_cache_failure", False)
                and has_job_log_signature(
                    self.baseline_dir,
                    job_id,
                    status.name,
                    STRICT_LOCAL_CACHE_FAILURE,
                )
            ):
                marker = f"{job_id}:{status.state}:{STRICT_LOCAL_CACHE_FAILURE}"
                handled = self.state.setdefault("known_local_cache_failures", {})
                blocked.pop(logical, None)
                if handled.get(logical) != marker:
                    self.record_failed_nodes(status.nodes)
                    attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
                    moved = quarantine_artifacts(
                        self.checkpoint_dir, self.run_tag, logical, attempt
                    )
                    handled[logical] = marker
                    self.log_action(
                        logical,
                        "ADOPT_STRICT_LOCAL_CACHE_FAILURE",
                        job_id,
                        f"quarantined={len(moved)} signature={STRICT_LOCAL_CACHE_FAILURE}",
                    )
                self.state["needs_resume"] = True
                self.state.setdefault("retry_causes", {})[logical] = (
                    "strict_local_cache_failure"
                )
                continue
            if (
                status.state == "FAILED"
                and logical.startswith("s2:")
                and getattr(self.args, "adopt_dispatcher_failure", False)
                and has_known_dispatcher_failure(
                    self.baseline_dir, job_id, status.name
                )
            ):
                marker = f"{job_id}:{status.state}:{KNOWN_MHTPW_DISPATCHER_FAILURE}"
                handled = self.state.setdefault("known_dispatcher_failures", {})
                blocked.pop(logical, None)
                if handled.get(logical) != marker:
                    attempt = int(self.state.get("retries", {}).get(logical, 0)) + 1
                    moved = quarantine_artifacts(
                        self.checkpoint_dir, self.run_tag, logical, attempt
                    )
                    handled[logical] = marker
                    self.log_action(
                        logical,
                        "ADOPT_KNOWN_DISPATCHER_FAILURE",
                        job_id,
                        f"quarantined={len(moved)} signature={KNOWN_MHTPW_DISPATCHER_FAILURE}",
                    )
                self.state["needs_resume"] = True
                self.state.setdefault("retry_causes", {})[logical] = (
                    "known_dispatcher_alias_failure"
                )
                continue
            self.cancel_dead_s1_children(logical, job_id, status.state)
            blocked[logical] = f"job {job_id} ended in {status.state}; automatic blind retry is disabled"

        self.save()
        if blocked:
            raise RuntimeError("watchdog blocked: " + json.dumps(blocked, ensure_ascii=False, sort_keys=True))
        if any_active:
            return False, True

        missing = self.missing_artifacts()
        if missing:
            if self.state.get("needs_resume") or self.args.adopt_missing:
                self.resume_chain(missing)
                return False, True
            raise RuntimeError(f"missing artifacts without an authorized retry cause: {missing[:12]}")

        if self.state.get("needs_resume"):
            # Training eventually produced valid files, but the original
            # downstream afterok chain was invalidated by a cancelled job.
            self.resume_chain([])
            return False, True

        done, post_active = self.post_chain_status()
        if done:
            self.log_action("chain", "ALL_RESULTS_COMPLETE", "", str(self.resolved_manifest))
            self.save()
            return True, False
        return False, post_active or True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=os.environ.get("RUN_TAG", ""), required=False)
    parser.add_argument("--baseline-dir", default=os.environ.get("BASELINE_DIR", "/public/home/putianshu/vis_mlp/ifs_baseline"))
    parser.add_argument("--eval-root", default=os.environ.get("EVAL_ROOT", ""))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CKPT_DIR", ""))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--once", action="store_true", default=os.environ.get("WATCH_ONCE", "0") == "1")
    parser.add_argument("--auto-retry", action="store_true", default=os.environ.get("WATCH_AUTO_RETRY", "0") == "1")
    parser.add_argument("--adopt-missing", action="store_true", default=os.environ.get("WATCH_ADOPT_MISSING", "0") == "1")
    parser.add_argument(
        "--adopt-current-manifest",
        action="store_true",
        default=os.environ.get("WATCH_ADOPT_CURRENT_MANIFEST", "0") == "1",
        help=(
            "archive stale watchdog state and adopt the exact JobID mapping in "
            "the current submission manifest; required when a new generation "
            "overwrites the manifest outside the running watchdog transaction"
        ),
    )
    parser.add_argument(
        "--adopt-dispatcher-failure",
        action="store_true",
        default=os.environ.get("WATCH_ADOPT_DISPATCHER_FAILURE", "0") == "1",
        help=(
            "adopt only FAILED MHTPW S2 jobs whose own Slurm log contains the "
            "exact historical unknown-EXPERIMENT signature"
        ),
    )
    parser.add_argument(
        "--force-end-current-generation",
        action="store_true",
        default=os.environ.get("WATCH_FORCE_END_CURRENT_GENERATION", "0") == "1",
        help=(
            "explicitly cancel every still-active training job in the currently "
            "tracked generation, quarantine its incomplete artifacts, and resume "
            "only missing artifact triplets"
        ),
    )
    parser.add_argument(
        "--adopt-local-cache-failure",
        action="store_true",
        default=os.environ.get("WATCH_ADOPT_LOCAL_CACHE_FAILURE", "0") == "1",
        help=(
            "adopt only FAILED training jobs whose own Slurm log contains the "
            "strict node-local-cache preflight error signature"
        ),
    )
    parser.add_argument(
        "--adopt-runtime-failure",
        action="store_true",
        default=os.environ.get("WATCH_ADOPT_RUNTIME_FAILURE", "0") == "1",
        help=(
            "adopt only FAILED/OOM training jobs whose own Slurm log contains "
            "a prespecified HIP-unavailable or cgroup-OOM infrastructure signature"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("WATCH_POLL_SECONDS", "180")))
    parser.add_argument("--startup-stale-minutes", type=int, default=int(os.environ.get("WATCH_STARTUP_STALE_MINUTES", "45")))
    parser.add_argument("--data-stale-minutes", type=int, default=int(os.environ.get("WATCH_DATA_STALE_MINUTES", "60")))
    parser.add_argument("--training-stale-minutes", type=int, default=int(os.environ.get("WATCH_TRAIN_STALE_MINUTES", "45")))
    parser.add_argument("--validation-stale-minutes", type=int, default=int(os.environ.get("WATCH_VALIDATION_STALE_MINUTES", "120")))
    parser.add_argument("--confirmations", type=int, default=int(os.environ.get("WATCH_CONFIRMATIONS", "2")))
    parser.add_argument("--recheck-seconds", type=int, default=int(os.environ.get("WATCH_RECHECK_SECONDS", "20")))
    parser.add_argument("--cancel-wait-seconds", type=int, default=int(os.environ.get("WATCH_CANCEL_WAIT_SECONDS", "180")))
    parser.add_argument("--checkpoint-grace-minutes", type=int, default=int(os.environ.get("WATCH_CHECKPOINT_GRACE_MINUTES", "30")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("WATCH_MAX_RETRIES", "2")))
    parser.add_argument(
        "--exclude-failed-nodes",
        action="store_true",
        default=os.environ.get("WATCH_EXCLUDE_FAILED_NODES", "1") == "1",
    )
    args = parser.parse_args()
    if not args.run_tag:
        parser.error("--run-tag or RUN_TAG is required")
    if not args.eval_root:
        args.eval_root = f"/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/{args.run_tag}"
    if not args.checkpoint_dir:
        args.checkpoint_dir = str(Path(args.baseline_dir) / "checkpoints")
    if args.auto_retry:
        if args.poll_seconds < 60 or args.confirmations < 2 or args.recheck_seconds < 10:
            parser.error("auto retry requires poll>=60 s, confirmations>=2, and recheck>=10 s")
        if min(
            args.startup_stale_minutes,
            args.data_stale_minutes,
            args.training_stale_minutes,
            args.validation_stale_minutes,
        ) < 30:
            parser.error("automatic stale limits must all be at least 30 minutes")
    return args


def require_commands(auto_retry: bool) -> None:
    commands = ["squeue", "sacct", "scontrol"]
    if auto_retry:
        commands.extend(["scancel", "sbatch", "bash"])
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise RuntimeError(f"required commands are unavailable: {missing}")


def main() -> int:
    args = parse_args()
    require_commands(args.auto_retry)
    watch = ChainWatch(args)
    try:
        if args.preflight:
            print(json.dumps(watch.preflight(), ensure_ascii=False, indent=2))
            return 0
        print(f"[watchdog] started {now_iso()} run_tag={args.run_tag}", flush=True)
        print(f"[watchdog] state={watch.state_path}", flush=True)
        while True:
            done, active = watch.cycle()
            if done:
                return 0
            if args.once:
                return 0
            if not active:
                return 2
            time.sleep(args.poll_seconds)
    finally:
        watch.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print("ERROR: another watchdog already holds this run's lock", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
