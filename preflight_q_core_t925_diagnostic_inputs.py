#!/usr/bin/env python3
"""Fail-fast provenance preflight for q-core T925 diagnostic datasets.

This check intentionally uses only the Python standard library so it can run
on a login node before any Slurm job is submitted.  It mirrors the immutable
layout/provenance requirements enforced by
``analyze_q_core_t925_joint_structure.py`` and also checks that the train/test
artifacts consumed by that analysis are present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


FEATURES = ("T_925", "Q_925", "RH_925")
SUPPLEMENT_FEATURES = ("T2M", "WSPD10")
REQUIRED_DYNAMIC_FEATURES = FEATURES + SUPPLEMENT_FEATURES
ALLOWED_FEATURE_SETS = {"source_full", "q_core_t925_no_rh2m"}
EXPECTED_UNIT_POLICY = "pmst_canonical_units_v2_20260630"
EXPECTED_UNITS = {"T_925": "K", "Q_925": "kg kg-1", "RH_925": "%"}
SOURCES = ("pangu", "tianji", "era5_reference_analysis")
REQUIRED_ARTIFACTS = tuple(
    f"{stem}_{split}.{suffix}"
    for split in ("train", "test")
    for stem, suffix in (("X", "npy"), ("y", "npy"), ("meta", "csv"))
)


class PreflightError(ValueError):
    """A dataset cannot be used for the pre-specified diagnosis."""


def _normalise_feature_set(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def validate_dataset(data_dir: Path, source: str) -> dict[str, object]:
    """Validate config provenance and required artifacts; return the config."""
    if source not in SOURCES:
        raise PreflightError(f"unsupported source={source!r}; expected one of {SOURCES}")
    data_dir = data_dir.expanduser().resolve()
    cfg_path = data_dir / "dataset_build_config.json"
    if not cfg_path.is_file() or cfg_path.stat().st_size == 0:
        raise PreflightError(f"missing non-empty config: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read valid JSON from {cfg_path}: {exc}") from exc
    if not isinstance(cfg, Mapping):
        raise PreflightError(f"{cfg_path} must contain a JSON object")

    order = tuple(str(value) for value in cfg.get("dynamic_feature_order", []))
    missing_features = [name for name in REQUIRED_DYNAMIC_FEATURES if name not in order]
    if missing_features:
        raise PreflightError(f"q-core+T925 layout is missing {missing_features}; order={order}")
    if len(order) != len(set(order)):
        raise PreflightError("dynamic_feature_order contains duplicate names")

    feature_set = _normalise_feature_set(cfg.get("feature_set"))
    if feature_set not in ALLOWED_FEATURE_SETS:
        raise PreflightError(
            f"feature_set={feature_set!r}; expected one of {sorted(ALLOWED_FEATURE_SETS)}"
        )
    policy = str(cfg.get("canonical_unit_policy", ""))
    if policy != EXPECTED_UNIT_POLICY:
        raise PreflightError(
            f"canonical_unit_policy={policy!r}; expected {EXPECTED_UNIT_POLICY}; rebuild required"
        )

    units = cfg.get("canonical_dynamic_units")
    if not isinstance(units, Mapping):
        raise PreflightError("canonical_dynamic_units metadata is missing")
    mismatches = {
        feature: (units.get(feature), expected)
        for feature, expected in EXPECTED_UNITS.items()
        if units.get(feature) != expected
    }
    if mismatches:
        raise PreflightError(f"canonical unit mismatch: {mismatches}")
    if str(cfg.get("time_coordinate", "")).upper() != "UTC":
        raise PreflightError("time_coordinate must be explicitly recorded as UTC")

    native = {str(value) for value in cfg.get("native_source_features", [])}
    derived = {str(value) for value in cfg.get("derived_source_features", [])}
    if "T_925" not in native:
        raise PreflightError("T_925 must be documented as a native source field")
    if source in {"pangu", "tianji"} and "Q_925" not in native:
        raise PreflightError(f"{source} Q_925 must be documented as native")
    if source == "era5_reference_analysis" and "Q_925" not in native:
        if "Q_925" not in derived or not {"T_925", "RH_925"}.issubset(native):
            raise PreflightError(
                "ERA5 Q_925 needs native Q or a documented derivation from native T925/RH925"
            )

    if source == "pangu":
        provenance = cfg.get("source_feature_provenance")
        if provenance is not None:
            if not isinstance(provenance, Mapping):
                raise PreflightError("source_feature_provenance must be a mapping when present")
            rh_lineage = str(provenance.get("RH_925", ""))
            if "derived" not in rh_lineage.lower() or "RH_925" in native or "RH_925" not in derived:
                raise PreflightError(
                    "new Pangu configs must classify RH_925 as derived from T_925/Q_925, not native"
                )
        lead = cfg.get("source_forecast_lead")
        if not isinstance(lead, Mapping) or not bool(lead.get("available")):
            raise PreflightError("Pangu forecast-lead provenance is missing")
        try:
            lead_min = float(lead.get("min_hours", math.nan))
            lead_max = float(lead.get("max_hours", math.nan))
        except (TypeError, ValueError) as exc:
            raise PreflightError(f"invalid Pangu forecast-lead provenance: {lead}") from exc
        if not (
            math.isclose(lead_min, 12.0, abs_tol=1.0e-6)
            and math.isclose(lead_max, 23.0, abs_tol=1.0e-6)
        ):
            raise PreflightError(f"expected canonical Pangu 12--23 h lead, got {lead}")

    try:
        window = int(cfg["window"])
        dyn_vars = int(cfg["dyn_vars"])
        fe_dim = int(cfg["fe_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError("window, dyn_vars and fe_dim must be integer metadata") from exc
    if window != 12 or dyn_vars != len(order) or fe_dim < 0:
        raise PreflightError(
            f"invalid layout metadata: window={window}, dyn_vars={dyn_vars}, "
            f"order_len={len(order)}, fe_dim={fe_dim}"
        )

    missing_artifacts = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (data_dir / name).is_file() or (data_dir / name).stat().st_size == 0
    ]
    if missing_artifacts:
        raise PreflightError(f"missing or empty analysis artifacts: {missing_artifacts}")
    return dict(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    try:
        cfg = validate_dataset(data_dir, args.source)
    except PreflightError as exc:
        print(f"[FAIL] {args.source}: {data_dir}: {exc}", file=sys.stderr)
        return 2
    print(
        f"[OK] {args.source}: {data_dir} "
        f"feature_set={_normalise_feature_set(cfg.get('feature_set'))} "
        f"unit_policy={cfg.get('canonical_unit_policy')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
