#!/usr/bin/env python3
"""Standard-library preflight for RH2M observation-quality inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


EXPECTED_UNIT_POLICY = "pmst_canonical_units_v2_20260630"
ROLES = ("tianji_product", "t2nd_raw", "era5_reference_analysis")
REQUIRED_ARTIFACTS = (
    "X_test.npy",
    "y_test.npy",
    "meta_test.csv",
    "dataset_build_config.json",
)


class PreflightError(ValueError):
    """The dataset is not admissible for the RH2M comparison."""


def normalise(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def validate_rh2m_dataset(data_dir: Path, role: str) -> dict[str, object]:
    if role not in ROLES:
        raise PreflightError(f"unsupported role={role!r}")
    data_dir = data_dir.expanduser().resolve()
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).is_file()]
    if missing_artifacts:
        raise PreflightError(f"{data_dir}: missing {missing_artifacts}")
    try:
        with (data_dir / "dataset_build_config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read dataset config under {data_dir}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise PreflightError("dataset_build_config.json must contain an object")

    order = tuple(str(value) for value in config.get("dynamic_feature_order", []))
    if "RH2M" not in order:
        raise PreflightError(f"{role}: RH2M is absent from dynamic_feature_order")
    if len(order) != len(set(order)):
        raise PreflightError(f"{role}: dynamic_feature_order contains duplicates")
    try:
        dyn_vars = int(config.get("dyn_vars", 0))
        window = int(config.get("window", 0))
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{role}: invalid layout metadata") from exc
    if dyn_vars != len(order) or window != 12:
        raise PreflightError(
            f"{role}: expected window=12 and dyn_vars=order length; "
            f"got window={window}, dyn_vars={dyn_vars}, order_len={len(order)}"
        )
    if str(config.get("canonical_unit_policy", "")) != EXPECTED_UNIT_POLICY:
        raise PreflightError(
            f"{role}: canonical_unit_policy must be {EXPECTED_UNIT_POLICY}; rebuild required"
        )
    units = config.get("canonical_dynamic_units")
    if not isinstance(units, Mapping) or units.get("RH2M") != "%":
        raise PreflightError(f"{role}: canonical RH2M unit metadata must be '%' ")
    if str(config.get("time_coordinate", "")).upper() != "UTC":
        raise PreflightError(f"{role}: time_coordinate must be UTC")

    rh2m_source = normalise(config.get("rh2m_source"))
    override_file = str(config.get("rh2m_override_file", "")).strip()
    allow_missing = bool(config.get("rh2m_override_allow_missing", False))
    if role == "tianji_product":
        if rh2m_source not in {"tianji_native", "tianji_product"}:
            raise PreflightError(
                f"Tianji product RH2M must be native/product data, got rh2m_source={rh2m_source!r}"
            )
        if override_file:
            raise PreflightError("Tianji product dataset unexpectedly records an RH2M override")
    elif role == "t2nd_raw":
        if "t2nd" not in rh2m_source:
            raise PreflightError(
                f"T2ND dataset must record T2ND RH2M provenance, got {rh2m_source!r}"
            )
        if not override_file:
            raise PreflightError("T2ND dataset does not record its raw RH2M override file")
        if not Path(override_file).expanduser().is_file():
            raise PreflightError(
                f"T2ND raw RH2M provenance file is missing: {override_file}"
            )
        if allow_missing:
            raise PreflightError(
                "T2ND RH2M override allowed fallback to Tianji product values; "
                "this is not admissible for the strict raw-source comparison"
            )
    else:
        source_tag = normalise(config.get("source_tag") or config.get("dataset"))
        if "era5" not in source_tag:
            raise PreflightError(
                f"ERA5 role lacks explicit ERA5 source provenance: {source_tag!r}"
            )
    return dict(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = validate_rh2m_dataset(Path(args.data_dir), args.role)
    except PreflightError as exc:
        print(f"PREFLIGHT_DATA_ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"PREFLIGHT_RUNTIME_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "role": args.role,
                "data_dir": str(Path(args.data_dir).expanduser().resolve()),
                "feature_set": config.get("feature_set"),
                "rh2m_source": config.get("rh2m_source", ""),
                "rh2m_override_file": config.get("rh2m_override_file", ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
