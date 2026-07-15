#!/usr/bin/env python3
"""Login-node preflight for the paired pressure-level quality diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preflight_q_core_t925_diagnostic_inputs import PreflightError, validate_dataset


SOURCES = ("pangu", "tianji", "era5_reference_analysis")
REQUIRED_FEATURES = (
    "T2M",
    "MSLP",
    "WSPD10",
    "T_925",
    "RH_925",
    "U_925",
    "V_925",
    "WSPD925",
    "DP_1000",
    "DP_925",
    "Q_1000",
    "Q_925",
)


def validate_quality_dataset(data_dir: Path, source: str) -> dict[str, object]:
    config = validate_dataset(data_dir, source)
    order = tuple(str(value) for value in config.get("dynamic_feature_order", []))
    missing = [feature for feature in REQUIRED_FEATURES if feature not in order]
    if missing:
        raise PreflightError(
            f"paired source-quality diagnosis requires {missing}; "
            "use a completed q_core_t925_no_rh2m diagnostic dataset"
        )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--data-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = validate_quality_dataset(Path(args.data_dir), args.source)
    except PreflightError as exc:
        print(f"PREFLIGHT_DATA_ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # infrastructure/runtime failure must not look rebuildable
        print(f"PREFLIGHT_RUNTIME_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "source": args.source,
                "data_dir": str(Path(args.data_dir).expanduser().resolve()),
                "feature_set": config.get("feature_set"),
                "required_features": list(REQUIRED_FEATURES),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
