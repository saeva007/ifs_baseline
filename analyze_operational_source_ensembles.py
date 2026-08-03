#!/usr/bin/env python3
"""Paired operational-ensemble analysis on the four-source test intersection.

The module keeps the trained source-full models unchanged and evaluates a
small, predeclared set of probability-fusion recipes:

* the current Tianji + Tianji-T2ND + IFS best effort;
* Tianji + Pangu, for the cleanest physics--AI pairing;
* all four operational members with equal member weights; and
* a source-family-balanced four-member recipe.

Every member and recipe is evaluated on the same exact station--time
intersection.  Compact source-data tables quantify skill relative to the best
individual member and decompose Pangu-related rescue and harm for observed
low-visibility, fog and non-low-visibility samples.  The evaluator imports
``write_analysis_bundle`` directly, so no large per-sample CSV is required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from analyze_best_effort_extended import metric_table


MEMBERS = (
    "tianji",
    "T2ND_rh2m_source_full",
    "ifs",
    "pangu2025_source_full",
)

CURRENT_ENSEMBLE = "tianji_t2nd_ifs_mean_softmax"
TIANJI_PANGU_ENSEMBLE = "tianji_pangu_mean_softmax"
ALL_FOUR_ENSEMBLE = "tianji_t2nd_ifs_pangu_mean_softmax"
FAMILY_BALANCED_ENSEMBLE = "tianji_family_ifs_pangu_balanced_softmax"

MEMBER_LABELS = {
    "tianji": "Tianji",
    "T2ND_rh2m_source_full": "Tianji T2ND",
    "ifs": "IFS-trained",
    "pangu2025_source_full": "Pangu",
}

RECIPE_LABELS = {
    CURRENT_ENSEMBLE: "Current best effort",
    TIANJI_PANGU_ENSEMBLE: "Tianji + Pangu",
    ALL_FOUR_ENSEMBLE: "All four (equal member)",
    FAMILY_BALANCED_ENSEMBLE: "All four (family-balanced)",
}

# Ordered dictionaries are guaranteed by modern Python and preserve the visual
# and source-data order used throughout the paper bundle.
RECIPES: Dict[str, Dict[str, float]] = {
    CURRENT_ENSEMBLE: {
        "tianji": 1.0 / 3.0,
        "T2ND_rh2m_source_full": 1.0 / 3.0,
        "ifs": 1.0 / 3.0,
        "pangu2025_source_full": 0.0,
    },
    TIANJI_PANGU_ENSEMBLE: {
        "tianji": 0.5,
        "T2ND_rh2m_source_full": 0.0,
        "ifs": 0.0,
        "pangu2025_source_full": 0.5,
    },
    ALL_FOUR_ENSEMBLE: {
        "tianji": 0.25,
        "T2ND_rh2m_source_full": 0.25,
        "ifs": 0.25,
        "pangu2025_source_full": 0.25,
    },
    FAMILY_BALANCED_ENSEMBLE: {
        "tianji": 1.0 / 6.0,
        "T2ND_rh2m_source_full": 1.0 / 6.0,
        "ifs": 1.0 / 3.0,
        "pangu2025_source_full": 1.0 / 3.0,
    },
}

GAIN_METRICS = (
    ("low_vis_ap", "Average precision"),
    ("low_vis_precision", "Precision"),
    ("low_vis_recall", "Recall"),
    ("low_vis_csi", "CSI"),
)

COMPARISON_METRICS = (
    *GAIN_METRICS,
    ("low_vis_fpr", "False-positive rate"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_probabilities(values: np.ndarray, source: str) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"{source}: expected N x 3 probabilities, got {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{source}: probabilities contain non-finite values")
    sums = probs.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError(f"{source}: probability rows must have positive sums")
    return probs / sums


def validate_member_payloads(
    payloads: Mapping[str, Mapping[str, object]],
    members: Sequence[str] = MEMBERS,
) -> None:
    missing = [source for source in members if source not in payloads]
    if missing:
        raise KeyError(f"Missing operational ensemble members: {missing}")
    reference = str(members[0])
    targets_ref = np.asarray(payloads[reference]["targets"], dtype=np.int64)
    raw_ref = np.asarray(payloads[reference]["raw_visibility_m"], dtype=np.float64)
    if len(targets_ref) == 0:
        raise ValueError("Operational ensemble intersection is empty")
    for source in members:
        payload = payloads[source]
        probs = normalized_probabilities(np.asarray(payload["probs"]), source)
        preds = np.asarray(payload["preds"], dtype=np.int64)
        targets = np.asarray(payload["targets"], dtype=np.int64)
        raw = np.asarray(payload["raw_visibility_m"], dtype=np.float64)
        if probs.shape[0] != len(targets_ref) or len(preds) != len(targets_ref):
            raise ValueError(f"{source}: payload length differs on the common intersection")
        if not np.array_equal(targets, targets_ref):
            raise ValueError(f"{source}: observed labels differ on the common intersection")
        finite = np.isfinite(raw) & np.isfinite(raw_ref)
        if finite.any() and not np.allclose(raw[finite], raw_ref[finite], atol=1.0e-3):
            raise ValueError(f"{source}: observed visibility differs after alignment")


def weighted_mean_probabilities(
    payloads: Mapping[str, Mapping[str, object]],
    weights: Mapping[str, float],
) -> np.ndarray:
    unknown = sorted(set(weights) - set(payloads))
    if unknown:
        raise KeyError(f"Recipe refers to missing source(s): {unknown}")
    total_weight = float(sum(float(value) for value in weights.values()))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Recipe weights must sum to a positive finite value")
    combined = None
    for source, weight in weights.items():
        value = float(weight)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{source}: invalid ensemble weight {weight}")
        if value == 0.0:
            continue
        probs = normalized_probabilities(np.asarray(payloads[source]["probs"]), source)
        combined = value * probs if combined is None else combined + value * probs
    if combined is None:
        raise ValueError("Recipe contains no positive weights")
    combined = combined / total_weight
    combined = combined / combined.sum(axis=1, keepdims=True)
    return combined.astype(np.float32)


def build_recipe_payloads(
    member_payloads: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    validate_member_payloads(member_payloads)
    reference = member_payloads[MEMBERS[0]]
    targets = np.asarray(reference["targets"], dtype=np.int64)
    raw = np.asarray(reference["raw_visibility_m"], dtype=np.float64)
    out: Dict[str, Dict[str, object]] = {}
    for source in MEMBERS:
        payload = member_payloads[source]
        out[source] = {
            "label": MEMBER_LABELS[source],
            "probs": normalized_probabilities(np.asarray(payload["probs"]), source),
            "preds": np.asarray(payload["preds"], dtype=np.int64),
            "targets": targets,
            "raw_visibility_m": raw,
        }
    for recipe, weights in RECIPES.items():
        probs = weighted_mean_probabilities(out, weights)
        out[recipe] = {
            "label": RECIPE_LABELS[recipe],
            "probs": probs,
            "preds": np.argmax(probs, axis=1).astype(np.int64),
            "targets": targets,
            "raw_visibility_m": raw,
        }
    return out


def recipe_weight_table() -> pd.DataFrame:
    rows = []
    for recipe, weights in RECIPES.items():
        for source in MEMBERS:
            rows.append(
                {
                    "recipe": recipe,
                    "recipe_label": RECIPE_LABELS[recipe],
                    "source": source,
                    "source_label": MEMBER_LABELS[source],
                    "weight": float(weights.get(source, 0.0)),
                    "decision_rule": "weighted mean post-softmax probabilities, then one argmax",
                }
            )
    return pd.DataFrame(rows)


def performance_table(payloads: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    source_order = [*MEMBERS, *RECIPES]
    table = metric_table(payloads, source_order)
    table["model_type"] = table["source"].map(
        lambda source: "ensemble_recipe" if source in RECIPES else "individual_member"
    )
    table["comparison_scope"] = "exact four-source station-time intersection"
    table["decision_rule"] = table["source"].map(
        lambda source: (
            "weighted mean post-softmax probabilities, then one argmax"
            if source in RECIPES
            else "frozen member prediction on the four-source intersection"
        )
    )
    return table


def relative_gain_table(performance: pd.DataFrame) -> pd.DataFrame:
    member_rows = performance[performance["source"].isin(MEMBERS)].set_index("source")
    recipe_rows = performance[performance["source"].isin(RECIPES)].set_index("source")
    rows = []
    for metric, label in GAIN_METRICS:
        member_values = pd.to_numeric(member_rows[metric], errors="coerce")
        if member_values.notna().sum() == 0:
            continue
        best_source = str(member_values.idxmax())
        best_value = float(member_values.loc[best_source])
        for recipe in RECIPES:
            value = float(recipe_rows.loc[recipe, metric])
            delta = value - best_value
            rows.append(
                {
                    "recipe": recipe,
                    "recipe_label": RECIPE_LABELS[recipe],
                    "metric": metric,
                    "metric_label": label,
                    "value": value,
                    "best_individual_source": best_source,
                    "best_individual_label": MEMBER_LABELS[best_source],
                    "best_individual_value": best_value,
                    "delta_absolute": delta,
                    "relative_change_percent": (
                        100.0 * delta / best_value if best_value != 0.0 else np.nan
                    ),
                    "positive_favours_ensemble": True,
                }
            )
    return pd.DataFrame(rows)


def anchor_delta_table(performance: pd.DataFrame) -> pd.DataFrame:
    indexed = performance.set_index("source")
    rows = []
    for recipe in RECIPES:
        if recipe == CURRENT_ENSEMBLE:
            continue
        for metric, label in COMPARISON_METRICS:
            candidate = float(indexed.loc[recipe, metric])
            anchor = float(indexed.loc[CURRENT_ENSEMBLE, metric])
            higher_is_better = metric != "low_vis_fpr"
            oriented = candidate - anchor if higher_is_better else anchor - candidate
            rows.append(
                {
                    "anchor": CURRENT_ENSEMBLE,
                    "anchor_label": RECIPE_LABELS[CURRENT_ENSEMBLE],
                    "candidate": recipe,
                    "candidate_label": RECIPE_LABELS[recipe],
                    "metric": metric,
                    "metric_label": label,
                    "anchor_value": anchor,
                    "candidate_value": candidate,
                    "raw_delta_candidate_minus_anchor": candidate - anchor,
                    "skill_oriented_delta": oriented,
                    "positive_favours_candidate": True,
                }
            )
    return pd.DataFrame(rows)


def transition_table(payloads: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    targets = np.asarray(payloads[CURRENT_ENSEMBLE]["targets"], dtype=np.int64)
    anchor_preds = np.asarray(payloads[CURRENT_ENSEMBLE]["preds"], dtype=np.int64)
    scopes = (
        ("observed_low_vis_lt1km", "Observed Low-vis (<1 km)", targets <= 1, lambda p: p <= 1),
        ("observed_fog_lt500m", "Observed fog (<500 m)", targets == 0, lambda p: p == 0),
        ("observed_non_low_vis_ge1km", "Observed non-Low-vis (>=1 km)", targets == 2, lambda p: p == 2),
    )
    rows = []
    for scope, scope_label, mask, success_rule in scopes:
        total = int(mask.sum())
        if total <= 0:
            continue
        anchor_success = np.asarray(success_rule(anchor_preds), dtype=bool) & mask
        for recipe in RECIPES:
            if recipe == CURRENT_ENSEMBLE:
                continue
            candidate_preds = np.asarray(payloads[recipe]["preds"], dtype=np.int64)
            candidate_success = np.asarray(success_rule(candidate_preds), dtype=bool) & mask
            categories = (
                ("shared_success", anchor_success & candidate_success),
                ("candidate_rescue", ~anchor_success & candidate_success & mask),
                ("candidate_harm", anchor_success & ~candidate_success & mask),
                ("shared_failure", ~anchor_success & ~candidate_success & mask),
            )
            for category, selected in categories:
                n = int(selected.sum())
                rows.append(
                    {
                        "scope": scope,
                        "scope_label": scope_label,
                        "anchor": CURRENT_ENSEMBLE,
                        "anchor_label": RECIPE_LABELS[CURRENT_ENSEMBLE],
                        "candidate": recipe,
                        "candidate_label": RECIPE_LABELS[recipe],
                        "category": category,
                        "n": n,
                        "share": n / total,
                        "scope_total": total,
                        "anchor_success_n": int(anchor_success.sum()),
                        "candidate_success_n": int(candidate_success.sum()),
                        "net_success_change_n": int(candidate_success.sum() - anchor_success.sum()),
                    }
                )
    return pd.DataFrame(rows)


def member_pattern_table(payloads: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    targets = np.asarray(payloads[MEMBERS[0]]["targets"], dtype=np.int64)
    scopes = (
        ("observed_low_vis_lt1km", "Observed Low-vis (<1 km)", targets <= 1, lambda p: p <= 1),
        ("observed_fog_lt500m", "Observed fog (<500 m)", targets == 0, lambda p: p == 0),
        ("observed_non_low_vis_ge1km", "Observed non-Low-vis (>=1 km)", targets == 2, lambda p: p == 2),
    )
    rows = []
    for scope, scope_label, mask, success_rule in scopes:
        total = int(mask.sum())
        if total <= 0:
            continue
        success = np.column_stack(
            [success_rule(np.asarray(payloads[source]["preds"], dtype=np.int64)) for source in MEMBERS]
        )[mask]
        for pattern in range(2 ** len(MEMBERS)):
            bits = np.asarray([(pattern >> index) & 1 for index in range(len(MEMBERS))], dtype=bool)
            selected = np.all(success == bits[None, :], axis=1)
            n = int(selected.sum())
            if n == 0:
                continue
            successful_members = [
                MEMBER_LABELS[source]
                for source, enabled in zip(MEMBERS, bits)
                if enabled
            ]
            rows.append(
                {
                    "scope": scope,
                    "scope_label": scope_label,
                    "pattern_mask": format(pattern, f"0{len(MEMBERS)}b"),
                    "member_pattern": " + ".join(successful_members) if successful_members else "No member",
                    "member_success_count": int(bits.sum()),
                    "pangu_success": bool(bits[MEMBERS.index("pangu2025_source_full")]),
                    "n": n,
                    "share": n / total,
                    "scope_total": total,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "n", "member_success_count"],
        ascending=[True, False, False],
        kind="stable",
    )


def write_analysis_bundle(
    member_payloads: Mapping[str, Mapping[str, object]],
    out_dir: Path | str,
) -> Dict[str, object]:
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payloads = build_recipe_payloads(member_payloads)
    performance = performance_table(payloads)
    tables = {
        "recipe_weights": recipe_weight_table(),
        "performance": performance,
        "relative_gain": relative_gain_table(performance),
        "anchor_delta": anchor_delta_table(performance),
        "transitions": transition_table(payloads),
        "member_patterns": member_pattern_table(payloads),
    }
    files: MutableMapping[str, str] = {}
    for name, frame in tables.items():
        path = output / f"operational_ensemble_{name}.csv"
        frame.to_csv(path, index=False)
        files[name] = path.name

    n = int(len(np.asarray(payloads[MEMBERS[0]]["targets"])))
    manifest = {
        "status": "completed",
        "core_claim": (
            "Pangu is tested for independent operational value beyond the current "
            "Tianji/T2ND/IFS best effort on one exact four-source intersection."
        ),
        "members": list(MEMBERS),
        "recipes": RECIPES,
        "recipe_labels": RECIPE_LABELS,
        "matched_rows": n,
        "paired_scope": "exact four-source station-time intersection",
        "decision": "weighted mean post-softmax probabilities, then one argmax",
        "selection_boundary": (
            "All recipes are predeclared sensitivity analyses. A replacement operational "
            "recipe must be selected on validation and frozen before formal test reporting."
        ),
        "outputs": {
            name: {"file": filename, "sha256": sha256_file(output / filename)}
            for name, filename in files.items()
        },
    }
    manifest_path = output / "operational_ensemble_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": manifest_path.name}


__all__ = [
    "MEMBERS",
    "RECIPES",
    "RECIPE_LABELS",
    "CURRENT_ENSEMBLE",
    "TIANJI_PANGU_ENSEMBLE",
    "ALL_FOUR_ENSEMBLE",
    "FAMILY_BALANCED_ENSEMBLE",
    "build_recipe_payloads",
    "write_analysis_bundle",
]
