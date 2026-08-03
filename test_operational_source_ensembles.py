from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_operational_source_ensembles as analysis


def _probabilities(preds: np.ndarray, confidence: float = 0.78) -> np.ndarray:
    probs = np.full((len(preds), 3), (1.0 - confidence) / 2.0, dtype=np.float64)
    probs[np.arange(len(preds)), preds] = confidence
    return probs


def synthetic_payloads() -> dict[str, dict[str, object]]:
    targets = np.asarray([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    raw = np.asarray([120, 360, 650, 880, 1300, 2400, 6000, 12000], dtype=np.float64)
    predictions = {
        "tianji": np.asarray([0, 2, 1, 2, 2, 1, 2, 2], dtype=np.int64),
        "T2ND_rh2m_source_full": np.asarray([0, 0, 2, 2, 2, 2, 1, 2], dtype=np.int64),
        "ifs": np.asarray([2, 0, 1, 2, 2, 2, 2, 1], dtype=np.int64),
        "pangu2025_source_full": np.asarray([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64),
    }
    return {
        source: {
            "label": source,
            "probs": _probabilities(preds),
            "preds": preds,
            "targets": targets,
            "raw_visibility_m": raw,
        }
        for source, preds in predictions.items()
    }


def test_recipe_probabilities_and_family_weights() -> None:
    members = synthetic_payloads()
    payloads = analysis.build_recipe_payloads(members)
    all_four_expected = np.mean(
        np.stack([np.asarray(members[source]["probs"]) for source in analysis.MEMBERS], axis=0),
        axis=0,
    )
    np.testing.assert_allclose(
        payloads[analysis.ALL_FOUR_ENSEMBLE]["probs"],
        all_four_expected,
        atol=1.0e-7,
    )
    weights = analysis.RECIPES[analysis.FAMILY_BALANCED_ENSEMBLE]
    assert np.isclose(weights["tianji"] + weights["T2ND_rh2m_source_full"], 1.0 / 3.0)
    assert np.isclose(weights["ifs"], 1.0 / 3.0)
    assert np.isclose(weights["pangu2025_source_full"], 1.0 / 3.0)


def test_transition_partitions_each_observed_regime() -> None:
    payloads = analysis.build_recipe_payloads(synthetic_payloads())
    table = analysis.transition_table(payloads)
    grouped = table.groupby(["scope", "candidate"], sort=False)
    for (_, _), part in grouped:
        assert int(part["n"].sum()) == int(part["scope_total"].iloc[0])
        assert np.isclose(float(part["share"].sum()), 1.0)
        assert set(part["category"]) == {
            "shared_success",
            "candidate_rescue",
            "candidate_harm",
            "shared_failure",
        }


def test_analysis_and_figure_bundle(tmp_path: Path) -> None:
    report = analysis.write_analysis_bundle(synthetic_payloads(), tmp_path)
    assert report["status"] == "completed"
    metrics = pd.read_csv(tmp_path / "operational_ensemble_performance.csv")
    assert set(analysis.RECIPES).issubset(set(metrics["source"]))
    manifest = json.loads(
        (tmp_path / "operational_ensemble_analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["matched_rows"] == 8

    from plot_operational_source_ensembles import plot_bundle

    figure_report = plot_bundle(
        tmp_path,
        tmp_path,
        formats=("svg", "png"),
        dpi=120,
    )
    assert figure_report["status"] == "completed"
    assert (tmp_path / "fig_operational_source_ensembles.svg").is_file()
    assert (tmp_path / "fig_operational_source_ensembles_b_relative_gain.png").is_file()
