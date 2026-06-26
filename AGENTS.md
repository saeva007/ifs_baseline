# Agent Rules for Source-Full / Best-Effort Evaluations

## Best-Effort Figure Contract

- Treat source-full best-effort as an upper-bound comparison in which each forecast source uses its own real available variables. Do not force missing variables into zero-filled common slots or present this as a controlled common-variable attribution experiment.
- Keep the source-full figure order explicit in every plotting path. When adding or renaming a row, update both `test_PMST_overlap_forecast_source_s2.py` and `merge_overlap_source_eval_metrics.py`; otherwise direct evaluation and merge-only redraws will disagree.
- The current source-full display order is: Tianji, Tianji T2ND, IFS-trained, Tianji+T2ND+IFS mean, ERA5, Pangu, IFS empirical VIS. Additions should be deliberately placed in this list, not appended by fallback order.
- Figure whitelists must be checked against the actual `overall_metrics.csv` rows. A successful evaluation is not enough if the row is omitted from `source_order`, `source_labels`, `source_colors`, `dedupe_sources`, or merge-only plotting logic.

## Probability-Ensemble Pitfalls

- Mean-softmax ensembles require saved or in-memory probabilities; they cannot be reconstructed from `overall_metrics.csv`. Re-run the evaluator when a new ensemble row is needed.
- Before averaging probabilities from different forecast-source models, align rows by `(time, station_id)` and validate that class labels and raw visibility agree on the common intersection. Never average arrays only because they have the same length.
- If the requested decision rule is "average probabilities then argmax", record the row with `threshold_source=argmax_mean_softmax` or an equally explicit value. Do not mix validation-threshold selection into an argmax-only best-effort figure.
- When adding an ensemble row, write enough provenance into `overall_metrics.csv` / `run_config.json`: member sources, matched row count, checkpoint paths, scaler paths, and data directories. This prevents later confusion about whether the row was a model, a diagnostic baseline, or a post-hoc ensemble.

## Submission Hygiene

- Use the Slurm-backed launcher for long source-full evaluations. Avoid interactive login-node runs for best-effort figures.
- Use a new `OUT_DIR` for redraws that change figure content or row definitions. Do not overwrite an older best-effort result directory unless explicitly instructed.
