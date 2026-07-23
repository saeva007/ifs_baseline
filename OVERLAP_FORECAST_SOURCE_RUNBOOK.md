# Overlap Forecast-Source Experiment Runbook

This runbook covers the controlled Tianji-input versus IFS-input Static-RNN overlap
experiment after the validation-split, PM10/PM2.5 layout, and UTC fixes.

## Scope

- S1 data-build submit script: `sub_s1_overlap_data.slurm`
- Tianji S2 data-build submit script: `sub_tianji_overlap_data.slurm`
- IFS S2 data-build submit script: `sub_ifs_data.slurm`
- Training submit script: `sub_ifs_overlap_baseline.slurm`
- S1 data builder: `build_s1_pm10_overlap_from_full.py`
- Tianji S2 data builder: `build_dataset_tianji_overlap_12h.py`
- IFS S2 data builder: `build_dataset_ifs_overlap_12h_fast.py`
- Static-RNN S1/S2 trainer wrapper: `train_static_rnn_overlap_baseline_s2.py`
- Static-RNN Slurm path: `sub_ifs_overlap_baseline.slurm` launches the main
  trainer at `/public/home/putianshu/vis_mlp/train/train_static_rnn_lowvis.py`
  directly for `MODEL_ARCH=static_rnn`.
- Pangu grid-to-station interpolation: `interpolate_pangu_to_stations.py`
- Generic station-source S2 builder for Pangu/ERA5: `build_dataset_station_source_overlap_12h.py`
- Single-source Static-RNN evaluator: `sub_static_rnn_overlap_single_eval.slurm`
- Multi-source key-variable quality analysis: `paper_eval/analyze_multi_source_rh2m_quality.py`
- Legacy PMST S1 trainer: `train_PMST_s1_overlap_baseline.py`
- Legacy PMST S2 Tianji trainer: `train_PMST_overlap_baseline_s2.py`
- Legacy PMST S2 IFS trainer: `train_PMST_overlap_baseline_s2_fast.py`
- Paired evaluator: `test_PMST_overlap_forecast_source_s2.py`

Each data-build path has its own Slurm entry point. Keep
`sub_ifs_data.slurm` for IFS-input data only.

## Feature-Set Policy

The source-family experiment has two tiers:

- `FEATURE_SET=common_core`: the fair comparison tier. It keeps only source
  variables shared with Pangu:
  `RH2M,T2M,MSLP,U10,WSPD10,V10,WDIR10,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,DPD`.
  It no longer writes unused PMST slots as zero-valued channels. Keep this tier
  for fairness diagnostics, not for the current best-effort all-variable main
  figure.
- `FEATURE_SET=source_full`: the best-effort all-variable tier. Each source
  fills all PMST slots it can physically provide; this tests operational
  potential under each source's native availability, not a clean
  data-source-only attribution.

Keep Tianji product RH2M and T2ND raw RH2M as separate sources. T2ND is included
in the main `common_core` comparison because the current Tianji product RH2M can
be less extreme than the raw-mode field.

For the best-effort source-full experiment, the main decision rule is `argmax`.
Do not use S1 zero-transfer, checkpoint thresholds, validation threshold search,
or unavailable-variable placeholder channels as the main source-full evidence. Always audit
`dataset_build_config.json` and interpret `available_pmst_features` as the true
variable list for each source.

Source-full variable availability:

| Source | Available PMST meteorological variables in source-full |
|---|---|
| Tianji product | `RH2M,T2M,PRECIP,MSLP,SW_RAD,U10,WSPD10,V10,WDIR10,CAPE,LCC,T_925,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,W_925,W_1000,DPD,INVERSION` |
| T2ND RH2M + Tianji | Same as Tianji product, with `RH2M` replaced by the T2ND station-interpolated field |
| IFS | `RH2M,T2M,PRECIP,MSLP,SW_RAD,U10,WSPD10,V10,WDIR10,LCC,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,W_925,W_1000,DPD`; no current `CAPE,T_925,INVERSION` |
| ERA5-2025 | `RH2M,T2M,PRECIP,MSLP,SW_RAD,U10,WSPD10,V10,WDIR10,CAPE,LCC,T_925,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,W_925,W_1000,DPD,INVERSION` |
| Pangu-2021 | `RH2M,T2M,MSLP,U10,WSPD10,V10,WDIR10,T_925,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,DPD,INVERSION`; no current `PRECIP,SW_RAD,CAPE,LCC,W_925,W_1000` |
| Pangu-2025 | `T2M,MSLP,U10,WSPD10,V10,WDIR10,T_925,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,INVERSION`; no `RH2M,D2M,DPD,PRECIP,SW_RAD,CAPE,LCC,W_925,W_1000` |

The legacy Pangu-2021 exporter derives `RH2M` from 1000 hPa humidity. It must be
labelled as a proxy if retained. The Pangu-2025 source-full path does not derive
`RH2M` from `Q_1000` and is the required path for the same-year mechanism study.
Pangu-2021 uses `CMA_visibility_2021_2023_GeoCoords_1.nc` by default, while the
Tianji, IFS, T2ND, and ERA5-2025 paths use 2025 labels. Treat the all-source
source-full figure as best-effort performance, not a strict same-year
source-only attribution. For a strict same-year figure, first obtain source data
for the same target year.

PM10/PM2.5 inputs must also match the source year and stations. Before training
Pangu-2021 or any non-2025 source, confirm `PM10_FILE/PM10_DIR` and
`PM25_FILE/PM25_DIR` have valid matches within the 90 min tolerance. Do not use
runs whose logs show missing PM files or all-unmatched PM channels as
"all-variable" evidence.

## Best-Effort Source-Full Order

Run from the remote overlap repository:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs
```

1. Build source-full S2 datasets. Each source keeps only its native available
PMST variables plus zenith, PM10, and PM2.5; missing variable slots are not
zero-filled.

```bash
sbatch --export=ALL,FEATURE_SET=source_full sub_tianji_overlap_data.slurm
sbatch --export=ALL,FEATURE_SET=source_full sub_ifs_data.slurm
sbatch sub_pangu_station_idw.slurm
sbatch --export=ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2021,YEAR=2021,FEATURE_SET=source_full sub_station_source_overlap_data.slurm
sbatch --export=ALL,SOURCE_KIND=era5_feature_dir,SOURCE_TAG=era5_2025,YEAR=2025,FEATURE_SET=source_full sub_station_source_overlap_data.slurm
sbatch --export=ALL,FEATURE_SET=source_full,RH2M_OVERRIDE_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/tianji_rh2m_station/T2ND_rh2m_station_2025.nc,RH2M_SOURCE_TAG=T2ND_rh2m sub_tianji_overlap_data.slurm
```

2. Build source-full S1 layouts. Tianji, T2ND RH2M, and ERA5 share
the dyn27 S1 layout; IFS uses dyn24; historical Pangu-2021 uses dyn21.
If you use the current Pangu-2025 ONNX/station product instead, train its
separate dyn19 profile and do not mix it with Pangu-2021.

```bash
sbatch --export=ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=tianji sub_s1_overlap_data.slurm
sbatch --export=ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=ifs sub_s1_overlap_data.slurm
sbatch --export=ALL,FEATURE_SET=source_full,SOURCE_FULL_PROFILE=pangu2025 sub_s1_overlap_data.slurm
```

Do not use `common_core`, `compact_common_core`, or historical `overlap_full`
S1 checkpoints as source-full initializers. Source-full channel counts and FE
dimensions can differ by source, so source-full S2 runs require the matching
source-full S1 checkpoint. Current Pangu-2025 uses the separate
`SOURCE_FULL_PROFILE=pangu2025` / `EXPERIMENT=s1_source_full_pangu2025` dyn19 S1.

3. After the S1 and S2 data-build jobs have completed, train source-full S1
checkpoints and queue matching S2 models with dependencies:

```bash
OVERLAP_CHAIN=source_full bash submit_ifs_overlap_training_chain.sh
```

This submits Tianji/dyn27, IFS/dyn24, and Pangu-2025/dyn19 S1 training jobs,
then queues each S2 job with `afterok` on its matching S1. Tianji, T2ND RH2M,
and ERA5 use the Tianji/dyn27 S1 checkpoint; IFS and Pangu-2025 use their own
layouts.

To run the best-effort set without Pangu first:

```bash
S2_EXPERIMENTS="s2_tianji_source_full s2_tianji_T2ND_rh2m_source_full s2_ifs_source_full s2_era5_2025_source_full" \
OVERLAP_CHAIN=source_full \
bash submit_ifs_overlap_training_chain.sh
```

The submitter will only create the Tianji/dyn27 and IFS/dyn24 S1 jobs needed by
those S2 runs.

4. Evaluate Figure 1 with `--threshold_mode argmax` using
`test_PMST_overlap_forecast_source_s2.py --independent_sources`, explicit
source-full data/checkpoint paths, `AUTO` scaler entries when using
`--extra_sources`, `--skip_ifs_forecast_baseline`, and an output directory such as
`paper_eval_results_pm10_pm25_journal/best_effort_source_full_argmax/figure1_all_sources`.

5. Evaluate Figure 2 in two pieces: run
`sub_static_rnn_overlap_softmax_ensemble.slurm` with source-full Tianji/IFS
paths and `SOURCE_THRESHOLD_MODE=argmax,ENSEMBLE_THRESHOLD_MODE=argmax`, then
run `test_PMST_overlap_forecast_source_s2.py --independent_sources` for only
`pangu2021_source_full` without `--skip_ifs_forecast_baseline`. Merge the two
output directories with `merge_overlap_source_eval_metrics.py`; it will write
`fig_forecast_source_key_metrics_pangu_ifs_ensemble.*`.

## Pangu-2025 Strict Common-Variable Rerun

The reproducible same-variable experiment is `FEATURE_SET=q_core_no_rh2m`, not
`source_full`, legacy `common_core`, or `compact_common_core`. It uses the exact
17-channel dynamic order
`T2M,MSLP,U10,WSPD10,V10,WDIR10,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,ZENITH,PM10_ugm3,PM25_ugm3`
for Tianji, IFS, Pangu-2025, and ERA5-2025. It never derives Pangu RH2M from
1000-hPa humidity. T2ND is intentionally omitted because its only difference
from Tianji is RH2M, which is absent from this input space.

This controls the input-variable layout, model architecture, labels, and paired
evaluation samples. The active canonical Pangu station product uses the same
documented 12--23 h stitched lead range as the Tianji/IFS comparison products.
ERA5 remains a reference analysis, not an operational forecast.

Use the end-to-end submitter rather than issuing the data and training jobs by
hand. The submitter creates unique run IDs and enforces this dependency graph:
all five data builds -> cross-source data audit -> shared S1 -> four S2 models ->
paired argmax evaluation. Any failed dependency prevents downstream jobs from
running. By default, every dataset is written below
`ifs_baseline/q_core_fair_datasets/<run_tag>/`, so a new Pangu-2025 rerun cannot
silently mix with or overwrite an older common-variable dataset.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

# Inspect every sbatch command without submitting it.
RUN_TAG=qcore_pangu2025_rerun DRY_RUN=1 bash submit_q_core_fair_experiment.sh

# Submit with automatic discovery of the canonical 12--23 h station product.
RUN_TAG=qcore_pangu2025_rerun bash submit_q_core_fair_experiment.sh

# If it is stored elsewhere, pass the verified real path explicitly:
# PANGU2025_STATION_FILE=/actual/path/pangu_station_2025_lead12_23h_canonical.nc
```

`BUILD_PANGU_STATION=1` is intentionally disabled in this launcher because the
generic interpolation job creates a different lead product. Supply the already
verified canonical station file. The downstream builder checks its explicit
12--23 h schedule and rejects filename-only provenance.

`audit_q_core_fair_datasets.py` checks the JSON-declared layout against the
actual arrays, rejects zero-filled, all-zero, or excessively non-finite
channels, checks broad physical ranges and 2025 valid times, verifies identical
month-tail split settings, and confirms that visibility labels agree on the
common `(valid_time, station_id)` intersection. Its outputs are written under
`paper_eval_results_pm10_pm25_journal/q_core_fair_pangu2025/<run_tag>/data_audit`.
The four S2 builders declare and enforce a `30000 m` visibility-label ceiling;
the audit reads this value from each `dataset_build_config.json` rather than
assuming that clear labels above 10 km are invalid.
The experiment year is the forecast-initialization year, not a strict
valid-time year. Late 31 December 2025 initializations can legitimately verify
on 1 January 2026. The audit therefore accepts valid times
from 1 January 2025 through the one-day next-year boundary spill, while still
rejecting dates beyond that physically permitted interval. It performs all
inexpensive structural/time/label/pairing checks before scanning the large
feature arrays and writes every issue found in the failing stage to
`data_audit/q_core_data_audit_failed.json`.

Datasets built before unit policy `pmst_canonical_units_v2_20260630` must not be
resumed. They contain legacy PM values scaled by `1e12` and may mix Tianji hPa
with Pa from the other sources. Use a fresh run tag so all five datasets are
rebuilt. `RESUME_FROM_AUDIT=1` is only valid when every config already declares
that exact unit policy; it is not valid for the failed legacy run shown above.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
RUN_TAG=qcore_pangu2025_units_v2_20260630 \
PANGU2025_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h_canonical.nc \
bash submit_q_core_fair_experiment.sh
```

The same unit defect affects every historical `source_full` dataset and its
checkpoint, not only Pangu. Rebuild and retrain all source-full profiles and
sources with the dedicated end-to-end launcher:

```bash
RUN_TAG=source_full_units_v2_20260701 \
PANGU2025_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h_canonical.nc \
bash submit_source_full_units_v2_experiment.sh
```

Its dependency graph is station verification -> three source-full S1 data
profiles plus five S2 datasets -> canonical-unit/physical-range/layout/paired
sample audit -> all three S1 models -> all five S2 models -> all-source argmax
evaluation. Do not reuse historical source-full data, scalers, S1 checkpoints,
or S2 checkpoints for this repaired result.

The final evaluation always uses each source-specific S2 checkpoint with
`argmax`. It first saves the ordinary per-source predictions, then recomputes
all reported comparison metrics on the four-source paired test intersection.
Uncertainty for source-minus-Pangu differences is obtained by paired bootstrap
resampling of UTC valid dates, preserving within-day spatial and temporal
dependence. Use `q_core_paired_common_metrics.csv` and
`q_core_paired_deltas_vs_pangu2025.csv` as the common-input result tables;
`overall_metrics.csv` retains unpaired full-source diagnostics and should not
be used for source attribution.

### Paper-grade q-core hybrid source attribution

The paired q-core score table identifies a performance gap but does not by
itself identify its source. Use `submit_q_core_hybrid_factorial.sh` to run the
controlled Pangu-base/Tianji-donor factorial. The three source packages are:

- `M`: `Q_1000,DP_1000,Q_925,DP_925,RH_925`;
- `T`: `T2M,MSLP`;
- `W`: `U10,V10,WSPD10,WDIR10,U_925,V_925,WSPD925`.

Masks are written in `MTW` order, so `000` is the recomputed Pangu endpoint,
`111` is the recomputed Tianji endpoint, and `100` replaces only the complete
moisture package. Every replacement covers the full 12 h sequence. The builder
recomputes all fog-derived features, preserves the shared zenith/PM/static/time
columns, and requires `000` and `111` to reproduce their source matrices. A
source row, label, unit-policy, PM-policy, lead-provenance, or shared-column
mismatch stops the chain before training.

Always pass the already-audited canonical q-core root explicitly. Commas are
not used inside exported seed/mask lists because Slurm treats them as
`--export` separators.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

# Inspect the full dependency graph without submitting anything.
RUN_TAG=qcore_hybrid_mtw_v1_20260702 \
SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_units_v2_20260701 \
DRY_RUN=1 \
bash submit_q_core_hybrid_factorial.sh

# Full paper experiment: 3 shared q-core S1 anchors + 24 hybrid S2 models,
# 3 common-core S1/S2 controls, a 27-model artifact gate, endpoint grouped
# permutation + ALE diagnostics, 3 seed evaluations, and one joint analysis.
RUN_TAG=qcore_hybrid_mtw_v1_20260702 \
SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_units_v2_20260701 \
SEEDS=42:2025:20260702 \
MASKS=000:001:010:011:100:101:110:111 \
OBS_ROOT=/path/to/hourly/station/csv/root \
bash submit_q_core_hybrid_factorial.sh
```

If the MTW analysis indicates that the thermal/pressure package should be
split, run the `mt2pw` follow-up instead of retraining a full 48-model
four-package matrix. In this profile the mask order is `M,T2,P,W`, where `T2`
is `T2M` and `P` is `MSLP`. The eight masks with `T2==P` are exactly the
already-trained MTW combinations, so the follow-up trains only the eight
T2M-only/MSLP-only masks:

- new S2 masks: `0010,0011,0100,0101,1010,1011,1100,1101`;
- reused MTW masks: `0000,0001,0110,0111,1000,1001,1110,1111`.

Use the completed MTW formal tag as `BASE_MTW_RUN_TAG`; this reuses its three
shared q-core S1 anchors and the eight coupled MTW S2 checkpoints, while
building/auditing the full 16-mask `mt2pw_*` dataset set for provenance.
Common-core, endpoint feature importance and ALE are disabled by default for
this follow-up; the main deliverable is the 4-group Shapley/interaction
analysis.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=qcore_hybrid_mt2pw_formal_v1_20260708 \
GROUP_PROFILE=mt2pw \
BASE_MTW_RUN_TAG=qcore_hybrid_mtw_formal_v1_20260703 \
SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_units_v2_20260701 \
SEEDS=42:2025:20260702 \
OBS_ROOT=/path/to/hourly/station/csv/root \
bash submit_q_core_hybrid_factorial.sh
```

For a scheduler dry run, add `DRY_RUN=1`. For a 2000-row smoke test of the new
profile, also set `RUN_TAG` to a smoke tag, `SEEDS=42`, `LIMIT_ROWS=2000`,
`LIMIT_SAMPLES=2000`, `BOOTSTRAP_ITERS=20`, and short S2 step counts. The
artifact audit for a formal `mt2pw` run expects 51 triplets: 3 reused S1
triplets, 24 newly trained split-only S2 triplets, and 24 reused coupled MTW S2
triplets.

Use a separate tag for the required 2000-row end-to-end smoke test. The short
step counts are inherited by every submitted training job:

```bash
RUN_TAG=qcore_hybrid_mtw_smoke_20260702 \
SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_units_v2_20260701 \
SEEDS=42 RUN_IMPORTANCE=0 RUN_ALE=0 \
LIMIT_ROWS=2000 LIMIT_SAMPLES=2000 \
RUN_COMMON_CORE=0 BOOTSTRAP_ITERS=20 \
LOWVIS_RNN_S1_STEPS=20 LOWVIS_RNN_S2_A_STEPS=20 LOWVIS_RNN_S2_B_STEPS=40 \
bash submit_q_core_hybrid_factorial.sh
```

The formal analysis uses Low-vis AP plus test CSI/recall at a common FPR chosen
only from validation data. It writes per-seed and seed-mean metrics, reliability
bins and a reliability figure, exact Shapley contributions, second-order interactions, UTC-date block
bootstrap intervals, Shapley efficiency checks, and compressed event
case-control samples under
`paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/<run_tag>/analysis/`.
Point AP is exact. To keep the 1000 date-block resamples tractable over all 24
models, bootstrap AP starts with 4096 score bins and doubles the resolution up
to 65536 bins when needed; the analysis still stops if the selected histogram
differs from exact point AP by more than `5e-4`.
The same chain also runs endpoint grouped permutation/model-reliance analysis
for `000` and `111` under `feature_importance/seed_<seed>/`. With `OBS_ROOT`
set, `observation_anchored_source_quality.csv` evaluates T2M, WSPD10 and MSLP
against station observations. `q_reference_analysis_quality_and_extreme_placement.csv`
compares Q1000/Q925 with the paired ERA5 reference analysis using both exact
reference thresholds and quantile-matched event placement; ERA5 is never
labelled as truth.
Before evaluation, `artifact_audit/primary_training_artifact_audit.*` must
verify all 3 S1 + 24 S2 checkpoint/scaler/config triplets, including seed,
dataset path, window, and the 12000/40000 S2 step protocol. Endpoint trajectory
ALE tables and plots are written under `ale/seed_<seed>/` and aggregated into
`analysis/q_core_trajectory_ale_*`; these are within-model response diagnostics,
not source-substitution effects.
`rh2m_dpd_information_package_*` compares the matched Tianji common-core model
against the Tianji `111` q-core endpoint and must be described as the value of
the added `RH2M+DPD` information package, not a general causal effect.

The source endpoint audit separates copied source fields from recomputed fog
features. Labels, row keys, dynamic trajectories, static/time fields and group
isolation retain the strict shared tolerances. Re-executing nonlinear float32
fog feature operations may differ from the source builder by a few ULP across
CPU/runtime paths, so only that derived block has a recorded absolute
compatibility ceiling of `5e-5`. The observed maximum is written to the hybrid
manifest; larger drift remains a hard failure.

If a formal chain stops in `qcore_hybrid_data` after its S1 jobs have already
completed, preserve the incomplete hybrid directory and resume with the same
run tag instead of retraining S1. `RESUME_AFTER_DATA_FAILURE=1` verifies every
existing S1 checkpoint/scaler/config triplet, reuses completed common-core S2
triplets, and resubmits the hybrid build plus all downstream jobs. It requires
the active `HYBRID_DATA_ROOT` to be absent; move the failed directory to a
forensic suffix first. Do not use resume mode if the S1 triplet check fails.

`moisture_followup_gate.json` permits the nested 1000/925-hPa moisture split
only when moisture is the largest Low-vis AP Shapley contribution, its paired
date-block interval excludes zero, and all three seed effects are positive. If
the gate fails, retain the null result and investigate the winning package;
do not continue to a moisture-only claim.

### T925--moisture joint-structure experiment

The original four-source q-core omitted `T_925` because it was not present in
the current IFS intersection. That omission is appropriate for the four-source
fair score table, but it cannot test whether the Pangu--Tianji performance gap
is associated with the joint low-level temperature--moisture state. The
targeted Pangu/Tianji follow-up therefore diagnoses native `T_925` directly in
the already-built source datasets. The default workflow trains no new model.
It combines the completed MTW/mt2pw performance intervention with a separate
quality-and-event diagnosis, avoiding a second 24-model matrix.

The optional confirmatory factorial has `m925b` masks in `M,H,B` order:

- `M`: `Q_1000,DP_1000,Q_925,DP_925,RH_925`, interpreted as the existing
  low-level moisture--thermodynamic state package rather than as pure moisture;
- `H`: explicit `T_925`;
- `B`: `T2M,MSLP,U10,V10,WSPD10,WDIR10,U_925,V_925,WSPD925`.

The default analysis pre-specifies three no-training diagnostic components and
reuses the existing trained performance layer:

1. **Reference-analysis quality.** On the exact paired
   `(valid_time, station_id)` test intersection, compare Pangu and Tianji with
   ERA5 reference analysis using marginal T925/Q925 errors, the Bolton-form
   925-hPa saturation-deficit error, temporal-tendency error, exact-reference
   and quantile-matched near-saturation placement, and source QC. ERA5 is not
   called truth.
2. **Dependence.** Remove train-fitted source
   marginals with empirical-CDF transforms, then compare the T925--Q925 copula
   with ERA5 using Gaussian-kernel MMD. A fixed-seed exact-kernel MMD on a
   2000-row paired subset must confirm the source ordering obtained from the
   full-sample random-feature approximation.
3. **Outcome linkage.** In true low-visibility cases, test whether the
   standardized T925--Q925 joint error is larger for Pangu than Tianji in
   `Tianji-hit/Pangu-miss` samples from the completed mt2pw analysis.

The completed MTW/mt2pw retraining remains the performance-intervention layer.
The new diagnostic does not reinterpret its T2M/MSLP effects as a direct T925
effect. Accordingly, the supported default claim is an association between a
larger joint-structure discrepancy and existing miss cases, not a new causal
T925 performance contribution.

Pangu's original architecture is a data-driven 3-D Earth-specific transformer,
not a numerical solver that explicitly advances the atmospheric governing
equations ([Bi et al., 2023](https://www.nature.com/articles/s41586-023-06185-3)).
That motivates the hypothesis but does not establish that Pangu is physically
inconsistent. MMD is a distributional two-sample distance
([Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html)); fixed
random Fourier features make the date bootstrap tractable
([Rahimi and Recht, 2007](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)); empirical copulas separate
dependence from marginal distributions
([Wilks, 2015](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2414)); and
the saturation calculation follows
[Bolton (1980)](https://journals.ametsoc.org/doi/10.1175/1520-0493%281980%29108%3C1046%3ATCOEPT%3E2.0.CO%3B2).
All confidence intervals resample paired UTC valid dates so station samples
within the same weather day remain together, following the case-resampling
principle of
[Hamill (1999)](https://journals.ametsoc.org/abstract/journals/wefo/14/2/1520-0434_1999_014_0155_htfenp_2_0_co_2.xml).

First run a scheduler dry run. The default `BUILD_DATA=auto` preflights every
source on the login node before `sbatch`. It prints `TRAINING_JOBS=0`, reuses
each compliant dataset, rebuilds only each noncompliant dataset, and finally
submits exactly one dependent analysis job:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=qcore_t925_diag_v1_20260714 \
DRY_RUN=1 \
bash submit_q_core_t925_diagnostics.sh
```

The launcher selects a Python 3.8+ interpreter for this preflight instead of
assuming that the login-node `python` command is Python 3. Dataset rejection
is the preflight's explicit exit code 2; syntax, interpreter, or other program
failures abort the chain before any `sbatch` rather than being misclassified as
three datasets needing rebuild.

Then submit the diagnosis by removing only `DRY_RUN=1`:

```bash
RUN_TAG=qcore_t925_diag_v1_20260714 \
BOOTSTRAP_ITERS=1000 TEST_MAX_ROWS=0 RFF_DIM=512 \
bash submit_q_core_t925_diagnostics.sh
```

The analyzer accepts different source-full feature orders but hard-fails unless
all three configs use the canonical unit policy, explicitly UTC time, native
T925 provenance, paired visibility labels, and the canonical Pangu 12--23 h
lead. The submitter mirrors all config and artifact checks before scheduling,
so an old unit policy is detected without consuming a compute allocation. In
auto mode, only failed sources are rebuilt. To force a clean rebuild of all
three diagnostic datasets, use:

```bash
RUN_TAG=qcore_t925_diag_rebuild_v1_20260714 \
BUILD_DATA=1 \
BOOTSTRAP_ITERS=1000 TEST_MAX_ROWS=0 RFF_DIM=512 \
bash submit_q_core_t925_diagnostics.sh
```

The diagnostic gate is conjunctive: (i) Pangu has significantly larger
saturation-deficit RMSE and copula MMD than Tianji, with the RFF ordering
confirmed by exact-kernel MMD, and (ii) the paired UTC-date interval for Pangu
minus Tianji joint error in `Tianji-hit/Pangu-miss` cases is above zero. Failure
of either layer is retained as a negative diagnostic result. Secondary
diagnostics cannot rescue the primary gate.

Outputs are written below
`paper_eval_results_pm10_pm25_journal/q_core_t925_diagnostics/<RUN_TAG>/`. The
analysis produces source-data CSV files, two JSON evidence reports, and three
single-claim figures in PNG/PDF/SVG/TIFF: saturation-deficit quality, empirical-
copula discrepancy, and asymmetric-event joint-state error. The report records
`analysis_mode=diagnostic_only` and `new_training_models_used=0`.

Only after this diagnosis has been reviewed may the direct `M x T925`
performance intervention be considered. It is not part of the default paper
workflow. `submit_q_core_t925_joint_structure.sh` refuses to run unless
`ENABLE_OPTIONAL_FACTORIAL=1` is explicitly supplied; that optional route is
the separate 3-S1 + 24-S2 experiment. Even then, it does not prove a governing-
equation violation and must not be generalized to all AI weather models.

### Two-source q-core + T925 fair rerun

Use `submit_q_core_t925_fair_experiment.sh` when the question is narrowly
whether the Pangu--Tianji fair-performance gap persists after adding their
shared `T_925` field. The input layout is `q_core_t925_no_rh2m`: the original
14 shared meteorological q-core fields plus `T_925`, followed by `ZENITH`,
`PM10_ugm3`, and `PM25_ugm3` (`dyn18`). RH2M remains excluded. This launcher
does not submit IFS, ERA5, hybrid masks, or the optional 24-S2 factorial.

The formal default submits three data builds, one hard data audit, three
seed-matched S1 models, six S2 models (Tianji and Pangu for each seed), three
paired validation/test inference jobs, and one joint analysis. Primary results
are threshold-free Low-vis AP and validation-matched-FPR test recall/CSI;
argmax performance, Brier/ECE, reliability, and seed dispersion are secondary.
The target FPR is the median of the three Pangu validation argmax FPR values,
and all confidence intervals jointly resample UTC valid dates before averaging
the three seed effects.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=qcore_t925_fair_formal_v1_20260721 \
DRY_RUN=1 \
bash submit_q_core_t925_fair_experiment.sh

RUN_TAG=qcore_t925_fair_formal_v1_20260721 \
BOOTSTRAP_ITERS=1000 \
bash submit_q_core_t925_fair_experiment.sh
```

Results are written to
`paper_eval_results_pm10_pm25_journal/q_core_t925_fair/<RUN_TAG>/analysis/`.
Use a fresh run tag for every formal submission; the launcher refuses to
overwrite existing data, results, or checkpoints.

### Final q-core + T925 five-package attribution

Use `submit_q_core_t925_mhtpw_factorial.sh` for the final source-block
attribution in the same `q_core_t925_no_rh2m` input space as the fair
comparison. The exact Shapley bit order is `M,H,T,P,W`:

- `M`: `Q_1000, DP_1000` (near-surface moisture);
- `H`: `T_925, Q_925, DP_925, RH_925` (one-source 925-hPa
  thermodynamic/moisture state);
- `T`: `T2M`;
- `P`: `MSLP`;
- `W`: `U10, V10, WSPD10, WDIR10, U_925, V_925, WSPD925` (the
  prespecified low-level wind/ventilation state).

MSLP is independent of wind in the factorial, so a near-zero or negative
pressure contribution cannot be hidden inside a favorable wind result. The
two wind layers remain one primary physical-process package: splitting them
would double the exact factorial from 32 to 64 masks and the formal S2 count
from 96 to 192. The endpoint grouped-permutation output instead includes
separate `native_surface_wind_ventilation` and `native_925_wind` rows. These
are secondary model-reliance diagnostics, not Shapley source contributions.

Reuse the completed q-core+T925 fair-data tree to avoid redundant source-data
rebuilding. First run the one-seed, short-step, all-32-mask smoke chain:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_t925_fair_datasets/qcore_t925_fair_formal_v1_20260721 \
RUN_TAG=qcore_t925_mhtpw_smoke_v1_20260722 \
SEEDS=42 \
LIMIT_ROWS=2000 \
LIMIT_SAMPLES=2000 \
LOWVIS_RNN_S1_STEPS=20 \
LOWVIS_RNN_S2_A_STEPS=20 \
LOWVIS_RNN_S2_B_STEPS=40 \
BOOTSTRAP_ITERS=50 \
RUN_IMPORTANCE=0 \
bash submit_q_core_t925_mhtpw_factorial.sh
```

After the smoke artifact audit, evaluation, and Shapley efficiency checks pass,
submit the formal three-seed matrix with a fresh tag:

```bash
SOURCE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_t925_fair_datasets/qcore_t925_fair_formal_v1_20260721 \
RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
SEEDS=42:2025:20260702 \
BOOTSTRAP_ITERS=1000 \
RUN_IMPORTANCE=1 \
bash submit_q_core_t925_mhtpw_factorial.sh
```

This schedules three shared S1 anchors and 96 S2 models. `00000` and `11111`
are the Pangu and Tianji fair endpoints on the exact common row intersection;
there is no separate post-hoc T925 performance model. Results are written to
`paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/<RUN_TAG>/`.
If a subset of training jobs fails, rerun the same command with
`RESUME_EXISTING_RUN=1`; completed checkpoint/scaler/config triplets are kept,
all 32 hybrid datasets are re-audited, and only incomplete models are
resubmitted.

#### Attach automatic stall recovery after the chain is already submitted

The manual resume flag does not detect a job that remains `RUNNING` while its
training step, validation marker, or initialization state has stopped moving.
For the 99-model formal matrix, attach the independent CPU watchdog after the
initial submission. It scopes itself to one `RUN_TAG` and reconstructs the
first generation from that run's recorded artifact-audit dependency; it never
scans or modifies unrelated training families.

The first command is inspection-only:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
bash attach_q_core_mhtpw_watchdog.sh
```

Check that the report identifies the intended run, 99 expected training rows,
and only `mhtpw_s1_*` / `mhtpw_<mask>_*` jobs. Then attach automatic recovery:

```bash
RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
CONFIRM_WATCH=YES \
bash attach_q_core_mhtpw_watchdog.sh
```

The watchdog requires two repeated semantic-stall confirmations plus a final
authoritative recheck before cancellation. Defaults are 45 min for startup,
60 min for data/model initialization, 45 min for training steps, and 120 min
for validation. `PENDING`, `CONFIGURING`, ordinary queue delay, and changing
training-step tokens are never cancelled. A stalled S2 is cancelled alone; a
stalled S1 is cancelled together with only its still-dependent S2 jobs. Partial
files for the cancelled run ID are moved under
`checkpoints/watchdog_quarantine/<RUN_TAG>/` rather than silently accepted.
The confirmed stalled allocation or transient failed NodeList is also excluded
from replacement training by default, reducing repeated RCCL hangs on the same
nodes.

After every other job in that generation becomes terminal, the watchdog calls
the tracked launcher with `RESUME_EXISTING_RUN=1`. Complete checkpoint/scaler/
config triplets are reused; only missing training is submitted. The launcher
then records fresh artifact-audit, evaluation, endpoint-importance, and final
analysis JobIDs. `NODE_FAIL`, `PREEMPTED`, and `BOOT_FAIL` are eligible for the
same bounded recovery. `FAILED`, OOM, TIMEOUT, ambiguous job identities, query
errors, and downstream analysis failures stop safely for diagnosis instead of
being retried blindly. The default limit is two attempts per logical model.

Persistent records are under `<EVAL_ROOT>/watchdog/`:

- `mhtpw_watchdog_state.json`: current generation, progress tokens, retry
  counts, and any blocking reason;
- `mhtpw_watchdog_actions.tsv`: append-only actions and cancellation evidence;
- `resolved_submission_manifest.txt`: the final dependency chain after a
  recovery;
- `watchdog_job.env`: the CPU watchdog JobID.

#### Recover the July 2026 MHTPW dispatcher-alias failure

The original formal launcher exported
`EXPERIMENT=s2_q_core_t925_mhtpw`, while an older
`sub_ifs_overlap_baseline.slurm` did not register that alias. Those S2 jobs
therefore failed in the dispatcher before model training. This is one
launcher/dispatcher integration defect, not independent optimization failure
of every affected model.

Pull a revision that accepts the alias before recovery. Then inspect the
recorded chain with opt-in recognition of only this exact historical error:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
ADOPT_MHTPW_DISPATCHER_FAILURE=YES \
bash attach_q_core_mhtpw_watchdog.sh
```

The inspection must report `adopt_dispatcher_failure: true`, the intended run
tag, and 99 expected training rows. To recover:

```bash
RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
ADOPT_MHTPW_DISPATCHER_FAILURE=YES \
CONFIRM_WATCH=YES \
WATCH_POLL_SECONDS=180 \
WATCH_TRAIN_STALE_MINUTES=60 \
WATCH_VALIDATION_STALE_MINUTES=120 \
WATCH_CONFIRMATIONS=2 \
WATCH_MAX_RETRIES=2 \
bash attach_q_core_mhtpw_watchdog.sh
```

This special adoption path applies only when an S2 job is `FAILED` and that
job's own `logs/<JobID>.out` or `.err` contains the exact
`Unknown EXPERIMENT=s2_q_core_t925_mhtpw` signature. Other failures remain
blocked for diagnosis. Complete S1/S2 artifact triplets are reused. An active
S1 is allowed to finish while its semantic progress changes; if it is
confirmed stalled, the ordinary watchdog path quarantines its incomplete
files and resubmits only that S1 and its dependent missing S2 models.

Do not run a second manual `RESUME_EXISTING_RUN=1` launcher, cancel all old
S2 jobs, or delete checkpoints while this watchdog owns the run.

If the original generation is already unusable and a remaining S1 allocation
is preventing replacement S2 submission, explicitly end only the training
jobs recorded in that generation and immediately resume the missing matrix.
First stop the old CPU watchdog (not the training jobs), then run the normal
attachment preflight with the force flag:

```bash
RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722
EVAL_ROOT=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/${RUN_TAG}

if [ -s "${EVAL_ROOT}/watchdog/watchdog_job.env" ]; then
  source "${EVAL_ROOT}/watchdog/watchdog_job.env"
  scancel "${WATCH_JOB}" 2>/dev/null || true
fi

RUN_TAG="${RUN_TAG}" \
ADOPT_MHTPW_DISPATCHER_FAILURE=YES \
FORCE_END_CURRENT_GENERATION=YES \
bash attach_q_core_mhtpw_watchdog.sh
```

The preflight must show `force_end_current_generation: true`. Confirm once:

```bash
RUN_TAG=qcore_t925_mhtpw_formal_v1_20260722 \
ADOPT_MHTPW_DISPATCHER_FAILURE=YES \
FORCE_END_CURRENT_GENERATION=YES \
CONFIRM_WATCH=YES \
bash attach_q_core_mhtpw_watchdog.sh
```

The replacement launcher reuses every complete artifact triplet. Every MHTPW
S1/S2 replacement also runs a strict per-node cache preflight before Torch:
all four train/validation arrays must be copied to and size-verified under
`/tmp` on every allocated node. There is no NFS fallback in this preflight.
An exact local-cache failure is eligible for bounded watchdog recovery; other
training failures remain blocked.

Stopping the watchdog does not cancel any managed training job. If its
240-hour allocation ends while the experiment is still queued/running, rerun
the same confirmed attach command; the persistent state and single-run lock
prevent a second active controller.

#### Redraw the evidence story from completed q-core+T925 endpoints

Do not use the old no-T925 endpoint figures as the final companion to the
MHTPW attribution. Performance and hit/miss-conditioned panels must be rebuilt
from the completed three-seed q-core+T925 fair run. Surface/pressure-level
quality and physical-QC panels are model-independent and may reuse their paired
diagnostic tables. The old Shapley panel is deliberately omitted until all 32
MHTPW masks are complete.

The paired-quality CPU job first recomputes exact paired RMSE-ratio confidence
intervals from joint UTC-date bootstrap draws. The event CPU job independently
defines Physics-only/AI-only Low-vis cases from the completed q-core+T925
three-seed mean probabilities at validation-matched FPR, then compares T925,
Q1000, Q925, and vector-derived 925-hPa wind speed point by point with ERA5
reference analysis at valid time. Both jobs are zero-training diagnoses and do
not depend on the unfinished 96 S2 jobs.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

QUALITY_TAG=qcore_paired_quality_pressure_scopes_v2_20260722
QUALITY_RAW=$(RUN_TAG=${QUALITY_TAG} bash submit_q_core_paired_source_quality.sh)
QUALITY_JOB=$(printf '%s\n' "${QUALITY_RAW}" | sed -n 's/^analysis_job=//p' | tail -n 1)

UPPER_JOB=$(sbatch --parsable \
  --export=ALL,SOURCE_RUN_TAG=qcore_t925_fair_formal_v1_20260721,RUN_TAG=qcore_t925_upper_air_disagreement_v3_20260722 \
  sub_q_core_t925_upper_air_disagreement.slurm)
UPPER_JOB=${UPPER_JOB%%;*}

STORY_JOB=$(sbatch --parsable \
  --dependency=afterok:${QUALITY_JOB}:${UPPER_JOB} \
  --export=ALL,SOURCE_RUN_TAG=qcore_t925_fair_formal_v1_20260721,UPPER_RUN_TAG=qcore_t925_upper_air_disagreement_v3_20260722,PAIRED_QUALITY_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/q_core_paired_source_quality/${QUALITY_TAG}/analysis,OUT_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/qcore_t925_fair_formal_v1_20260721/evidence_story_figures_t925_nc_v3 \
  sub_pangu_qcore_t925_evidence_story.slurm)
STORY_JOB=${STORY_JOB%%;*}

echo "QUALITY_JOB=${QUALITY_JOB}"
echo "UPPER_JOB=${UPPER_JOB}"
echo "STORY_JOB=${STORY_JOB}"
```

The redraw is written under the completed fair run as
`evidence_story_figures_t925_nc_v3/`. Panel 07 is a unit-free, log-scale
Tianji/Pangu RMSE-ratio overview for T925, Q1000, Q925 and 925-hPa vector wind
in both all-test and observed-Low-vis samples. Panels 07a--07d show the same
four quantities as native-unit absolute RMSE comparisons. Panel 09d uses the
same upper-air coverage for event-conditioned signed bias; its wind panel uses
speed because a signed vector-error direction is not defined. The report also
records the intentional Shapley omission. Every figure is exported as SVG,
PDF, PNG, and 600-dpi TIFF.

### Corrected canonical-station rerun (fair + best effort)

The earlier corrected-Pangu launcher reused q-core S1/Tianji/IFS datasets. Do
not use that reuse path for the repaired fair experiment because those datasets
predate the canonical PM/MSLP policy. Use `submit_q_core_fair_experiment.sh`
with a fresh run tag. `submit_corrected_pangu2025_experiments.sh` remains only
for diagnosing the historical coordinate-only rerun. It first runs a compute-node
preflight that compares the old and new Pangu station products against
`merged_final_all_vars.nc`. The preflight requires all of the following before
any large data build or GPU job can start:

- the new station IDs and coordinates equal the canonical target station table;
- at least one old station coordinate differs from the corrected coordinate;
- sampled interpolated meteorological values differ between old and new files;
- valid times are unique and hourly;
- per-time `init_time`/`forecast_lead_hours` evidence is internally consistent;
  for the legacy stitched product only, an explicit 00/12 UTC schedule may
  reconstruct leads 12--23 h from valid-time hour.

The chain then builds only two Pangu datasets (`q_core_no_rh2m` and
`source_full`), runs the four-source q-core pairing audit, trains a fresh shared
q-core S1 and all four fair S2 models, and evaluates the paired fair result. In
parallel, it reuses the unaffected Pangu-2025 dyn19 source-full S1 checkpoint,
trains only a corrected Pangu source-full S2, and reruns the all-source
best-effort argmax evaluation with the existing Tianji/IFS/T2ND/ERA5 models.

Inspect the exact submission graph first:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
git pull

RUN_TAG=pangu2025_canonical_20260630 \
OLD_PANGU_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h.nc \
PANGU2025_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h_canonical.nc \
REUSED_QCORE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_pangu2025_rerun_20260629 \
EXPECTED_PANGU_LEAD_MIN_HOURS=12 \
EXPECTED_PANGU_LEAD_MAX_HOURS=23 \
DRY_RUN=1 \
bash submit_corrected_pangu2025_experiments.sh
```

If the paths and run IDs are correct, submit the real chain by removing only
`DRY_RUN=1`:

```bash
RUN_TAG=pangu2025_canonical_20260630 \
OLD_PANGU_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h.nc \
PANGU2025_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead12_23h_canonical.nc \
REUSED_QCORE_DATA_ROOT=/public/home/putianshu/vis_mlp/ifs_baseline/q_core_fair_datasets/qcore_pangu2025_rerun_20260629 \
EXPECTED_PANGU_LEAD_MIN_HOURS=12 \
EXPECTED_PANGU_LEAD_MAX_HOURS=23 \
bash submit_corrected_pangu2025_experiments.sh
```

The lead range above is an assertion, not a filename inference. The corrected
launcher defaults to `INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1` for the known
legacy hourly stitched product. It reconstructs the documented schedule as:
valid hours 12--23 use the same-day 00 UTC initialization, and valid hours
00--11 use the previous-day 12 UTC initialization. Disable this option for any
other Pangu product; such products must carry their own lead metadata.

## Required Order

Run from the remote overlap repository:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs
```

### 1. Build Or Refresh The Overlap S1 Dataset

Run this when the full S1 PM10+PM2.5 source dataset changed, when the overlap S1
dataset is missing, or when you need to verify the 27-dyn layout from scratch.

```bash
sbatch sub_s1_overlap_data.slurm
```

For the five-source `common_core` comparison and the S1 zero-transfer response
diagnostic, build a separate S1 dataset whose masked variables exactly match the
Pangu-compatible common-core layout:

```bash
sbatch --export=ALL,FEATURE_SET=common_core sub_s1_overlap_data.slurm
```

The common-core S1 build derives from
`/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap`
by default, so the plain `sbatch sub_s1_overlap_data.slurm` step above must
finish successfully first.

Optional explicit paths:

```bash
sbatch --export=ALL,SOURCE_DIR=/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25,OUT_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap sub_s1_overlap_data.slurm

sbatch --export=ALL,FEATURE_SET=common_core,SOURCE_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap,OUT_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_common_core sub_s1_overlap_data.slurm
```

Do not use `--merge_train_val` for the paper experiment.

### 2. Train The Overlap S1 Checkpoint

```bash
sbatch --export=ALL,EXPERIMENT=s1_overlap sub_ifs_overlap_baseline.slurm
```

For common-core source-family experiments, train the matching S1 checkpoint
instead of reusing the overlap-full S1 checkpoint:

```bash
sbatch --export=ALL,EXPERIMENT=s1_common_core sub_ifs_overlap_baseline.slurm
```

If the common-core S2 datasets are already built and you want to queue S1 and
all common-core S2 jobs together, use
`OVERLAP_CHAIN=common_core bash submit_ifs_overlap_training_chain.sh` instead.
The submitter runs `EXPERIMENT=s1_common_core` first and queues the five
common-core S2 jobs with `afterok:<s1_jobid>`.

The default `MODEL_ARCH=static_rnn` trains
`exp_overlap_static_rnn_s1_pm10_pm25_S1_best_score.pt`. The Slurm launcher uses
the same direct main-trainer path and stable knobs as
`sub_static_rnn_lowvis_main.slurm`: GRU, mean pooling, one RNN layer,
5 nodes x 4 DCU, `LOWVIS_RNN_BATCH_SIZE=512`, `LOWVIS_RNN_GRAD_ACCUM=2`,
`LOWVIS_RNN_NUM_WORKERS=0`, and recall/CSI validation selection unless
overridden. The trainer requires explicit `X_train/y_train` and `X_val/y_val`,
and it reads native `dyn_vars`, feature order, and FE dimensions from each
dataset build config when present. Use `MODEL_ARCH=pmst` only for legacy PMST
audits.

`EXPERIMENT=s1_common_core` writes
`exp_overlap_static_rnn_s1_common_core_pm10_pm25_S1_best_score.pt` and
`robust_scaler_exp_overlap_static_rnn_s1_common_core_pm10_pm25_s1_w12_dyn19_pm.pkl`.
Use this pair for S1 zero-transfer tests against common-core forecast sources.

### 3. Rebuild Tianji-Input S2 Overlap Data

```bash
sbatch sub_tianji_overlap_data.slurm
```

This uses `merged_final_all_vars.nc` raw times as UTC and writes
`tianji_raw_time_alignment=raw_utc_no_shift` into `dataset_build_config.json`.
The overlap builder now fills the shared PMST slots
`RH2M,T2M,PRECIP,MSLP,SW_RAD,U10,WSPD10,V10,WDIR10,LCC,RH_925,U_925,WSPD925,V_925,DP_1000,DP_925,Q_1000,Q_925,W_925,W_1000,DPD`.
Tianji `PRECIP` is treated as an accumulated amount and converted to hourly
increments before window construction.

To build the Tianji-input variant whose `RH2M` slot is replaced by the T2ND
station interpolation, first create the station file from the completed fregrid
tree and then pass it into the same data builder:

```bash
cd /public/home/putianshu/vis_mlp
python tianji_regrid/rh2m_station_IDW.py \
  --input_root /public/home/putianshu/vis_mlp/src_data \
  --mode T2ND \
  --res 0p1 \
  --var rh2m \
  --output /public/home/putianshu/vis_mlp/ifs_baseline/tianji_rh2m_station/T2ND_rh2m_station_2025.nc

sbatch tianji_regrid/sub_rh2m_station_idw.slurm

cd /public/home/putianshu/vis_mlp/ifs_baseline
sbatch --export=ALL,RH2M_OVERRIDE_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/tianji_rh2m_station/T2ND_rh2m_station_2025.nc,RH2M_SOURCE_TAG=T2ND_rh2m sub_tianji_overlap_data.slurm
```

`rh2m_station_IDW.py` defaults to per-init stitching with
`12 <= lead_hour < 24`, matching the IFS overlap interpolation convention and
avoiding duplicate lead-24 collisions. Use `--lead_end_inclusive` only for a
diagnostic run where the 24h endpoint is intentionally retained.

When `RH2M_OVERRIDE_FILE` is supplied and `OUT_DIR` is not, the output dataset
defaults to `ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m`.
The RH2M-override Slurm path also defaults `CHUNK_WINS=64` to reduce peak RAM.
If the job fails with disk quota or no-space errors, set `STAGING_DIR` to a
large temporary filesystem; the staging memmaps alone are about 26 GiB for the
2025 S2 overlap build.

For the fair source-family run, build Tianji product RH2M and T2ND raw RH2M
with the common-core feature set:

```bash
sbatch --export=ALL,FEATURE_SET=common_core sub_tianji_overlap_data.slurm

sbatch --export=ALL,\
FEATURE_SET=common_core,\
RH2M_OVERRIDE_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/tianji_rh2m_station/T2ND_rh2m_station_2025.nc,\
RH2M_SOURCE_TAG=T2ND_rh2m \
sub_tianji_overlap_data.slurm
```

### 4. Rebuild IFS-Input S2 Overlap Data

```bash
sbatch sub_ifs_data.slurm
```

By default this uses `build_dataset_ifs_overlap_12h_fast.py`, auto-discovers
station-interpolated IFS inputs, uses raw Tianji UTC times, and writes the same
`raw_utc_no_shift` marker. Use `IFS_INTERP_GLOB` for explicit IFS inputs:

```bash
sbatch "--export=ALL,IFS_INTERP_GLOB=/public/home/putianshu/vis_mlp/ifs_baseline/ifs_interp_out/**/ifs_interp_*_2025.nc" sub_ifs_data.slurm
```

IFS station-interpolated inputs should include the source variables
`T2M,D2M,PRECIP,MSLP,SW_RAD,U10,V10,LCC,RH_925,U_925,V_925,Q_1000,Q_925,W_925,W_1000`.
The dataset builder derives `RH2M` from `T2M+D2M`, `DP_1000/DP_925` from
specific humidity and pressure level, `DPD` from `T2M-D2M`, and wind speed or
direction from U/V. IFS `PRECIP` is kept as an hourly amount/rate and is not
differenced.

For the fair source-family run:

```bash
sbatch --export=ALL,FEATURE_SET=common_core sub_ifs_data.slurm
```

### 4.1 Build Pangu-2021 And ERA5-2025 Source Datasets

Pangu first needs grid-to-station interpolation:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
sbatch sub_pangu_station_idw.slurm
```

Then build its overlap dataset:

```bash
sbatch --export=ALL,\
SOURCE_KIND=station_nc,\
SOURCE_TAG=pangu2021,\
YEAR=2021,\
FEATURE_SET=common_core \
sub_station_source_overlap_data.slurm
```

The legacy Pangu-2021 `RH2M` field is a proxy derived in `pangu_data.py` from
1000 hPa humidity. Pangu-2025 does not use this proxy and must stay excluded
from RH2M quality figures.

ERA5-2025 can be built directly from the station feature directory:

```bash
sbatch --export=ALL,\
SOURCE_KIND=era5_feature_dir,\
SOURCE_TAG=era5_2025,\
YEAR=2025,\
FEATURE_SET=common_core \
sub_station_source_overlap_data.slurm
```

For the best-effort all-variable experiment, use the source-full build order
above and keep its outputs separate from the common-core fairness table.

### 5. Train Tianji-Input S2

```bash
sbatch --export=ALL,EXPERIMENT=s2_tianji sub_ifs_overlap_baseline.slurm
```

For the T2ND-rh2m replacement dataset:

```bash
sbatch --export=ALL,EXPERIMENT=s2_tianji_T2ND_rh2m sub_ifs_overlap_baseline.slurm
```

Its default Static-RNN output names are
`exp_overlap_static_rnn_s2_T2ND_rh2m_pm10_pm25_S2_PhaseB_best_score.pt` and
`robust_scaler_exp_overlap_static_rnn_s2_T2ND_rh2m_pm10_pm25_s2_w12_dyn27_pm.pkl`.

For the common-core source-family comparison:

```bash
OVERLAP_CHAIN=common_core bash submit_ifs_overlap_training_chain.sh
```

Run this one-shot queue only after the common-core S2 datasets are present. If
the common-core S1 checkpoint already exists and you do not want to retrain it,
submit the S2 experiments directly.

For supplementary upper-bound runs:

```bash
OVERLAP_CHAIN=source_full bash submit_ifs_overlap_training_chain.sh
```

Run this after the source-full S1 and S2 data directories have been built. The
submitter trains the matching S1 layouts and queues each S2 on the correct S1
checkpoint.

If `s1_source_full_ifs` logs
`[Data-Copy] Insufficient space on /tmp, using NFS.`, cancel and replace only
that IFS chain. Its old dependent `s2_ifs_source_full` must also be cancelled;
the Tianji/T2ND/ERA5 chains are independent and can remain queued/running:

```bash
OLD_S1_JOBID=<slow_ifs_s1_job_id> bash resubmit_source_full_ifs_chain.sh
```

The replacement defaults `LOWVIS_RNN_LOCAL_CACHE_DIR=/dev/shm`. Check the new
log for `Copying X_train.npy to /dev/shm` or a cache hit. If `/dev/shm` also
reports insufficient space, use another large node-local cache directory
instead of repeatedly resubmitting the same NFS-backed job.

The replacement S1 also defaults `LOWVIS_RNN_CLEAN_LOCAL_CACHE=1`. After Slurm
allocates its exclusive nodes, the launcher deletes only user-owned
`X_train/X_val/y_train/y_val` cache files and fallback markers directly under
`/tmp` on those nodes. It does not remove the whole `/tmp` directory or files
owned by other users. The dependent S2 keeps the fresh cache and does not clean
again.

### 6. Train IFS-Input S2

```bash
sbatch --export=ALL,EXPERIMENT=s2_ifs sub_ifs_overlap_baseline.slurm
```

Both S2 runs default to Static-RNN and require
`exp_overlap_static_rnn_s1_pm10_pm25_S1_best_score.pt` as the pretrained
checkpoint. Override only with an explicit
`OVERLAP_STATIC_RNN_PRETRAINED_CKPT=/path/to/S1_best_score.pt`; otherwise the
launcher stops instead of silently training S2 from scratch. The expected best
outputs are `exp_overlap_static_rnn_s2_tianji_pm10_pm25_S2_PhaseB_best_score.pt`
and `exp_overlap_static_rnn_s2_ifs_pm10_pm25_S2_PhaseB_best_score.pt`. Both S2
trainers require explicit month-tail validation files and fail on legacy PM10-only
or wrong FE layouts. The overlap S2 launcher defaults to a longer fine-tuning
budget than the main quick path: `LOWVIS_RNN_S2_A_STEPS=12000`,
`LOWVIS_RNN_S2_B_STEPS=40000`, and `LOWVIS_RNN_PATIENCE=18`, so Tianji-input
training is less likely to stop before the validation score has saturated.

### 7. Run Paired Forecast-Source Evaluation

After both S2 checkpoints exist:

```bash
python test_PMST_overlap_forecast_source_s2.py \
  --tianji_ckpt /public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/exp_overlap_static_rnn_s2_tianji_pm10_pm25_S2_PhaseB_best_score.pt \
  --ifs_ckpt /public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/exp_overlap_static_rnn_s2_ifs_pm10_pm25_S2_PhaseB_best_score.pt \
  --out_dir /public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/overlap_forecast_source
```

The evaluator refuses datasets without `tianji_raw_time_alignment=raw_utc_no_shift`
unless `--allow_legacy_time_alignment` is passed. Scenario day/night grouping
uses UTC+8 by default through `--local_time_offset_hours 8`. By default it reads
the decision thresholds stored in each selected `*_best_score.pt` checkpoint;
`--threshold_mode val_search` is available only when you intentionally want to
rerun validation threshold selection inside the evaluator.
For the T2ND-rh2m replacement model, add `--tianji_source_tag T2ND_rh2m` or pass
explicit `--tianji_data_dir`, `--tianji_ckpt`, and `--tianji_scaler` paths.
Feature replacement runs by default for
`RH2M,Q_1000,DP_1000,RH_925,PRECIP` when those slots are populated in both
overlap datasets.

### 8. Test S1 Zero-Transfer Response to Forecast Sources

This diagnostic asks whether the S1-only Static-RNN model responds at all to
forecast-source inputs before any S2 transfer. It uses the same S1 checkpoint and
S1 RobustScaler for every source, then reports full-test Fog, Mist, and low-vis
recall/CSI/precision/FPR.

The default decision rule is `THRESHOLD_MODE=argmax`, because this experiment is
a response diagnostic rather than a deployment test. Reusing the S1 checkpoint's
saved thresholds on forecast-source inputs can collapse all low-visibility
predictions to clear under domain shift. To test "threshold recalibration only,
no weight transfer", run with `THRESHOLD_MODE=val_search` instead. The
zero-transfer launcher defaults that validation search to
`THRESHOLD_SEARCH_POLICY=response`, which relaxes the operational precision and
clear-recall guards and asks whether any low-visibility response can be recovered
without updating model weights.

Run all five common-core sources in one CPU job:

```bash
sbatch --export=ALL,SOURCE_GROUP=all,DEVICE=cpu sub_static_rnn_s1_zero_transfer_eval.slurm
```

Optional threshold-recalibration audit:

```bash
sbatch --export=ALL,SOURCE_GROUP=all,DEVICE=cpu,THRESHOLD_MODE=val_search,SKIP_VALIDATION_INFERENCE=0,THRESHOLD_SEARCH_POLICY=response sub_static_rnn_s1_zero_transfer_eval.slurm
```

Or split the inference by source and merge the metric tables afterwards:

```bash
deps=""
for src in tianji_common_core ifs_common_core T2ND_rh2m_common_core pangu2021_common_core era5_2025_common_core; do
  jid=$(sbatch --parsable --export=ALL,SOURCE_GROUP=${src},DEVICE=cpu sub_static_rnn_s1_zero_transfer_eval.slurm)
  deps="${deps:+${deps}:}${jid}"
done
sbatch --dependency=afterok:${deps} --export=ALL,SOURCE_GROUP=merge sub_static_rnn_s1_zero_transfer_eval.slurm
```

Default inputs are:

- checkpoint:
  `ifs_baseline/checkpoints/exp_overlap_static_rnn_s1_common_core_pm10_pm25_S1_best_score.pt`
- scaler:
  `ifs_baseline/checkpoints/robust_scaler_exp_overlap_static_rnn_s1_common_core_pm10_pm25_s1_w12_dyn19_pm.pkl`
- output root:
  `paper_eval_results_pm10_pm25_journal/zero_transfer_s1_forecast_sources`

Override `S1_CKPT`, `S1_SCALER`, `OUT_ROOT`, `BATCH_SIZE`, or `LIMIT_SAMPLES`
through `--export=ALL,...` for audit runs.

For a single-source smoke test of a trained Static-RNN source model:

```bash
sbatch --export=ALL,SOURCE_TAG=T2ND_rh2m,FEATURE_SET=common_core,LIMIT_SAMPLES=2000 sub_static_rnn_overlap_single_eval.slurm
```

For key-variable extremeness and observation-anchored quality across all
common-core sources:

```bash
sbatch --export=ALL,FEATURE_SET=common_core sub_rh2m_multi_source_quality.slurm
```

The default key-variable list is `RH2M,Q_1000,DP_1000,RH_925,PRECIP`; override
it with `FEATURES=RH2M,Q_1000,DP_1000` if needed.

The key-variable analysis writes:

- `key_variable_source_quality_metrics.csv`: per-source quantiles, and observation-anchored MAE/RMSE/correlation where station observations exist.
- `key_variable_source_pairwise_distribution.csv`: paired source-source differences for RH2M, Q_1000, DP_1000, RH_925, and PRECIP within the same year group.
- `fig_key_variable_tail_<feature>_<group>.*`: tail-frequency curves for each key variable.
- legacy RH2M-specific files are still written for compatibility: `rh2m_source_quality_metrics.csv`, `rh2m_source_pairwise_distribution.csv`, `rh2m_tail_curve_<group>.csv`, and `fig_rh2m_tail_multi_source_<group>.*`.

### 8. Run The Mean-Softmax Ensemble Check

This experiment keeps the trained Tianji-input and IFS-input Static-RNN models
fixed. It runs each model on its matching paired source dataset, averages the
post-softmax class probabilities for the same `(time, station_id)` rows, selects
the ensemble fog/mist decision thresholds from the paired validation split, and
then evaluates the ensemble on the held-out paired test split.

```bash
sbatch sub_static_rnn_overlap_softmax_ensemble.slurm
```

The default checkpoints are:

```text
/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/exp_overlap_static_rnn_s2_tianji_pm10_pm25_S2_PhaseB_best_score.pt
/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/exp_overlap_static_rnn_s2_ifs_pm10_pm25_S2_PhaseB_best_score.pt
```

For a quick smoke test:

```bash
sbatch --export=ALL,LIMIT_SAMPLES=2000,SKIP_BOOTSTRAP=1,NO_FIGURES=1,OUT_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/overlap_softmax_ensemble_smoke sub_static_rnn_overlap_softmax_ensemble.slurm
```

For the T2ND-rh2m replacement model, set
`OVERLAP_TIANJI_SOURCE_TAG=T2ND_rh2m`; override `OVERLAP_TIANJI_DATA_DIR`,
`TIANJI_CKPT`, or `TIANJI_SCALER` only when using non-default paths.

Important outputs:

- `overall_metrics.csv`: Tianji single model, IFS single model, and mean-softmax ensemble test metrics.
- `metric_deltas_ensemble_minus_tianji.csv` and `metric_deltas_ensemble_minus_ifs.csv`: direct gain/loss tables with metric direction.
- `scenario_metrics.csv`: All/day/night/season split metrics for the two single models and the ensemble.
- `per_sample_softmax_ensemble_eval.csv`: paired probabilities, predictions, correctness, and ensemble win/loss flags.
- `softmax_ensemble_report.txt`: compact human-readable summary.
- `fig_overlap_softmax_ensemble_key_metrics.*`: Tianji, IFS, and ensemble key-metric bars when matplotlib is available.

## Q1000 Extreme Verification And Multi-Source Model Reliance

For the corrected canonical-station Pangu product, use the dedicated data-check
chain. It verifies the new file against both the old Pangu product and the
canonical Tianji station table, rebuilds only the Pangu `source_full` dataset,
then runs the lineage/elevation audit and the complete Q1000 mechanism analysis.
It does not train a model.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline

RUN_TAG=q1000_pangu2025_canonical_20260630 \
DRY_RUN=1 \
bash submit_corrected_pangu_q1000_checks.sh

RUN_TAG=q1000_pangu2025_canonical_20260630 \
bash submit_corrected_pangu_q1000_checks.sh
```

The defaults use
`pangu_station/pangu_station_2025_lead12_23h_canonical.nc`, assert leads
12--23 h, and enable the documented 00/12 UTC stitched-schedule reconstruction.
Override those defaults only if the new product path or lead metadata differs.

Before interpreting any Q1000 result, run the metadata-backed lineage audit.
It reads actual `valid_time-init_time`, checks hourly cadence, verifies native
Q provenance, and reports Q1000 skill separately below 100 m, 100--500 m, and
above 500 m because 1000 hPa is below ground at many elevated stations.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
sbatch --export=ALL,\
PANGU_DATASET_DIR=/path/to/pangu2025_dataset,\
TIANJI_DATASET_DIR=/path/to/tianji_dataset,\
IFS_DATASET_DIR=/path/to/ifs_dataset,\
ERA5_DATASET_DIR=/path/to/era5_dataset \
sub_q1000_lineage_audit.slurm
```

Use `q1000_lineage_audit.json` for provenance decisions and
`q1000_paired_complete_case_elevation_metrics.csv` for descriptive paired
errors. All sources in the latter use the same finite `(valid_time, station_id)`
rows. ERA5 Q1000/Q925 must be native pressure-level specific humidity; the
q-core audit rejects T/RH-reconstructed ERA5 Q as the reference product.

Compute Q1000 MAE only after taking the common finite complete-case intersection:
`mean(abs(Q_source - Q_ERA5))` on identical `(valid_time, station_id)` rows for
every source. A shared index alone is insufficient if one source still contains
non-finite Q values. Overall MAE measures agreement with the ERA5 reference
analysis, but ordinary humidity cases dominate it. Use
`q1000_extreme_spatiotemporal_metrics.csv` for the high-tail question. Its
`exact_reference_threshold` rows apply the monthly ERA5 P90/P95/P99 threshold
to both fields and jointly test amplitude calibration and concurrence. Its
`quantile_matched` rows apply each source's own monthly quantile and test
whether equal-frequency extremes occur at the same station and valid time.
Interpret SEDI together with POD, FAR, CSI, ETS, and frequency bias. Conditional
bias/MAE/RMSE on ERA5-extreme cases are descriptive only; do not rank sources
from those conditional errors alone because of the forecaster's dilemma.

To draw only the paired multi-source Q1000 and DP1000 probability-density
figure on the current source-full data:

```bash
sbatch --export=ALL,DISTRIBUTION_ONLY=1,REQUIRE_CASE_CONTROL=0,FEATURE_SET=source_full,OUT_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/q1000_dp1000_distribution_source_full sub_q1000_mechanism_analysis.slurm
```

This writes `fig_q1000_dp1000_probability_distribution.*`, editable density
source data in `q1000_dp1000_probability_density.csv`, and P50/P90/P95/P99 in
`q1000_dp1000_distribution_quantiles.csv`. Q1000 and DP1000 use separate panels
and physical units. ERA5 is labelled as reference analysis, not truth. At fixed
1000 hPa, DP1000 is a monotonic coordinate transform of Q1000 and therefore is
an interpretation/consistency panel rather than independent mechanism evidence.

For a paired feature-importance smoke test:

```bash
sbatch --export=ALL,LIMIT_ROWS=20000,SAMPLE_SIZE=10000,MIN_LOW_VIS=20,REPEATS=2,BOOTSTRAP_ITERS=100,GROUP_SCOPE=dynamic,MAX_GROUPS=2,FEATURE_IMPORTANCE_OUT_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/multi_source_feature_importance_smoke sub_multi_source_feature_importance.slurm
```

For the full source-full analysis:

```bash
sbatch --export=ALL,SAMPLE_SIZE=50000,REPEATS=5,BOOTSTRAP_ITERS=1000,GROUP_SCOPE=all,MAX_GROUPS=0,FEATURE_IMPORTANCE_OUT_DIR=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/multi_source_feature_importance_source_full sub_multi_source_feature_importance.slurm
```

`MAX_GROUPS` is only a smoke-test truncation switch. Any positive value evaluates
only the first N groups for each source. `MAX_GROUPS=0` evaluates every group.
`GROUP_SCOPE=dynamic` covers all source meteorological/aerosol time-series
variables plus physical packages; `GROUP_SCOPE=all` additionally covers static
and engineered feature groups. Use the dedicated `FEATURE_IMPORTANCE_OUT_DIR`
variable so an unrelated exported `OUT_DIR` cannot redirect these results.

`analyze_multi_source_feature_importance.py` uses the same common
`(valid time, station)` rows for Tianji, T2ND, IFS, Pangu-2025, and ERA5. It
samples uniformly, preserving the observed event rate, and permutes each
dynamic predictor as a complete 12 h sequence. Marginal grouped permutation is
the primary Fisher et al. model-reliance analysis. The season/hour/region and
moisture-state stratified permutation is a dependence-aware sensitivity check,
not an exact reimplementation of Strobl et al.'s random-forest algorithm.
Uncertainty and source-to-source differences use the same valid-date bootstrap
draws; marginal donor maps are also shared across models.

Use `multi_source_grouped_permutation_importance.csv` for within-model
reliance, `multi_source_shared_feature_importance.csv` for inputs shared by all
sources, and `multi_source_pairwise_feature_importance_differences.csv` for
direct paired differences. `global_shared` rows support all-source comparison;
`pair_shared` rows support controlled Tianji-T2ND RH2M comparison. Never compare
`native_*` package magnitudes across sources when their members differ.
`feature_importance_group_manifest.csv` records every available group and
whether it was selected, so a truncated smoke run cannot be mistaken for the
full analysis.

## Pangu q-core evidence-story figures for PPT and paper

After the formal `mt2pw` analysis and the 51-triplet artifact audit have both
passed, run the zero-training chain below. It first refreshes the paired source
quality report with the model-consistent low-visibility definition
`visibility < 1000 m`, then renders figures only after that diagnosis succeeds.
It reuses all factorial checkpoints and inference outputs; no S1 or S2 model is
trained. If the strict paired-quality report already exists, the submitter
reuses it and schedules only the plotting job.

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs

RUN_TAG=qcore_hybrid_mt2pw_formal_v1_20260708 \
QUALITY_RUN_TAG=qcore_paired_quality_strictlt_v1_20260716 \
bash submit_pangu_qcore_evidence_story.sh
```

The default output is
`paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/<RUN_TAG>/evidence_story_figures_nc_v9`.

The manuscript palette is source-stable across every panel: Tianji is dark
blue (`#2E5A87`), Pangu is mid-light violet (`#8E6BBE`), and baseline/ERA5 is
grey (`#9A9A9A`).  The shared contract lives in `paper_source_palette.py`;
marker shape is retained as a second cue. The two endpoint-performance plots
use the 89-mm Nature single-column width; denser comparison figures use the
183-mm two-column width. Height is selected per figure to avoid empty canvas
space in sparse comparisons. The entire q-core series retains the mainline
layout grammar: centered bold titles, parenthesized panel letters for true
multi-panel figures, complete boxed axes, and light direction-appropriate
grids. Typography follows Nature-family figure conventions with an
Arial/Helvetica-first sans-serif stack and Linux-compatible fallbacks.
Every claim is a separate figure, in presentation order:

1. complete experimental logic and claim boundary;
2. threshold-free Low-vis AP;
3. validation-matched-FPR Low-vis recall;
4. optional argmax Low-vis Precision/Recall/CSI/FPR overview for PPT or supplement;
5. exact four-package Shapley attribution from the 16 retrained combinations;
6. station-observation T2M quality, showing both all samples and observed
   visibility below 1 km;
7. station-observation 10-m wind-speed quality for the same two regimes;
8. paired pressure-level quality against ERA5 reference analysis;
9. validation-matched endpoint-specific Low-vis hits;
10. observation-anchored Tianji source advantage within Tianji-only hits;
11. paired Tianji-minus-Pangu forecast-state contrasts in Tianji-only hits,
    with Pangu-only hits as the reverse-disagreement control.

The optional argmax overview uses the formal `0000` Pangu and `1111` Tianji
q-core endpoints, bars for three-seed means, and open circles for individual
seeds. F1 is intentionally omitted because it duplicates CSI ordering for the
same binary event. AP and matched-FPR recall remain the primary endpoints.

The new event-conditioned figure reports
`100 * (RMSE_Pangu - RMSE_Tianji) / RMSE_Pangu` for T2M, WSPD10 and MSLP,
with 1000 joint UTC-valid-date bootstrap draws. T2M is converted from K to °C
and MSLP from Pa to hPa before matching automatic-station observations. It is a
descriptive analysis of endpoint-selected station-time samples, not a causal
source intervention.

The forecast-state figure uses the same paired station-time samples to report
T2M, WSPD10, RH925 and MSLP differences (`Tianji minus Pangu`). Its primary
diagnostic is the between-group difference: the mean source difference within
Tianji-only hits minus the corresponding difference within Pangu-only hits.
All variables and both categories share the same 1000 UTC-valid-date bootstrap
draws. This reverse-disagreement comparison tests whether the cool, moist and
weak-wind contrast is specific to Tianji-only hits, but it remains conditioned
on model outcomes and therefore is not a causal fog-mechanism estimate.

MSLP quality, the endpoint-specific hit-count figure, and pressure-level
physical QC are supplementary candidates. Subtle vertical dashed major grids
are limited to horizontal numerical comparisons; the endpoint plots use
horizontal value grids, and the workflow has no grid. All figures use
content-specific compact heights, minimal in-figure annotation, a stable
source-color mapping, and are exported as editable SVG/PDF, 600-dpi PNG/TIFF,
plus one source-data CSV per figure.
`qcore_evidence_story_manifest.json` records evidence roles and file hashes;
`QCORE_EVIDENCE_STORY_GUIDE.md` records the slide order, scope, and bounded
wording. Pressure-level comparisons use ERA5 as a reference analysis, not as
truth. Package Shapley is a controlled source attribution and is not proof of
single-variable causality or governing-equation violation.

## Completion Checklist

1. `dataset_build_config.json` for Tianji and IFS both contain
   `tianji_raw_time_alignment=raw_utc_no_shift`.
2. Train/val/test files exist for both overlap S2 datasets.
3. S1, Tianji S2, and IFS S2 checkpoints exist under
   `/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints`.
4. The paired evaluator writes `overall_metrics.csv`, `validation_metrics.csv`,
   `scenario_metrics.csv`, and `run_config.json`.
5. The softmax ensemble evaluator writes `overall_metrics.csv`,
   `metric_deltas_ensemble_minus_tianji.csv`,
   `metric_deltas_ensemble_minus_ifs.csv`, and `softmax_ensemble_report.txt`.
6. Old results generated before the UTC fix are not used in the paper.
