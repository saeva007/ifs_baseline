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
evaluation samples; it does not control forecast lead. The current Pangu ONNX
model advances exactly 24 h, whereas the existing Tianji/IFS products use
`12 <= lead_hour < 24`. ERA5 remains a reference analysis, not an operational
forecast. Therefore this is a common-input product comparison, not a causal
forecast-source attribution experiment. A strict lead-matched experiment must
rebuild Tianji/IFS at lead24h or generate Pangu 12--23 h fields with the official
1/3/6 h models and verified hierarchical stepping.

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

# Submit with automatic discovery of the current lead24h station product.
RUN_TAG=qcore_pangu2025_rerun bash submit_q_core_fair_experiment.sh

# If it is stored elsewhere, pass the verified real path explicitly:
# PANGU2025_STATION_FILE=/actual/path/pangu_station_2025_lead24h.nc
```

If the station product must be regenerated first, pass
`BUILD_PANGU_STATION=1`, `INPUT_DIR` for the current Pangu-2025 China NetCDF
files, and the desired `PANGU2025_STATION_FILE`. The interpolation job then
becomes an explicit dependency of the Pangu dataset build. Do not silently use
the historical Pangu-2021 station file. The station interpolator preserves
per-time `init_time` and `forecast_lead_hours`; the downstream builder verifies
those values and rejects filename-only provenance. Bulk inference must use
`INIT_HOURS=all` so a 12-step input sequence is actually 12 consecutive hours.

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
valid-time year. With a 24 h Pangu lead, 31 December 2025 initializations can
legitimately verify on 1 January 2026. The audit therefore accepts valid times
from 1 January 2025 through the one-day next-year boundary spill, while still
rejecting dates beyond that physically permitted interval. It performs all
inexpensive structural/time/label/pairing checks before scanning the large
feature arrays and writes every issue found in the failing stage to
`data_audit/q_core_data_audit_failed.json`.

If all data builds completed but the audit failed, do not delete or rebuild the
versioned datasets. After fixing and syncing the audit code, resume from the
existing datasets with the same run tag:

```bash
cd /public/home/putianshu/vis_mlp/ifs_baseline
RUN_TAG=qcore_pangu2025_rerun_20260629 \
PANGU2025_STATION_FILE=/public/home/putianshu/vis_mlp/ifs_baseline/pangu_station/pangu_station_2025_lead24h.nc \
RESUME_FROM_AUDIT=1 \
bash submit_q_core_fair_experiment.sh
```

This mode verifies that every expected S1/S2 array, metadata file, and build
config is present, submits a fresh audit with no stale Slurm dependency, and
then recreates only the `audit -> S1 -> four S2 -> paired evaluation` chain.
It writes a timestamped resume manifest so the original failed submission
record is preserved. It refuses existing checkpoints unless
`ALLOW_EXISTING_RUN=1` is deliberately supplied.

The final evaluation always uses each source-specific S2 checkpoint with
`argmax`. It first saves the ordinary per-source predictions, then recomputes
all reported comparison metrics on the four-source paired test intersection.
Uncertainty for source-minus-Pangu differences is obtained by paired bootstrap
resampling of UTC valid dates, preserving within-day spatial and temporal
dependence. Use `q_core_paired_common_metrics.csv` and
`q_core_paired_deltas_vs_pangu2025.csv` as the common-input result tables;
`overall_metrics.csv` retains unpaired full-source diagnostics and should not
be used for source attribution.

### Corrected canonical-station rerun (fair + best effort)

When only the Pangu station coordinates were wrong, do not rebuild the already
completed q-core S1, Tianji, IFS, or ERA5 datasets. Use
`submit_corrected_pangu2025_experiments.sh`. It first runs a compute-node
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
