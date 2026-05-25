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
- Legacy PMST S1 trainer: `train_PMST_s1_overlap_baseline.py`
- Legacy PMST S2 Tianji trainer: `train_PMST_overlap_baseline_s2.py`
- Legacy PMST S2 IFS trainer: `train_PMST_overlap_baseline_s2_fast.py`
- Paired evaluator: `test_PMST_overlap_forecast_source_s2.py`

Each data-build path has its own Slurm entry point. Keep
`sub_ifs_data.slurm` for IFS-input data only.

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

Optional explicit paths:

```bash
sbatch --export=ALL,SOURCE_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25,OUT_DIR=/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap sub_s1_overlap_data.slurm
```

Do not use `--merge_train_val` for the paper experiment.

### 2. Train The Overlap S1 Checkpoint

```bash
sbatch --export=ALL,EXPERIMENT=s1_overlap sub_ifs_overlap_baseline.slurm
```

The default `MODEL_ARCH=static_rnn` trains
`exp_overlap_static_rnn_s1_pm10_pm25_S1_best_score.pt`. The Slurm launcher uses
the same direct main-trainer path and stable knobs as
`sub_static_rnn_lowvis_main.slurm`: GRU, mean pooling, one RNN layer,
5 nodes x 4 DCU, `LOWVIS_RNN_BATCH_SIZE=512`, `LOWVIS_RNN_GRAD_ACCUM=2`,
`LOWVIS_RNN_NUM_WORKERS=0`, and recall/CSI validation selection unless
overridden. The trainer requires explicit `X_train/y_train` and `X_val/y_val`,
and it fails if the row layout is not `27 dyn + 36 FE`. Use `MODEL_ARCH=pmst`
only for legacy PMST audits.

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
