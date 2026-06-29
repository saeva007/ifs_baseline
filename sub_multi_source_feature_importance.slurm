#!/bin/bash
#SBATCH -J src_feat_imp
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
SCRIPT="${SCRIPT:-${BASELINE_DIR}/analyze_multi_source_feature_importance.py}"
FEATURE_IMPORTANCE_OUT_DIR="${FEATURE_IMPORTANCE_OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/multi_source_feature_importance_source_full}"

TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full}"
T2ND_DATA_DIR="${T2ND_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_source_full}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_pangu2025_12h_pm10_pm25_source_full}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_source_full}"

TIANJI_CKPT="${TIANJI_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_tianji_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
T2ND_CKPT="${T2ND_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_T2ND_rh2m_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
IFS_CKPT="${IFS_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_ifs_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
PANGU_CKPT="${PANGU_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_pangu2025_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"
ERA5_CKPT="${ERA5_CKPT:-${CKPT_DIR}/exp_overlap_static_rnn_s2_era5_2025_source_full_pm10_pm25_S2_PhaseB_best_score.pt}"

SOURCES="${SOURCES:-tianji=${TIANJI_DATA_DIR}|${TIANJI_CKPT}|AUTO|Tianji;T2ND_rh2m_source_full=${T2ND_DATA_DIR}|${T2ND_CKPT}|AUTO|Tianji T2ND;ifs=${IFS_DATA_DIR}|${IFS_CKPT}|AUTO|IFS;pangu2025_source_full=${PANGU_DATA_DIR}|${PANGU_CKPT}|AUTO|Pangu;era5_2025_source_full=${ERA5_DATA_DIR}|${ERA5_CKPT}|AUTO|ERA5}"

DEVICE="${DEVICE:-auto}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
MIN_LOW_VIS="${MIN_LOW_VIS:-200}"
REPEATS="${REPEATS:-5}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
MODES="${MODES:-marginal,stratified}"
GROUP_SCOPE="${GROUP_SCOPE:-all}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
MAX_GROUPS="${MAX_GROUPS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${BASELINE_DIR}/logs" "${FEATURE_IMPORTANCE_OUT_DIR}"

CMD=(
  python "${SCRIPT}"
  --sources "${SOURCES}"
  --out_dir "${FEATURE_IMPORTANCE_OUT_DIR}"
  --reference_source tianji
  --static_rnn_train_dir "${TRAIN_DIR}"
  --device "${DEVICE}"
  --sample_size "${SAMPLE_SIZE}"
  --min_low_vis "${MIN_LOW_VIS}"
  --repeats "${REPEATS}"
  --bootstrap_iters "${BOOTSTRAP_ITERS}"
  --modes "${MODES}"
  --group_scope "${GROUP_SCOPE}"
  --limit_rows "${LIMIT_ROWS}"
  --max_groups "${MAX_GROUPS}"
)

if [[ "${ALLOW_LEGACY_TIME_ALIGNMENT:-0}" == "1" ]]; then
  CMD+=(--allow_legacy_time_alignment)
fi
if [[ "${NO_PLOT:-0}" == "1" ]]; then
  CMD+=(--no_plot)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGV=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Multi-source feature importance started at: $(date)"
echo "Output directory: ${FEATURE_IMPORTANCE_OUT_DIR}"
echo "Group scope: ${GROUP_SCOPE}; max groups: ${MAX_GROUPS} (0 means all)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Finished at: $(date)"
