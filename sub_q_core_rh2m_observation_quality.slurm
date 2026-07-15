#!/bin/bash
#SBATCH -J qcore_rh2m_qa
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
FEATURE_SET="${FEATURE_SET:-common_core}"
TIANJI_RH2M_DATA_DIR="${TIANJI_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_${FEATURE_SET}}"
T2ND_RH2M_DATA_DIR="${T2ND_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_${FEATURE_SET}}"
ERA5_RH2M_DATA_DIR="${ERA5_RH2M_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_era5_2025_12h_pm10_pm25_${FEATURE_SET}}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_rh2m_observation_quality/${RUN_TAG}/analysis}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260715}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jarvis226/miniconda3/envs/torch/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/analyze_multi_source_rh2m_quality.py}"

mkdir -p "${BASELINE_DIR}/logs" "${OUT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: analysis Python is missing: ${PYTHON_BIN}" >&2; exit 2; }
[[ -s "${SCRIPT_PATH}" ]] || { echo "ERROR: RH2M analysis script is missing: ${SCRIPT_PATH}" >&2; exit 2; }

SOURCES="tianji=${TIANJI_RH2M_DATA_DIR}|Tianji product RH2M|2025;T2ND=${T2ND_RH2M_DATA_DIR}|Tianji T2ND raw RH2M|2025;era5_2025=${ERA5_RH2M_DATA_DIR}|ERA5 reference analysis|2025"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --sources "${SOURCES}"
  --obs_root "${OBS_ROOT}"
  --out_dir "${OUT_DIR}"
  --bootstrap_iters "${BOOTSTRAP_ITERS}"
  --bootstrap_seed "${BOOTSTRAP_SEED}"
)
if [[ "${LIMIT_SAMPLES}" != "0" ]]; then
  CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi

echo "RH2M observation-quality diagnosis (training jobs: 0)"
echo "Pangu RH2M proxy: excluded"
echo "T2ND role: raw Tianji-model RH2M override, not Tianji product RH2M"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
