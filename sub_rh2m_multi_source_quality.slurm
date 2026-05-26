#!/bin/bash
#SBATCH -J rh2m_quality
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "RH2M multi-source quality analysis started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASE}/paper_eval/analyze_multi_source_rh2m_quality.py}"
FEATURE_SET="${FEATURE_SET:-common_core}"
OBS_ROOT="${OBS_ROOT:-${BASE}/auto_station}"
OUT_DIR="${OUT_DIR:-${BASE}/paper_eval_results_pm10_pm25_journal/rh2m_multi_source_quality_${FEATURE_SET}}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SOURCES_DEFAULT="tianji=${BASE}/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_${FEATURE_SET}|Tianji product|2025;T2ND=${BASE}/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_T2ND_rh2m_${FEATURE_SET}|T2ND raw RH2M|2025;ifs=${BASE}/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_${FEATURE_SET}|IFS|2025;era5_2025=${BASE}/ifs_baseline/ml_dataset_overlap_era5_2025_12h_pm10_pm25_${FEATURE_SET}|ERA5 2025|2025;pangu2021=${BASE}/ifs_baseline/ml_dataset_overlap_pangu2021_12h_pm10_pm25_${FEATURE_SET}|Pangu RH2M proxy|2021|RH2M proxy from 1000 hPa humidity"
SOURCES="${SOURCES:-${SOURCES_DEFAULT}}"

mkdir -p "${BASE}/ifs_baseline/logs"

CMD=(python "${SCRIPT_PATH}" --sources "${SOURCES}" --obs_root "${OBS_ROOT}" --out_dir "${OUT_DIR}")
if [[ -n "${LIMIT_SAMPLES}" ]]; then
    CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "RH2M multi-source quality analysis finished at: $(date)"
