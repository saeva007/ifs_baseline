#!/bin/bash
#SBATCH -J tianji_overlap_data
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "Tianji overlap data build started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_dataset_tianji_overlap_12h.py"
WINDOW="${WINDOW:-12}"
STEP="${STEP:-1}"
INPUT_FILE="${INPUT_FILE:-}"
VEG_FILE="${VEG_FILE:-}"
ORO_FILE="${ORO_FILE:-}"
PM10_FILE="${PM10_FILE:-}"
PM10_DIR="${PM10_DIR:-}"
PM25_FILE="${PM25_FILE:-}"
PM25_DIR="${PM25_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
VAL_LAST_DAYS="${VAL_LAST_DAYS:-}"
TEST_LAST_DAYS="${TEST_LAST_DAYS:-}"
GAP_HOURS="${GAP_HOURS:-}"
CHUNK_WINS="${CHUNK_WINS:-}"
KEEP_STAGING="${KEEP_STAGING:-0}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

PY_ARGS=(--window "${WINDOW}" --step "${STEP}")

if [[ -n "${INPUT_FILE}" ]]; then
    PY_ARGS+=(--input_file "${INPUT_FILE}")
fi
if [[ -n "${VEG_FILE}" ]]; then
    PY_ARGS+=(--veg_file "${VEG_FILE}")
fi
if [[ -n "${ORO_FILE}" ]]; then
    PY_ARGS+=(--oro_file "${ORO_FILE}")
fi
if [[ -n "${PM10_FILE}" ]]; then
    PY_ARGS+=(--pm10_file "${PM10_FILE}")
fi
if [[ -n "${PM10_DIR}" ]]; then
    PY_ARGS+=(--pm10_dir "${PM10_DIR}")
fi
if [[ -n "${PM25_FILE}" ]]; then
    PY_ARGS+=(--pm25_file "${PM25_FILE}")
fi
if [[ -n "${PM25_DIR}" ]]; then
    PY_ARGS+=(--pm25_dir "${PM25_DIR}")
fi
if [[ -n "${OUT_DIR}" ]]; then
    PY_ARGS+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${VAL_LAST_DAYS}" ]]; then
    PY_ARGS+=(--val_last_days "${VAL_LAST_DAYS}")
fi
if [[ -n "${TEST_LAST_DAYS}" ]]; then
    PY_ARGS+=(--test_last_days "${TEST_LAST_DAYS}")
fi
if [[ -n "${GAP_HOURS}" ]]; then
    PY_ARGS+=(--gap_hours "${GAP_HOURS}")
fi
if [[ -n "${CHUNK_WINS}" ]]; then
    PY_ARGS+=(--chunk_wins "${CHUNK_WINS}")
fi
if [[ "${KEEP_STAGING}" == "1" || "${KEEP_STAGING}" == "true" ]]; then
    PY_ARGS+=(--keep_staging)
fi

echo "Running Python script: ${SCRIPT_PATH}"
if [[ -n "${DATA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_TOKENS=( ${DATA_EXTRA_ARGS} )
    PY_ARGS+=("${EXTRA_TOKENS[@]}")
fi

echo "PY_ARGS=${PY_ARGS[*]}"
python "${SCRIPT_PATH}" "${PY_ARGS[@]}"

echo "Tianji overlap data build finished at: $(date)"
