#!/bin/bash
#SBATCH -J pangu_station_idw
#SBATCH --partition=kshcexclu04
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

echo "Pangu station interpolation started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
WORK_DIR="${WORK_DIR:-${BASE}}"
cd "${BASELINE_DIR}"
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-${BASELINE_DIR}/interpolate_pangu_to_stations.py}"
INPUT_DIR="${INPUT_DIR:-${BASE}/pangu_2025_china_chunks_lead12_23h}"
INPUT_GLOB="${INPUT_GLOB:-}"
INPUT_FILES="${INPUT_FILES:-}"
STATION_FILE="${STATION_FILE:-}"
TARGET_FILE="${TARGET_FILE:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
YEAR="${YEAR:-2025}"
OUT_FILE="${OUT_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h.nc}"
K="${K:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --input_dir "${INPUT_DIR}"
    --target_file "${TARGET_FILE}"
    --year "${YEAR}"
    --out_file "${OUT_FILE}"
    --k "${K}"
)
if [[ -n "${STATION_FILE}" ]]; then
    CMD+=(--station_file "${STATION_FILE}")
else
    CMD+=(--station_file "")
fi
if [[ -n "${INPUT_GLOB}" ]]; then
    CMD+=(--input_glob "${INPUT_GLOB}")
fi
if [[ -n "${INPUT_FILES}" ]]; then
    CMD+=(--input_files "${INPUT_FILES}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Pangu station interpolation finished at: $(date)"
