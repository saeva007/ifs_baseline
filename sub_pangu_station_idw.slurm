#!/bin/bash
#SBATCH -J pangu_station_idw
#SBATCH --partition=cpu_parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /data2/share/chenxi/PuTS/mlp/ifs_baseline
#SBATCH -o /data2/share/chenxi/PuTS/mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

echo "Pangu station interpolation started at: $(date)"
echo "Node: $(hostname)"

source /lustre/chenxi/yliang/miniforge3/bin/activate
module purge
module load compiler/dtk/23.10
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
source /lustre/chenxi/yliang/work/cast3/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

WORK_DIR="${WORK_DIR:-/data2/share/chenxi/PuTS/mlp}"
cd "${WORK_DIR}/ifs_baseline"
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-${WORK_DIR}/ifs_baseline/interpolate_pangu_to_stations.py}"
INPUT_DIR="${INPUT_DIR:-${WORK_DIR}/pangu_2025_china_chunks}"
INPUT_GLOB="${INPUT_GLOB:-}"
INPUT_FILES="${INPUT_FILES:-}"
TARGET_FILE="${TARGET_FILE:-${WORK_DIR}/tianji_auto_station/merged_final_all_vars.nc}"
YEAR="${YEAR:-2025}"
OUT_FILE="${OUT_FILE:-${WORK_DIR}/ifs_baseline/pangu_station/pangu_station_2025_lead24h.nc}"
K="${K:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(python "${SCRIPT_PATH}" --input_dir "${INPUT_DIR}" --target_file "${TARGET_FILE}" --year "${YEAR}" --out_file "${OUT_FILE}" --k "${K}")
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
