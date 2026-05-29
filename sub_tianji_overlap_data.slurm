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
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
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
RH2M_OVERRIDE_FILE="${RH2M_OVERRIDE_FILE:-}"
RH2M_OVERRIDE_VAR="${RH2M_OVERRIDE_VAR:-}"
RH2M_SOURCE_TAG="${RH2M_SOURCE_TAG:-}"
RH2M_TIME_TOLERANCE_MINUTES="${RH2M_TIME_TOLERANCE_MINUTES:-}"
RH2M_OVERRIDE_ALLOW_MISSING="${RH2M_OVERRIDE_ALLOW_MISSING:-0}"
FEATURE_SET="${FEATURE_SET:-overlap_full}"
FEATURE_SUFFIX="${FEATURE_SET}"
if [[ "${FEATURE_SET}" == compact_common_core* ]]; then
    FEATURE_SUFFIX="compact_common_core_no_rh2m"
fi
OUT_DIR="${OUT_DIR:-}"
VAL_LAST_DAYS="${VAL_LAST_DAYS:-}"
TEST_LAST_DAYS="${TEST_LAST_DAYS:-}"
GAP_HOURS="${GAP_HOURS:-}"
CHUNK_WINS="${CHUNK_WINS:-}"
STAGING_DIR="${STAGING_DIR:-}"
KEEP_STAGING="${KEEP_STAGING:-0}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

if [[ -z "${OUT_DIR}" ]]; then
    if [[ -n "${RH2M_OVERRIDE_FILE}" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_${RH2M_SOURCE_TAG:-T2ND_rh2m}"
    elif [[ "${FEATURE_SET}" == "common_core" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_common_core"
    elif [[ "${FEATURE_SET}" == compact_common_core* ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_${FEATURE_SUFFIX}"
    elif [[ "${FEATURE_SET}" == "source_full" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_tianji_12h_pm10_pm25_source_full"
    fi
fi
if [[ -n "${RH2M_OVERRIDE_FILE}" && "${FEATURE_SET}" != "overlap_full" && "${OUT_DIR}" != *"${FEATURE_SUFFIX}" ]]; then
    OUT_DIR="${OUT_DIR}_${FEATURE_SUFFIX}"
fi
if [[ -z "${CHUNK_WINS}" && -n "${RH2M_OVERRIDE_FILE}" ]]; then
    CHUNK_WINS=64
fi

PY_ARGS=(--window "${WINDOW}" --step "${STEP}")
PY_ARGS+=(--feature_set "${FEATURE_SET}")

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
if [[ -n "${RH2M_OVERRIDE_FILE}" ]]; then
    PY_ARGS+=(--rh2m_override_file "${RH2M_OVERRIDE_FILE}")
fi
if [[ -n "${RH2M_OVERRIDE_VAR}" ]]; then
    PY_ARGS+=(--rh2m_override_var "${RH2M_OVERRIDE_VAR}")
fi
if [[ -n "${RH2M_SOURCE_TAG}" ]]; then
    PY_ARGS+=(--rh2m_source_tag "${RH2M_SOURCE_TAG}")
fi
if [[ -n "${RH2M_TIME_TOLERANCE_MINUTES}" ]]; then
    PY_ARGS+=(--rh2m_time_tolerance_minutes "${RH2M_TIME_TOLERANCE_MINUTES}")
fi
if [[ "${RH2M_OVERRIDE_ALLOW_MISSING}" == "1" || "${RH2M_OVERRIDE_ALLOW_MISSING}" == "true" ]]; then
    PY_ARGS+=(--rh2m_override_allow_missing)
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
if [[ -n "${STAGING_DIR}" ]]; then
    PY_ARGS+=(--staging_dir "${STAGING_DIR}")
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
