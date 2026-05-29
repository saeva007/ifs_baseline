#!/bin/bash
#SBATCH -J ifs_data_process
#SBATCH -p kshcexclu04          # 使用你指定的队列节点
#SBATCH -N 1                    # 数据处理只需单节点
#SBATCH --ntasks-per-node=1     # 单个任务
#SBATCH --cpus-per-task=32      # 充分利用 32 个 CPU 核心
#SBATCH --mem=0                 # 0 表示申请节点上的所有可用内存（防止数据量大时 OOM）
#SBATCH -t 48:00:00             # 数据处理时间限制
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err
#SBATCH --exclusive             # 独占节点，避免与其他任务抢占内存和 IO

set -euo pipefail

echo "Data Processing Job started at: $(date)"
echo "Node: $(hostname)"

# 1. 环境加载 (与你的训练环境对齐)
module purge
module load compiler/devtoolset/7.3.1
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

# 2. 核心性能环境变量优化
export PYTHONUNBUFFERED=1
# 限制底层线性代数库的线程数，对齐申请的 32 核心，极大加速 numpy 矩阵运算
export OMP_NUM_THREADS=32       
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

cd /public/home/putianshu/vis_mlp/ifs_baseline
mkdir -p logs

# 3. IFS overlap S2 data build entry point.
# This submit script intentionally only builds the IFS-input dataset.
SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_dataset_ifs_overlap_12h_fast.py"
YEAR="${YEAR:-2025}"
WINDOW="${WINDOW:-12}"
STEP="${STEP:-1}"
IFS_INTERP_NC="${IFS_INTERP_NC:-}"
IFS_INTERP_GLOB="${IFS_INTERP_GLOB:-}"
IFS_INTERP_DIR="${IFS_INTERP_DIR:-}"
IFS_ROOT="${IFS_ROOT:-}"
TIANJI_FILE="${TIANJI_FILE:-}"
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
WINDOW_CHUNK="${WINDOW_CHUNK:-}"
NO_AUTO_IFS_INTERP="${NO_AUTO_IFS_INTERP:-0}"
FEATURE_SET="${FEATURE_SET:-overlap_full}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

PY_ARGS=(--year "${YEAR}" --window "${WINDOW}" --step "${STEP}")
PY_ARGS+=(--feature_set "${FEATURE_SET}")

if [[ -z "${OUT_DIR}" ]]; then
    if [[ "${FEATURE_SET}" == "common_core" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_common_core"
    elif [[ "${FEATURE_SET}" == "compact_common_core" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_compact_common_core"
    elif [[ "${FEATURE_SET}" == "source_full" ]]; then
        OUT_DIR="/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full"
    fi
fi

if [[ -n "${IFS_INTERP_NC}" ]]; then
    PY_ARGS+=(--ifs_interp_nc "${IFS_INTERP_NC}")
fi
if [[ -n "${IFS_INTERP_GLOB}" ]]; then
    PY_ARGS+=(--ifs_interp_glob "${IFS_INTERP_GLOB}")
fi
if [[ -n "${IFS_INTERP_DIR}" ]]; then
    PY_ARGS+=(--ifs_interp_dir "${IFS_INTERP_DIR}")
fi
if [[ -n "${IFS_ROOT}" ]]; then
    PY_ARGS+=(--ifs_root "${IFS_ROOT}")
fi
if [[ -n "${TIANJI_FILE}" ]]; then
    PY_ARGS+=(--tianji_file "${TIANJI_FILE}")
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
if [[ -n "${WINDOW_CHUNK}" ]]; then
    PY_ARGS+=(--window_chunk "${WINDOW_CHUNK}")
fi
if [[ "${NO_AUTO_IFS_INTERP}" == "1" || "${NO_AUTO_IFS_INTERP}" == "true" ]]; then
    PY_ARGS+=(--no_auto_ifs_interp)
fi

# 4. 执行数据处理
echo "Running Python script: ${SCRIPT_PATH}"
if [[ -n "${DATA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_TOKENS=( ${DATA_EXTRA_ARGS} )
    PY_ARGS+=("${EXTRA_TOKENS[@]}")
fi

echo "PY_ARGS=${PY_ARGS[*]}"
python "${SCRIPT_PATH}" "${PY_ARGS[@]}"

echo "Job finished at: $(date)"
