#!/bin/bash
#SBATCH -J ifs_data_process
#SBATCH -p kshcexclu04          # 使用你指定的队列节点
#SBATCH -N 1                    # 数据处理只需单节点
#SBATCH --ntasks-per-node=1     # 单个任务
#SBATCH --cpus-per-task=32      # 充分利用 32 个 CPU 核心
#SBATCH --mem=0                 # 0 表示申请节点上的所有可用内存（防止数据量大时 OOM）
#SBATCH -t 48:00:00             # 数据处理时间限制
#SBATCH -o logs/process_%j.out
#SBATCH -e logs/process_%j.err
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

# 3. 脚本路径
# DATA_SOURCE=s1_overlap -> derive overlap S1 dataset from the full 27-dyn S1 dataset
# DATA_SOURCE=tianji     -> build Tianji-input overlap dataset with the same split and layout
# DATA_SOURCE=ifs        -> build IFS-input overlap dataset (default)
DATA_SOURCE="${DATA_SOURCE:-ifs}"
DATA_EXTRA_ARGS="${DATA_EXTRA_ARGS:-}"

case "${DATA_SOURCE}" in
    s1_overlap)
        SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_s1_pm10_overlap_from_full.py"
        PY_ARGS=()
        ;;
    ifs)
        SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_dataset_ifs_overlap_12h_fast.py"
        PY_ARGS=(--year 2025 --window 12 --step 1)
        ;;
    tianji)
        SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_dataset_tianji_overlap_12h.py"
        PY_ARGS=(--window 12 --step 1)
        ;;
    *)
        echo "Unknown DATA_SOURCE=${DATA_SOURCE} (use s1_overlap | tianji | ifs)" >&2
        exit 1
        ;;
esac

# shellcheck disable=SC2206
EXTRA_TOKENS=( ${DATA_EXTRA_ARGS} )

# 4. 执行数据处理
echo "Running Python script: ${SCRIPT_PATH}"
echo "DATA_SOURCE=${DATA_SOURCE}"
echo "PY_ARGS=${PY_ARGS[*]:-} ${EXTRA_TOKENS[*]:-}"
python "${SCRIPT_PATH}" "${PY_ARGS[@]}" "${EXTRA_TOKENS[@]}"

echo "Job finished at: $(date)"
