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
SCRIPT_PATH="/public/home/putianshu/vis_mlp/ifs_baseline/build_dataset_ifs_overlap_12h_fast.py"

# 4. 执行数据处理
echo "Running Python script: ${SCRIPT_PATH}"
python "${SCRIPT_PATH}" \
    --year 2025 \
    --window 12 \
    --step 1

echo "Job finished at: $(date)"