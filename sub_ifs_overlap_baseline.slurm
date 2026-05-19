#!/bin/bash
#SBATCH -J pmst_ifs_overlap
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive
#
# IFS / Tianji overlap 基线训练，环境与通信配置对齐完美运行版本。

set -euo pipefail

echo "Job started at: $(date)"
echo "EXPERIMENT=${EXPERIMENT:-s1_overlap}"
echo "MODEL_ARCH=${MODEL_ARCH:-static_rnn}"

MAX_STABLE_NNODES=10
if [ "${SLURM_NNODES}" -gt "${MAX_STABLE_NNODES}" ]; then
    echo "Refusing to launch on ${SLURM_NNODES} nodes"
    exit 1
fi

BASELINE_DIR="/public/home/putianshu/vis_mlp/ifs_baseline"
SCRIPT_DIR="${BASELINE_DIR}"
STATIC_RNN_TRAIN_DIR="${STATIC_RNN_TRAIN_DIR:-/public/home/putianshu/vis_mlp/train}"
STATIC_RNN_TRAIN_SCRIPT="${STATIC_RNN_TRAIN_SCRIPT:-${STATIC_RNN_TRAIN_DIR}/train_static_rnn_lowvis.py}"
LAUNCH_DIR="${SCRIPT_DIR}"

EXPERIMENT="${EXPERIMENT:-s1_overlap}"
MODEL_ARCH="${MODEL_ARCH:-static_rnn}"
TORCHRUN_EXTRA_ARGS="${TORCHRUN_EXTRA_ARGS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
LOWVIS_RNN_EXTRA_ARGS="${LOWVIS_RNN_EXTRA_ARGS:-}"

export LOWVIS_RNN_S1_STEPS="${LOWVIS_RNN_S1_STEPS:-15000}"
export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-8000}"
export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-22000}"
export LOWVIS_RNN_VAL_INTERVAL="${LOWVIS_RNN_VAL_INTERVAL:-500}"
export LOWVIS_RNN_BATCH_SIZE="${LOWVIS_RNN_BATCH_SIZE:-512}"
export LOWVIS_RNN_GRAD_ACCUM="${LOWVIS_RNN_GRAD_ACCUM:-2}"
export LOWVIS_RNN_NUM_WORKERS="${LOWVIS_RNN_NUM_WORKERS:-0}"
export LOWVIS_RNN_PATIENCE="${LOWVIS_RNN_PATIENCE:-10}"

OVERLAP_CKPT_DIR="${OVERLAP_CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
OVERLAP_S1_DATA_DIR="${OVERLAP_S1_DATA_DIR:-${BASELINE_DIR}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap}"
OVERLAP_TIANJI_S2_DATA_DIR="${OVERLAP_TIANJI_S2_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_baseline}"
OVERLAP_IFS_S2_DATA_DIR="${OVERLAP_IFS_S2_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_baseline}"

case "${EXPERIMENT}" in
    s1_overlap)
        if [ "${MODEL_ARCH}" = "static_rnn" ]; then
            TRAIN_SCRIPT="${STATIC_RNN_TRAIN_SCRIPT}"
            LAUNCH_DIR="${STATIC_RNN_TRAIN_DIR}"
            LOWVIS_RNN_MODE_EFFECTIVE="s1"
            LOWVIS_RNN_RUN_ID="${LOWVIS_RNN_RUN_ID:-exp_overlap_static_rnn_s1_pm10_pm25}"
            STATIC_RNN_S2_DATA_DIR="${OVERLAP_S2_DATA_DIR:-${OVERLAP_TIANJI_S2_DATA_DIR}}"
        else
            TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_s1_overlap_baseline.py"
            PY_ARGS=()
        fi
        ;;
    s2_tianji)
        if [ "${MODEL_ARCH}" = "static_rnn" ]; then
            TRAIN_SCRIPT="${STATIC_RNN_TRAIN_SCRIPT}"
            LAUNCH_DIR="${STATIC_RNN_TRAIN_DIR}"
            LOWVIS_RNN_MODE_EFFECTIVE="s2"
            LOWVIS_RNN_RUN_ID="${LOWVIS_RNN_RUN_ID:-exp_overlap_static_rnn_s2_tianji_pm10_pm25}"
            STATIC_RNN_S2_DATA_DIR="${OVERLAP_S2_DATA_DIR:-${OVERLAP_TIANJI_S2_DATA_DIR}}"
        else
            TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_overlap_baseline_s2.py"
            PY_ARGS=(--data_source tianji)
        fi
        ;;
    s2_ifs)
        if [ "${MODEL_ARCH}" = "static_rnn" ]; then
            TRAIN_SCRIPT="${STATIC_RNN_TRAIN_SCRIPT}"
            LAUNCH_DIR="${STATIC_RNN_TRAIN_DIR}"
            LOWVIS_RNN_MODE_EFFECTIVE="s2"
            LOWVIS_RNN_RUN_ID="${LOWVIS_RNN_RUN_ID:-exp_overlap_static_rnn_s2_ifs_pm10_pm25}"
            STATIC_RNN_S2_DATA_DIR="${OVERLAP_S2_DATA_DIR:-${OVERLAP_IFS_S2_DATA_DIR}}"
        else
            TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_overlap_baseline_s2_fast.py"
            PY_ARGS=(--data_source ifs)
        fi
        ;;
    *)
        echo "Unknown EXPERIMENT=${EXPERIMENT} (use s1_overlap | s2_tianji | s2_ifs)"
        exit 1
        ;;
esac

if [ "${MODEL_ARCH}" = "static_rnn" ]; then
    STATIC_RNN_PRETRAINED_CKPT="${OVERLAP_STATIC_RNN_PRETRAINED_CKPT:-}"
    if [ "${LOWVIS_RNN_MODE_EFFECTIVE}" = "s2" ] && [ -z "${STATIC_RNN_PRETRAINED_CKPT}" ]; then
        DEFAULT_STATIC_RNN_S1_CKPT="${OVERLAP_CKPT_DIR}/exp_overlap_static_rnn_s1_pm10_pm25_S1_best_score.pt"
        if [ -f "${DEFAULT_STATIC_RNN_S1_CKPT}" ]; then
            STATIC_RNN_PRETRAINED_CKPT="${DEFAULT_STATIC_RNN_S1_CKPT}"
        fi
    fi

    PY_ARGS=(
        --mode "${LOWVIS_RNN_MODE_EFFECTIVE}"
        --encoder gru
        --run-id "${LOWVIS_RNN_RUN_ID}"
        --base-path /public/home/putianshu/vis_mlp
        --s1-data-dir "${OVERLAP_S1_DATA_DIR}"
        --s2-data-dir "${STATIC_RNN_S2_DATA_DIR}"
        --ckpt-dir "${OVERLAP_CKPT_DIR}"
        --pooling mean
        --rnn-layers 1
        --hidden-dim 256
        --static-hidden-dim 96
        --fusion-hidden-dim 256
        --dropout 0.2
        --selection-metric recall_csi
        --s1-steps "${LOWVIS_RNN_S1_STEPS}"
        --s2-phase-a-steps "${LOWVIS_RNN_S2_A_STEPS}"
        --s2-phase-b-steps "${LOWVIS_RNN_S2_B_STEPS}"
        --val-interval "${LOWVIS_RNN_VAL_INTERVAL}"
        --batch-size "${LOWVIS_RNN_BATCH_SIZE}"
        --grad-accum "${LOWVIS_RNN_GRAD_ACCUM}"
        --num-workers "${LOWVIS_RNN_NUM_WORKERS}"
        --patience "${LOWVIS_RNN_PATIENCE}"
    )
    if [ -n "${STATIC_RNN_PRETRAINED_CKPT}" ]; then
        PY_ARGS+=(--pretrained-ckpt "${STATIC_RNN_PRETRAINED_CKPT}")
    fi
fi

# shellcheck disable=SC2206
EXTRA_TOKENS=( ${LOWVIS_RNN_EXTRA_ARGS} ${TRAIN_EXTRA_ARGS} )

# ==========================================
# 1. Environment（与 sub_s1.slurm 一致）
# ==========================================
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 计算节点上的 srun 步常不继承登录式 PATH；用绝对路径调用 torchrun，避免 “command not found”。
TORCHRUN_BIN="$(command -v torchrun)"
# 【已修复：添加空格 || [ 】
if [ -z "${TORCHRUN_BIN}" ] || [ ! -x "${TORCHRUN_BIN}" ]; then
    echo "ERROR: torchrun not found after conda activate torch (PATH=${PATH:-})" >&2
    exit 1
fi
echo "TORCHRUN_BIN=${TORCHRUN_BIN}"

# ==========================================
# 2. RCCL / NCCL Communication (求生模式)
# ==========================================
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# ==========================================
# 3. Node Discovery
# ==========================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# 使用 /bin/bash 绝对路径，避免计算节点丢失环境
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" /bin/bash -c '
  for c in /sbin/ip /usr/sbin/ip; do
    if [ -x "$c" ]; then
      v=$("$c" -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
      if [ -n "$v" ]; then echo "$v"; exit 0; fi
    fi
  done
  if command -v ip >/dev/null 2>&1; then
    v=$(ip -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
    # 【已修复：添加空格 if[ 】
    if [ -n "$v" ]; then echo "$v"; exit 0; fi
  fi
  h=$(hostname -I 2>/dev/null | awk "{print \$1}");[ -n "$h" ] && echo "$h" && exit 0
  exit 1
' 2>/dev/null | head -1 | tr -d "\r") || true

if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=${head_node}
fi

BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
RDZV_PORT=${BASE_PORT}
DIST_STORE_PORT=$(expr ${BASE_PORT} + 1)

export MASTER_ADDR
export DIST_STORE_PORT

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}"
echo "DIST_STORE_PORT : ${DIST_STORE_PORT}"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "TORCHRUN_BIN    : ${TORCHRUN_BIN}"
echo "LAUNCH_DIR      : ${LAUNCH_DIR}"
echo "TRAIN_SCRIPT    : ${TRAIN_SCRIPT}"
echo "PY_ARGS         : ${PY_ARGS[*]:-} ${EXTRA_TOKENS[*]:-}"
if [ "${MODEL_ARCH}" = "static_rnn" ]; then
    echo "RUN_ID          : ${LOWVIS_RNN_RUN_ID}"
    echo "S1_STEPS        : ${LOWVIS_RNN_S1_STEPS}"
    echo "S2_A_STEPS      : ${LOWVIS_RNN_S2_A_STEPS}"
    echo "S2_B_STEPS      : ${LOWVIS_RNN_S2_B_STEPS}"
    echo "BATCH/ACCUM     : ${LOWVIS_RNN_BATCH_SIZE}/${LOWVIS_RNN_GRAD_ACCUM}"
    echo "NUM_WORKERS     : ${LOWVIS_RNN_NUM_WORKERS}"
fi

# ==========================================
# 4. Clean /dev/shm
# ==========================================
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     /bin/bash -c "rm -rf /dev/shm/torch_* /dev/shm/*X_train* /dev/shm/*y_train* /dev/shm/*static_rnn_lowvis* /dev/shm/*lowvis_rnn* 2>/dev/null; echo \"\$(hostname) SHM Cleaned\""

# ==========================================
# 5. Launch Training
# ==========================================
echo "Starting ${EXPERIMENT} on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."

# --export=ALL 保证节点间的子进程环境变量（如 LD_LIBRARY_PATH、NCCL_ 等）完全一致
srun --export=ALL \
     --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     --kill-on-bad-exit=1 \
     /bin/bash -c "
         cd \"${LAUNCH_DIR}\" || exit 1
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\${MIOPEN_USER_DB_PATH}
         mkdir -p \${MIOPEN_USER_DB_PATH}

         \"${TORCHRUN_BIN}\" \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=4 \
             --node_rank=\${SLURM_NODEID} \
             --rdzv_backend=static \
             --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --max_restarts=0 \
             ${TORCHRUN_EXTRA_ARGS} \
             \"${TRAIN_SCRIPT}\" \
             ${PY_ARGS[*]:-} \
             ${EXTRA_TOKENS[*]:-}
     "

echo "Job finished at: $(date)"
