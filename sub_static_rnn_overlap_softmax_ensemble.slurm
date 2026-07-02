#!/bin/bash
#SBATCH -J overlap_ens
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/overlap_ens_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/overlap_ens_%j.err
#SBATCH --exclusive

set -euo pipefail

echo "Overlap softmax ensemble job started at: $(date)"
echo "Node: $(hostname)"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/vis_mlp/ifs_baseline/activate_eval_torch_runtime.sh
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

BASELINE_DIR="${BASELINE_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline}"
SCRIPT_PATH="${SCRIPT_PATH:-${BASELINE_DIR}/test_static_rnn_overlap_softmax_ensemble_s2.py}"
STATIC_RNN_TRAIN_DIR="${STATIC_RNN_TRAIN_DIR:-/public/home/putianshu/vis_mlp/train}"
MODEL_ARCH="${MODEL_ARCH:-static_rnn}"
OVERLAP_CKPT_DIR="${OVERLAP_CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
OVERLAP_TIANJI_SOURCE_TAG="${OVERLAP_TIANJI_SOURCE_TAG:-tianji}"
OVERLAP_TIANJI_DATA_DIR="${OVERLAP_TIANJI_DATA_DIR:-}"
OVERLAP_IFS_DATA_DIR="${OVERLAP_IFS_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_ifs_12h_pm10_pm25_baseline}"
OUT_DIR="${OUT_DIR:-/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/overlap_softmax_ensemble}"

TIANJI_CKPT="${TIANJI_CKPT:-}"
IFS_CKPT="${IFS_CKPT:-}"
TIANJI_SCALER="${TIANJI_SCALER:-}"
IFS_SCALER="${IFS_SCALER:-}"
DEVICE="${DEVICE:-cpu}"
SOURCE_THRESHOLD_MODE="${SOURCE_THRESHOLD_MODE:-checkpoint}"
ENSEMBLE_THRESHOLD_MODE="${ENSEMBLE_THRESHOLD_MODE:-val_search}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BOOTSTRAP="${BOOTSTRAP:-500}"
BOOTSTRAP_SIZE="${BOOTSTRAP_SIZE:-200000}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
NO_PER_SAMPLE_CSV="${NO_PER_SAMPLE_CSV:-0}"
NO_FIGURES="${NO_FIGURES:-0}"
ALLOW_LEGACY_TIME_ALIGNMENT="${ALLOW_LEGACY_TIME_ALIGNMENT:-0}"
STRICT_META="${STRICT_META:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

CMD=(
  python "${SCRIPT_PATH}"
  --model_arch "${MODEL_ARCH}"
  --static_rnn_train_dir "${STATIC_RNN_TRAIN_DIR}"
  --ckpt_dir "${OVERLAP_CKPT_DIR}"
  --tianji_source_tag "${OVERLAP_TIANJI_SOURCE_TAG}"
  --ifs_data_dir "${OVERLAP_IFS_DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --source_threshold_mode "${SOURCE_THRESHOLD_MODE}"
  --ensemble_threshold_mode "${ENSEMBLE_THRESHOLD_MODE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --bootstrap "${BOOTSTRAP}"
  --bootstrap_size "${BOOTSTRAP_SIZE}"
)

if [[ -n "${OVERLAP_TIANJI_DATA_DIR}" ]]; then
  CMD+=(--tianji_data_dir "${OVERLAP_TIANJI_DATA_DIR}")
fi
if [[ -n "${TIANJI_CKPT}" ]]; then
  CMD+=(--tianji_ckpt "${TIANJI_CKPT}")
fi
if [[ -n "${IFS_CKPT}" ]]; then
  CMD+=(--ifs_ckpt "${IFS_CKPT}")
fi
if [[ -n "${TIANJI_SCALER}" ]]; then
  CMD+=(--tianji_scaler "${TIANJI_SCALER}")
fi
if [[ -n "${IFS_SCALER}" ]]; then
  CMD+=(--ifs_scaler "${IFS_SCALER}")
fi
if [[ -n "${LIMIT_SAMPLES}" ]]; then
  CMD+=(--limit_samples "${LIMIT_SAMPLES}")
fi
if [[ "${SKIP_BOOTSTRAP}" == "1" || "${SKIP_BOOTSTRAP}" == "true" ]]; then
  CMD+=(--skip_bootstrap)
fi
if [[ "${NO_PER_SAMPLE_CSV}" == "1" || "${NO_PER_SAMPLE_CSV}" == "true" ]]; then
  CMD+=(--no_per_sample_csv)
fi
if [[ "${NO_FIGURES}" == "1" || "${NO_FIGURES}" == "true" ]]; then
  CMD+=(--no_figures)
fi
if [[ "${ALLOW_LEGACY_TIME_ALIGNMENT}" == "1" || "${ALLOW_LEGACY_TIME_ALIGNMENT}" == "true" ]]; then
  CMD+=(--allow_legacy_time_alignment)
fi
if [[ "${STRICT_META}" == "1" || "${STRICT_META}" == "true" ]]; then
  CMD+=(--strict_meta)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGV=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Overlap softmax ensemble job finished at: $(date)"
