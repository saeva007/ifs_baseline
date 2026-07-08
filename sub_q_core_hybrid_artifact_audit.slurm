#!/bin/bash
#SBATCH -J qcore_artifacts
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -D /public/home/putianshu/vis_mlp/ifs_baseline
#SBATCH -o /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/ifs_baseline/logs/%j_%x.err

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
SEEDS="${SEEDS:-42:2025:20260702}"
MASKS="${MASKS:-}"
TRAIN_MASKS="${TRAIN_MASKS:-}"
GROUP_PROFILE="${GROUP_PROFILE:-mtw}"
HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-}"
RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-}"
S1_RUN_TAG="${S1_RUN_TAG:-}"
REUSE_COUPLED_MTW_RUN_TAG="${REUSE_COUPLED_MTW_RUN_TAG:-}"
REUSE_COUPLED_MTW_HYBRID_ROOT="${REUSE_COUPLED_MTW_HYBRID_ROOT:-}"
S1_DATA_DIR="${S1_DATA_DIR:?S1_DATA_DIR is required}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-12000}"
S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-40000}"
REUSED_S2_A_STEPS="${REUSED_LOWVIS_RNN_S2_A_STEPS:-12000}"
REUSED_S2_B_STEPS="${REUSED_LOWVIS_RNN_S2_B_STEPS:-40000}"
ALLOW_SMOKE="${ALLOW_SMOKE:-0}"

case "${GROUP_PROFILE}" in
  mtw)
    MASKS="${MASKS:-000:001:010:011:100:101:110:111}"
    ;;
  mt2pw)
    MASKS="${MASKS:-0000:0001:0010:0011:0100:0101:0110:0111:1000:1001:1010:1011:1100:1101:1110:1111}"
    TRAIN_MASKS="${TRAIN_MASKS:-0010:0011:0100:0101:1010:1011:1100:1101}"
    ;;
  *)
    echo "ERROR: unsupported GROUP_PROFILE=${GROUP_PROFILE}" >&2
    exit 2
    ;;
esac

mkdir -p "${BASELINE_DIR}/logs" "${EVAL_ROOT}/artifact_audit"
# This verifier reads JSON/CSV paths only.  Use the stable data-analysis
# Python without importing Torch/HIP on the CPU audit partition.
source "${BASELINE_DIR}/activate_torch_runtime.sh"
CMD=(python "${BASELINE_DIR}/verify_q_core_hybrid_artifacts.py" \
  --run-tag "${RUN_TAG}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --hybrid-data-root "${HYBRID_DATA_ROOT}" \
  --s1-data-dir "${S1_DATA_DIR}" \
  --out-dir "${EVAL_ROOT}/artifact_audit" \
  --group-profile "${GROUP_PROFILE}" \
  --seeds "${SEEDS}" \
  --masks "${MASKS}" \
  --s2-a-steps "${S2_A_STEPS}" \
  --s2-b-steps "${S2_B_STEPS}" \
  --reused-s2-a-steps "${REUSED_S2_A_STEPS}" \
  --reused-s2-b-steps "${REUSED_S2_B_STEPS}")
if [[ -n "${S1_RUN_TAG}" ]]; then
  CMD+=(--s1-run-tag "${S1_RUN_TAG}")
fi
if [[ -n "${HYBRID_DATASET_PREFIX}" ]]; then
  CMD+=(--dataset-prefix "${HYBRID_DATASET_PREFIX}")
fi
if [[ -n "${RUN_MASK_PREFIX}" ]]; then
  CMD+=(--run-mask-prefix "${RUN_MASK_PREFIX}")
fi
if [[ -n "${TRAIN_MASKS}" ]]; then
  CMD+=(--train-masks "${TRAIN_MASKS}")
fi
if [[ -n "${REUSE_COUPLED_MTW_RUN_TAG}" ]]; then
  CMD+=(--reuse-coupled-mtw-run-tag "${REUSE_COUPLED_MTW_RUN_TAG}")
fi
if [[ -n "${REUSE_COUPLED_MTW_HYBRID_ROOT}" ]]; then
  CMD+=(--reuse-coupled-mtw-hybrid-root "${REUSE_COUPLED_MTW_HYBRID_ROOT}")
fi
if [[ "${ALLOW_SMOKE}" == "1" ]]; then
  CMD+=(--allow-smoke)
fi
"${CMD[@]}"
