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
MASKS="${MASKS:-000:001:010:011:100:101:110:111}"
S1_DATA_DIR="${S1_DATA_DIR:?S1_DATA_DIR is required}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-12000}"
S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-40000}"
ALLOW_SMOKE="${ALLOW_SMOKE:-0}"

mkdir -p "${BASELINE_DIR}/logs" "${EVAL_ROOT}/artifact_audit"
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
CMD=(python "${BASELINE_DIR}/verify_q_core_hybrid_artifacts.py" \
  --run-tag "${RUN_TAG}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --hybrid-data-root "${HYBRID_DATA_ROOT}" \
  --s1-data-dir "${S1_DATA_DIR}" \
  --out-dir "${EVAL_ROOT}/artifact_audit" \
  --seeds "${SEEDS}" \
  --masks "${MASKS}" \
  --s2-a-steps "${S2_A_STEPS}" \
  --s2-b-steps "${S2_B_STEPS}")
if [[ "${ALLOW_SMOKE}" == "1" ]]; then
  CMD+=(--allow-smoke)
fi
"${CMD[@]}"
