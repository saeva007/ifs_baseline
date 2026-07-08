#!/bin/bash
#SBATCH -J qcore_hybrid_eval
#SBATCH -p kshcexclu04
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

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
TRAIN_DIR="${TRAIN_DIR:-${BASE}/train}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
MODE="${MODE:-evaluate_seed}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-42:2025:20260702}"
MASKS="${MASKS:-}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
GROUP_PROFILE="${GROUP_PROFILE:-mtw}"
HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-}"
RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-}"
SOURCE_PREFIX="${SOURCE_PREFIX:-}"
REUSE_COUPLED_MTW_RUN_TAG="${REUSE_COUPLED_MTW_RUN_TAG:-}"
REUSE_COUPLED_MTW_HYBRID_ROOT="${REUSE_COUPLED_MTW_HYBRID_ROOT:-}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EVAL_ROOT}/analysis}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260702}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
OBS_ROOT="${OBS_ROOT:-}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-}"
REQUIRE_ALE="${REQUIRE_ALE:-0}"
INCLUDE_COMMON_CORE="${INCLUDE_COMMON_CORE:-1}"
COMMON_CORE_DATA_DIR="${COMMON_CORE_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_common_core}"

case "${GROUP_PROFILE}" in
    mtw)
        MASKS="${MASKS:-000:001:010:011:100:101:110:111}"
        HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-mtw}"
        RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-mtw}"
        SOURCE_PREFIX="${SOURCE_PREFIX:-qcore_hybrid_}"
        ;;
    mt2pw)
        MASKS="${MASKS:-0000:0001:0010:0011:0100:0101:0110:0111:1000:1001:1010:1011:1100:1101:1110:1111}"
        HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-mt2pw}"
        RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-mt2pw}"
        SOURCE_PREFIX="${SOURCE_PREFIX:-qcore_hybrid_mt2pw_}"
        ;;
    *)
        echo "ERROR: unsupported GROUP_PROFILE=${GROUP_PROFILE}" >&2
        exit 2
        ;;
esac

coupled_mtw_mask() {
    local mask="$1"
    if [[ "${GROUP_PROFILE}" == "mt2pw" && "${#mask}" -eq 4 && "${mask:1:1}" == "${mask:2:1}" ]]; then
        echo "${mask:0:1}${mask:1:1}${mask:3:1}"
    fi
}

mkdir -p "${BASELINE_DIR}/logs" "${EVAL_ROOT}"
source "${BASELINE_DIR}/activate_eval_torch_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

if [[ "${MODE}" == "evaluate_seed" ]]; then
    OUT_DIR="${EVAL_ROOT}/seed_${SEED}"
    mkdir -p "${OUT_DIR}"
    EXTRA_SOURCES=""
    SOURCE_SUBSET=""
    IFS=':' read -ra MASK_ARRAY <<< "${MASKS//,/:}"
    for mask in "${MASK_ARRAY[@]}"; do
        mask="${mask//[[:space:]]/}"
        [[ -n "${mask}" ]] || continue
        tag="${SOURCE_PREFIX}${mask}"
        data_dir="${HYBRID_DATA_ROOT}/${HYBRID_DATASET_PREFIX}_${mask}"
        run_id="exp_qcore_hybrid_${RUN_TAG}_${RUN_MASK_PREFIX}${mask}_seed${SEED}_pm10_pm25"
        label="Hybrid ${GROUP_PROFILE} ${mask}"
        old_mask="$(coupled_mtw_mask "${mask}")"
        if [[ -n "${old_mask}" && -n "${REUSE_COUPLED_MTW_RUN_TAG}" ]]; then
            [[ -n "${REUSE_COUPLED_MTW_HYBRID_ROOT}" ]] || {
                echo "ERROR: REUSE_COUPLED_MTW_HYBRID_ROOT is required when reusing coupled MTW artifacts" >&2
                exit 2
            }
            data_dir="${REUSE_COUPLED_MTW_HYBRID_ROOT}/mtw_${old_mask}"
            run_id="exp_qcore_hybrid_${REUSE_COUPLED_MTW_RUN_TAG}_mtw${old_mask}_seed${SEED}_pm10_pm25"
            label="Reused MTW ${old_mask} as ${GROUP_PROFILE} ${mask}"
        fi
        ckpt="${CKPT_DIR}/${run_id}_S2_PhaseB_best_score.pt"
        spec="${tag}=${data_dir}|${ckpt}|AUTO|${label}"
        EXTRA_SOURCES="${EXTRA_SOURCES:+${EXTRA_SOURCES};}${spec}"
        SOURCE_SUBSET="${SOURCE_SUBSET:+${SOURCE_SUBSET},}${tag}"
    done
    if [[ "${INCLUDE_COMMON_CORE}" == "1" ]]; then
        common_run_id="exp_qcore_hybrid_${RUN_TAG}_tianji_common_core_seed${SEED}_pm10_pm25"
        common_ckpt="${CKPT_DIR}/${common_run_id}_S2_PhaseB_best_score.pt"
        EXTRA_SOURCES="${EXTRA_SOURCES};tianji_common_core=${COMMON_CORE_DATA_DIR}|${common_ckpt}|AUTO|Tianji common-core (+RH2M+DPD)"
        SOURCE_SUBSET="${SOURCE_SUBSET},tianji_common_core"
    fi
    CMD=(
        python "${BASELINE_DIR}/test_PMST_overlap_forecast_source_s2.py"
        --model_arch static_rnn
        --static_rnn_train_dir "${TRAIN_DIR}"
        --independent_sources
        --threshold_mode argmax
        --no_temp_scaling
        --skip_bootstrap
        --skip_ifs_forecast_baseline
        --no_figures
        --extra_sources "${EXTRA_SOURCES}"
        --source_subset "${SOURCE_SUBSET}"
        --out_dir "${OUT_DIR}"
        --device "${DEVICE}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
    )
    if [[ "${LIMIT_SAMPLES}" != "0" ]]; then
        CMD+=(--limit_samples "${LIMIT_SAMPLES}")
    fi
elif [[ "${MODE}" == "analyze" ]]; then
    CMD=(
        python "${BASELINE_DIR}/analyze_q_core_hybrid_factorial.py"
        --eval-root "${EVAL_ROOT}"
        --out-dir "${ANALYSIS_DIR}"
        --seeds "${SEEDS}"
        --masks "${MASKS}"
        --group-profile "${GROUP_PROFILE}"
        --source-prefix "${SOURCE_PREFIX}"
        --bootstrap-iters "${BOOTSTRAP_ITERS}"
        --bootstrap-seed "${BOOTSTRAP_SEED}"
        --bootstrap-max-rows "${BOOTSTRAP_MAX_ROWS}"
        --paper-eval-dir "${BASE}/paper_eval"
    )
    if [[ -n "${OBS_ROOT}" ]]; then
        CMD+=(--obs-root "${OBS_ROOT}")
    fi
    if [[ -n "${ERA5_DATA_DIR}" ]]; then
        CMD+=(--era5-data-dir "${ERA5_DATA_DIR}")
    fi
    if [[ "${REQUIRE_ALE}" == "1" ]]; then
        CMD+=(--require-ale)
    fi
else
    echo "ERROR: MODE must be evaluate_seed or analyze, got ${MODE}" >&2
    exit 2
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
