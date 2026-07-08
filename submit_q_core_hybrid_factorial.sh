#!/bin/bash
# Submit the audited three-seed Pangu/Tianji q-core hybrid factorial chain.
#
# Required:
#   RUN_TAG=qcore_hybrid_YYYYMMDD
#   SOURCE_DATA_ROOT=/.../q_core_fair_datasets/<canonical_run_tag>
#
# Optional smoke test:
#   DRY_RUN=1 LIMIT_ROWS=2000 LIMIT_SAMPLES=2000 \
#   LOWVIS_RNN_S1_STEPS=20 LOWVIS_RNN_S2_A_STEPS=20 LOWVIS_RNN_S2_B_STEPS=40 ...

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be a fresh result tag}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT must point to the audited canonical q-core dataset root}"
SEEDS="${SEEDS:-42:2025:20260702}"
GROUP_PROFILE="${GROUP_PROFILE:-mtw}"
case "${GROUP_PROFILE}" in
    mtw)
        MASK_WIDTH=3
        DEFAULT_MASKS="000:001:010:011:100:101:110:111"
        DEFAULT_TRAIN_MASKS="${DEFAULT_MASKS}"
        DEFAULT_RUN_COMMON_CORE=1
        DEFAULT_RUN_IMPORTANCE=1
        DEFAULT_RUN_ALE=1
        HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-mtw}"
        RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-mtw}"
        SOURCE_PREFIX="${SOURCE_PREFIX:-qcore_hybrid_}"
        ;;
    mt2pw)
        MASK_WIDTH=4
        DEFAULT_MASKS="0000:0001:0010:0011:0100:0101:0110:0111:1000:1001:1010:1011:1100:1101:1110:1111"
        DEFAULT_TRAIN_MASKS="0010:0011:0100:0101:1010:1011:1100:1101"
        DEFAULT_RUN_COMMON_CORE=0
        DEFAULT_RUN_IMPORTANCE=0
        DEFAULT_RUN_ALE=0
        HYBRID_DATASET_PREFIX="${HYBRID_DATASET_PREFIX:-mt2pw}"
        RUN_MASK_PREFIX="${RUN_MASK_PREFIX:-mt2pw}"
        SOURCE_PREFIX="${SOURCE_PREFIX:-qcore_hybrid_mt2pw_}"
        ;;
    *)
        echo "ERROR: unsupported GROUP_PROFILE=${GROUP_PROFILE}" >&2
        exit 2
        ;;
esac
MASKS="${MASKS:-${DEFAULT_MASKS}}"
TRAIN_MASKS="${TRAIN_MASKS:-${DEFAULT_TRAIN_MASKS}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_COMMON_CORE="${RUN_COMMON_CORE:-${DEFAULT_RUN_COMMON_CORE}}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_IMPORTANCE="${RUN_IMPORTANCE:-${DEFAULT_RUN_IMPORTANCE}}"
RUN_ALE="${RUN_ALE:-${DEFAULT_RUN_ALE}}"
RESUME_AFTER_DATA_FAILURE="${RESUME_AFTER_DATA_FAILURE:-0}"
RESUME_EXISTING_RUN="${RESUME_EXISTING_RUN:-0}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
OBS_ROOT="${OBS_ROOT:-}"
BASE_MTW_RUN_TAG="${BASE_MTW_RUN_TAG:-}"

S1_DATA_DIR="${S1_DATA_DIR:-${SOURCE_DATA_ROOT}/s1}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${SOURCE_DATA_ROOT}/tianji}"
IFS_DATA_DIR="${IFS_DATA_DIR:-${SOURCE_DATA_ROOT}/ifs}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${SOURCE_DATA_ROOT}/pangu2025}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${SOURCE_DATA_ROOT}/era5_2025}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_hybrid_factorial/${RUN_TAG}}"
COMMON_CORE_S1_DATA_DIR="${COMMON_CORE_S1_DATA_DIR:-${BASELINE_DIR}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_common_core}"
COMMON_CORE_DATA_DIR="${COMMON_CORE_DATA_DIR:-${BASELINE_DIR}/ml_dataset_overlap_tianji_12h_pm10_pm25_common_core}"
S1_RUN_TAG="${S1_RUN_TAG:-${BASE_MTW_RUN_TAG:-${RUN_TAG}}}"
REUSE_COUPLED_MTW_RUN_TAG="${REUSE_COUPLED_MTW_RUN_TAG:-${BASE_MTW_RUN_TAG}}"
REUSE_COUPLED_MTW_HYBRID_ROOT="${REUSE_COUPLED_MTW_HYBRID_ROOT:-}"
if [[ -n "${REUSE_COUPLED_MTW_RUN_TAG}" && -z "${REUSE_COUPLED_MTW_HYBRID_ROOT}" ]]; then
    REUSE_COUPLED_MTW_HYBRID_ROOT="${BASELINE_DIR}/q_core_hybrid_datasets/${REUSE_COUPLED_MTW_RUN_TAG}"
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

case "${RUN_TAG}" in
    *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
if [[ "${SOURCE_DATA_ROOT}" == "${HYBRID_DATA_ROOT}" ]]; then
    echo "ERROR: source and hybrid data roots must differ" >&2
    exit 2
fi
if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" && "${RESUME_EXISTING_RUN}" == "1" ]]; then
    echo "ERROR: use either RESUME_AFTER_DATA_FAILURE=1 or RESUME_EXISTING_RUN=1, not both" >&2
    exit 2
fi
if [[ "${GROUP_PROFILE}" != "mtw" && "${RUN_IMPORTANCE}" == "1" ]]; then
    echo "ERROR: RUN_IMPORTANCE=1 is currently supported only for GROUP_PROFILE=mtw; keep it at 0 for mt2pw." >&2
    exit 2
fi
if [[ "${GROUP_PROFILE}" != "mtw" && "${RUN_ALE}" == "1" ]]; then
    echo "ERROR: RUN_ALE=1 is currently supported only for GROUP_PROFILE=mtw; keep it at 0 for mt2pw." >&2
    exit 2
fi

mask_in_colon_list() {
    local needle="$1" list="$2" item
    local -a items
    IFS=':' read -ra items <<< "${list//,/:}"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

coupled_mtw_mask() {
    local mask="$1"
    if [[ "${GROUP_PROFILE}" == "mt2pw" && "${#mask}" -eq 4 && "${mask:1:1}" == "${mask:2:1}" ]]; then
        echo "${mask:0:1}${mask:1:1}${mask:3:1}"
    fi
}

require_dataset() {
    local label="$1" dir="$2"; shift 2
    [[ -f "${dir}/dataset_build_config.json" ]] || { echo "ERROR: ${label} missing config: ${dir}" >&2; exit 2; }
    local split
    for split in "$@"; do
        [[ -f "${dir}/X_${split}.npy" && -f "${dir}/y_${split}.npy" ]] || {
            echo "ERROR: ${label} missing ${split} arrays: ${dir}" >&2; exit 2;
        }
    done
}

require_dataset qcore_s1 "${S1_DATA_DIR}" train val
require_dataset qcore_tianji "${TIANJI_DATA_DIR}" train val test
require_dataset qcore_ifs "${IFS_DATA_DIR}" train val test
require_dataset qcore_pangu "${PANGU_DATA_DIR}" train val test
require_dataset qcore_era5 "${ERA5_DATA_DIR}" train val test
if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
    require_dataset common_core_s1 "${COMMON_CORE_S1_DATA_DIR}" train val
    require_dataset common_core_tianji "${COMMON_CORE_DATA_DIR}" train val test
fi

artifact_triplet_complete() {
    local run_id="$1" stage="$2" checkpoint_tag
    local -a scalers
    if [[ "${stage}" == "s1" ]]; then
        checkpoint_tag="S1_best_score"
    else
        checkpoint_tag="S2_PhaseB_best_score"
    fi
    scalers=("${CKPT_DIR}/robust_scaler_${run_id}_${stage}_w12_dyn"*_pm.pkl)
    [[ -s "${CKPT_DIR}/${run_id}_${checkpoint_tag}.pt" \
        && ${#scalers[@]} -eq 1 \
        && -s "${scalers[0]}" \
        && -s "${CKPT_DIR}/${run_id}_static_rnn_config.json" ]]
}

require_artifact_triplet() {
    local run_id="$1" stage="$2"
    if ! artifact_triplet_complete "${run_id}" "${stage}"; then
        echo "ERROR: resume requested but completed ${stage} checkpoint/scaler/config triplet is missing: ${run_id}" >&2
        exit 2
    fi
}

append_dep() {
    local ids="$1" new_id="$2"
    if [[ -z "${new_id}" ]]; then
        echo "${ids}"
    elif [[ -z "${ids}" ]]; then
        echo "${new_id}"
    else
        echo "${ids}:${new_id}"
    fi
}

if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" ]]; then
    if [[ -e "${HYBRID_DATA_ROOT}" ]]; then
        echo "ERROR: resume requires a clean HYBRID_DATA_ROOT; preserve/move the incomplete directory first: ${HYBRID_DATA_ROOT}" >&2
        exit 2
    fi
    IFS=':' read -ra RESUME_SEED_ARRAY <<< "${SEEDS//,/:}"
    for seed_raw in "${RESUME_SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        require_artifact_triplet "exp_qcore_hybrid_${S1_RUN_TAG}_s1_seed${seed}_pm10_pm25" s1
        if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
            require_artifact_triplet "exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25" s1
        fi
    done
    echo "[RESUME] verified completed S1 triplets; S1 jobs will not be resubmitted"
fi

if [[ "${RESUME_EXISTING_RUN}" == "1" ]]; then
    [[ -d "${HYBRID_DATA_ROOT}" ]] || {
        echo "ERROR: RESUME_EXISTING_RUN=1 requires an existing HYBRID_DATA_ROOT: ${HYBRID_DATA_ROOT}" >&2
        exit 2
    }
    IFS=':' read -ra RESUME_SEED_ARRAY <<< "${SEEDS//,/:}"
    for seed_raw in "${RESUME_SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        require_artifact_triplet "exp_qcore_hybrid_${S1_RUN_TAG}_s1_seed${seed}_pm10_pm25" s1
        if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
            require_artifact_triplet "exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25" s1
        fi
    done
    IFS=':' read -ra RESUME_MASK_ARRAY <<< "${MASKS//,/:}"
    for mask_raw in "${RESUME_MASK_ARRAY[@]}"; do
        mask="${mask_raw//[[:space:]]/}"
        require_dataset "hybrid_${HYBRID_DATASET_PREFIX}_${mask}" "${HYBRID_DATA_ROOT}/${HYBRID_DATASET_PREFIX}_${mask}" train val test
    done
    echo "[RESUME] verified existing hybrid datasets and S1 triplets; completed S2 triplets will be reused"
fi

submit() {
    local label="$1"; shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY-RUN] ${label}: sbatch $*" >&2
        echo "dry_${label}"
    else
        local job_id
        job_id="$(sbatch --parsable "$@")"
        job_id="${job_id%%;*}"
        echo "[SUBMITTED] ${label}: ${job_id}" >&2
        echo "${job_id}"
    fi
}

dependency_arg() {
    local ids="$1"
    if [[ -z "${ids}" || "${DRY_RUN}" == "1" ]]; then
        echo ""
    else
        echo "--dependency=afterok:${ids}"
    fi
}

echo "q-core hybrid factorial chain"
echo "RUN_TAG=${RUN_TAG}"
echo "GROUP_PROFILE=${GROUP_PROFILE}"
echo "SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}"
echo "HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT}"
echo "SEEDS=${SEEDS}"
echo "MASKS=${MASKS}"
echo "TRAIN_MASKS=${TRAIN_MASKS}"
echo "S1_RUN_TAG=${S1_RUN_TAG}"
echo "REUSE_COUPLED_MTW_RUN_TAG=${REUSE_COUPLED_MTW_RUN_TAG}"
echo "RESUME_AFTER_DATA_FAILURE=${RESUME_AFTER_DATA_FAILURE}"
echo "RESUME_EXISTING_RUN=${RESUME_EXISTING_RUN}"

if [[ "${RESUME_EXISTING_RUN}" == "1" ]]; then
    base_audit_job=""
    hybrid_build_job="skipped_existing"
    hybrid_audit_job=""
else
    base_audit_job=$(submit base_audit \
        --export="ALL,RUN_TAG=${RUN_TAG},S1_DATA_DIR=${S1_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},IFS_DATA_DIR=${IFS_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_DATA_DIR},AUDIT_OUT_DIR=${EVAL_ROOT}/base_data_audit" \
        sub_q_core_fair_data_audit.slurm)

    base_dep="$(dependency_arg "${base_audit_job}")"
    build_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=build,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=${HYBRID_DATASET_PREFIX},MASKS=${MASKS},LIMIT_ROWS=${LIMIT_ROWS}")
    [[ -z "${base_dep}" ]] || build_args+=("${base_dep}")
    hybrid_build_job=$(submit hybrid_build "${build_args[@]}" sub_q_core_hybrid_factorial_data.slurm)

    build_dep="$(dependency_arg "${hybrid_build_job}")"
    hybrid_audit_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=audit,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=${HYBRID_DATASET_PREFIX},MASKS=${MASKS}")
    [[ -z "${build_dep}" ]] || hybrid_audit_args+=("${build_dep}")
    hybrid_audit_job=$(submit hybrid_audit "${hybrid_audit_args[@]}" sub_q_core_hybrid_factorial_data.slurm)
fi

IFS=':' read -ra SEED_ARRAY <<< "${SEEDS//,/:}"
IFS=':' read -ra MASK_ARRAY <<< "${MASKS//,/:}"
IFS=':' read -ra TRAIN_MASK_ARRAY <<< "${TRAIN_MASKS//,/:}"
declare -A S1_JOBS
declare -A COMMON_S1_JOBS
declare -A S2_JOBS
declare -A COMMON_S2_JOBS
declare -A IMPORTANCE_JOBS
declare -A ALE_JOBS

for seed_raw in "${SEED_ARRAY[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed ${seed}" >&2; exit 2; }
    s1_run_id="exp_qcore_hybrid_${S1_RUN_TAG}_s1_seed${seed}_pm10_pm25"
    if [[ "${S1_RUN_TAG}" != "${RUN_TAG}" ]]; then
        require_artifact_triplet "${s1_run_id}" s1
        S1_JOBS[${seed}]=""
        echo "[REUSE] using existing S1 triplet: ${s1_run_id}"
    elif [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" || "${RESUME_EXISTING_RUN}" == "1" ]]; then
        S1_JOBS[${seed}]=""
    else
        s1_args=(--export="ALL,EXPERIMENT=s1_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
        [[ -z "${base_dep}" ]] || s1_args+=("${base_dep}")
        S1_JOBS[${seed}]=$(submit "s1_seed${seed}" "${s1_args[@]}" sub_ifs_overlap_baseline.slurm)
    fi

    if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
        common_s1_run_id="exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25"
        if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" || "${RESUME_EXISTING_RUN}" == "1" ]]; then
            COMMON_S1_JOBS[${seed}]=""
        else
            common_s1_args=(--export="ALL,EXPERIMENT=s1_common_core,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${common_s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${COMMON_CORE_S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_common_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
            COMMON_S1_JOBS[${seed}]=$(submit "common_s1_seed${seed}" "${common_s1_args[@]}" sub_ifs_overlap_baseline.slurm)
        fi
    fi
done

for seed_raw in "${SEED_ARRAY[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    s1_run_id="exp_qcore_hybrid_${S1_RUN_TAG}_s1_seed${seed}_pm10_pm25"
    s1_ckpt="${CKPT_DIR}/${s1_run_id}_S1_best_score.pt"
    for mask_raw in "${TRAIN_MASK_ARRAY[@]}"; do
        mask="${mask_raw//[[:space:]]/}"
        [[ "${#mask}" -eq "${MASK_WIDTH}" && "${mask}" != *[!01]* ]] || { echo "ERROR: invalid mask ${mask}" >&2; exit 2; }
        mask_in_colon_list "${mask}" "${MASKS}" || { echo "ERROR: TRAIN_MASK ${mask} is not included in MASKS=${MASKS}" >&2; exit 2; }
        run_id="exp_qcore_hybrid_${RUN_TAG}_${RUN_MASK_PREFIX}${mask}_seed${seed}_pm10_pm25"
        if [[ "${RESUME_EXISTING_RUN}" == "1" ]] && artifact_triplet_complete "${run_id}" s2; then
            S2_JOBS[${seed}_${mask}]=""
            echo "[RESUME] reusing completed S2 triplet: ${run_id}"
            continue
        fi
        deps="${hybrid_audit_job}"
        if [[ -n "${S1_JOBS[${seed}]}" ]]; then
            deps="$(append_dep "${deps}" "${S1_JOBS[${seed}]}")"
        fi
        dep="$(dependency_arg "${deps}")"
        s2_args=(--export="ALL,EXPERIMENT=s2_pangu2025_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S2_DATA_DIR=${HYBRID_DATA_ROOT}/${HYBRID_DATASET_PREFIX}_${mask},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_${RUN_MASK_PREFIX}${mask}_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
        [[ -z "${dep}" ]] || s2_args+=("${dep}")
        S2_JOBS[${seed}_${mask}]=$(submit "s2_${mask}_seed${seed}" "${s2_args[@]}" sub_ifs_overlap_baseline.slurm)
    done

    for mask_raw in "${MASK_ARRAY[@]}"; do
        mask="${mask_raw//[[:space:]]/}"
        [[ "${#mask}" -eq "${MASK_WIDTH}" && "${mask}" != *[!01]* ]] || { echo "ERROR: invalid mask ${mask}" >&2; exit 2; }
        if mask_in_colon_list "${mask}" "${TRAIN_MASKS}"; then
            continue
        fi
        old_mask="$(coupled_mtw_mask "${mask}")"
        if [[ -z "${old_mask}" || -z "${REUSE_COUPLED_MTW_RUN_TAG}" ]]; then
            echo "ERROR: mask ${mask} is not scheduled for training and cannot be mapped to a reused MTW artifact" >&2
            exit 2
        fi
        require_artifact_triplet "exp_qcore_hybrid_${REUSE_COUPLED_MTW_RUN_TAG}_mtw${old_mask}_seed${seed}_pm10_pm25" s2
        require_dataset "reused_mtw_${old_mask}" "${REUSE_COUPLED_MTW_HYBRID_ROOT}/mtw_${old_mask}" train val test
        S2_JOBS[${seed}_${mask}]=""
        echo "[REUSE] mask ${mask} uses MTW ${old_mask} checkpoint/data from ${REUSE_COUPLED_MTW_RUN_TAG}"
    done

    if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
        common_s1_run_id="exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25"
        common_s1_ckpt="${CKPT_DIR}/${common_s1_run_id}_S1_best_score.pt"
        common_run_id="exp_qcore_hybrid_${RUN_TAG}_tianji_common_core_seed${seed}_pm10_pm25"
        if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" || "${RESUME_EXISTING_RUN}" == "1" ]] && artifact_triplet_complete "${common_run_id}" s2; then
            COMMON_S2_JOBS[${seed}]=""
            echo "[RESUME] reusing completed common-core S2 triplet: ${common_run_id}"
        else
            dep=""
            if [[ -n "${COMMON_S1_JOBS[${seed}]}" ]]; then
                dep="$(dependency_arg "${COMMON_S1_JOBS[${seed}]}")"
            fi
            common_s2_args=(--export="ALL,EXPERIMENT=s2_tianji_common_core,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${common_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S2_DATA_DIR=${COMMON_CORE_DATA_DIR},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${common_s1_ckpt},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_common_s2_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
            [[ -z "${dep}" ]] || common_s2_args+=("${dep}")
            COMMON_S2_JOBS[${seed}]=$(submit "common_s2_seed${seed}" "${common_s2_args[@]}" sub_ifs_overlap_baseline.slurm)
        fi
    fi
done

training_deps=""
for seed_raw in "${SEED_ARRAY[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    for mask_raw in "${MASK_ARRAY[@]}"; do
        mask="${mask_raw//[[:space:]]/}"
        training_deps="$(append_dep "${training_deps}" "${S2_JOBS[${seed}_${mask}]}")"
    done
done
artifact_dep="$(dependency_arg "${training_deps}")"
allow_smoke=0
if [[ "${#SEED_ARRAY[@]}" -ne 3 || "${#MASK_ARRAY[@]}" -ne 8 ]]; then
    allow_smoke=1
fi
if [[ "${GROUP_PROFILE}" == "mt2pw" && "${#SEED_ARRAY[@]}" -eq 3 && "${#MASK_ARRAY[@]}" -eq 16 ]]; then
    allow_smoke=0
fi
artifact_args=(--export="ALL,RUN_TAG=${RUN_TAG},S1_RUN_TAG=${S1_RUN_TAG},SEEDS=${SEEDS},MASKS=${MASKS},TRAIN_MASKS=${TRAIN_MASKS},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=${HYBRID_DATASET_PREFIX},RUN_MASK_PREFIX=${RUN_MASK_PREFIX},REUSE_COUPLED_MTW_RUN_TAG=${REUSE_COUPLED_MTW_RUN_TAG},REUSE_COUPLED_MTW_HYBRID_ROOT=${REUSE_COUPLED_MTW_HYBRID_ROOT},S1_DATA_DIR=${S1_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},CKPT_DIR=${CKPT_DIR},EVAL_ROOT=${EVAL_ROOT},ALLOW_SMOKE=${allow_smoke}")
[[ -z "${artifact_dep}" ]] || artifact_args+=("${artifact_dep}")
artifact_audit_job=$(submit artifact_audit "${artifact_args[@]}" sub_q_core_hybrid_artifact_audit.slurm)

if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        pangu_run_id="exp_qcore_hybrid_${RUN_TAG}_mtw000_seed${seed}_pm10_pm25"
        tianji_run_id="exp_qcore_hybrid_${RUN_TAG}_mtw111_seed${seed}_pm10_pm25"
        sources="tianji=${HYBRID_DATA_ROOT}/mtw_111|${CKPT_DIR}/${tianji_run_id}_S2_PhaseB_best_score.pt|AUTO|Tianji q-core;pangu2025_q_core_no_rh2m=${HYBRID_DATA_ROOT}/mtw_000|${CKPT_DIR}/${pangu_run_id}_S2_PhaseB_best_score.pt|AUTO|Pangu q-core"
        deps=""
        deps="$(append_dep "${deps}" "${S2_JOBS[${seed}_000]}")"
        deps="$(append_dep "${deps}" "${S2_JOBS[${seed}_111]}")"
        dep="$(dependency_arg "${deps}")"
        importance_out="${EVAL_ROOT}/feature_importance/seed_${seed}"
        importance_args=(--export="ALL,SOURCES=${sources},FEATURE_IMPORTANCE_OUT_DIR=${importance_out},SAMPLE_SIZE=50000,MIN_LOW_VIS=200,REPEATS=5,BOOTSTRAP_ITERS=1000,GROUP_SCOPE=all")
        [[ -z "${dep}" ]] || importance_args+=("${dep}")
        IMPORTANCE_JOBS[${seed}]=$(submit "importance_seed${seed}" "${importance_args[@]}" sub_multi_source_feature_importance.slurm)
    done
fi

if [[ "${RUN_ALE}" == "1" ]]; then
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        deps=""
        deps="$(append_dep "${deps}" "${S2_JOBS[${seed}_000]}")"
        deps="$(append_dep "${deps}" "${S2_JOBS[${seed}_111]}")"
        dep="$(dependency_arg "${deps}")"
        ale_args=(--export="ALL,RUN_TAG=${RUN_TAG},SEED=${seed},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},SAMPLE_SIZE=50000,MIN_LOW_VIS=200,BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},LIMIT_ROWS=${LIMIT_SAMPLES}")
        [[ -z "${dep}" ]] || ale_args+=("${dep}")
        ALE_JOBS[${seed}]=$(submit "ale_seed${seed}" "${ale_args[@]}" sub_q_core_hybrid_ale.slurm)
    done
fi

declare -A EVAL_JOBS
if [[ "${RUN_EVAL}" == "1" ]]; then
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        deps="${artifact_audit_job}"
        for mask_raw in "${MASK_ARRAY[@]}"; do
            mask="${mask_raw//[[:space:]]/}"
            deps="$(append_dep "${deps}" "${S2_JOBS[${seed}_${mask}]}")"
        done
        if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
            if [[ -n "${COMMON_S2_JOBS[${seed}]}" ]]; then
                deps="$(append_dep "${deps}" "${COMMON_S2_JOBS[${seed}]}")"
            fi
        fi
        if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
            deps="$(append_dep "${deps}" "${IMPORTANCE_JOBS[${seed}]}")"
        fi
        if [[ "${RUN_ALE}" == "1" ]]; then
            deps="$(append_dep "${deps}" "${ALE_JOBS[${seed}]}")"
        fi
        dep="$(dependency_arg "${deps}")"
        eval_args=(--job-name="qcore_eval_s${seed}" --export="ALL,RUN_TAG=${RUN_TAG},MODE=evaluate_seed,SEED=${seed},SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=${HYBRID_DATASET_PREFIX},RUN_MASK_PREFIX=${RUN_MASK_PREFIX},SOURCE_PREFIX=${SOURCE_PREFIX},REUSE_COUPLED_MTW_RUN_TAG=${REUSE_COUPLED_MTW_RUN_TAG},REUSE_COUPLED_MTW_HYBRID_ROOT=${REUSE_COUPLED_MTW_HYBRID_ROOT},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},LIMIT_SAMPLES=${LIMIT_SAMPLES},INCLUDE_COMMON_CORE=${RUN_COMMON_CORE},COMMON_CORE_DATA_DIR=${COMMON_CORE_DATA_DIR}")
        [[ -z "${dep}" ]] || eval_args+=("${dep}")
        EVAL_JOBS[${seed}]=$(submit "eval_seed${seed}" "${eval_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)
    done
    eval_deps=""
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        eval_deps="$(append_dep "${eval_deps}" "${EVAL_JOBS[${seed}]}")"
    done
    dep="$(dependency_arg "${eval_deps}")"
    analysis_args=(--job-name="qcore_analysis" --export="ALL,RUN_TAG=${RUN_TAG},MODE=analyze,SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=${GROUP_PROFILE},SOURCE_PREFIX=${SOURCE_PREFIX},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_MAX_ROWS=${BOOTSTRAP_MAX_ROWS},OBS_ROOT=${OBS_ROOT},ERA5_DATA_DIR=${ERA5_DATA_DIR},REQUIRE_ALE=${RUN_ALE}")
    [[ -z "${dep}" ]] || analysis_args+=("${dep}")
    analysis_job=$(submit analysis "${analysis_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)
else
    analysis_job="not_submitted"
fi

manifest="${EVAL_ROOT}/submission_manifest_${RUN_TAG}.txt"
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${EVAL_ROOT}"
    {
        echo "status=scheduled"
        echo "run_tag=${RUN_TAG}"
        echo "group_profile=${GROUP_PROFILE}"
        echo "source_data_root=${SOURCE_DATA_ROOT}"
        echo "hybrid_data_root=${HYBRID_DATA_ROOT}"
        echo "seeds=${SEEDS}"
        echo "masks=${MASKS}"
        echo "train_masks=${TRAIN_MASKS}"
        echo "s1_run_tag=${S1_RUN_TAG}"
        echo "hybrid_dataset_prefix=${HYBRID_DATASET_PREFIX}"
        echo "run_mask_prefix=${RUN_MASK_PREFIX}"
        echo "source_prefix=${SOURCE_PREFIX}"
        echo "reuse_coupled_mtw_run_tag=${REUSE_COUPLED_MTW_RUN_TAG}"
        echo "reuse_coupled_mtw_hybrid_root=${REUSE_COUPLED_MTW_HYBRID_ROOT}"
        echo "resume_after_data_failure=${RESUME_AFTER_DATA_FAILURE}"
        echo "resume_existing_run=${RESUME_EXISTING_RUN}"
        echo "base_audit_job=${base_audit_job}"
        echo "hybrid_build_job=${hybrid_build_job}"
        echo "hybrid_audit_job=${hybrid_audit_job}"
        echo "artifact_audit_job=${artifact_audit_job}"
        echo "analysis_job=${analysis_job}"
        echo "scientific_role=controlled source-block retraining attribution"
    } > "${manifest}"
fi

echo "analysis_job=${analysis_job}"
echo "results=${EVAL_ROOT}"
