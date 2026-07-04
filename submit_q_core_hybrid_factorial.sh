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
MASKS="${MASKS:-000:001:010:011:100:101:110:111}"
DRY_RUN="${DRY_RUN:-0}"
RUN_COMMON_CORE="${RUN_COMMON_CORE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_IMPORTANCE="${RUN_IMPORTANCE:-1}"
RUN_ALE="${RUN_ALE:-1}"
RESUME_AFTER_DATA_FAILURE="${RESUME_AFTER_DATA_FAILURE:-0}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
OBS_ROOT="${OBS_ROOT:-}"

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

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

case "${RUN_TAG}" in
    *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
if [[ "${SOURCE_DATA_ROOT}" == "${HYBRID_DATA_ROOT}" ]]; then
    echo "ERROR: source and hybrid data roots must differ" >&2
    exit 2
fi

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

dataset_dyn() {
    # Keep this login-node preflight compatible with both the cluster's legacy
    # `python` command and Python 3; the JSON field is numeric/ASCII.
    python -c 'import json,sys; print(int(json.load(open(sys.argv[1] + "/dataset_build_config.json"))["dyn_vars"]))' "$1"
}

artifact_triplet_complete() {
    local run_id="$1" stage="$2" dyn="$3" checkpoint_tag
    if [[ "${stage}" == "s1" ]]; then
        checkpoint_tag="S1_best_score"
    else
        checkpoint_tag="S2_PhaseB_best_score"
    fi
    [[ -s "${CKPT_DIR}/${run_id}_${checkpoint_tag}.pt" \
        && -s "${CKPT_DIR}/robust_scaler_${run_id}_${stage}_w12_dyn${dyn}_pm.pkl" \
        && -s "${CKPT_DIR}/${run_id}_static_rnn_config.json" ]]
}

require_artifact_triplet() {
    local run_id="$1" stage="$2" dyn="$3"
    if ! artifact_triplet_complete "${run_id}" "${stage}" "${dyn}"; then
        echo "ERROR: resume requested but completed ${stage} checkpoint/scaler/config triplet is missing: ${run_id}" >&2
        exit 2
    fi
}

S1_DYN="$(dataset_dyn "${S1_DATA_DIR}")"
COMMON_S1_DYN=""
COMMON_S2_DYN=""
if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
    COMMON_S1_DYN="$(dataset_dyn "${COMMON_CORE_S1_DATA_DIR}")"
    COMMON_S2_DYN="$(dataset_dyn "${COMMON_CORE_DATA_DIR}")"
fi

if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" ]]; then
    if [[ -e "${HYBRID_DATA_ROOT}" ]]; then
        echo "ERROR: resume requires a clean HYBRID_DATA_ROOT; preserve/move the incomplete directory first: ${HYBRID_DATA_ROOT}" >&2
        exit 2
    fi
    IFS=':' read -ra RESUME_SEED_ARRAY <<< "${SEEDS//,/:}"
    for seed_raw in "${RESUME_SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        require_artifact_triplet "exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25" s1 "${S1_DYN}"
        if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
            require_artifact_triplet "exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25" s1 "${COMMON_S1_DYN}"
        fi
    done
    echo "[RESUME] verified completed S1 triplets; S1 jobs will not be resubmitted"
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
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo ""
    else
        echo "--dependency=afterok:${ids}"
    fi
}

echo "q-core hybrid factorial chain"
echo "RUN_TAG=${RUN_TAG}"
echo "SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}"
echo "HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT}"
echo "SEEDS=${SEEDS}"
echo "MASKS=${MASKS}"
echo "RESUME_AFTER_DATA_FAILURE=${RESUME_AFTER_DATA_FAILURE}"

base_audit_job=$(submit base_audit \
    --export="ALL,RUN_TAG=${RUN_TAG},S1_DATA_DIR=${S1_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},IFS_DATA_DIR=${IFS_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_DATA_DIR},ERA5_2025_DATA_DIR=${ERA5_DATA_DIR},AUDIT_OUT_DIR=${EVAL_ROOT}/base_data_audit" \
    sub_q_core_fair_data_audit.slurm)

base_dep="$(dependency_arg "${base_audit_job}")"
build_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=build,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},MASKS=${MASKS},LIMIT_ROWS=${LIMIT_ROWS}")
[[ -z "${base_dep}" ]] || build_args+=("${base_dep}")
hybrid_build_job=$(submit hybrid_build "${build_args[@]}" sub_q_core_hybrid_factorial_data.slurm)

build_dep="$(dependency_arg "${hybrid_build_job}")"
hybrid_audit_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=audit,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},MASKS=${MASKS}")
[[ -z "${build_dep}" ]] || hybrid_audit_args+=("${build_dep}")
hybrid_audit_job=$(submit hybrid_audit "${hybrid_audit_args[@]}" sub_q_core_hybrid_factorial_data.slurm)

IFS=':' read -ra SEED_ARRAY <<< "${SEEDS//,/:}"
IFS=':' read -ra MASK_ARRAY <<< "${MASKS//,/:}"
declare -A S1_JOBS
declare -A COMMON_S1_JOBS
declare -A S2_JOBS
declare -A COMMON_S2_JOBS
declare -A IMPORTANCE_JOBS
declare -A ALE_JOBS

for seed_raw in "${SEED_ARRAY[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed ${seed}" >&2; exit 2; }
    s1_run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
    if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" ]]; then
        S1_JOBS[${seed}]=""
    else
        s1_args=(--export="ALL,EXPERIMENT=s1_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
        [[ -z "${base_dep}" ]] || s1_args+=("${base_dep}")
        S1_JOBS[${seed}]=$(submit "s1_seed${seed}" "${s1_args[@]}" sub_ifs_overlap_baseline.slurm)
    fi

    if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
        common_s1_run_id="exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25"
        if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" ]]; then
            COMMON_S1_JOBS[${seed}]=""
        else
            common_s1_args=(--export="ALL,EXPERIMENT=s1_common_core,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${common_s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${COMMON_CORE_S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_common_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
            COMMON_S1_JOBS[${seed}]=$(submit "common_s1_seed${seed}" "${common_s1_args[@]}" sub_ifs_overlap_baseline.slurm)
        fi
    fi
done

for seed_raw in "${SEED_ARRAY[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    s1_run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
    s1_ckpt="${CKPT_DIR}/${s1_run_id}_S1_best_score.pt"
    for mask_raw in "${MASK_ARRAY[@]}"; do
        mask="${mask_raw//[[:space:]]/}"
        [[ "${mask}" =~ ^[01]{3}$ ]] || { echo "ERROR: invalid mask ${mask}" >&2; exit 2; }
        run_id="exp_qcore_hybrid_${RUN_TAG}_mtw${mask}_seed${seed}_pm10_pm25"
        deps="${hybrid_audit_job}"
        if [[ -n "${S1_JOBS[${seed}]}" ]]; then
            deps="${deps}:${S1_JOBS[${seed}]}"
        fi
        dep="$(dependency_arg "${deps}")"
        s2_args=(--export="ALL,EXPERIMENT=s2_pangu2025_q_core_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S2_DATA_DIR=${HYBRID_DATA_ROOT}/mtw_${mask},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_mtw${mask}_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1")
        [[ -z "${dep}" ]] || s2_args+=("${dep}")
        S2_JOBS[${seed}_${mask}]=$(submit "s2_${mask}_seed${seed}" "${s2_args[@]}" sub_ifs_overlap_baseline.slurm)
    done

    if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
        common_s1_run_id="exp_qcore_hybrid_${RUN_TAG}_common_core_s1_seed${seed}_pm10_pm25"
        common_s1_ckpt="${CKPT_DIR}/${common_s1_run_id}_S1_best_score.pt"
        common_run_id="exp_qcore_hybrid_${RUN_TAG}_tianji_common_core_seed${seed}_pm10_pm25"
        if [[ "${RESUME_AFTER_DATA_FAILURE}" == "1" ]] && artifact_triplet_complete "${common_run_id}" s2 "${COMMON_S2_DYN}"; then
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
        training_deps="${training_deps:+${training_deps}:}${S2_JOBS[${seed}_${mask}]}"
    done
done
artifact_dep="$(dependency_arg "${training_deps}")"
allow_smoke=0
if [[ "${#SEED_ARRAY[@]}" -ne 3 || "${#MASK_ARRAY[@]}" -ne 8 ]]; then
    allow_smoke=1
fi
artifact_args=(--export="ALL,RUN_TAG=${RUN_TAG},SEEDS=${SEEDS},MASKS=${MASKS},S1_DATA_DIR=${S1_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},CKPT_DIR=${CKPT_DIR},EVAL_ROOT=${EVAL_ROOT},ALLOW_SMOKE=${allow_smoke}")
[[ -z "${artifact_dep}" ]] || artifact_args+=("${artifact_dep}")
artifact_audit_job=$(submit artifact_audit "${artifact_args[@]}" sub_q_core_hybrid_artifact_audit.slurm)

if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        pangu_run_id="exp_qcore_hybrid_${RUN_TAG}_mtw000_seed${seed}_pm10_pm25"
        tianji_run_id="exp_qcore_hybrid_${RUN_TAG}_mtw111_seed${seed}_pm10_pm25"
        sources="tianji=${HYBRID_DATA_ROOT}/mtw_111|${CKPT_DIR}/${tianji_run_id}_S2_PhaseB_best_score.pt|AUTO|Tianji q-core;pangu2025_q_core_no_rh2m=${HYBRID_DATA_ROOT}/mtw_000|${CKPT_DIR}/${pangu_run_id}_S2_PhaseB_best_score.pt|AUTO|Pangu q-core"
        deps="${S2_JOBS[${seed}_000]}:${S2_JOBS[${seed}_111]}"
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
        deps="${S2_JOBS[${seed}_000]}:${S2_JOBS[${seed}_111]}"
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
            deps="${deps:+${deps}:}${S2_JOBS[${seed}_${mask}]}"
        done
        if [[ "${RUN_COMMON_CORE}" == "1" ]]; then
            if [[ -n "${COMMON_S2_JOBS[${seed}]}" ]]; then
                deps="${deps}:${COMMON_S2_JOBS[${seed}]}"
            fi
        fi
        if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
            deps="${deps}:${IMPORTANCE_JOBS[${seed}]}"
        fi
        if [[ "${RUN_ALE}" == "1" ]]; then
            deps="${deps}:${ALE_JOBS[${seed}]}"
        fi
        dep="$(dependency_arg "${deps}")"
        eval_args=(--job-name="qcore_eval_s${seed}" --export="ALL,RUN_TAG=${RUN_TAG},MODE=evaluate_seed,SEED=${seed},SEEDS=${SEEDS},MASKS=${MASKS},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},LIMIT_SAMPLES=${LIMIT_SAMPLES},INCLUDE_COMMON_CORE=${RUN_COMMON_CORE},COMMON_CORE_DATA_DIR=${COMMON_CORE_DATA_DIR}")
        [[ -z "${dep}" ]] || eval_args+=("${dep}")
        EVAL_JOBS[${seed}]=$(submit "eval_seed${seed}" "${eval_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)
    done
    eval_deps=""
    for seed_raw in "${SEED_ARRAY[@]}"; do
        seed="${seed_raw//[[:space:]]/}"
        eval_deps="${eval_deps:+${eval_deps}:}${EVAL_JOBS[${seed}]}"
    done
    dep="$(dependency_arg "${eval_deps}")"
    analysis_args=(--job-name="qcore_analysis" --export="ALL,RUN_TAG=${RUN_TAG},MODE=analyze,SEEDS=${SEEDS},MASKS=${MASKS},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_MAX_ROWS=${BOOTSTRAP_MAX_ROWS},OBS_ROOT=${OBS_ROOT},ERA5_DATA_DIR=${ERA5_DATA_DIR},REQUIRE_ALE=${RUN_ALE}")
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
        echo "source_data_root=${SOURCE_DATA_ROOT}"
        echo "hybrid_data_root=${HYBRID_DATA_ROOT}"
        echo "seeds=${SEEDS}"
        echo "masks=${MASKS}"
        echo "resume_after_data_failure=${RESUME_AFTER_DATA_FAILURE}"
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
