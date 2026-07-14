#!/bin/bash
# End-to-end Pangu/Tianji T925--moisture joint-structure experiment.
#
# Formal design:
#   3 shared q-core+T925 S1 anchors + 8 m925b hybrids x 3 seeds = 27 trainings.
#   m925b order is M,H,B:
#     M = Q1000/DP1000/Q925/DP925/RH925
#     H = explicit T925
#     B = remaining source-dependent q-core variables
#
# Optional confirmatory smoke example (diagnosis must be reviewed first):
#   ENABLE_OPTIONAL_FACTORIAL=1 \
#   RUN_TAG=qcore_t925_joint_smoke_$(date +%Y%m%d_%H%M%S) \
#   SEEDS=42 LIMIT_ROWS=2000 LIMIT_SAMPLES=2000 \
#   LOWVIS_RNN_S1_STEPS=20 LOWVIS_RNN_S2_A_STEPS=20 LOWVIS_RNN_S2_B_STEPS=40 \
#   BOOTSTRAP_ITERS=50 TEST_MAX_ROWS=2000 bash submit_q_core_t925_joint_structure.sh
#
# Optional confirmatory formal example:
#   ENABLE_OPTIONAL_FACTORIAL=1 \
#   RUN_TAG=qcore_t925_joint_formal_v1_20260714 \
#   PANGU2025_STATION_FILE=/.../pangu_station_2025_lead12_23h_canonical.nc \
#   bash submit_q_core_t925_joint_structure.sh

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be a fresh tag}"
ENABLE_OPTIONAL_FACTORIAL="${ENABLE_OPTIONAL_FACTORIAL:-0}"
SEEDS="${SEEDS:-42:2025:20260702}"
MASKS="000:001:010:011:100:101:110:111"
FEATURE_SET="q_core_t925_no_rh2m"
GROUP_PROFILE="m925b"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
RESUME_EXISTING_DATA="${RESUME_EXISTING_DATA:-0}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
FIT_MAX_ROWS="${FIT_MAX_ROWS:-200000}"
TEST_MAX_ROWS="${TEST_MAX_ROWS:-0}"
RFF_DIM="${RFF_DIM:-512}"
OBS_ROOT="${OBS_ROOT:-}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"

if [[ "${ENABLE_OPTIONAL_FACTORIAL}" != "1" ]]; then
  echo "ERROR: this launcher submits 3 S1 + 24 S2 optional confirmatory trainings." >&2
  echo "Use submit_q_core_t925_diagnostics.sh for the default zero-training diagnosis." >&2
  echo "Set ENABLE_OPTIONAL_FACTORIAL=1 only after the diagnostic evidence has been reviewed." >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_joint_datasets/${RUN_TAG}}"
S1_DATA_DIR="${S1_DATA_DIR:-${DATA_ROOT}/s1}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-${DATA_ROOT}/era5_2025}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_joint/${RUN_TAG}}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN}" != "1" && ! -s "${PANGU2025_STATION_FILE}" ]]; then
  echo "ERROR: canonical Pangu station product is missing: ${PANGU2025_STATION_FILE}" >&2
  exit 2
fi
if [[ "${RESUME_EXISTING_DATA}" == "1" && ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: RESUME_EXISTING_DATA=1 but DATA_ROOT is absent: ${DATA_ROOT}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && "${ALLOW_EXISTING_RUN}" != "1" && "${RESUME_EXISTING_DATA}" != "1" ]]; then
  if [[ -e "${DATA_ROOT}" || -e "${HYBRID_DATA_ROOT}" || -e "${EVAL_ROOT}" ]]; then
    echo "ERROR: RUN_TAG would reuse data or results; choose a fresh tag or set the explicit resume flags." >&2
    exit 2
  fi
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

submit() {
  local label="$1"; shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] %s: sbatch' "${label}" >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf 'dry_%s\n' "${label}"
  else
    local job_id
    job_id="$(sbatch --parsable "$@")"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id ${job_id}" >&2; exit 2; }
    echo "[SUBMITTED] ${label}: ${job_id}" >&2
    printf '%s\n' "${job_id}"
  fi
}

dep_arg() {
  local ids="$1"
  if [[ -z "${ids}" || "${DRY_RUN}" == "1" ]]; then
    printf '%s' ""
  else
    printf '%s' "--dependency=afterok:${ids}"
  fi
}

append_dep() {
  local left="$1" right="$2"
  if [[ -z "${right}" ]]; then printf '%s\n' "${left}"
  elif [[ -z "${left}" ]]; then printf '%s\n' "${right}"
  else printf '%s:%s\n' "${left}" "${right}"
  fi
}

require_dataset() {
  local label="$1" dir="$2"; shift 2
  [[ -s "${dir}/dataset_build_config.json" ]] || { echo "ERROR: ${label} missing config under ${dir}" >&2; exit 2; }
  local split
  for split in "$@"; do
    [[ -s "${dir}/X_${split}.npy" && -s "${dir}/y_${split}.npy" ]] || {
      echo "ERROR: ${label} missing ${split} arrays under ${dir}" >&2; exit 2;
    }
  done
}

echo "q-core+T925 joint-structure chain"
echo "RUN_TAG=${RUN_TAG}"
echo "FEATURE_SET=${FEATURE_SET}"
echo "GROUP_PROFILE=${GROUP_PROFILE}"
echo "SEEDS=${SEEDS}"
echo "PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"

if [[ "${RESUME_EXISTING_DATA}" == "1" ]]; then
  require_dataset s1 "${S1_DATA_DIR}" train val
  require_dataset tianji "${TIANJI_DATA_DIR}" train val test
  require_dataset pangu "${PANGU_DATA_DIR}" train val test
  require_dataset era5 "${ERA5_DATA_DIR}" train val test
  data_jobs=""
else
  s1_data_job=$(submit s1_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${S1_DATA_DIR}" \
    sub_s1_overlap_data.slurm)
  tianji_data_job=$(submit tianji_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${TIANJI_DATA_DIR}" \
    sub_tianji_overlap_data.slurm)
  pangu_data_job=$(submit pangu_data \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},SOURCE_FILE=${PANGU2025_STATION_FILE},OUT_DIR=${PANGU_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=12,EXPECTED_LEAD_MAX_HOURS=23,INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1" \
    sub_station_source_overlap_data.slurm)
  era5_data_job=$(submit era5_data \
    --export="ALL,SOURCE_KIND=era5_feature_dir,SOURCE_TAG=era5_2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},OUT_DIR=${ERA5_DATA_DIR}" \
    sub_station_source_overlap_data.slurm)
  data_jobs="${s1_data_job}:${tianji_data_job}:${pangu_data_job}:${era5_data_job}"
fi

data_dep="$(dep_arg "${data_jobs}")"
audit_args=(
  --export="ALL,RUN_TAG=${RUN_TAG},AUDIT_PROFILE=qcore_t925,SOURCES=tianji=${TIANJI_DATA_DIR};pangu2025=${PANGU_DATA_DIR};era5_2025=${ERA5_DATA_DIR},S1_DATA_DIR=${S1_DATA_DIR},AUDIT_OUT_DIR=${EVAL_ROOT}/base_data_audit,EXPECTED_PANGU_LEAD_MIN_HOURS=12,EXPECTED_PANGU_LEAD_MAX_HOURS=23"
)
[[ -z "${data_dep}" ]] || audit_args+=("${data_dep}")
audit_job=$(submit base_data_audit "${audit_args[@]}" sub_q_core_fair_data_audit.slurm)

audit_dep="$(dep_arg "${audit_job}")"
hybrid_build_args=(
  --export="ALL,RUN_TAG=${RUN_TAG},MODE=build,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=m925b,MASKS=${MASKS},LIMIT_ROWS=${LIMIT_ROWS}"
)
[[ -z "${audit_dep}" ]] || hybrid_build_args+=("${audit_dep}")
hybrid_build_job=$(submit hybrid_build "${hybrid_build_args[@]}" sub_q_core_hybrid_factorial_data.slurm)

build_dep="$(dep_arg "${hybrid_build_job}")"
hybrid_audit_args=(
  --export="ALL,RUN_TAG=${RUN_TAG},MODE=audit,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=m925b,MASKS=${MASKS}"
)
[[ -z "${build_dep}" ]] || hybrid_audit_args+=("${build_dep}")
hybrid_audit_job=$(submit hybrid_audit "${hybrid_audit_args[@]}" sub_q_core_hybrid_factorial_data.slurm)

IFS=':' read -ra seed_array <<< "${SEEDS//,/:}"
IFS=':' read -ra mask_array <<< "${MASKS}"
declare -A s1_jobs
declare -A s2_jobs
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed ${seed}" >&2; exit 2; }
  run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
  s1_args=(
    --export="ALL,EXPERIMENT=s1_q_core_t925_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_t925_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1"
  )
  [[ -z "${audit_dep}" ]] || s1_args+=("${audit_dep}")
  s1_jobs[${seed}]=$(submit "s1_seed${seed}" "${s1_args[@]}" sub_ifs_overlap_baseline.slurm)
done

training_jobs=""
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  s1_run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
  s1_ckpt="${CKPT_DIR}/${s1_run_id}_S1_best_score.pt"
  for mask in "${mask_array[@]}"; do
    run_id="exp_qcore_hybrid_${RUN_TAG}_m925b${mask}_seed${seed}_pm10_pm25"
    deps="$(append_dep "${hybrid_audit_job}" "${s1_jobs[${seed}]}")"
    dep="$(dep_arg "${deps}")"
    s2_args=(
      --export="ALL,EXPERIMENT=s2_q_core_t925_joint,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S2_DATA_DIR=${HYBRID_DATA_ROOT}/m925b_${mask},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_m925b${mask}_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1"
    )
    [[ -z "${dep}" ]] || s2_args+=("${dep}")
    s2_jobs[${seed}_${mask}]=$(submit "s2_${mask}_seed${seed}" "${s2_args[@]}" sub_ifs_overlap_baseline.slurm)
    training_jobs="$(append_dep "${training_jobs}" "${s2_jobs[${seed}_${mask}]}")"
  done
done

allow_smoke=0
if [[ "${#seed_array[@]}" -ne 3 || "${LOWVIS_RNN_S1_STEPS:-15000}" -ne 15000 || "${LOWVIS_RNN_S2_A_STEPS:-12000}" -ne 12000 || "${LOWVIS_RNN_S2_B_STEPS:-40000}" -ne 40000 ]]; then
  allow_smoke=1
fi
artifact_dep="$(dep_arg "${training_jobs}")"
artifact_args=(
  --export="ALL,RUN_TAG=${RUN_TAG},S1_RUN_TAG=${RUN_TAG},SEEDS=${SEEDS},MASKS=${MASKS},TRAIN_MASKS=${MASKS},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=m925b,RUN_MASK_PREFIX=m925b,S1_DATA_DIR=${S1_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},CKPT_DIR=${CKPT_DIR},EVAL_ROOT=${EVAL_ROOT},ALLOW_SMOKE=${allow_smoke}"
)
[[ -z "${artifact_dep}" ]] || artifact_args+=("${artifact_dep}")
artifact_job=$(submit artifact_audit "${artifact_args[@]}" sub_q_core_hybrid_artifact_audit.slurm)

declare -A eval_jobs
eval_deps=""
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  dep="$(dep_arg "${artifact_job}")"
  eval_args=(
    --job-name="qcore_t925_eval_s${seed}"
    --export="ALL,RUN_TAG=${RUN_TAG},MODE=evaluate_seed,SEED=${seed},SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=${GROUP_PROFILE},HYBRID_DATASET_PREFIX=m925b,RUN_MASK_PREFIX=m925b,SOURCE_PREFIX=qcore_joint_,HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},LIMIT_SAMPLES=${LIMIT_SAMPLES},INCLUDE_COMMON_CORE=0"
  )
  [[ -z "${dep}" ]] || eval_args+=("${dep}")
  eval_jobs[${seed}]=$(submit "eval_seed${seed}" "${eval_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)
  eval_deps="$(append_dep "${eval_deps}" "${eval_jobs[${seed}]}")"
done

eval_dep="$(dep_arg "${eval_deps}")"
analysis_args=(
  --job-name="qcore_t925_factorial"
  --export="ALL,RUN_TAG=${RUN_TAG},MODE=analyze,SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=${GROUP_PROFILE},SOURCE_PREFIX=qcore_joint_,HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},ANALYSIS_DIR=${EVAL_ROOT}/factorial_analysis,BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_MAX_ROWS=${BOOTSTRAP_MAX_ROWS},OBS_ROOT=${OBS_ROOT},ERA5_DATA_DIR=${ERA5_DATA_DIR},INCLUDE_COMMON_CORE=0"
)
[[ -z "${eval_dep}" ]] || analysis_args+=("${eval_dep}")
analysis_job=$(submit factorial_analysis "${analysis_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)

analysis_dep="$(dep_arg "${analysis_job}")"
joint_args=(
  --export="ALL,RUN_TAG=${RUN_TAG},ANALYSIS_MODE=factorial_confirmatory,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},ERA5_DATA_DIR=${ERA5_DATA_DIR},EVAL_ROOT=${EVAL_ROOT},EVENT_ANALYSIS_DIR=${EVAL_ROOT}/factorial_analysis,FACTORIAL_ANALYSIS_DIR=${EVAL_ROOT}/factorial_analysis,OUT_DIR=${EVAL_ROOT}/joint_structure_analysis,FIT_MAX_ROWS=${FIT_MAX_ROWS},TEST_MAX_ROWS=${TEST_MAX_ROWS},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},RFF_DIM=${RFF_DIM}"
)
[[ -z "${analysis_dep}" ]] || joint_args+=("${analysis_dep}")
joint_job=$(submit joint_structure_analysis "${joint_args[@]}" sub_q_core_t925_joint_structure.slurm)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${EVAL_ROOT}"
  {
    echo "status=scheduled"
    echo "run_tag=${RUN_TAG}"
    echo "feature_set=${FEATURE_SET}"
    echo "group_profile=${GROUP_PROFILE}"
    echo "factorial_groups=M:Q1000+DP1000+Q925+DP925+RH925;H:T925;B:remaining_qcore"
    echo "seeds=${SEEDS}"
    echo "masks=${MASKS}"
    echo "data_root=${DATA_ROOT}"
    echo "hybrid_data_root=${HYBRID_DATA_ROOT}"
    echo "base_data_audit_job=${audit_job}"
    echo "hybrid_build_job=${hybrid_build_job}"
    echo "hybrid_audit_job=${hybrid_audit_job}"
    echo "artifact_audit_job=${artifact_job}"
    echo "factorial_analysis_job=${analysis_job}"
    echo "joint_structure_analysis_job=${joint_job}"
    echo "era5_role=reference_analysis_not_truth"
    echo "claim_limit=interaction_is_predictive_complementarity_not_proof_of_governing_equation_consistency"
  } > "${EVAL_ROOT}/submission_manifest_${RUN_TAG}.txt"
fi

echo "joint_structure_analysis_job=${joint_job}"
echo "results=${EVAL_ROOT}"
