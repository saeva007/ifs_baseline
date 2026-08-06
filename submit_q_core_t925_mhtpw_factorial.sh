#!/bin/bash
# Final two-source q-core+T925 physical-package attribution experiment.
#
# Exact retraining factorial (bit order M,H,T,P,W):
#   M = Q1000 + DP1000
#   H = T925 + Q925 + DP925 + RH925
#   T = T2M
#   P = MSLP
#   W = U10 + V10 + WSPD10 + WDIR10 + U925 + V925 + WSPD925
#
# Formal training count: 3 shared S1 anchors + 32 masks x 3 seeds = 99.
# The W block is deliberately kept whole for exact Shapley attribution. The
# endpoint grouped-permutation diagnostics separately report 10 m and 925 hPa
# wind reliance without increasing the factorial to 64 masks.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be fresh}"
FEATURE_SET="q_core_t925_no_rh2m"
GROUP_PROFILE="mhtpw"
MASKS="00000:00001:00010:00011:00100:00101:00110:00111:01000:01001:01010:01011:01100:01101:01110:01111:10000:10001:10010:10011:10100:10101:10110:10111:11000:11001:11010:11011:11100:11101:11110:11111"
SEEDS="${SEEDS:-42:2025:20260702}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_EXISTING_RUN="${RESUME_EXISTING_RUN:-0}"
REUSE_COMPLETED_AUDITS="${REUSE_COMPLETED_AUDITS:-0}"
RUN_IMPORTANCE="${RUN_IMPORTANCE:-1}"
TRAIN_EXCLUDE_NODES="${TRAIN_EXCLUDE_NODES:-}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
OBS_ROOT="${OBS_ROOT:-}"
ERA5_DATA_DIR="${ERA5_DATA_DIR:-}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"

# Set SOURCE_DATA_ROOT to reuse a completed audited q-core+T925 fair-data tree
# containing s1/, tianji/, and pangu2025/. If omitted, this launcher builds a
# fresh three-source tree under DATA_ROOT before constructing the intersections.
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_factorial_source_datasets/${RUN_TAG}}"
if [[ -n "${SOURCE_DATA_ROOT}" ]]; then
  DATA_ROOT="${SOURCE_DATA_ROOT}"
  BUILD_SOURCE_DATA=0
else
  BUILD_SOURCE_DATA=1
fi
S1_DATA_DIR="${S1_DATA_DIR:-${DATA_ROOT}/s1}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
HYBRID_DATA_ROOT="${HYBRID_DATA_ROOT:-${BASELINE_DIR}/q_core_hybrid_datasets/${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_factorial/${RUN_TAG}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EVAL_ROOT}/analysis}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
for flag in DRY_RUN RESUME_EXISTING_RUN REUSE_COMPLETED_AUDITS RUN_IMPORTANCE; do
  value="${!flag}"
  [[ "${value}" == "0" || "${value}" == "1" ]] || { echo "ERROR: ${flag} must be 0 or 1" >&2; exit 2; }
done
if [[ "${REUSE_COMPLETED_AUDITS}" == "1" && "${RESUME_EXISTING_RUN}" != "1" ]]; then
  echo "ERROR: REUSE_COMPLETED_AUDITS=1 requires RESUME_EXISTING_RUN=1" >&2
  exit 2
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

submit() {
  local label="$1"; shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] %s: sbatch --parsable' "${label}" >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf 'dry_%s\n' "${label}"
    return
  fi
  local raw job_id
  raw="$(sbatch --parsable "$@")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid sbatch id for ${label}: ${raw}" >&2; exit 2; }
  echo "[SUBMITTED] ${label}: ${job_id}" >&2
  printf '%s\n' "${job_id}"
}

append_dep() {
  local left="$1" right="$2"
  if [[ -z "${right}" ]]; then printf '%s' "${left}"
  elif [[ -z "${left}" ]]; then printf '%s' "${right}"
  else printf '%s:%s' "${left}" "${right}"
  fi
}

dep_arg() {
  local ids="$1"
  if [[ -n "${ids}" && "${DRY_RUN}" != "1" ]]; then printf '%s' "--dependency=afterok:${ids}"; fi
}

require_dataset() {
  local label="$1" dir="$2"; shift 2
  [[ -s "${dir}/dataset_build_config.json" ]] || { echo "ERROR: ${label} lacks dataset_build_config.json: ${dir}" >&2; exit 2; }
  local split
  for split in "$@"; do
    [[ -s "${dir}/X_${split}.npy" && -s "${dir}/y_${split}.npy" && -s "${dir}/meta_${split}.csv" ]] || {
      echo "ERROR: ${label} lacks ${split} arrays/metadata: ${dir}" >&2; exit 2;
    }
  done
}

artifact_complete() {
  local run_id="$1" stage="$2" checkpoint_tag
  local -a scalers
  if [[ "${stage}" == "s1" ]]; then checkpoint_tag="S1_best_score"; else checkpoint_tag="S2_PhaseB_best_score"; fi
  scalers=("${CKPT_DIR}/robust_scaler_${run_id}_${stage}_w12_dyn"*_pm.pkl)
  [[ -s "${CKPT_DIR}/${run_id}_${checkpoint_tag}.pt" \
     && -s "${CKPT_DIR}/${run_id}_static_rnn_config.json" \
     && ${#scalers[@]} -eq 1 && -s "${scalers[0]}" ]]
}

IFS=':' read -ra seed_array <<< "${SEEDS//,/:}"
IFS=':' read -ra mask_array <<< "${MASKS}"
[[ ${#mask_array[@]} -eq 32 ]] || { echo "ERROR: mhtpw requires all 32 masks" >&2; exit 2; }
declare -A seen_seed
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed ${seed}" >&2; exit 2; }
  [[ -z "${seen_seed[${seed}]:-}" ]] || { echo "ERROR: duplicate seed ${seed}" >&2; exit 2; }
  seen_seed[${seed}]=1
done

if [[ "${DRY_RUN}" != "1" && "${RESUME_EXISTING_RUN}" != "1" ]]; then
  [[ ! -e "${HYBRID_DATA_ROOT}" && ! -e "${EVAL_ROOT}" ]] || {
    echo "ERROR: RUN_TAG reuses hybrid/results paths; choose a fresh tag or set RESUME_EXISTING_RUN=1" >&2
    exit 2
  }
  if [[ "${BUILD_SOURCE_DATA}" == "1" && -e "${DATA_ROOT}" ]]; then
    echo "ERROR: fresh source DATA_ROOT already exists: ${DATA_ROOT}" >&2
    exit 2
  fi
  for seed_raw in "${seed_array[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
    if compgen -G "${CKPT_DIR}/${run_id}_*" >/dev/null; then
      echo "ERROR: checkpoint artifacts already exist for ${run_id}" >&2
      exit 2
    fi
    for mask in "${mask_array[@]}"; do
      run_id="exp_qcore_hybrid_${RUN_TAG}_mhtpw${mask}_seed${seed}_pm10_pm25"
      if compgen -G "${CKPT_DIR}/${run_id}_*" >/dev/null; then
        echo "ERROR: checkpoint artifacts already exist for ${run_id}" >&2
        exit 2
      fi
    done
  done
fi

echo "q-core+T925 final five-package factorial"
echo "RUN_TAG=${RUN_TAG}"
echo "GROUP_PROFILE=${GROUP_PROFILE}; order=M,H,T,P,W"
echo "GROUPS=M:Q1000+DP1000;H:T925+Q925+DP925+RH925;T:T2M;P:MSLP;W:10m+925hPa_wind"
echo "SEEDS=${SEEDS}; S2_MODELS=$((32 * ${#seed_array[@]}))"
echo "BUILD_SOURCE_DATA=${BUILD_SOURCE_DATA}; DATA_ROOT=${DATA_ROOT}"
echo "HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"

data_jobs=""
if [[ "${RESUME_EXISTING_RUN}" == "1" || "${BUILD_SOURCE_DATA}" == "0" ]]; then
  require_dataset s1 "${S1_DATA_DIR}" train val
  require_dataset tianji "${TIANJI_DATA_DIR}" train val test
  require_dataset pangu "${PANGU_DATA_DIR}" train val test
else
  [[ "${DRY_RUN}" == "1" || -s "${PANGU2025_STATION_FILE}" ]] || {
    echo "ERROR: canonical Pangu station file missing: ${PANGU2025_STATION_FILE}" >&2; exit 2;
  }
  s1_data_job="$(submit s1_data --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${S1_DATA_DIR}" sub_s1_overlap_data.slurm)"
  tianji_data_job="$(submit tianji_data --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${TIANJI_DATA_DIR}" sub_tianji_overlap_data.slurm)"
  pangu_data_job="$(submit pangu_data --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},SOURCE_FILE=${PANGU2025_STATION_FILE},OUT_DIR=${PANGU_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1" sub_station_source_overlap_data.slurm)"
  data_jobs="${s1_data_job}:${tianji_data_job}:${pangu_data_job}"
fi

# A resume must prove that every immutable hybrid dataset exists before it
# submits any Slurm job.  Previously this check happened after base/runtime
# jobs were submitted, leaving orphaned audits when HYBRID_DATA_ROOT was wrong.
if [[ "${RESUME_EXISTING_RUN}" == "1" ]]; then
  for mask in "${mask_array[@]}"; do
    require_dataset "hybrid_${mask}" "${HYBRID_DATA_ROOT}/mhtpw_${mask}" train val test
  done
fi

data_dep="$(dep_arg "${data_jobs}")"
if [[ "${REUSE_COMPLETED_AUDITS}" == "1" ]]; then
  base_audit_job=""
  runtime_gate_job=""
  echo "[RESUME] Reusing previously completed base-data audit and shared runtime gate."
else
  audit_args=(--export="ALL,RUN_TAG=${RUN_TAG},AUDIT_PROFILE=qcore_t925,SOURCES=tianji=${TIANJI_DATA_DIR};pangu2025=${PANGU_DATA_DIR},S1_DATA_DIR=${S1_DATA_DIR},AUDIT_OUT_DIR=${EVAL_ROOT}/base_data_audit,EXPECTED_PANGU_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_PANGU_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS}")
  [[ -z "${data_dep}" ]] || audit_args+=("${data_dep}")
  base_audit_job="$(submit base_data_audit "${audit_args[@]}" sub_q_core_fair_data_audit.slurm)"
  runtime_gate_job="$(submit runtime_gate --export="ALL,BASELINE_DIR=${BASELINE_DIR}" sub_q_core_mhtpw_runtime_gate.slurm)"
fi

base_deps=""
base_deps="$(append_dep "${base_deps}" "${base_audit_job}")"
base_deps="$(append_dep "${base_deps}" "${runtime_gate_job}")"
base_dep="$(dep_arg "${base_deps}")"
if [[ "${RESUME_EXISTING_RUN}" == "1" ]]; then
  hybrid_build_job=""
else
  build_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=build,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=mhtpw,HYBRID_DATASET_PREFIX=mhtpw,MASKS=${MASKS},LIMIT_ROWS=${LIMIT_ROWS}")
  [[ -z "${base_dep}" ]] || build_args+=("${base_dep}")
  hybrid_build_job="$(submit hybrid_build "${build_args[@]}" sub_q_core_hybrid_factorial_data.slurm)"
fi

hybrid_deps=""
hybrid_deps="$(append_dep "${hybrid_deps}" "${base_audit_job}")"
hybrid_deps="$(append_dep "${hybrid_deps}" "${hybrid_build_job}")"
hybrid_dep="$(dep_arg "${hybrid_deps}")"
if [[ "${REUSE_COMPLETED_AUDITS}" == "1" ]]; then
  hybrid_audit_job=""
  echo "[RESUME] Reusing previously completed hybrid audit."
else
  hybrid_audit_args=(--export="ALL,RUN_TAG=${RUN_TAG},MODE=audit,PANGU_DATA_DIR=${PANGU_DATA_DIR},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},GROUP_PROFILE=mhtpw,HYBRID_DATASET_PREFIX=mhtpw,MASKS=${MASKS}")
  [[ -z "${hybrid_dep}" ]] || hybrid_audit_args+=("${hybrid_dep}")
  hybrid_audit_job="$(submit hybrid_audit "${hybrid_audit_args[@]}" sub_q_core_hybrid_factorial_data.slurm)"
fi

declare -A s1_jobs s2_jobs importance_jobs eval_jobs
scheduled_training=""
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  s1_run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
  if [[ "${RESUME_EXISTING_RUN}" == "1" ]] && artifact_complete "${s1_run_id}" s1; then
    s1_jobs[${seed}]=""
    echo "[RESUME] S1 ${seed}"
  else
    s1_args=(--job-name="mhtpw_s1_s${seed}" --export="ALL,EXPERIMENT=s1_q_core_t925_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_s1_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_REQUIRE_LOCAL_CACHE=1,LOWVIS_RNN_DCU_PREFLIGHT=1")
    [[ -z "${TRAIN_EXCLUDE_NODES}" ]] || s1_args+=(--exclude="${TRAIN_EXCLUDE_NODES}")
    [[ -z "${base_dep}" ]] || s1_args+=("${base_dep}")
    s1_jobs[${seed}]="$(submit "s1_seed${seed}" "${s1_args[@]}" sub_ifs_overlap_baseline.slurm)"
    scheduled_training="$(append_dep "${scheduled_training}" "${s1_jobs[${seed}]}")"
  fi
done

for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  s1_run_id="exp_qcore_hybrid_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
  s1_ckpt="${CKPT_DIR}/${s1_run_id}_S1_best_score.pt"
  for mask in "${mask_array[@]}"; do
    run_id="exp_qcore_hybrid_${RUN_TAG}_mhtpw${mask}_seed${seed}_pm10_pm25"
    if [[ "${RESUME_EXISTING_RUN}" == "1" ]] && artifact_complete "${run_id}" s2; then
      s2_jobs[${seed}_${mask}]=""
      echo "[RESUME] S2 seed=${seed} mask=${mask}"
      continue
    fi
    deps=""
    deps="$(append_dep "${deps}" "${hybrid_audit_job}")"
    deps="$(append_dep "${deps}" "${runtime_gate_job}")"
    deps="$(append_dep "${deps}" "${s1_jobs[${seed}]}")"
    dep="$(dep_arg "${deps}")"
    s2_args=(--job-name="mhtpw_${mask}_s${seed}" --export="ALL,EXPERIMENT=s2_q_core_t925_mhtpw,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S2_DATA_DIR=${HYBRID_DATA_ROOT}/mhtpw_${mask},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_mhtpw${mask}_seed${seed},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_REQUIRE_LOCAL_CACHE=1,LOWVIS_RNN_DCU_PREFLIGHT=1")
    [[ -z "${TRAIN_EXCLUDE_NODES}" ]] || s2_args+=(--exclude="${TRAIN_EXCLUDE_NODES}")
    [[ -z "${dep}" ]] || s2_args+=("${dep}")
    s2_jobs[${seed}_${mask}]="$(submit "s2_${mask}_seed${seed}" "${s2_args[@]}" sub_ifs_overlap_baseline.slurm)"
    scheduled_training="$(append_dep "${scheduled_training}" "${s2_jobs[${seed}_${mask}]}")"
  done
done

allow_smoke=0
if [[ ${#seed_array[@]} -ne 3 || "${LOWVIS_RNN_S1_STEPS:-15000}" -ne 15000 || "${LOWVIS_RNN_S2_A_STEPS:-12000}" -ne 12000 || "${LOWVIS_RNN_S2_B_STEPS:-40000}" -ne 40000 ]]; then allow_smoke=1; fi
artifact_dep="$(dep_arg "${scheduled_training}")"
artifact_args=(--export="ALL,RUN_TAG=${RUN_TAG},S1_RUN_TAG=${RUN_TAG},SEEDS=${SEEDS},MASKS=${MASKS},TRAIN_MASKS=${MASKS},GROUP_PROFILE=mhtpw,HYBRID_DATASET_PREFIX=mhtpw,RUN_MASK_PREFIX=mhtpw,S1_DATA_DIR=${S1_DATA_DIR},HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},CKPT_DIR=${CKPT_DIR},EVAL_ROOT=${EVAL_ROOT},ALLOW_SMOKE=${allow_smoke}")
[[ -z "${artifact_dep}" ]] || artifact_args+=("${artifact_dep}")
artifact_job="$(submit artifact_audit "${artifact_args[@]}" sub_q_core_hybrid_artifact_audit.slurm)"

post_jobs=""
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  dep="$(dep_arg "${artifact_job}")"
  eval_args=(--job-name="mhtpw_eval_s${seed}" --export="ALL,RUN_TAG=${RUN_TAG},MODE=evaluate_seed,SEED=${seed},SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=mhtpw,HYBRID_DATASET_PREFIX=mhtpw,RUN_MASK_PREFIX=mhtpw,SOURCE_PREFIX=qcore_hybrid_mhtpw_,HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},LIMIT_SAMPLES=${LIMIT_SAMPLES},INCLUDE_COMMON_CORE=0")
  [[ -z "${dep}" ]] || eval_args+=("${dep}")
  eval_jobs[${seed}]="$(submit "eval_seed${seed}" "${eval_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)"
  post_jobs="$(append_dep "${post_jobs}" "${eval_jobs[${seed}]}")"

  if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
    pangu_run_id="exp_qcore_hybrid_${RUN_TAG}_mhtpw00000_seed${seed}_pm10_pm25"
    tianji_run_id="exp_qcore_hybrid_${RUN_TAG}_mhtpw11111_seed${seed}_pm10_pm25"
    sources="tianji=${HYBRID_DATA_ROOT}/mhtpw_11111|${CKPT_DIR}/${tianji_run_id}_S2_PhaseB_best_score.pt|AUTO|Tianji q-core+T925;pangu2025_q_core_t925=${HYBRID_DATA_ROOT}/mhtpw_00000|${CKPT_DIR}/${pangu_run_id}_S2_PhaseB_best_score.pt|AUTO|Pangu q-core+T925"
    importance_args=(--job-name="mhtpw_imp_s${seed}" --export="ALL,SOURCES=${sources},FEATURE_IMPORTANCE_OUT_DIR=${EVAL_ROOT}/feature_importance/seed_${seed},SAMPLE_SIZE=50000,MIN_LOW_VIS=200,REPEATS=5,BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},GROUP_SCOPE=dynamic")
    [[ -z "${dep}" ]] || importance_args+=("${dep}")
    importance_jobs[${seed}]="$(submit "importance_seed${seed}" "${importance_args[@]}" sub_multi_source_feature_importance.slurm)"
  fi
done

analysis_dep="$(dep_arg "${post_jobs}")"
analysis_args=(--job-name=mhtpw_analysis --export="ALL,RUN_TAG=${RUN_TAG},MODE=analyze,SEEDS=${SEEDS},MASKS=${MASKS},GROUP_PROFILE=mhtpw,HYBRID_DATASET_PREFIX=mhtpw,SOURCE_PREFIX=qcore_hybrid_mhtpw_,HYBRID_DATA_ROOT=${HYBRID_DATA_ROOT},EVAL_ROOT=${EVAL_ROOT},ANALYSIS_DIR=${ANALYSIS_DIR},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_MAX_ROWS=${BOOTSTRAP_MAX_ROWS},OBS_ROOT=${OBS_ROOT},ERA5_DATA_DIR=${ERA5_DATA_DIR},INCLUDE_COMMON_CORE=0")
[[ -z "${analysis_dep}" ]] || analysis_args+=("${analysis_dep}")
analysis_job="$(submit factorial_analysis "${analysis_args[@]}" sub_q_core_hybrid_factorial_eval.slurm)"

eval_job_ids=""
importance_job_ids=""
for seed_raw in "${seed_array[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  eval_job_ids="$(append_dep "${eval_job_ids}" "${eval_jobs[${seed}]}")"
  if [[ "${RUN_IMPORTANCE}" == "1" ]]; then
    importance_job_ids="$(append_dep "${importance_job_ids}" "${importance_jobs[${seed}]}")"
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${EVAL_ROOT}"
  {
    echo "status=scheduled"
    echo "run_tag=${RUN_TAG}"
    echo "feature_set=${FEATURE_SET}"
    echo "group_profile=mhtpw"
    echo "group_order=M,H,T,P,W"
    echo "factorial_groups=M:Q1000+DP1000;H:T925+Q925+DP925+RH925;T:T2M;P:MSLP;W:U10+V10+WSPD10+WDIR10+U925+V925+WSPD925"
    echo "wind_secondary_diagnostics=surface_wind,925hPa_wind_grouped_model_reliance"
    echo "seeds=${SEEDS}"
    echo "masks=${MASKS}"
    echo "source_data_root=${DATA_ROOT}"
    echo "hybrid_data_root=${HYBRID_DATA_ROOT}"
    echo "base_data_audit_job=${base_audit_job}"
    echo "runtime_gate_job=${runtime_gate_job}"
    echo "hybrid_build_job=${hybrid_build_job}"
    echo "hybrid_audit_job=${hybrid_audit_job}"
    echo "training_job_ids=${scheduled_training}"
    echo "artifact_audit_job=${artifact_job}"
    echo "eval_job_ids=${eval_job_ids}"
    echo "importance_job_ids=${importance_job_ids}"
    echo "factorial_analysis_job=${analysis_job}"
    echo "analysis_dir=${ANALYSIS_DIR}"
    echo "run_importance=${RUN_IMPORTANCE}"
    echo "reuse_completed_audits=${REUSE_COMPLETED_AUDITS}"
    echo "bootstrap_iters=${BOOTSTRAP_ITERS}"
    echo "bootstrap_max_rows=${BOOTSTRAP_MAX_ROWS}"
    echo "limit_samples=${LIMIT_SAMPLES}"
    echo "obs_root=${OBS_ROOT}"
    echo "era5_data_dir=${ERA5_DATA_DIR}"
    echo "s1_steps=${LOWVIS_RNN_S1_STEPS:-15000}"
    echo "s2_a_steps=${LOWVIS_RNN_S2_A_STEPS:-12000}"
    echo "s2_b_steps=${LOWVIS_RNN_S2_B_STEPS:-40000}"
    echo "train_exclude_nodes=${TRAIN_EXCLUDE_NODES}"
    echo "claim_limit=source_block_predictive_attribution_not_isolated_variable_causality"
  } > "${EVAL_ROOT}/submission_manifest_${RUN_TAG}.txt"
fi

echo "factorial_analysis_job=${analysis_job}"
echo "results=${ANALYSIS_DIR}"
