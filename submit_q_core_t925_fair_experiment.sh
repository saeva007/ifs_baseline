#!/bin/bash
# Formal two-source q-core+T925 fair experiment:
#   3 data builds -> hard audit -> one S1 and two S2 per seed
#   -> paired validation/test inference per seed -> joint date-block analysis.
# No IFS, ERA5, hybrid masks, feature replacement, or joint-structure training.

set -euo pipefail

BASE="${BASE:-/public/home/putianshu/vis_mlp}"
BASELINE_DIR="${BASELINE_DIR:-${BASE}/ifs_baseline}"
CKPT_DIR="${CKPT_DIR:-${BASELINE_DIR}/checkpoints}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required and must be fresh}"
FEATURE_SET="q_core_t925_no_rh2m"
SEEDS="${SEEDS:-42:2025:20260702}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260721}"
BOOTSTRAP_MAX_ROWS="${BOOTSTRAP_MAX_ROWS:-0}"
# Colon-separated so the value remains one field inside Slurm --export.
FORMATS="${FORMATS:-svg:pdf:png:tiff}"
DPI="${DPI:-600}"
EXPECTED_PANGU_LEAD_MIN_HOURS="${EXPECTED_PANGU_LEAD_MIN_HOURS:-12}"
EXPECTED_PANGU_LEAD_MAX_HOURS="${EXPECTED_PANGU_LEAD_MAX_HOURS:-23}"
PANGU2025_STATION_FILE="${PANGU2025_STATION_FILE:-${BASELINE_DIR}/pangu_station/pangu_station_2025_lead12_23h_canonical.nc}"

DATA_ROOT="${DATA_ROOT:-${BASELINE_DIR}/q_core_t925_fair_datasets/${RUN_TAG}}"
S1_DATA_DIR="${S1_DATA_DIR:-${DATA_ROOT}/s1}"
TIANJI_DATA_DIR="${TIANJI_DATA_DIR:-${DATA_ROOT}/tianji}"
PANGU_DATA_DIR="${PANGU_DATA_DIR:-${DATA_ROOT}/pangu2025}"
EVAL_ROOT="${EVAL_ROOT:-${BASE}/paper_eval_results_pm10_pm25_journal/q_core_t925_fair/${RUN_TAG}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EVAL_ROOT}/analysis}"
PANGU_SOURCE_TAG="pangu2025_${FEATURE_SET}"

case "${RUN_TAG}" in
    *[!A-Za-z0-9_.-]*|"") echo "ERROR: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN}" != "1" && ! -s "${PANGU2025_STATION_FILE}" ]]; then
    echo "ERROR: canonical Pangu station product is missing: ${PANGU2025_STATION_FILE}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -e "${DATA_ROOT}" || -e "${EVAL_ROOT}" ]]; then
        echo "ERROR: RUN_TAG=${RUN_TAG} would reuse an existing data/result directory." >&2
        echo "Choose a fresh RUN_TAG; do not overwrite formal q-core outputs." >&2
        exit 2
    fi
fi

mkdir -p "${BASELINE_DIR}/logs"
cd "${BASELINE_DIR}"

submit() {
    local label="$1"
    shift
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
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid sbatch job id for ${label}: ${raw}" >&2
        exit 2
    }
    echo "[SUBMITTED] ${label}: ${job_id}" >&2
    printf '%s\n' "${job_id}"
}

join_by_colon() {
    local IFS=:
    echo "$*"
}

IFS=':' read -ra seed_array <<< "${SEEDS//,/:}"
if (( ${#seed_array[@]} == 0 )); then
    echo "ERROR: SEEDS is empty" >&2
    exit 2
fi
declare -A seen_seeds
for seed_raw in "${seed_array[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed ${seed}" >&2; exit 2; }
    [[ -z "${seen_seeds[${seed}]:-}" ]] || { echo "ERROR: duplicate seed ${seed}" >&2; exit 2; }
    seen_seeds[${seed}]=1
    if [[ "${DRY_RUN}" != "1" ]]; then
        for run_id in \
            "exp_qcore_t925_fair_${RUN_TAG}_s1_seed${seed}_pm10_pm25" \
            "exp_qcore_t925_fair_${RUN_TAG}_tianji_seed${seed}_pm10_pm25" \
            "exp_qcore_t925_fair_${RUN_TAG}_pangu2025_seed${seed}_pm10_pm25"
        do
            if compgen -G "${CKPT_DIR}/${run_id}_*" >/dev/null; then
                echo "ERROR: checkpoint artifacts already exist for ${run_id}; choose a fresh RUN_TAG." >&2
                exit 2
            fi
        done
    fi
done

echo "Pangu--Tianji q-core+T925 fair chain"
echo "RUN_TAG=${RUN_TAG}"
echo "FEATURE_SET=${FEATURE_SET} (dyn18: 15 shared meteorological + ZENITH + PM10 + PM2.5)"
echo "SEEDS=${SEEDS}"
echo "SOURCE_SCOPE=pangu_tianji (IFS/ERA5 disabled)"
echo "PANGU2025_STATION_FILE=${PANGU2025_STATION_FILE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"

s1_data_job="$(submit s1_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${S1_DATA_DIR}" \
    sub_s1_overlap_data.slurm)"
tianji_data_job="$(submit tianji_data \
    --export="ALL,FEATURE_SET=${FEATURE_SET},OUT_DIR=${TIANJI_DATA_DIR}" \
    sub_tianji_overlap_data.slurm)"
pangu_data_job="$(submit pangu_data \
    --export="ALL,SOURCE_KIND=station_nc,SOURCE_TAG=pangu2025,YEAR=2025,FEATURE_SET=${FEATURE_SET},SOURCE_FILE=${PANGU2025_STATION_FILE},OUT_DIR=${PANGU_DATA_DIR},EXPECTED_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS},INFER_PANGU_LEAD12_23_FROM_VALID_TIME=1" \
    sub_station_source_overlap_data.slurm)"

data_jobs="$(join_by_colon "${s1_data_job}" "${tianji_data_job}" "${pangu_data_job}")"
audit_job="$(submit data_audit \
    --dependency="afterok:${data_jobs}" \
    --export="ALL,RUN_TAG=${RUN_TAG},AUDIT_PROFILE=qcore_t925,SOURCES=tianji=${TIANJI_DATA_DIR};pangu2025=${PANGU_DATA_DIR},S1_DATA_DIR=${S1_DATA_DIR},AUDIT_OUT_DIR=${EVAL_ROOT}/data_audit,EXPECTED_PANGU_LEAD_MIN_HOURS=${EXPECTED_PANGU_LEAD_MIN_HOURS},EXPECTED_PANGU_LEAD_MAX_HOURS=${EXPECTED_PANGU_LEAD_MAX_HOURS}" \
    sub_q_core_fair_data_audit.slurm)"

declare -A s1_jobs
declare -A tianji_jobs
declare -A pangu_jobs
declare -A eval_jobs
all_s2_jobs=()
all_eval_jobs=()

for seed_raw in "${seed_array[@]}"; do
    seed="${seed_raw//[[:space:]]/}"
    s1_run_id="exp_qcore_t925_fair_${RUN_TAG}_s1_seed${seed}_pm10_pm25"
    tianji_run_id="exp_qcore_t925_fair_${RUN_TAG}_tianji_seed${seed}_pm10_pm25"
    pangu_run_id="exp_qcore_t925_fair_${RUN_TAG}_pangu2025_seed${seed}_pm10_pm25"
    s1_ckpt="${CKPT_DIR}/${s1_run_id}_S1_best_score.pt"

    s1_jobs[${seed}]="$(submit "s1_seed${seed}" \
        --job-name="qct925_s1_s${seed}" \
        --dependency="afterok:${audit_job}" \
        --export="ALL,EXPERIMENT=s1_q_core_t925_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${s1_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_S1_DATA_DIR=${S1_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_s1_seed${seed}" \
        sub_ifs_overlap_baseline.slurm)"

    tianji_jobs[${seed}]="$(submit "tianji_s2_seed${seed}" \
        --job-name="qct925_tj_s${seed}" \
        --dependency="afterok:${s1_jobs[${seed}]}" \
        --export="ALL,EXPERIMENT=s2_tianji_q_core_t925_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${tianji_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},OVERLAP_S2_DATA_DIR=${TIANJI_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_tianji_seed${seed}" \
        sub_ifs_overlap_baseline.slurm)"
    pangu_jobs[${seed}]="$(submit "pangu_s2_seed${seed}" \
        --job-name="qct925_pg_s${seed}" \
        --dependency="afterok:${s1_jobs[${seed}]}" \
        --export="ALL,EXPERIMENT=s2_pangu2025_q_core_t925_no_rh2m,MODEL_ARCH=static_rnn,LOWVIS_RNN_RUN_ID=${pangu_run_id},LOWVIS_RNN_SEED=${seed},OVERLAP_STATIC_RNN_PRETRAINED_CKPT=${s1_ckpt},OVERLAP_S2_DATA_DIR=${PANGU_DATA_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=1,LOWVIS_RNN_LOCAL_CACHE_ID=${RUN_TAG}_pangu_seed${seed}" \
        sub_ifs_overlap_baseline.slurm)"
    all_s2_jobs+=("${tianji_jobs[${seed}]}" "${pangu_jobs[${seed}]}")

    pair_dep="$(join_by_colon "${tianji_jobs[${seed}]}" "${pangu_jobs[${seed}]}")"
    eval_jobs[${seed}]="$(submit "paired_eval_seed${seed}" \
        --job-name="qcore_t925_eval_s${seed}" \
        --dependency="afterok:${pair_dep}" \
        --export="ALL,MODE=evaluate,RUN_TAG=${RUN_TAG},SEED=${seed},FEATURE_SET=${FEATURE_SET},SOURCE_SCOPE=pangu_tianji,TIANJI_RUN_ID=${tianji_run_id},PANGU2025_RUN_ID=${pangu_run_id},TIANJI_DATA_DIR=${TIANJI_DATA_DIR},PANGU2025_DATA_DIR=${PANGU_DATA_DIR},PANGU_SOURCE_TAG=${PANGU_SOURCE_TAG},EVAL_ROOT=${EVAL_ROOT},OUT_DIR=${EVAL_ROOT}/seed_${seed},SKIP_VALIDATION_INFERENCE=0,NO_TEMP_SCALING=1,RUN_PAIRED_SUMMARY=0,LIMIT_SAMPLES=${LIMIT_SAMPLES}" \
        sub_static_rnn_q_core_fair_eval.slurm)"
    all_eval_jobs+=("${eval_jobs[${seed}]}")
done

eval_deps="$(join_by_colon "${all_eval_jobs[@]}")"
analysis_job="$(submit joint_analysis \
    --job-name=qcore_t925_analysis \
    --dependency="afterok:${eval_deps}" \
    --export="ALL,MODE=analyze,RUN_TAG=${RUN_TAG},FEATURE_SET=${FEATURE_SET},SOURCE_SCOPE=pangu_tianji,SEEDS=${SEEDS},PANGU_SOURCE_TAG=${PANGU_SOURCE_TAG},EVAL_ROOT=${EVAL_ROOT},ANALYSIS_DIR=${ANALYSIS_DIR},BOOTSTRAP_ITERS=${BOOTSTRAP_ITERS},BOOTSTRAP_SEED=${BOOTSTRAP_SEED},BOOTSTRAP_MAX_ROWS=${BOOTSTRAP_MAX_ROWS},FORMATS=${FORMATS},DPI=${DPI}" \
    sub_static_rnn_q_core_fair_eval.slurm)"

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${EVAL_ROOT}"
    {
        echo "status=scheduled"
        echo "run_tag=${RUN_TAG}"
        echo "feature_set=${FEATURE_SET}"
        echo "dynamic_feature_count=18"
        echo "sources=tianji,pangu2025"
        echo "excluded_sources=ifs,era5"
        echo "seeds=${SEEDS}"
        echo "pangu_lead_hours=${EXPECTED_PANGU_LEAD_MIN_HOURS}..${EXPECTED_PANGU_LEAD_MAX_HOURS}"
        echo "threshold_protocol=argmax_plus_validation_matched_fpr"
        echo "bootstrap_unit=UTC_valid_date"
        echo "data_jobs=${data_jobs}"
        echo "audit_job=${audit_job}"
        echo "s2_jobs=$(join_by_colon "${all_s2_jobs[@]}")"
        echo "eval_jobs=${eval_deps}"
        echo "analysis_job=${analysis_job}"
        echo "analysis_dir=${ANALYSIS_DIR}"
    } > "${EVAL_ROOT}/submission_manifest_${RUN_TAG}.txt"
fi

echo "analysis_job=${analysis_job}"
echo "results=${ANALYSIS_DIR}"
