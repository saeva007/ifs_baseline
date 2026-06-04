#!/bin/bash
#
# Cancel one slow source-full IFS S1 job and its queued afterok dependents, then
# submit a fresh IFS S1 -> S2 chain using a configurable node-local cache.
#
# Usage:
#   OLD_S1_JOBID=123456789 bash resubmit_source_full_ifs_chain.sh
#
# Default cache is /dev/shm because the source-full IFS X_train.npy may not fit
# in /tmp. Override only with a large node-local filesystem.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

OLD_S1_JOBID="${OLD_S1_JOBID:-}"
LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/dev/shm}"
CLEAN_LOCAL_CACHE="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_ifs_overlap_baseline.slurm}"

if [ -z "${OLD_S1_JOBID}" ]; then
    echo "ERROR: set OLD_S1_JOBID to the currently running s1_source_full_ifs job id." >&2
    echo "Example: OLD_S1_JOBID=123456789 bash resubmit_source_full_ifs_chain.sh" >&2
    exit 2
fi

case "${OLD_S1_JOBID}" in
    *[!0-9]*)
        echo "ERROR: OLD_S1_JOBID must be numeric, got ${OLD_S1_JOBID}" >&2
        exit 2
        ;;
esac

case "${LOCAL_CACHE_DIR}" in
    *","*|*" "*)
        echo "ERROR: LOWVIS_RNN_LOCAL_CACHE_DIR must not contain spaces or commas." >&2
        exit 2
        ;;
esac

mapfile -t OLD_DEPENDENTS < <(
    squeue -u "${USER}" -t PENDING -h -o "%A|%E" |
        awk -F'|' -v dep="${OLD_S1_JOBID}" '$2 ~ ("(^|:)" dep "([(:]|$)") {print $1}'
)

if ((${#OLD_DEPENDENTS[@]} > 0)); then
    echo "Cancelling queued afterok dependents of old IFS S1: ${OLD_DEPENDENTS[*]}"
    scancel "${OLD_DEPENDENTS[@]}"
else
    echo "No queued afterok dependents found for old IFS S1 job ${OLD_S1_JOBID}."
fi

if squeue -h -j "${OLD_S1_JOBID}" -o "%A" | grep -qx "${OLD_S1_JOBID}"; then
    echo "Cancelling slow old IFS S1 job ${OLD_S1_JOBID}"
    scancel "${OLD_S1_JOBID}"
else
    echo "Old IFS S1 job ${OLD_S1_JOBID} is no longer active; continuing with resubmit."
fi

echo "Submitting replacement IFS source-full S1 with LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR}"
NEW_S1_JOB="$(
    sbatch --parsable \
        --export=ALL,EXPERIMENT=s1_source_full_ifs,MODEL_ARCH=static_rnn,LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=${CLEAN_LOCAL_CACHE} \
        "${SBATCH_SCRIPT}"
)"

NEW_S2_JOB="$(
    sbatch --parsable \
        --dependency=afterok:${NEW_S1_JOB} \
        --export=ALL,EXPERIMENT=s2_ifs_source_full,MODEL_ARCH=static_rnn,LOWVIS_RNN_LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR},LOWVIS_RNN_CLEAN_LOCAL_CACHE=0 \
        "${SBATCH_SCRIPT}"
)"

echo "Replacement chain submitted:"
echo "  IFS source-full S1: ${NEW_S1_JOB}"
echo "  IFS source-full S2: ${NEW_S2_JOB} (afterok:${NEW_S1_JOB})"
echo "Other Tianji/T2ND/ERA5/Pangu jobs were not cancelled."
