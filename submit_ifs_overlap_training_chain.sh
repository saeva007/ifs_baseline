#!/bin/bash
#
# Submit Static-RNN overlap training with explicit S1 -> S2 dependencies.
#
# Examples:
#   OVERLAP_CHAIN=common_core bash submit_ifs_overlap_training_chain.sh
#   OVERLAP_CHAIN=source_full bash submit_ifs_overlap_training_chain.sh
#   OVERLAP_CHAIN=overlap_full bash submit_ifs_overlap_training_chain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

OVERLAP_CHAIN="${OVERLAP_CHAIN:-common_core}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sub_ifs_overlap_baseline.slurm}"
MODEL_ARCH="${MODEL_ARCH:-static_rnn}"

case "${MODEL_ARCH}" in
    static_rnn) ;;
    *)
        echo "ERROR: dependency submitter is intended for MODEL_ARCH=static_rnn, got ${MODEL_ARCH}" >&2
        exit 2
        ;;
esac

submit_s1() {
    local exp="$1"
    sbatch --parsable \
        --export=ALL,EXPERIMENT=${exp},MODEL_ARCH=${MODEL_ARCH} \
        "${SBATCH_SCRIPT}"
}

submit_s2_after() {
    local dep="$1"
    local exp="$2"
    sbatch --parsable \
        --dependency=afterok:${dep} \
        --export=ALL,EXPERIMENT=${exp},MODEL_ARCH=${MODEL_ARCH} \
        "${SBATCH_SCRIPT}"
}

echo "Submitting IFS/Tianji overlap Static-RNN chain"
echo "OVERLAP_CHAIN=${OVERLAP_CHAIN}"
echo "SBATCH_SCRIPT=${SBATCH_SCRIPT}"

case "${OVERLAP_CHAIN}" in
    common_core)
        s1_job="$(submit_s1 s1_common_core)"
        echo "common_core S1: job ${s1_job}"
        s2_list="${S2_EXPERIMENTS:-s2_tianji_common_core s2_tianji_T2ND_rh2m_common_core s2_ifs_common_core s2_pangu2021_common_core s2_era5_2025_common_core}"
        for exp in ${s2_list}; do
            s2_job="$(submit_s2_after "${s1_job}" "${exp}")"
            echo "${exp}: afterok:${s1_job} -> job ${s2_job}"
        done
        ;;

    compact_common_core|compact)
        s1_job="$(submit_s1 s1_compact_common_core)"
        echo "compact_common_core S1: job ${s1_job}"
        s2_list="${S2_EXPERIMENTS:-s2_tianji_compact_common_core s2_ifs_compact_common_core s2_pangu2021_compact_common_core s2_era5_2025_compact_common_core}"
        for exp in ${s2_list}; do
            s2_job="$(submit_s2_after "${s1_job}" "${exp}")"
            echo "${exp}: afterok:${s1_job} -> job ${s2_job}"
        done
        ;;

    overlap_full|overlap)
        s1_job="$(submit_s1 s1_overlap)"
        echo "overlap_full S1: job ${s1_job}"
        s2_list="${S2_EXPERIMENTS:-s2_tianji s2_tianji_T2ND_rh2m s2_ifs}"
        for exp in ${s2_list}; do
            s2_job="$(submit_s2_after "${s1_job}" "${exp}")"
            echo "${exp}: afterok:${s1_job} -> job ${s2_job}"
        done
        ;;

    source_full)
        s2_list="${S2_EXPERIMENTS:-s2_tianji_source_full s2_tianji_T2ND_rh2m_source_full s2_ifs_source_full s2_pangu2025_source_full s2_era5_2025_source_full}"
        need_tianji=0
        need_ifs=0
        need_pangu=0
        need_pangu2025=0
        for exp in ${s2_list}; do
            case "${exp}" in
                s2_tianji_source_full|s2_tianji_T2ND_rh2m_source_full|s2_era5_2025_source_full)
                    need_tianji=1
                    ;;
                s2_ifs_source_full)
                    need_ifs=1
                    ;;
                s2_pangu2021_source_full|s2_pangu_source_full)
                    need_pangu=1
                    ;;
                s2_pangu2025_source_full)
                    need_pangu2025=1
                    ;;
                *)
                    echo "ERROR: no source_full S1 dependency is known for ${exp}; override the script before submitting it." >&2
                    exit 2
                    ;;
            esac
        done

        declare -A deps=()
        if [ "${need_tianji}" = "1" ]; then
            s1_tianji="$(submit_s1 s1_source_full_tianji)"
            echo "source_full S1 tianji/dyn27: job ${s1_tianji}"
            deps[s2_tianji_source_full]="${s1_tianji}"
            deps[s2_tianji_T2ND_rh2m_source_full]="${s1_tianji}"
            deps[s2_era5_2025_source_full]="${s1_tianji}"
        fi
        if [ "${need_ifs}" = "1" ]; then
            s1_ifs="$(submit_s1 s1_source_full_ifs)"
            echo "source_full S1 ifs/dyn24: job ${s1_ifs}"
            deps[s2_ifs_source_full]="${s1_ifs}"
        fi
        if [ "${need_pangu}" = "1" ]; then
            s1_pangu="$(submit_s1 s1_source_full_pangu)"
            echo "source_full S1 pangu2021/dyn21: job ${s1_pangu}"
            deps[s2_pangu2021_source_full]="${s1_pangu}"
            deps[s2_pangu_source_full]="${s1_pangu}"
        fi
        if [ "${need_pangu2025}" = "1" ]; then
            s1_pangu2025="$(submit_s1 s1_source_full_pangu2025)"
            echo "source_full S1 pangu2025/dyn19: job ${s1_pangu2025}"
            deps[s2_pangu2025_source_full]="${s1_pangu2025}"
        fi

        for exp in ${s2_list}; do
            dep="${deps[$exp]:-}"
            if [ -z "${dep}" ]; then
                echo "ERROR: no source_full S1 dependency is known for ${exp}; override the script before submitting it." >&2
                exit 2
            fi
            s2_job="$(submit_s2_after "${dep}" "${exp}")"
            echo "${exp}: afterok:${dep} -> job ${s2_job}"
        done
        ;;

    *)
        echo "ERROR: unknown OVERLAP_CHAIN=${OVERLAP_CHAIN}" >&2
        echo "Use one of: common_core, compact_common_core, overlap_full, source_full" >&2
        exit 2
        ;;
esac

echo "Submitted overlap training chain. With 10 nodes and 5 nodes/job, Slurm can run two jobs concurrently."
