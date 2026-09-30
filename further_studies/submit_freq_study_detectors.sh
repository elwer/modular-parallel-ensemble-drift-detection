#!/bin/bash
# =============================================================================
# CPU frequency study with multiple detector types on AMD EPYC 7702.
#
# Same methodology as submit_freq_study.sh but runs the balanced scenario
# with each detector type (UDetect, SPLL, D3, OCDD, CSDDM, IBDD) to compare
# scaling behavior across detector implementations.
#
# AMD EPYC 7702: base 2.0 GHz, boost up to ~3.35 GHz, min ~1.5 GHz
# Available scaling frequencies: 2000000 1800000 1500000 (kHz)
# We use:
#   mid  = userspace governor at 2000000 kHz (2.0 GHz, no boost)
#   low  = userspace governor at 1500000 kHz (1.5 GHz, no boost)
#
# Slurm --cpu-freq does NOT work on this cluster. We set frequency manually
# via sysfs (scaling_governor + scaling_setspeed) on all cores.
#
# Results are stored in separate directories per detector to avoid overwriting.
#
# Usage:
#   bash further_studies/submit_freq_study_detectors.sh                # submit all detectors
#   bash further_studies/submit_freq_study_detectors.sh --dry-run      # generate sbatch only
#   bash further_studies/submit_freq_study_detectors.sh --only-detectors "UDetect SPLL"
#   bash further_studies/submit_freq_study_detectors.sh --only "UDetect_baseline_pin_mid"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Common benchmark parameters ----
STREAM_LENGTH=2000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
N_DIM=32
WALL_TIME="06:00:00"
MEM_PER_CPU=1024

# ---- Detectors to study ----
ALL_DETECTORS="UDetect SPLL D3 OCDD CSDDM IBDD"

# ---- Frequency levels ----
declare -A FREQ_GOVERNOR
FREQ_GOVERNOR[mid]="userspace"
FREQ_GOVERNOR[low]="userspace"

declare -A FREQ_KHZ
FREQ_KHZ[mid]="2000000"    # 2.0 GHz (base frequency, no boost)
FREQ_KHZ[low]="1500000"    # 1.5 GHz (reduced frequency, no boost)

declare -A FREQ_BOOST
FREQ_BOOST[mid]="0"
FREQ_BOOST[low]="0"

declare -A FREQ_LABELS
FREQ_LABELS[mid]="2000MHz"
FREQ_LABELS[low]="1500MHz"

# ---- Parse CLI flags ----
DRY_RUN=0
ONLY=""
ONLY_DETECTORS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=1; shift ;;
        --only)            ONLY="$2"; shift 2 ;;
        --only-detectors)  ONLY_DETECTORS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -n "$ONLY_DETECTORS" ]]; then
    ALL_DETECTORS="$ONLY_DETECTORS"
fi

# ---- Helper: generate sbatch file and submit ----
submit_freq_config() {
    local detector="$1"      # detector name
    local base_name="$2"     # config name without freq suffix
    local slurm_extra="$3"   # extra #SBATCH lines
    local cpus="$4"           # --cpus-per-task
    local numactl_cmd="$5"   # numactl prefix (empty = none)
    local srun_cmd="$6"      # srun prefix (empty = none)
    local py_extra="$7"      # extra python args (e.g. --pin-cpus)
    local freq_governor="$8"  # scaling governor
    local freq_khz="$9"       # frequency in kHz
    local freq_boost="${10}"  # boost flag (0 or 1)
    local freq_label="${11}"  # "mid" | "low"

    local name="${detector}_${base_name}_${freq_label}"
    local lower_detector=$(echo "$detector" | tr 'A-Z' 'a-z')
    local gridsearch_dir="${SCRIPT_DIR}/results_freq_study_${lower_detector}"
    local sbatch_dir="${SCRIPT_DIR}/sbatch_freq_study_${lower_detector}"
    mkdir -p "$gridsearch_dir" "$sbatch_dir"

    # Skip if --only filter is set and this name is not in the list
    if [[ -n "$ONLY" ]] && [[ " $ONLY " != *" $name "* ]]; then
        return
    fi

    local sbatch_file="${sbatch_dir}/${name}.sbatch"
    local output_dir="${gridsearch_dir}/${name}"

    local py_common="--stream-length ${STREAM_LENGTH} --drift-frequency ${DRIFT_FREQ} \
--n-repeats ${N_REPEATS} --seed ${SEED} --scenario ${SCENARIO} \
--n-dimensions ${N_DIM} --detector ${detector}"

    # Build the command line
    local cmd_prefix=""
    [[ -n "$numactl_cmd" ]] && cmd_prefix="${numactl_cmd}"
    [[ -n "$srun_cmd" ]] && cmd_prefix="${cmd_prefix:+${cmd_prefix} }${srun_cmd}"
    cmd_prefix="${cmd_prefix:+${cmd_prefix} }python -u further_studies/run_scalability_study.py"

    local py_args="${py_common}"
    [[ -n "$py_extra" ]] && py_args="${py_args} ${py_extra}"

    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=FREQD-${name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${WALL_TIME}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --exclusive
${slurm_extra}

set -euo pipefail
cd "\${SLURM_SUBMIT_DIR:-${PROJECT_ROOT}}"
source setup.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${output_dir}"
mkdir -p "\${OUTPUT_DIR}"

# ---- Set CPU frequency on ALL cores via sysfs ----
FREQ_GOVERNOR="${freq_governor}"
FREQ_KHZ_VAL="${freq_khz}"
FREQ_BOOST_VAL="${freq_boost}"

echo "Setting CPU frequency: governor=\${FREQ_GOVERNOR}, freq=\${FREQ_KHZ_VAL:-auto}, boost=\${FREQ_BOOST_VAL}"

if [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
    echo "\${FREQ_BOOST_VAL}" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
fi

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
    gov_file="\$cpu_dir/cpufreq/scaling_governor"
    setspeed_file="\$cpu_dir/cpufreq/scaling_setspeed"
    if [[ -w "\$gov_file" ]]; then
        echo "\${FREQ_GOVERNOR}" > "\$gov_file" 2>/dev/null || true
    fi
    if [[ "\${FREQ_GOVERNOR}" == "userspace" ]] && [[ -n "\${FREQ_KHZ_VAL}" ]] && [[ -w "\$setspeed_file" ]]; then
        echo "\${FREQ_KHZ_VAL}" > "\$setspeed_file" 2>/dev/null || true
    fi
done

# ---- Log environment info ----
echo "=== NUMA topology ===" > "\${OUTPUT_DIR}/env_info.txt"
numactl --hardware >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== lscpu (relevant) ===" >> "\${OUTPUT_DIR}/env_info.txt"
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line|Thread|Core|Socket|NUMA|NUMA node|CPU min|CPU max|MHz)' >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== Slurm env ===" >> "\${OUTPUT_DIR}/env_info.txt"
env | grep -E '^(SLURM_|OMP_|OPENBLAS|MKL|NUMEXPR)' | sort >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== CPU affinity ===" >> "\${OUTPUT_DIR}/env_info.txt"
taskset -cp \$\$ >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== numactl --show ===" >> "\${OUTPUT_DIR}/env_info.txt"
numactl --show >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== cpufreq available frequencies ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== boost flag ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_governor (cpu0) ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_setspeed (cpu0) ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== CPU frequency (all cores, AFTER setting) ===" >> "\${OUTPUT_DIR}/env_info.txt"
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    if [[ -r "\$f" ]]; then
        core=\$(echo "\$f" | grep -oP 'cpu\K[0-9]+')
        freq=\$(cat "\$f" 2>/dev/null || echo "N/A")
        echo "cpu\${core}: \${freq} kHz"
    fi
done >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true

echo "Running config: ${name} (detector=${detector}, governor=\${FREQ_GOVERNOR}, freq=\${FREQ_KHZ_VAL:-auto}, boost=\${FREQ_BOOST_VAL})"
echo "numactl: [${numactl_cmd}]"
echo "srun:    [${srun_cmd}]"
echo "py_extra:[${py_extra}]"

${cmd_prefix} \\
    ${py_args} \\
    --output-dir "\${OUTPUT_DIR}" \\
    2>&1 | tee "\${OUTPUT_DIR}/scalability.log"
EOF

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Generated: ${sbatch_file}"
    else
        local job_id
        job_id=$(sbatch --parsable "$sbatch_file")
        echo "Submitted ${name} -> JobID ${job_id}"
    fi
}

# =============================================================================
# CONFIGURATION GRID
# =============================================================================
# Same configs as the BNDM frequency study, tested at 2 frequency levels each.
# All configs request the full node (128 cpus, --exclusive).
# =============================================================================

submit_all_freqs() {
    local detector="$1"
    local name="$2"
    local slurm_extra="$3"
    local cpus="$4"
    local numactl_cmd="$5"
    local srun_cmd="$6"
    local py_extra="$7"

    for freq_label in mid low; do
        submit_freq_config "$detector" "$name" "$slurm_extra" "$cpus" \
            "$numactl_cmd" "$srun_cmd" "$py_extra" \
            "${FREQ_GOVERNOR[$freq_label]}" \
            "${FREQ_KHZ[$freq_label]}" \
            "${FREQ_BOOST[$freq_label]}" \
            "$freq_label"
    done
}

# =============================================================================
# Submit for each detector
# =============================================================================
for detector in $ALL_DETECTORS; do
    echo ""
    echo "===== Submitting frequency study for detector: ${detector} ====="

    # ---- Config 1: Baseline (no NUMA control, no pinning) ----
    submit_all_freqs "$detector" "baseline" \
        "#SBATCH --hint=nomultithread" 128 "" "" ""

    # ---- Config 2: Baseline with --pin-cpus ----
    submit_all_freqs "$detector" "baseline_pin" \
        "#SBATCH --hint=nomultithread" 128 "" "" "--pin-cpus"

    # ---- Config 3: numa_phys_0_31 ----
    submit_all_freqs "$detector" "numa_phys_0_31" \
        "#SBATCH --hint=nomultithread" 128 \
        "numactl --physcpubind=0-31 --membind=0" "" ""

    # ---- Config 4: numa_phys_0_31 with --pin-cpus ----
    submit_all_freqs "$detector" "numa_phys_0_31_pin" \
        "#SBATCH --hint=nomultithread" 128 \
        "numactl --physcpubind=0-31 --membind=0" "" "--pin-cpus"

    # ---- Config 5: Single NUMA node 0 ----
    submit_all_freqs "$detector" "numa_n0" \
        "#SBATCH --hint=nomultithread" 128 \
        "numactl --cpunodebind=0 --membind=0" "" ""

    # ---- Config 6: Single NUMA node 0 with --pin-cpus ----
    submit_all_freqs "$detector" "numa_n0_pin" \
        "#SBATCH --hint=nomultithread" 128 \
        "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

    # ---- Config 7: Interleave across both NUMA nodes ----
    submit_all_freqs "$detector" "numa_interleave" \
        "#SBATCH --hint=nomultithread" 128 \
        "numactl --interleave=all" "" ""

    # ---- Config 8: taskset cores 0-31 ----
    submit_all_freqs "$detector" "taskset_0_31" \
        "#SBATCH --hint=nomultithread" 128 \
        "taskset -c 0-31" "" ""

    # ---- Config 9: srun_sockets_pin ----
    submit_all_freqs "$detector" "srun_sockets_pin" \
        "#SBATCH --hint=nomultithread" 128 "" \
        "srun --cpu-bind=sockets" "--pin-cpus"
done

# =============================================================================
echo ""
echo "============================================"
echo "Multi-detector frequency study submission complete."
echo "Detectors: ${ALL_DETECTORS}"
echo ""
for detector in $ALL_DETECTORS; do
    lower_detector=$(echo "$detector" | tr 'A-Z' 'a-z')
    sbatch_dir="${SCRIPT_DIR}/sbatch_freq_study_${lower_detector}"
    results_dir="${SCRIPT_DIR}/results_freq_study_${lower_detector}"
    count=$(ls "$sbatch_dir"/*.sbatch 2>/dev/null | wc -l)
    echo "  ${detector}: ${count} jobs, results -> ${results_dir}/"
done
echo ""
echo "After jobs complete, analyze with:"
for detector in $ALL_DETECTORS; do
    lower_detector=$(echo "$detector" | tr 'A-Z' 'a-z')
    echo "  python further_studies/analyze_freq_study.py --results-dir further_studies/results_freq_study_${lower_detector}"
done
echo "============================================"
