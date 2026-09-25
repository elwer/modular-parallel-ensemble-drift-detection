#!/bin/bash
# =============================================================================
# CPU frequency study: run scalability benchmark at different clock speeds
# to test whether the workload is memory-bound.
#
# Hypothesis: if memory-bound, reducing CPU frequency increases speedup
# because the compute/memory speed ratio shifts in favor of memory bandwidth.
#
# AMD EPYC 7702: base 2.0 GHz, boost up to ~3.35 GHz, min ~1.5 GHz
# Available scaling frequencies: 2000000 1800000 1500000 (kHz)
# Boost frequencies (up to ~3350 MHz) are controlled by the boost flag,
# not by scaling_setspeed. We use:
#   high = performance governor (boost enabled, up to ~3350 MHz)
#   mid  = userspace governor at 2000000 kHz (2.0 GHz, no boost)
#   low  = userspace governor at 1500000 kHz (1.5 GHz, no boost)
#
# Slurm --cpu-freq does NOT work on this cluster. We set frequency manually
# via sysfs (scaling_governor + scaling_setspeed) on all cores.
#
# Usage:
#   bash further_studies/submit_freq_study.sh           # submit all
#   bash further_studies/submit_freq_study.sh --dry-run  # generate sbatch only
#   bash further_studies/submit_freq_study.sh --only "numa_phys_0_31_high numa_phys_0_31_low"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GRIDSEARCH_DIR="${SCRIPT_DIR}/results_freq_study"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_freq_study"
mkdir -p "$GRIDSEARCH_DIR" "$SBATCH_DIR"

# ---- Common benchmark parameters ----
STREAM_LENGTH=2000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
N_DIM=32
WALL_TIME="06:00:00"
MEM_PER_CPU=1024

PY_COMMON="--stream-length ${STREAM_LENGTH} --drift-frequency ${DRIFT_FREQ} \
--n-repeats ${N_REPEATS} --seed ${SEED} --scenario ${SCENARIO} \
--n-dimensions ${N_DIM}"

# ---- Frequency levels ----
# Each level defines: governor, frequency in kHz (for userspace), boost flag
# high: performance governor, boost=1 (CPU free to boost up to ~3350 MHz)
# mid:  userspace governor, 2000000 kHz (2.0 GHz), boost=0
# low:  userspace governor, 1500000 kHz (1.5 GHz), boost=0
declare -A FREQ_GOVERNOR
FREQ_GOVERNOR[high]="performance"
FREQ_GOVERNOR[mid]="userspace"
FREQ_GOVERNOR[low]="userspace"

declare -A FREQ_KHZ
FREQ_KHZ[high]=""           # performance governor — no fixed freq
FREQ_KHZ[mid]="2000000"    # 2.0 GHz
FREQ_KHZ[low]="1500000"    # 1.5 GHz

declare -A FREQ_BOOST
FREQ_BOOST[high]="1"        # boost enabled
FREQ_BOOST[mid]="0"        # boost disabled
FREQ_BOOST[low]="0"        # boost disabled

declare -A FREQ_LABELS
FREQ_LABELS[high]="perf-boost"
FREQ_LABELS[mid]="2000MHz"
FREQ_LABELS[low]="1500MHz"

# ---- Parse CLI flags ----
DRY_RUN=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --only)     ONLY="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Helper: generate sbatch file and submit ----
submit_freq_config() {
    local base_name="$1"     # config name without freq suffix
    local slurm_extra="$2"   # extra #SBATCH lines
    local cpus="$3"           # --cpus-per-task
    local numactl_cmd="$4"   # numactl prefix (empty = none)
    local srun_cmd="$5"      # srun prefix (empty = none)
    local py_extra="$6"     # extra python args (e.g. --pin-cpus)
    local freq_governor="$7"  # scaling governor
    local freq_khz="$8"       # frequency in kHz (empty for performance)
    local freq_boost="$9"     # boost flag (0 or 1)
    local freq_label="${10}"  # "high" | "mid" | "low"

    local name="${base_name}_${freq_label}"

    # Skip if --only filter is set and this name is not in the list
    if [[ -n "$ONLY" ]] && [[ " $ONLY " != *" $name "* ]]; then
        return
    fi

    local sbatch_file="${SBATCH_DIR}/${name}.sbatch"
    local output_dir="${GRIDSEARCH_DIR}/${name}"

    # Build the command line (avoid leading/trailing spaces)
    local cmd_prefix=""
    [[ -n "$numactl_cmd" ]] && cmd_prefix="${numactl_cmd}"
    [[ -n "$srun_cmd" ]] && cmd_prefix="${cmd_prefix:+${cmd_prefix} }${srun_cmd}"
    cmd_prefix="${cmd_prefix:+${cmd_prefix} }python -u further_studies/run_scalability_study.py"

    # Build the full py command with optional --pin-cpus folded in
    local py_args="${PY_COMMON}"
    [[ -n "$py_extra" ]] && py_args="${py_args} ${py_extra}"

    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=FREQ-${name}
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

# Set boost flag (0=disabled, 1=enabled)
if [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
    echo "\${FREQ_BOOST_VAL}" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
fi

# Set governor and frequency on all cores
for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
    cpu_num=\$(echo "\$cpu_dir" | grep -oP 'cpu\K[0-9]+')
    gov_file="\$cpu_dir/cpufreq/scaling_governor"
    setspeed_file="\$cpu_dir/cpufreq/scaling_setspeed"

    # Set governor
    if [[ -w "\$gov_file" ]]; then
        echo "\${FREQ_GOVERNOR}" > "\$gov_file" 2>/dev/null || true
    fi

    # Set frequency (only for userspace governor)
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
        echo "cpu\${core}: \${freq} kHz (\$(echo "scale=3; \${freq}/1000000" | bc 2>/dev/null || echo '?') GHz)"
    fi
done >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== cpuinfo MHz (first 8 cores) ===" >> "\${OUTPUT_DIR}/env_info.txt"
grep "cpu MHz" /proc/cpuinfo | head -8 >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true

echo "Running config: ${name} (governor=\${FREQ_GOVERNOR}, freq=\${FREQ_KHZ_VAL:-auto}, boost=\${FREQ_BOOST_VAL})"
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
# Top configs from NUMA grid search, tested at 3 frequency levels each.
# All configs request the full node (128 cpus, --exclusive).
#
# Each call: submit_freq_config <name> <slurm_extra> <cpus> <numactl> <srun> <py_extra> <freq_mhz> <freq_label>
# =============================================================================

# Define the base configs to test (subset of best from NUMA grid search)
# We use a function to submit one config at all 3 frequencies
submit_all_freqs() {
    local name="$1"
    local slurm_extra="$2"
    local cpus="$3"
    local numactl_cmd="$4"
    local srun_cmd="$5"
    local py_extra="$6"

    for freq_label in high mid low; do
        submit_freq_config "$name" "$slurm_extra" "$cpus" \
            "$numactl_cmd" "$srun_cmd" "$py_extra" \
            "${FREQ_GOVERNOR[$freq_label]}" \
            "${FREQ_KHZ[$freq_label]}" \
            "${FREQ_BOOST[$freq_label]}" \
            "$freq_label"
    done
}

# ---- Config 1: Baseline (no NUMA control, no pinning) ----
submit_all_freqs "baseline" \
    "#SBATCH --hint=nomultithread" 128 "" "" ""

# ---- Config 2: Baseline with --pin-cpus ----
submit_all_freqs "baseline_pin" \
    "#SBATCH --hint=nomultithread" 128 "" "" "--pin-cpus"

# ---- Config 3: Best from grid search — numa_phys_0_31 ----
# numactl --physcpubind=0-31 --membind=0 (no Python pinning)
submit_all_freqs "numa_phys_0_31" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --physcpubind=0-31 --membind=0" "" ""

# ---- Config 4: numa_phys_0_31 with --pin-cpus ----
submit_all_freqs "numa_phys_0_31_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --physcpubind=0-31 --membind=0" "" "--pin-cpus"

# ---- Config 5: Single NUMA node 0 ----
submit_all_freqs "numa_n0" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" "" ""

# ---- Config 6: Single NUMA node 0 with --pin-cpus ----
submit_all_freqs "numa_n0_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

# ---- Config 7: Interleave across both NUMA nodes ----
submit_all_freqs "numa_interleave" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --interleave=all" "" ""

# ---- Config 8: taskset cores 0-31 (second-best from grid search) ----
submit_all_freqs "taskset_0_31" \
    "#SBATCH --hint=nomultithread" 128 \
    "taskset -c 0-31" "" ""

# ---- Config 9: srun_sockets_pin (best for K=128) ----
submit_all_freqs "srun_sockets_pin" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=sockets" "--pin-cpus"

# ---- Config 10: SMT on baseline (for K=128 comparison) ----
submit_all_freqs "baseline_smt" \
    "" 128 "" "" ""

# =============================================================================
echo ""
echo "============================================"
echo "Frequency study submission complete."
echo "Total configurations: $(ls "$SBATCH_DIR"/*.sbatch 2>/dev/null | wc -l)"
echo "Results will appear in: $GRIDSEARCH_DIR/"
echo "Sbatch files in:        $SBATCH_DIR/"
echo ""
echo "After jobs complete, analyze with:"
echo "  python further_studies/analyze_freq_study.py"
echo "============================================"
