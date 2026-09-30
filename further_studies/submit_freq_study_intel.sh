#!/bin/bash
# =============================================================================
# CPU frequency study: Intel Xeon Platinum 8470 (Sapphire Rapids)
#
# 2 x Intel Xeon Platinum 8470 (52 cores) @ 2.00 GHz
# 104 cores total, 2 NUMA domains (1 per socket), hyperthreading disabled
#
# Intel frequency control:
#   - Turbo disable via /sys/devices/system/cpu/intel_pstate/no_turbo
#   - userspace governor + scaling_setspeed for fixed frequency
#   - Available frequencies depend on the CPU; verify with:
#       cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
#
# Frequency levels (both with turbo disabled for determinism):
#   mid = userspace governor at base frequency (2000 MHz, no turbo)
#   low = userspace governor at reduced frequency (1500 MHz, no turbo)
#
# Usage:
#   bash further_studies/submit_freq_study_intel.sh           # submit all
#   bash further_studies/submit_freq_study_intel.sh --dry-run  # generate sbatch only
#   bash further_studies/submit_freq_study_intel.sh --only "baseline_pin_mid baseline_pin_low"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results_freq_study_intel"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_freq_study_intel"
mkdir -p "$RESULTS_DIR" "$SBATCH_DIR"

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
# Both with turbo disabled for deterministic, reproducible measurements.
# NOTE: Verify available frequencies on the target node:
#   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
declare -A FREQ_GOVERNOR
FREQ_GOVERNOR[mid]="userspace"
FREQ_GOVERNOR[low]="userspace"
FREQ_GOVERNOR[ultra_low]="userspace"

declare -A FREQ_KHZ
FREQ_KHZ[mid]="2000000"    # 2.0 GHz (base frequency, no turbo)
FREQ_KHZ[low]="1500000"    # 1.5 GHz (reduced frequency, no turbo)
FREQ_KHZ[ultra_low]="800000"  # 800 MHz (minimum frequency, no turbo)

declare -A FREQ_LABELS
FREQ_LABELS[mid]="2000MHz"
FREQ_LABELS[low]="1500MHz"
FREQ_LABELS[ultra_low]="800MHz"

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
    local freq_khz="$8"       # frequency in kHz
    local freq_label="$9"    # "mid" | "low"

    local name="${base_name}_${freq_label}"

    # Skip if --only filter is set and this name is not in the list
    if [[ -n "$ONLY" ]] && [[ " $ONLY " != *" $name "* ]]; then
        return
    fi

    local sbatch_file="${SBATCH_DIR}/${name}.sbatch"
    local output_dir="${RESULTS_DIR}/${name}"

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
#SBATCH --job-name=FREQI-${name}
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

# ---- Disable Intel Turbo Boost ----
if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    echo "Intel Turbo Boost disabled"
fi

# ---- Set CPU frequency on ALL cores via sysfs ----
FREQ_GOVERNOR="${freq_governor}"
FREQ_KHZ_VAL="${freq_khz}"

echo "Setting CPU frequency: governor=\${FREQ_GOVERNOR}, freq=\${FREQ_KHZ_VAL} kHz"

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
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line|Thread|Core|Socket|NUMA|NUMA node|CPU min|CPU max|MHz|Model name)' >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== Slurm env ===" >> "\${OUTPUT_DIR}/env_info.txt"
env | grep -E '^(SLURM_|OMP_|OPENBLAS|MKL|NUMEXPR)' | sort >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== CPU affinity ===" >> "\${OUTPUT_DIR}/env_info.txt"
taskset -cp \$\$ >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== numactl --show ===" >> "\${OUTPUT_DIR}/env_info.txt"
numactl --show >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== cpufreq available frequencies ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== intel_pstate no_turbo ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
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

echo "Running config: ${name} (governor=\${FREQ_GOVERNOR}, freq=\${FREQ_KHZ_VAL} kHz, turbo=disabled)"
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
# Intel Xeon Platinum 8470: 104 cores (52 per socket), 2 NUMA domains
# K values: 2, 4, 8, 16, 26, 52, 104
#   K=26  = half socket
#   K=52  = full socket (1 NUMA domain)
#   K=104 = full node (both sockets)
#
# All configs request the full node (104 cpus, --exclusive).
# =============================================================================

submit_all_freqs() {
    local name="$1"
    local slurm_extra="$2"
    local cpus="$3"
    local numactl_cmd="$4"
    local srun_cmd="$5"
    local py_extra="$6"

    for freq_label in mid low ultra_low; do
        submit_freq_config "$name" "$slurm_extra" "$cpus" \
            "$numactl_cmd" "$srun_cmd" "$py_extra" \
            "${FREQ_GOVERNOR[$freq_label]}" \
            "${FREQ_KHZ[$freq_label]}" \
            "$freq_label"
    done
}

# ---- Config 1: Baseline (no NUMA control, no pinning) ----
submit_all_freqs "baseline" \
    "#SBATCH --hint=nomultithread" 104 "" "" ""

# ---- Config 2: Baseline with --pin-cpus ----
submit_all_freqs "baseline_pin" \
    "#SBATCH --hint=nomultithread" 104 "" "" "--pin-cpus"

# ---- Config 3: Single NUMA node 0 (one socket) ----
submit_all_freqs "numa_n0" \
    "#SBATCH --hint=nomultithread" 104 \
    "numactl --cpunodebind=0 --membind=0" "" ""

# ---- Config 4: Single NUMA node 0 with --pin-cpus ----
submit_all_freqs "numa_n0_pin" \
    "#SBATCH --hint=nomultithread" 104 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

# ---- Config 5: Interleave across both NUMA nodes ----
submit_all_freqs "numa_interleave" \
    "#SBATCH --hint=nomultithread" 104 \
    "numactl --interleave=all" "" ""

# ---- Config 6: Interleave with --pin-cpus ----
submit_all_freqs "numa_interleave_pin" \
    "#SBATCH --hint=nomultithread" 104 \
    "numactl --interleave=all" "" "--pin-cpus"

# ---- Config 7: taskset first socket (cores 0-51) ----
submit_all_freqs "taskset_0_51" \
    "#SBATCH --hint=nomultithread" 104 \
    "taskset -c 0-51" "" ""

# ---- Config 8: srun_sockets_pin (spread across both sockets) ----
submit_all_freqs "srun_sockets_pin" \
    "#SBATCH --hint=nomultithread" 104 "" \
    "srun --cpu-bind=sockets" "--pin-cpus"

# =============================================================================
echo ""
echo "============================================"
echo "Intel frequency study submission complete."
echo "Total configurations: $(ls "$SBATCH_DIR"/*.sbatch 2>/dev/null | wc -l)"
echo "Results will appear in: $RESULTS_DIR/"
echo "Sbatch files in:        $SBATCH_DIR/"
echo ""
echo "After jobs complete, analyze with:"
echo "  python further_studies/analyze_freq_study.py --results-dir $RESULTS_DIR"
echo "============================================"
