#!/bin/bash
# =============================================================================
# CPU frequency study: run scalability benchmark at different clock speeds
# to test whether the workload is memory-bound.
#
# Hypothesis: if memory-bound, reducing CPU frequency increases speedup
# because the compute/memory speed ratio shifts in favor of memory bandwidth.
#
# AMD EPYC 7702: base 2.0 GHz, boost up to ~3.35 GHz, min ~1.5 GHz
# Slurm --cpu-freq accepts MHz values (e.g., 3350 = 3.35 GHz)
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

# ---- Frequency levels (in MHz for Slurm --cpu-freq) ----
# AMD EPYC 7702: min ~1500 MHz, base 2000 MHz, boost ~3350 MHz
declare -A FREQ_LABELS
FREQ_LABELS[high]="3350"    # max boost
FREQ_LABELS[mid]="2000"     # base clock
FREQ_LABELS[low]="1500"     # minimum frequency

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
    local freq_mhz="$7"      # CPU frequency in MHz
    local freq_label="$8"    # "high" | "mid" | "low"

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
#SBATCH --cpu-freq=${freq_mhz}
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

# Log NUMA topology, CPU frequency, and affinity for this job
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
echo "=== CPU frequency (all cores) ===" >> "\${OUTPUT_DIR}/env_info.txt"
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    if [[ -r "\$f" ]]; then
        core=\$(echo "\$f" | grep -oP 'cpu\K[0-9]+')
        freq=\$(cat "\$f" 2>/dev/null || echo "N/A")
        echo "cpu\${core}: \${freq} kHz (\$(echo "scale=3; \${freq}/1000000" | bc 2>/dev/null || echo '?') GHz)"
    fi
done >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== cpufreq available frequencies ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_governor ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== SLURM_CPU_FREQ_REQ ===" >> "\${OUTPUT_DIR}/env_info.txt"
echo "\${SLURM_CPU_FREQ_REQ:-not_set}" >> "\${OUTPUT_DIR}/env_info.txt"

echo "Running config: ${name} (freq=${freq_mhz} MHz, label=${freq_label})"
echo "numactl: [${numactl_cmd}]"
echo "srun:    [${srun_cmd}]"
echo "py_extra:[${py_extra}]"
echo "CPU freq target: ${freq_mhz} MHz"

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
            "${FREQ_LABELS[$freq_label]}" "$freq_label"
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
