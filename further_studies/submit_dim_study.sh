#!/bin/bash
# =============================================================================
# Dimension sweep study: test whether higher compute density improves speedup.
#
# If the bottleneck is synchronization overhead (not compute), then increasing
# n_dimensions should shift the compute/sync ratio and improve speedup.
#
# Uses the best config from L3 study: 2 NUMA domains, pinned, high frequency.
# Sweeps n_dimensions = 4, 8, 16, 32, 64, 128.
#
# Usage:
#   bash further_studies/submit_dim_study.sh           # submit all
#   bash further_studies/submit_dim_study.sh --dry-run  # generate sbatch only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results_dim_study"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_dim_study"
mkdir -p "$RESULTS_DIR" "$SBATCH_DIR"

# ---- Common benchmark parameters ----
STREAM_LENGTH=2000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
WALL_TIME="24:00:00"
MEM_PER_CPU=1024

# Use full node: test scaling across all hardware boundaries
# K=3 (1 CCX), K=7 (2 CCX), K=15 (1 NUMA), K=31 (2 NUMA), K=63 (1 socket), K=127 (full node)
CPU_LIST="0-127"
NUMACTL_MEM="--interleave=all"
ENSEMBLE_SIZES="3,7,15,31,63,127"

# ---- Parse CLI flags ----
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Dimensions to sweep ----
DIMS_LIST=(16 32 64 128 256)

# ---- Helper: set CPU frequency to high (performance governor) ----
set_cpu_freq_high() {
    if [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
        echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
    fi
    for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
        gov_file="$cpu_dir/cpufreq/scaling_governor"
        if [[ -w "$gov_file" ]]; then
            echo "performance" > "$gov_file" 2>/dev/null || true
        fi
    done
}

# ---- Submit one config ----
submit_dim_config() {
    local n_dim="$1"

    local name="dim${n_dim}"
    local sbatch_file="${SBATCH_DIR}/${name}.sbatch"
    local output_dir="${RESULTS_DIR}/${name}"

    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=DIM-${name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=${WALL_TIME}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --exclusive
#SBATCH --hint=nomultithread

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

# ---- Set CPU frequency to high ----
$(declare -f set_cpu_freq_high)
set_cpu_freq_high

# ---- Log environment ----
echo "=== NUMA topology ===" > "\${OUTPUT_DIR}/env_info.txt"
numactl --hardware >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== CPU list ===" >> "\${OUTPUT_DIR}/env_info.txt"
echo "${CPU_LIST}" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== n_dimensions ===" >> "\${OUTPUT_DIR}/env_info.txt"
echo "${n_dim}" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_governor (cpu0) ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== cpuinfo MHz (first 8 cores) ===" >> "\${OUTPUT_DIR}/env_info.txt"
grep "cpu MHz" /proc/cpuinfo | head -8 >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true

echo "Running config: ${name} (n_dim=${n_dim}, cpus=${CPU_LIST})"

numactl ${NUMACTL_MEM} taskset -c ${CPU_LIST} \\
    python -u further_studies/run_scalability_study.py \\
    --stream-length ${STREAM_LENGTH} --drift-frequency ${DRIFT_FREQ} \\
    --n-repeats ${N_REPEATS} --seed ${SEED} --scenario ${SCENARIO} \\
    --n-dimensions ${n_dim} --pin-cpus --ensemble-sizes ${ENSEMBLE_SIZES} \\
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

# ---- Submit all dimension configs ----
echo "=== Submitting dimension sweep (${#DIMS_LIST[@]} configs) ==="
for n_dim in "${DIMS_LIST[@]}"; do
    submit_dim_config "$n_dim"
done

echo ""
echo "============================================"
echo "Dimension sweep study submission complete."
echo "Total configurations: ${#DIMS_LIST[@]}"
echo "Results will appear in: ${RESULTS_DIR}/"
echo ""
echo "After jobs complete, analyze with:"
echo "  python further_studies/analyze_dim_study.py"
echo ""
echo "Expected: speedup should increase with n_dimensions"
echo "  (higher compute density -> sync overhead becomes proportionally smaller)"
echo "============================================"
