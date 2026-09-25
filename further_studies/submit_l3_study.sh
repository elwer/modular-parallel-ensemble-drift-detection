#!/bin/bash
# =============================================================================
# L3 complex spread study: test whether L3 cache bandwidth is the bottleneck
# by spreading K workers across 2, 4, or 8 L3 complexes (NUMA nodes).
#
# AMD EPYC 7702: 8 L3 complexes (NUMA nodes), each 16 cores, 64MB L3
# Each L3 complex has its own L3 bandwidth.
#
# If L3 cache bandwidth is the bottleneck:
#   - Spreading workers across more complexes → higher speedup
#   - Effect stronger at low CPU frequency (memory pressure dominates)
#
# If DRAM bandwidth is the bottleneck:
#   - Spreading across 4 complexes on same socket → no improvement
#   - Only spreading across both sockets (4+4) would help (double DRAM BW)
#
# CPU lists are constructed so that --pin-cpus distributes workers evenly:
#   CPU 0 reserved for main thread, workers on remaining CPUs
#
# Usage:
#   bash further_studies/submit_l3_study.sh           # submit all
#   bash further_studies/submit_l3_study.sh --dry-run  # generate sbatch only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results_l3_study"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_l3_study"
mkdir -p "$RESULTS_DIR" "$SBATCH_DIR"

# ---- Common benchmark parameters ----
STREAM_LENGTH=2000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
N_DIM=32
WALL_TIME="02:00:00"
MEM_PER_CPU=1024

PY_COMMON="--stream-length ${STREAM_LENGTH} --drift-frequency ${DRIFT_FREQ} \
--n-repeats ${N_REPEATS} --seed ${SEED} --scenario ${SCENARIO} \
--n-dimensions ${N_DIM} --pin-cpus"

# ---- Parse CLI flags ----
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Helper: set CPU frequency via sysfs ----
set_cpu_freq() {
    local freq_label="$1"

    case "$freq_label" in
        high)
            # Performance governor, boost enabled
            if [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
                echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
            fi
            for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
                gov_file="$cpu_dir/cpufreq/scaling_governor"
                if [[ -w "$gov_file" ]]; then
                    echo "performance" > "$gov_file" 2>/dev/null || true
                fi
            done
            ;;
        low)
            # Userspace governor at 1500 MHz, boost disabled
            if [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
                echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
            fi
            for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
                gov_file="$cpu_dir/cpufreq/scaling_governor"
                setspeed_file="$cpu_dir/cpufreq/scaling_setspeed"
                if [[ -w "$gov_file" ]]; then
                    echo "userspace" > "$gov_file" 2>/dev/null || true
                fi
                if [[ -w "$setspeed_file" ]]; then
                    echo "1500000" > "$setspeed_file" 2>/dev/null || true
                fi
            done
            ;;
    esac
}

# ---- Helper: generate sbatch file and submit ----
submit_l3_config() {
    local name="$1"          # config name
    local cpu_list="$2"      # taskset CPU list
    local numactl_mem="$3"   # numactl memory policy (e.g., "--interleave=0-3")
    local freq_label="$4"    # "high" or "low"

    local sbatch_file="${SBATCH_DIR}/${name}_${freq_label}.sbatch"
    local output_dir="${RESULTS_DIR}/${name}_${freq_label}"

    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=L3-${name}-${freq_label}
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

# ---- Set CPU frequency ----
$(declare -f set_cpu_freq)
set_cpu_freq "${freq_label}"

# ---- Log environment ----
echo "=== NUMA topology ===" > "\${OUTPUT_DIR}/env_info.txt"
numactl --hardware >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== CPU list ===" >> "\${OUTPUT_DIR}/env_info.txt"
echo "${cpu_list}" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_governor (cpu0) ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== scaling_setspeed (cpu0) ===" >> "\${OUTPUT_DIR}/env_info.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed 2>/dev/null >> "\${OUTPUT_DIR}/env_info.txt" || echo "N/A" >> "\${OUTPUT_DIR}/env_info.txt"
echo "=== cpuinfo MHz (first 8 cores) ===" >> "\${OUTPUT_DIR}/env_info.txt"
grep "cpu MHz" /proc/cpuinfo | head -8 >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== taskset check ===" >> "\${OUTPUT_DIR}/env_info.txt"
taskset -c ${cpu_list} true 2>&1 >> "\${OUTPUT_DIR}/env_info.txt" || true

echo "Running config: ${name} (freq=${freq_label}, cpus=${cpu_list}, mem=${numactl_mem})"

numactl ${numactl_mem} taskset -c ${cpu_list} \\
    python -u further_studies/run_scalability_study.py \\
    ${PY_COMMON} \\
    --output-dir "\${OUTPUT_DIR}" \\
    2>&1 | tee "\${OUTPUT_DIR}/scalability.log"
EOF

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Generated: ${sbatch_file}"
    else
        local job_id
        job_id=$(sbatch --parsable "$sbatch_file")
        echo "Submitted ${name}_${freq_label} -> JobID ${job_id}"
    fi
}

# =============================================================================
# Config definitions
# Each config specifies: CPU list, numactl memory policy
# CPU lists are designed so --pin-cpus distributes workers evenly across L3
# complexes. CPU 0 is reserved for the main thread.
# =============================================================================

# ---- K=32 configs ----
# 2 L3 complexes: 16 workers per complex (1 main + 15 on complex 0, 16 on complex 1)
# CPUs: 0-31 (33 CPUs for clean pinning: 0-8 on complex 0, 16-31 on complex 1)
# Actually use 0-31 (32 CPUs). Main on 0, workers 1-31 (31 workers). Need 32, so
# add one extra from complex 0: 0-8,16-31 = 33 CPUs, workers 1-8,16-31 = 8+16 = 24... no.
# Clean approach: 0-8 (9 CPUs, complex 0), 16-31 (16 CPUs, complex 1) = 25 CPUs. Not enough.
# Best: 0-15 (16 CPUs, complex 0), 16-31 (16 CPUs, complex 1) = 32 CPUs.
# Main on 0, workers 1-31 = 31 workers. Accept sharing CPU 0 for worker 0.
L3_2COMPLEX_K32_CPUS="0-31"
L3_2COMPLEX_K32_MEM="--interleave=0-1"

# 4 L3 complexes: 8 workers per complex
# Complex 0: CPUs 0-8 (9 CPUs, 1 main + 8 workers)
# Complex 1: CPUs 16-23 (8 CPUs, 8 workers)
# Complex 2: CPUs 32-39 (8 CPUs, 8 workers)
# Complex 3: CPUs 48-55 (8 CPUs, 8 workers)
# Total: 33 CPUs, 32 workers
L3_4COMPLEX_K32_CPUS="0-8,16-23,32-39,48-55"
L3_4COMPLEX_K32_MEM="--interleave=0-3"

# 8 L3 complexes: 4 workers per complex
# Complex 0: CPUs 0-4 (5 CPUs, 1 main + 4 workers)
# Complex 1-7: 4 CPUs each
# Total: 33 CPUs, 32 workers
L3_8COMPLEX_K32_CPUS="0-4,16-19,32-35,48-51,64-67,80-83,96-99,112-115"
L3_8COMPLEX_K32_MEM="--interleave=0-7"

# ---- K=64 configs ----
# 4 L3 complexes: 16 workers per complex
# Complex 0: CPUs 0-15 (16 CPUs, 1 main + 15 workers)
# Complex 1: CPUs 16-31 (16 CPUs, 16 workers)
# Complex 2: CPUs 32-47 (16 CPUs, 16 workers)
# Complex 3: CPUs 48-63 (16 CPUs, 16 workers)
# Total: 64 CPUs, 64 workers (main shares CPU 0 with worker 0)
L3_4COMPLEX_K64_CPUS="0-63"
L3_4COMPLEX_K64_MEM="--interleave=0-3"

# 8 L3 complexes: 8 workers per complex
# Complex 0: CPUs 0-8 (9 CPUs, 1 main + 8 workers)
# Complex 1-7: 8 CPUs each
# Total: 65 CPUs, 64 workers
L3_8COMPLEX_K64_CPUS="0-8,16-23,32-39,48-55,64-71,80-87,96-103,112-119"
L3_8COMPLEX_K64_MEM="--interleave=0-7"

# ---- Submit all configs at both frequencies ----
for freq in high low; do
    echo "=== Submitting ${freq} frequency configs ==="

    submit_l3_config "l3_2complex_k32" "$L3_2COMPLEX_K32_CPUS" "$L3_2COMPLEX_K32_MEM" "$freq"
    submit_l3_config "l3_4complex_k32" "$L3_4COMPLEX_K32_CPUS" "$L3_4COMPLEX_K32_MEM" "$freq"
    submit_l3_config "l3_8complex_k32" "$L3_8COMPLEX_K32_CPUS" "$L3_8COMPLEX_K32_MEM" "$freq"

    submit_l3_config "l3_4complex_k64" "$L3_4COMPLEX_K64_CPUS" "$L3_4COMPLEX_K64_MEM" "$freq"
    submit_l3_config "l3_8complex_k64" "$L3_8COMPLEX_K64_CPUS" "$L3_8COMPLEX_K64_MEM" "$freq"
done

echo ""
echo "============================================"
echo "L3 complex spread study submission complete."
echo "Total configurations: 10 (5 configs x 2 frequencies)"
echo "Results will appear in: ${RESULTS_DIR}/"
echo ""
echo "After jobs complete, compare speedup at each K:"
echo "  - If speedup increases with more L3 complexes → L3 cache bandwidth bottleneck"
echo "  - If speedup is flat → DRAM bandwidth bottleneck"
echo "============================================"
