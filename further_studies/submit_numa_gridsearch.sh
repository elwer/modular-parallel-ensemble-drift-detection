#!/bin/bash
# =============================================================================
# Grid search over NUMA / Slurm / srun / CPU-pinning configurations
# for the ThreadsDeployment scalability study.
#
# Submits one sbatch job per configuration. Each job runs the same
# run_scalability_study.py benchmark but with different CPU/memory
# placement strategies. Results land in results_numa_gridsearch/<config_name>/.
#
# Usage:
#   bash further_studies/submit_numa_gridsearch.sh           # submit all
#   bash further_studies/submit_numa_gridsearch.sh --dry-run  # generate sbatch files only
#   bash further_studies/submit_numa_gridsearch.sh --only "baseline numa_n0"  # subset
#
# Architecture (target):
#   2 x AMD EPYC 7702 (64 cores/socket, 128 cores total, SMT available → 256 logical)
#   2 NUMA nodes (one per socket), 256 GB per socket
#   Full node is always requested (--cpus-per-task=128 --exclusive) so that
#   numactl/taskset/srun can reliably place threads on any subset of cores.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GRIDSEARCH_DIR="${SCRIPT_DIR}/results_numa_gridsearch"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_numa_gridsearch"
mkdir -p "$GRIDSEARCH_DIR" "$SBATCH_DIR"

# ---- Common benchmark parameters (keep identical across all configs) ----
STREAM_LENGTH=2000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
N_DIM=32
MAX_K=32
WALL_TIME="00:30:00"
MEM_PER_CPU=1024

PY_COMMON="--stream-length ${STREAM_LENGTH} --drift-frequency ${DRIFT_FREQ} \
--n-repeats ${N_REPEATS} --seed ${SEED} --scenario ${SCENARIO} \
--n-dimensions ${N_DIM} --max-ensemble-size ${MAX_K}"

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
submit_config() {
    local name="$1"
    local slurm_extra="$2"     # extra #SBATCH lines
    local cpus="$3"             # --cpus-per-task
    local numactl_cmd="$4"     # numactl prefix (empty = none)
    local srun_cmd="$5"        # srun prefix (empty = none)
    local py_extra="$6"        # extra python args (e.g. --pin-cpus)

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
#SBATCH --job-name=NUMA-${name}
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

# Log NUMA topology and CPU affinity for this job
echo "=== NUMA topology ===" > "\${OUTPUT_DIR}/env_info.txt"
numactl --hardware >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== lscpu (relevant) ===" >> "\${OUTPUT_DIR}/env_info.txt"
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line|Thread|Core|Socket|NUMA|NUMA node)' >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== Slurm env ===" >> "\${OUTPUT_DIR}/env_info.txt"
env | grep -E '^(SLURM_|OMP_|OPENBLAS|MKL|NUMEXPR)' | sort >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== CPU affinity ===" >> "\${OUTPUT_DIR}/env_info.txt"
taskset -cp \$\$ >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true
echo "=== numactl --show ===" >> "\${OUTPUT_DIR}/env_info.txt"
numactl --show >> "\${OUTPUT_DIR}/env_info.txt" 2>&1 || true

echo "Running config: ${name}"
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
# All configs request the full node (128 cpus, --exclusive) so that
# numactl/taskset/srun can place threads on any subset of cores.
# SMT is controlled via --hint=nomultithread (off) or its absence (on).
#
# Each call: submit_config <name> <slurm_extra> <cpus> <numactl> <srun> <py_extra>
# =============================================================================

# ---- Group 1: Baselines (full node, no placement control) ----
# SMT off, no pinning — OS scheduler decides where threads land
submit_config "baseline" \
    "#SBATCH --hint=nomultithread" 128 "" "" ""

submit_config "baseline_pin" \
    "#SBATCH --hint=nomultithread" 128 "" "" "--pin-cpus"

# SMT on (256 logical CPUs) — threads may share physical cores
submit_config "baseline_smt" \
    "" 128 "" "" ""

submit_config "baseline_smt_pin" \
    "" 128 "" "" "--pin-cpus"

# ---- Group 2: numactl single-NUMA-node (all threads + memory on one socket) ----
# Full node requested, numactl restricts to 64 cores on socket 0
submit_config "numa_n0" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" "" ""

submit_config "numa_n0_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

submit_config "numa_n1" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=1 --membind=1" "" ""

submit_config "numa_n1_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=1 --membind=1" "" "--pin-cpus"

# ---- Group 3: numactl interleave (spread memory across both NUMA nodes) ----
submit_config "numa_interleave" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --interleave=all" "" ""

submit_config "numa_interleave_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --interleave=all" "" "--pin-cpus"

# ---- Group 4: numactl --localalloc (prefer local allocation per thread) ----
submit_config "numa_localalloc" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --localalloc" "" ""

submit_config "numa_localalloc_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --localalloc" "" "--pin-cpus"

# ---- Group 5: Slurm --distribution (controls CPU allocation order across sockets) ----
submit_config "slurm_block" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 "" "" ""

submit_config "slurm_block_pin" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 "" "" "--pin-cpus"

submit_config "slurm_cyclic" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=cyclic:cyclic" 128 "" "" ""

submit_config "slurm_cyclic_pin" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=cyclic:cyclic" 128 "" "" "--pin-cpus"

# ---- Group 6: srun with --cpu-bind (task-level binding) ----
submit_config "srun_cores" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=cores" ""

submit_config "srun_cores_pin" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=cores" "--pin-cpus"

submit_config "srun_sockets" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=sockets" ""

submit_config "srun_sockets_pin" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=sockets" "--pin-cpus"

# ---- Group 7: srun with --distribution + --cpu-bind ----
submit_config "srun_block_cores" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 "" \
    "srun --cpu-bind=cores --distribution=block:block" ""

submit_config "srun_cyclic_cores" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=cyclic:cyclic" 128 "" \
    "srun --cpu-bind=cores --distribution=cyclic:cyclic" ""

# ---- Group 8: Combinations (numactl + Slurm distribution) ----
submit_config "combo_n0_block" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 \
    "numactl --cpunodebind=0 --membind=0" "" ""

submit_config "combo_n0_block_pin" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

submit_config "combo_n0_srun_cores" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" \
    "srun --cpu-bind=cores" ""

submit_config "combo_n0_srun_cores_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --cpunodebind=0 --membind=0" \
    "srun --cpu-bind=cores" "--pin-cpus"

submit_config "combo_interleave_block" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 \
    "numactl --interleave=all" "" ""

submit_config "combo_interleave_block_pin" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 128 \
    "numactl --interleave=all" "" "--pin-cpus"

# ---- Group 9: numactl + srun (srun wraps numactl) ----
submit_config "combo_srun_n0" \
    "#SBATCH --hint=nomultithread" 128 "" \
    "srun --cpu-bind=cores numactl --cpunodebind=0 --membind=0" ""

# ---- Group 10: Exact CPU count (33 = 32 workers + 1 main) ----
# Tests whether over-allocation (128 cpus for 33 threads) causes scheduler noise
submit_config "exact33_n0" \
    "#SBATCH --hint=nomultithread" 33 \
    "numactl --cpunodebind=0 --membind=0" "" ""

submit_config "exact33_n0_pin" \
    "#SBATCH --hint=nomultithread" 33 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

submit_config "exact33_block" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 33 "" "" ""

submit_config "exact33_block_pin" \
    "#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block" 33 "" "" "--pin-cpus"

# ---- Group 11: taskset (alternative CPU restriction, full node allocated) ----
submit_config "taskset_0_31" \
    "#SBATCH --hint=nomultithread" 128 \
    "taskset -c 0-31" "" ""

submit_config "taskset_0_31_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "taskset -c 0-31" "" "--pin-cpus"

# ---- Group 12: numactl --physcpubind (explicit physical CPU range, full node) ----
submit_config "numa_phys_0_31" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --physcpubind=0-31 --membind=0" "" ""

submit_config "numa_phys_0_31_pin" \
    "#SBATCH --hint=nomultithread" 128 \
    "numactl --physcpubind=0-31 --membind=0" "" "--pin-cpus"

# ---- Group 13: SMT on variants for key configs ----
# With SMT on, 256 logical CPUs — tests if hyperthread siblings hurt or help
submit_config "smt_n0_pin" \
    "" 128 \
    "numactl --cpunodebind=0 --membind=0" "" "--pin-cpus"

submit_config "smt_interleave_pin" \
    "" 128 \
    "numactl --interleave=all" "" "--pin-cpus"

submit_config "smt_taskset_pin" \
    "" 128 \
    "taskset -c 0-31" "" "--pin-cpus"

# =============================================================================
echo ""
echo "============================================"
echo "Grid search submission complete."
echo "Total configurations: $(ls "$SBATCH_DIR"/*.sbatch 2>/dev/null | wc -l)"
echo "Results will appear in: $GRIDSEARCH_DIR/"
echo "Sbatch files in:        $SBATCH_DIR/"
echo ""
echo "After jobs complete, analyze with:"
echo "  python further_studies/analyze_numa_gridsearch.py"
echo "============================================"
