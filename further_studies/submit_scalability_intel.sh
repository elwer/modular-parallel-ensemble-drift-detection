#!/bin/bash
# =============================================================================
# Scalability frequency study on Intel Xeon Platinum 8470 (Sapphire Rapids).
#
# 2 x Intel Xeon Platinum 8470 (52 cores) @ 2.00 GHz
# 104 cores total, 2 NUMA domains (1 per socket), hyperthreading disabled
#
# Runs D3, IBDD, OCDD, SPLL, and CSDDM one after another at three CPU frequency
# levels (2.0 GHz, 1.5 GHz, 800 MHz) on a single exclusive node.
#
# BNDM is excluded (already measured). UDetect is excluded (not viable).
#
# Stream: SineClustersHighDim, 256 dimensions, 10 000 samples.
# Ensemble sizes: 2, 4, 8, 16, 26, 52, 104 (matching Intel hardware boundaries).
# CPU pinning: --pin-cpus (CPU 0 reserved for main thread).
#
# Intel frequency control:
#   - Turbo disable via /sys/devices/system/cpu/intel_pstate/no_turbo
#   - userspace governor + scaling_setspeed for fixed frequency
#
# Frequency levels (all with turbo disabled for determinism):
#   mid       = userspace governor at 2000000 kHz (2.0 GHz, no turbo)
#   low       = userspace governor at 1500000 kHz (1.5 GHz, no turbo)
#   ultra_low = userspace governor at 800000 kHz  (800 MHz, no turbo)
#
# Usage:
#   bash further_studies/submit_scalability_intel.sh             # submit all frequencies
#   bash further_studies/submit_scalability_intel.sh --dry-run   # generate sbatch only
#   bash further_studies/submit_scalability_intel.sh --only-freq low       # only 1.5 GHz
#   bash further_studies/submit_scalability_intel.sh --only-freq mid       # only 2.0 GHz
#   bash further_studies/submit_scalability_intel.sh --only-freq ultra_low # only 800 MHz
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Common benchmark parameters ----
STREAM_LENGTH=10000
DRIFT_FREQ=100
N_REPEATS=3
SEED=42
SCENARIO="balanced"
N_DIM=256
ENSEMBLE_SIZES="2,4,8,16,26,52,104"
WALL_TIME="24:00:00"
MEM_PER_CPU=1024
N_CPUS=104

# ---- Detectors to study (no BNDM, no UDetect) ----
DETECTORS="D3 IBDD OCDD SPLL CSDDM"

# ---- Detector parameters (heavy configs for 256 dims) ----
declare -A DETECTOR_PARAMS
DETECTOR_PARAMS[D3]='{"n_reference_samples": 5000, "recent_samples_proportion": 0.01, "threshold": 0.8, "recent_samples_size": 5000}'
DETECTOR_PARAMS[IBDD]='{"n_samples": 500, "n_consecutive_deviations": 1, "n_permutations": 100, "update_interval": 50, "recent_samples_size": 500}'
DETECTOR_PARAMS[OCDD]='{"n_samples": 2000, "threshold": 0.5, "recent_samples_size": 2000}'
DETECTOR_PARAMS[SPLL]='{"n_samples": 50, "n_clusters": 5, "threshold": 0.5, "recent_samples_size": 50}'
DETECTOR_PARAMS[CSDDM]='{"n_samples": 50, "n_clusters": 5, "confidence": 0.05, "feature_proportion": 0.1, "recent_samples_size": 50}'

# ---- Frequency levels ----
declare -A FREQ_GOVERNOR
FREQ_GOVERNOR[mid]="userspace"
FREQ_GOVERNOR[low]="userspace"
FREQ_GOVERNOR[ultra_low]="userspace"

declare -A FREQ_KHZ
FREQ_KHZ[mid]="2000000"       # 2.0 GHz (base frequency, no turbo)
FREQ_KHZ[low]="1500000"       # 1.5 GHz (reduced frequency, no turbo)
FREQ_KHZ[ultra_low]="800000"  # 800 MHz (minimum frequency, no turbo)

declare -A FREQ_LABELS
FREQ_LABELS[mid]="2000MHz"
FREQ_LABELS[low]="1500MHz"
FREQ_LABELS[ultra_low]="800MHz"

# ---- Parse CLI flags ----
DRY_RUN=0
ONLY_FREQ=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --only-freq)    ONLY_FREQ="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---- Output base directory ----
RESULTS_BASE="${SCRIPT_DIR}/results_scalability_intel"
SBATCH_DIR="${SCRIPT_DIR}/sbatch_scalability_intel"
mkdir -p "$RESULTS_BASE" "$SBATCH_DIR"

# ---- Log detector configs ----
CONFIGS_FILE="${RESULTS_BASE}/detector_configs.json"
{
echo '{'
echo '  "stream_length": 10000,'
echo '  "drift_frequency": 100,'
echo '  "n_dimensions": 256,'
echo '  "n_repeats": 3,'
echo '  "ensemble_sizes": [2, 4, 8, 16, 26, 52, 104],'
echo '  "scenario": "balanced",'
echo '  "pin_cpus": true,'
echo '  "architecture": "Intel Xeon Platinum 8470 (Sapphire Rapids), 104 cores",'
echo '  "detectors": {'
FIRST=1
for det in $DETECTORS; do
    if [[ $FIRST -eq 0 ]]; then echo ','; fi
    FIRST=0
    printf '    "%s": %s' "$det" "${DETECTOR_PARAMS[$det]}"
done
echo ''
echo '  }'
echo '}'
} > "$CONFIGS_FILE"
echo "Detector configs written to: $CONFIGS_FILE"

# ---- Generate and submit sbatch for one frequency ----
submit_freq() {
    local freq_label="$1"
    local freq_governor="${FREQ_GOVERNOR[$freq_label]}"
    local freq_khz="${FREQ_KHZ[$freq_label]}"
    local freq_name="${FREQ_LABELS[$freq_label]}"

    local sbatch_file="${SBATCH_DIR}/scalability_intel_${freq_label}.sbatch"

    # ---- Part 1: Write header and frequency setup ----
    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --job-name=SCAL-INTEL-${freq_label}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_CPUS}
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

# ---- Log environment info ----
ENV_INFO="${RESULTS_BASE}/env_info_${freq_label}.txt"
echo "=== NUMA topology ===" > "\${ENV_INFO}"
numactl --hardware >> "\${ENV_INFO}" 2>&1 || true
echo "=== lscpu (relevant) ===" >> "\${ENV_INFO}"
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line|Thread|Core|Socket|NUMA|NUMA node|CPU min|CPU max|MHz|Model name)' >> "\${ENV_INFO}" 2>&1 || true
echo "=== Slurm env ===" >> "\${ENV_INFO}"
env | grep -E '^(SLURM_|OMP_|OPENBLAS|MKL|NUMEXPR|PYTHON_GIL)' | sort >> "\${ENV_INFO}" 2>&1 || true
echo "=== CPU affinity ===" >> "\${ENV_INFO}"
taskset -cp \$\$ >> "\${ENV_INFO}" 2>&1 || true
echo "=== numactl --show ===" >> "\${ENV_INFO}"
numactl --show >> "\${ENV_INFO}" 2>&1 || true
echo "=== cpufreq available frequencies ===" >> "\${ENV_INFO}"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>/dev/null >> "\${ENV_INFO}" || echo "N/A" >> "\${ENV_INFO}"

# ---- Disable Intel Turbo Boost ----
if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    echo "Intel Turbo Boost disabled"
fi

# ---- Set CPU frequency on ALL cores via sysfs ----
FREQ_GOVERNOR_VAL="${freq_governor}"
FREQ_KHZ_VAL="${freq_khz}"

echo "Setting CPU frequency: governor=\${FREQ_GOVERNOR_VAL}, freq=\${FREQ_KHZ_VAL} kHz"

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
    gov_file="\$cpu_dir/cpufreq/scaling_governor"
    setspeed_file="\$cpu_dir/cpufreq/scaling_setspeed"
    if [[ -w "\$gov_file" ]]; then
        echo "\${FREQ_GOVERNOR_VAL}" > "\$gov_file" 2>/dev/null || true
    fi
    if [[ "\${FREQ_GOVERNOR_VAL}" == "userspace" ]] && [[ -n "\${FREQ_KHZ_VAL}" ]] && [[ -w "\$setspeed_file" ]]; then
        echo "\${FREQ_KHZ_VAL}" > "\$setspeed_file" 2>/dev/null || true
    fi
done

# Verify frequency on all cores
echo "=== CPU frequency (all cores, AFTER setting) ===" >> "\${ENV_INFO}"
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    if [[ -r "\$f" ]]; then
        core=\$(echo "\$f" | grep -oP 'cpu\K[0-9]+')
        freq=\$(cat "\$f" 2>/dev/null || echo "N/A")
        echo "cpu\${core}: \${freq} kHz"
    fi
done >> "\${ENV_INFO}" 2>&1 || true

echo "=== intel_pstate no_turbo ===" >> "\${ENV_INFO}"
cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null >> "\${ENV_INFO}" || echo "N/A" >> "\${ENV_INFO}"
echo "=== scaling_governor (cpu0) ===" >> "\${ENV_INFO}"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null >> "\${ENV_INFO}" || echo "N/A" >> "\${ENV_INFO}"
echo "=== scaling_setspeed (cpu0) ===" >> "\${ENV_INFO}"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed 2>/dev/null >> "\${ENV_INFO}" || echo "N/A" >> "\${ENV_INFO}"
echo "=== cpuinfo MHz (first 8 cores) ===" >> "\${ENV_INFO}"
grep "cpu MHz" /proc/cpuinfo | head -8 >> "\${ENV_INFO}" 2>&1 || true

echo ""
echo "============================================================"
echo "Scalability Intel study: frequency=${freq_name} (\${FREQ_KHZ_VAL} kHz)"
echo "Architecture: Intel Xeon Platinum 8470 (Sapphire Rapids), 104 cores"
echo "Detectors: ${DETECTORS}"
echo "Stream: ${STREAM_LENGTH} samples, ${N_DIM} dims, drift_freq=${DRIFT_FREQ}"
echo "Ensemble sizes: ${ENSEMBLE_SIZES}"
echo "Repeats: ${N_REPEATS}, Scenario: ${SCENARIO}, Pin CPUs: yes"
echo "============================================================"
echo ""

# ---- Run all detectors one after another ----
EOF

    # ---- Part 2: Append detector command blocks ----
    for det in $DETECTORS; do
        local out_dir="${RESULTS_BASE}/${det}_${freq_label}"
        local params="${DETECTOR_PARAMS[$det]}"

        cat >> "$sbatch_file" << EOF

echo ""
echo "==========================================="
echo "=== Detector: ${det} | Frequency: ${freq_name} ==="
echo "==========================================="
echo "Params: ${params}"
echo ""
mkdir -p "${out_dir}"
# Log detector config
echo '${params}' > "${out_dir}/detector_config.json"
python -u further_studies/run_scalability_study.py \\
    --stream-length ${STREAM_LENGTH} \\
    --drift-frequency ${DRIFT_FREQ} \\
    --n-repeats ${N_REPEATS} \\
    --seed ${SEED} \\
    --scenario ${SCENARIO} \\
    --n-dimensions ${N_DIM} \\
    --ensemble-sizes ${ENSEMBLE_SIZES} \\
    --pin-cpus \\
    --detector ${det} \\
    --detector-params '${params}' \\
    --output-dir "${out_dir}" \\
    2>&1 | tee "${out_dir}/scalability.log"
EOF
    done

    # ---- Part 3: Append footer ----
    cat >> "$sbatch_file" << EOF

echo ""
echo "============================================================"
echo "All detectors completed for frequency=${freq_name}"
echo "Results in: ${RESULTS_BASE}"
echo "============================================================"
EOF

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Generated: ${sbatch_file}"
    else
        local job_id
        job_id=$(sbatch --parsable "$sbatch_file")
        echo "Submitted scalability_intel_${freq_label} -> JobID ${job_id}"
    fi
}

# ---- Submit for each frequency ----
for freq_label in mid low ultra_low; do
    if [[ -n "$ONLY_FREQ" ]] && [[ "$ONLY_FREQ" != "$freq_label" ]]; then
        continue
    fi
    submit_freq "$freq_label"
done

echo ""
echo "============================================"
echo "Scalability Intel study submission complete."
echo "Results base: ${RESULTS_BASE}"
echo "Detector configs: ${CONFIGS_FILE}"
echo ""
echo "Architecture: Intel Xeon Platinum 8470 (Sapphire Rapids), 104 cores"
echo "Detectors: ${DETECTORS}"
echo "Frequencies: 2000MHz (mid), 1500MHz (low), 800MHz (ultra_low)"
echo ""
for det in $DETECTORS; do
    echo "  ${det}: ${DETECTOR_PARAMS[$det]}"
done
echo ""
echo "Output structure:"
echo "  ${RESULTS_BASE}/{detector}_{freq_label}/"
echo "    scalability_summary.csv"
echo "    scalability_results.json"
echo "    scalability.log"
echo "    detector_config.json"
echo "============================================"
