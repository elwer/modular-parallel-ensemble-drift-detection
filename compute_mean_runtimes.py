#!/usr/bin/env python3
"""
Compute mean runtimes from runtime experiment results.

Run this after all SLURM jobs from submit_runtime_experiments.sh have finished.

Usage:
    python compute_mean_runtimes.py
"""

import os
import csv
import glob
import numpy as np

BASE_DIR = "runtime_experiments"
CORE_COUNTS = [1, 2, 4]
REMOVE_OUTLIERS = True  # set to False to use all data points


def collect_runtimes(detector, config_id, cores):
    """Read all run_*.csv files for a given detector/config/cores combo."""
    pattern = os.path.join(BASE_DIR, f"{detector}_{config_id}",
                           f"{cores}_cores", "run_*.csv")
    runtimes = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                runtimes.append(float(row["runtime_s"]))
    return runtimes


def remove_outliers_iqr(values):
    """Remove outliers using the IQR method (1.5 * IQR rule)."""
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)]
    return filtered.tolist()


def main():
    # Discover all detector_configid directories
    if not os.path.exists(BASE_DIR):
        print(f"Error: {BASE_DIR} not found. Have the jobs finished?")
        return

    entries = sorted(os.listdir(BASE_DIR))
    if not entries:
        print(f"No results found in {BASE_DIR}.")
        return

    if REMOVE_OUTLIERS:
        print("Outlier removal: ON (IQR method, 1.5 * IQR)\n")
    print(f"{'Config':<20} {'Cores':>5} {'Runs':>5} {'Removed':>8} {'Mean(s)':>10} "
          f"{'Std(s)':>10} {'Min(s)':>10} {'Max(s)':>10} {'Speedup':>8}")
    print("-" * 90)

    for entry in entries:
        entry_path = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(entry_path):
            continue

        parts = entry.rsplit("_", 1)
        if len(parts) != 2:
            continue
        detector, config_id = parts[0], parts[1]

        baseline_mean = None
        results = []

        for cores in CORE_COUNTS:
            runtimes = collect_runtimes(detector, config_id, cores)
            if not runtimes:
                results.append((cores, None, None, None, None, None, 0))
                continue
            n_raw = len(runtimes)
            if REMOVE_OUTLIERS:
                runtimes = remove_outliers_iqr(runtimes)
            n_removed = n_raw - len(runtimes)
            arr = np.array(runtimes)
            mean_rt = np.mean(arr)
            std_rt = np.std(arr)
            min_rt = np.min(arr)
            max_rt = np.max(arr)
            if cores == CORE_COUNTS[0]:
                baseline_mean = mean_rt
            results.append((cores, len(runtimes), mean_rt, std_rt, min_rt, max_rt, n_removed))

        label = f"{detector.upper()} ID {config_id}"
        for cores, n, mean_rt, std_rt, min_rt, max_rt, n_removed in results:
            if mean_rt is None:
                print(f"{label:<20} {cores:>5} {'N/A':>5} {'N/A':>8} {'N/A':>10} "
                      f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            else:
                speedup = baseline_mean / mean_rt if baseline_mean and baseline_mean > 0 else float('nan')
                print(f"{label:<20} {cores:>5} {n:>5} {n_removed:>8} {mean_rt:>10.2f} "
                      f"{std_rt:>10.2f} {min_rt:>10.2f} {max_rt:>10.2f} {speedup:>8.2f}")
        print()


if __name__ == "__main__":
    main()
