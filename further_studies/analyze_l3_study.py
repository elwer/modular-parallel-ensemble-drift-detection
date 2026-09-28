#!/usr/bin/env python3
"""
Analyze L3 complex spread study results.

Loads scalability_results.json from each config/freq combination,
computes speedup from sequential vs ensemble runs, and compares
across L3 complex counts.

Expected outcome:
  - If L3 cache bandwidth is the bottleneck: speedup increases with more complexes
  - If DRAM bandwidth is the bottleneck: speedup is flat (only socket count matters)
"""

import json
import os
import sys
import glob
import csv
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_l3_study")


def load_and_compute(config_dir, name):
    """Load JSON, compute speedup from sequential vs ensemble runs."""
    json_file = os.path.join(config_dir, "scalability_results.json")
    if not os.path.exists(json_file):
        return []

    with open(json_file) as f:
        data = json.load(f)

    # Group runs by (mode, K), averaging across reps
    # Handles duplicate entries from reruns by averaging all
    seq_times = defaultdict(list)   # K -> [elapsed_sec, ...]
    ens_times = defaultdict(list)    # K -> [elapsed_sec, ...]
    ens_tps = defaultdict(list)      # K -> [throughput_sps, ...]

    for entry in data:
        mode = entry.get("mode", "")
        k = entry.get("n_detectors", 0)
        if k == 0:
            continue
        if mode == "sequential":
            seq_times[k].append(entry["elapsed_sec"])
        elif mode == "ensemble":
            ens_times[k].append(entry["elapsed_sec"])
            ens_tps[k].append(entry.get("throughput_sps", 0))

    # Parse config name: l3_Xcomplex_kY_freq
    parts = name.split("_")
    if len(parts) < 4:
        return []
    n_complexes = int(parts[1].replace("complex", ""))
    k_target = int(parts[2].replace("k", ""))
    freq_label = parts[3]

    rows = []
    for k in sorted(ens_times.keys()):
        mean_ens_time = np.mean(ens_times[k])
        mean_ens_tps = np.mean(ens_tps[k])

        if k in seq_times and seq_times[k]:
            seq_mean = np.mean(seq_times[k])
            speedup = seq_mean / mean_ens_time if mean_ens_time > 0 else 0
        else:
            speedup = 0

        rows.append({
            "config": name,
            "n_l3_complexes": n_complexes,
            "k_target": k_target,
            "freq_label": freq_label,
            "K": k,
            "speedup": speedup,
            "throughput_sps": mean_ens_tps,
            "wall_time_sec": mean_ens_time,
            "n_reps": len(ens_times[k]),
        })

    return rows


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    all_rows = []
    for config_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
        if not os.path.isdir(config_dir):
            continue
        name = os.path.basename(config_dir)
        rows = load_and_compute(config_dir, name)
        if not rows:
            print(f"  [WARN] No results in {name}")
        all_rows.extend(rows)

    if not all_rows:
        print("No results found. Are the jobs complete?")
        sys.exit(1)

    # Print results for target K values
    for k_target in [32, 64]:
        sub = [r for r in all_rows if r["K"] == k_target]
        if not sub:
            print(f"\nNo results for K={k_target}")
            continue

        print(f"\n{'='*80}")
        print(f"K={k_target}: Speedup vs L3 Complex Count")
        print(f"{'='*80}")

        for freq in ["high", "low"]:
            freq_sub = [r for r in sub if r["freq_label"] == freq]
            if not freq_sub:
                continue

            print(f"\n  Frequency: {freq}")
            print(f"  {'Complexes':>10} {'Speedup':>10} {'Throughput':>12} {'Wall time':>10} {'Reps':>5}")
            print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*5}")

            for n_cx in sorted(set(r["n_l3_complexes"] for r in freq_sub)):
                matching = [r for r in freq_sub if r["n_l3_complexes"] == n_cx]
                avg_sp = np.mean([r["speedup"] for r in matching])
                avg_tps = np.mean([r["throughput_sps"] for r in matching])
                avg_wt = np.mean([r["wall_time_sec"] for r in matching])
                n_reps = matching[0]["n_reps"]
                print(f"  {n_cx:>10} {avg_sp:>10.2f} {avg_tps:>12.1f} {avg_wt:>10.2f} {n_reps:>5}")

        # Compare high vs low frequency
        high_rows = {r["n_l3_complexes"]: r for r in sub if r["freq_label"] == "high"}
        low_rows = {r["n_l3_complexes"]: r for r in sub if r["freq_label"] == "low"}

        common = sorted(set(high_rows.keys()) & set(low_rows.keys()))
        if common:
            print(f"\n  Frequency effect (speedup increase from high->low):")
            print(f"  {'Complexes':>10} {'High':>10} {'Low':>10} {'Delta':>10} {'Delta%':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for n_cx in common:
                high_sp = high_rows[n_cx]["speedup"]
                low_sp = low_rows[n_cx]["speedup"]
                delta = low_sp - high_sp
                pct = (delta / high_sp * 100) if high_sp > 0 else 0
                print(f"  {n_cx:>10} {high_sp:>10.2f} {low_sp:>10.2f} {delta:>10.2f} {pct:>9.1f}%")

    # Also show all K values for completeness
    print(f"\n{'='*80}")
    print("All K values (for reference)")
    print(f"{'='*80}")
    for freq in ["low", "high"]:
        freq_rows = [r for r in all_rows if r["freq_label"] == freq]
        if not freq_rows:
            continue
        print(f"\n  Frequency: {freq}")
        print(f"  {'Config':>25} {'K':>5} {'Speedup':>10} {'Throughput':>12} {'Wall time':>10}")
        print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*12} {'-'*10}")
        for r in sorted(freq_rows, key=lambda x: (x["n_l3_complexes"], x["K"])):
            print(f"  {r['config']:>25} {r['K']:>5} {r['speedup']:>10.2f} {r['throughput_sps']:>12.1f} {r['wall_time_sec']:>10.2f}")

    # Interpretation
    print(f"\n{'='*80}")
    print("Interpretation Guide")
    print(f"{'='*80}")
    print("""
If speedup INCREASES with more L3 complexes:
  -> L3 cache bandwidth is the bottleneck (each complex has its own L3 BW)

If speedup is FLAT across complex counts:
  -> DRAM bandwidth is the bottleneck (shared across all complexes on a socket)

If the frequency effect (high->low speedup increase) SHRINKS with more complexes:
  -> L3 bandwidth was the bottleneck, now relieved by spreading
  -> Strong evidence for L3 cache bandwidth contention
""")

    # Save CSV
    out_csv = os.path.join(os.path.dirname(__file__), "l3_study_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n_l3_complexes", "k_target",
                                          "freq_label", "K", "speedup",
                                          "throughput_sps", "wall_time_sec", "n_reps"])
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["K"], x["freq_label"], x["n_l3_complexes"])):
            w.writerow(r)
    print(f"Saved summary to: {out_csv}")


if __name__ == "__main__":
    main()
