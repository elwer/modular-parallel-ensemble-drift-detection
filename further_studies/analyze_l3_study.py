#!/usr/bin/env python3
"""
Analyze L3 complex spread study results.

Loads scalability_results.json from each config/freq combination,
extracts speedup for K=32 and K=64, and compares across L3 complex counts.

Expected outcome:
  - If L3 cache bandwidth is the bottleneck: speedup increases with more complexes
  - If DRAM bandwidth is the bottleneck: speedup is flat (only socket count matters)
"""

import json
import os
import sys
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_l3_study")

def load_results():
    rows = []
    for config_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
        if not os.path.isdir(config_dir):
            continue
        name = os.path.basename(config_dir)

        # Parse config name: l3_Xcomplex_kY_freq
        parts = name.split("_")
        if len(parts) < 4:
            continue
        n_complexes = int(parts[1].replace("complex", ""))
        k_target = int(parts[2].replace("k", ""))
        freq_label = parts[3]

        json_file = os.path.join(config_dir, "scalability_results.json")
        if not os.path.exists(json_file):
            print(f"  [WARN] No results JSON in {name}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        for entry in data:
            k = entry.get("n_detectors", entry.get("K", 0))
            if entry.get("mode") != "ensemble":
                continue
            speedup = entry.get("speedup", 0)
            throughput = entry.get("throughput_sps", 0)
            wall_time = entry.get("elapsed_sec", 0)

            rows.append({
                "config": name,
                "n_l3_complexes": n_complexes,
                "k_target": k_target,
                "freq_label": freq_label,
                "K": k,
                "speedup": speedup,
                "throughput_sps": throughput,
                "wall_time_sec": wall_time,
            })

    return pd.DataFrame(rows)


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    df = load_results()
    if df.empty:
        print("No results found. Are the jobs complete?")
        sys.exit(1)

    # Filter to ensemble sizes that match our targets
    for k_target in [32, 64]:
        sub = df[df["K"] == k_target].copy()
        if sub.empty:
            print(f"\nNo results for K={k_target}")
            continue

        print(f"\n{'='*80}")
        print(f"K={k_target}: Speedup vs L3 Complex Count")
        print(f"{'='*80}")

        for freq in ["high", "low"]:
            freq_sub = sub[sub["freq_label"] == freq]
            if freq_sub.empty:
                continue

            print(f"\n  Frequency: {freq}")
            print(f"  {'Complexes':>10} {'Speedup':>10} {'Throughput':>12} {'Wall time':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

            for n_cx in sorted(freq_sub["n_l3_complexes"].unique()):
                row = freq_sub[freq_sub["n_l3_complexes"] == n_cx].iloc[0]
                print(f"  {n_cx:>10} {row['speedup']:>10.2f} {row['throughput_sps']:>12.1f} {row['wall_time_sec']:>10.2f}")

        # Compare high vs low frequency
        if "high" in sub["freq_label"].values and "low" in sub["freq_label"].values:
            print(f"\n  Frequency effect (speedup increase from high→low):")
            print(f"  {'Complexes':>10} {'High':>10} {'Low':>10} {'Δ':>10} {'Δ%':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for n_cx in sorted(sub["n_l3_complexes"].unique()):
                high_row = sub[(sub["n_l3_complexes"] == n_cx) & (sub["freq_label"] == "high")]
                low_row = sub[(sub["n_l3_complexes"] == n_cx) & (sub["freq_label"] == "low")]
                if high_row.empty or low_row.empty:
                    continue
                high_sp = high_row.iloc[0]["speedup"]
                low_sp = low_row.iloc[0]["speedup"]
                delta = low_sp - high_sp
                pct = (delta / high_sp * 100) if high_sp > 0 else 0
                print(f"  {n_cx:>10} {high_sp:>10.2f} {low_sp:>10.2f} {delta:>10.2f} {pct:>9.1f}%")

    # Interpretation
    print(f"\n{'='*80}")
    print("Interpretation Guide")
    print(f"{'='*80}")
    print("""
If speedup INCREASES with more L3 complexes:
  → L3 cache bandwidth is the bottleneck (each complex has its own L3 BW)

If speedup is FLAT across complex counts:
  → DRAM bandwidth is the bottleneck (shared across all complexes on a socket)

If the frequency effect (high→low speedup increase) SHRINKS with more complexes:
  → L3 bandwidth was the bottleneck, now relieved by spreading
  → Strong evidence for L3 cache bandwidth contention
""")

    # Save CSV
    out_csv = os.path.join(os.path.dirname(__file__), "l3_study_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved summary to: {out_csv}")


if __name__ == "__main__":
    main()
