#!/usr/bin/env python3
"""
Check if stream 3 in evaluation always has 0 F1 score in N=1 Optuna results.
This helps verify if the stream never terminated.
"""

import os
import sys
import csv
import glob
from pathlib import Path

def check_stream_3_f1(csv_pattern: str, stream_idx: int = 3):
    """Check F1 scores for a specific stream across all CSV files."""
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Checking stream {stream_idx} F1 scores...")
    print("=" * 80)
    
    f1_scores = []
    zero_count = 0
    non_zero_count = 0
    missing_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check for per-stream F1 column (e.g., per_stream_f1)
                    per_stream_f1_str = row.get('per_stream_f1', '')
                    if per_stream_f1_str:
                        try:
                            # Parse the list-like string
                            per_stream_f1 = eval(per_stream_f1_str)
                            if isinstance(per_stream_f1, list) and len(per_stream_f1) > stream_idx:
                                f1 = per_stream_f1[stream_idx]
                                f1_scores.append(f1)
                                if f1 == 0.0:
                                    zero_count += 1
                                else:
                                    non_zero_count += 1
                            else:
                                missing_count += 1
                        except:
                            error_count += 1
                    else:
                        missing_count += 1
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            error_count += 1
    
    print(f"\nResults for stream {stream_idx}:")
    print(f"  Total entries checked: {len(f1_scores)}")
    print(f"  Zero F1 scores: {zero_count}")
    print(f"  Non-zero F1 scores: {non_zero_count}")
    print(f"  Missing/invalid entries: {missing_count}")
    print(f"  Error entries: {error_count}")
    
    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"  Min F1: {min(f1_scores):.4f}")
        print(f"  Max F1: {max(f1_scores):.4f}")
    
    print("=" * 80)
    
    if zero_count == len(f1_scores) and len(f1_scores) > 0:
        print(f"⚠️  ALL {zero_count} entries have F1=0 for stream {stream_idx}")
        print("This suggests the stream may never have terminated.")
    elif zero_count > 0:
        print(f"⚠️  {zero_count}/{len(f1_scores)} entries have F1=0 for stream {stream_idx}")
    else:
        print(f"✓ No zero F1 scores found for stream {stream_idx}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_stream_3_f1.py <csv_pattern> [stream_idx]")
        print("Example: python check_stream_3_f1.py 'synthetic_multistream_results_2h/Mix_SineClusters+WaveformDrift2/synthF1ms_Mix_SineClusters+WaveformDrift2_N1_S10_w*.csv' 3")
        sys.exit(1)
    
    csv_pattern = sys.argv[1]
    stream_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    check_stream_3_f1(csv_pattern, stream_idx)
