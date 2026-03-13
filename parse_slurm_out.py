#!/usr/bin/env python3
"""
Parse SLURM .out files from Optuna optimization runs.

Extracts:
  - Config: dataset, drift detector, classifier, training samples, seed
  - Per-trial: trial ID, accuracy, runtime, drifts, and all hyperparameters

Outputs:
  - <Detector>_<Dataset>.csv   — trial results with parameters
  - <Detector>_<Dataset>.config — run configuration
"""

import re
import csv
import ast
import sys
import os
from argparse import ArgumentParser


def parse_slurm_out(filepath):
    """Parse a single SLURM .out file and return config dict and list of trial dicts."""

    with open(filepath, 'r', errors='replace') as f:
        raw = f.read()

    # Strip carriage-return noise from tqdm progress bars
    lines = raw.split('\n')

    # ── 1. Extract configuration ──────────────────────────────────────────
    config = {}

    m = re.search(r'-\s*Dataset:\s*(\S+)', raw)
    if m:
        config['dataset'] = m.group(1)

    m = re.search(r'-\s*Classifier:\s*(\S+)', raw)
    if m:
        config['classifier'] = m.group(1)

    m = re.search(r'-\s*Training samples:\s*(\d+)', raw)
    if m:
        config['training_samples'] = int(m.group(1))

    m = re.search(r'-\s*Seed:\s*(\d+)', raw)
    if m:
        config['seed'] = int(m.group(1))

    # Detector name — from "Optimizing <NAME>" or "Detectors to optimize: ['NAME']"
    # For single-DD runs
    m = re.search(r'Optimizing\s+(\w+)', raw)
    if m:
        config['detector'] = m.group(1)
    else:
        # For MOPEDDS runs
        m = re.search(r'MOPEDDS Hyperparameter Optimization', raw)
        if m:
            config['detector'] = 'MOPEDDS'

    # Also try to get detector from the INFO lines if not found yet
    if 'detector' not in config:
        m = re.search(r'INFO:__main__:(\w+)\s+Trial\s+\d+:', raw)
        if m:
            config['detector'] = m.group(1)

    # ── 2. Extract trial data ─────────────────────────────────────────────
    # Two sources per trial:
    #   a) INFO line:  INFO:__main__:IBDD Trial 0: accuracy=0.9215, runtime=1634.8s, drifts=8589
    #      or MOPEDDS:    INFO:__main__:Trial 0: accuracy=0.9215, runtime=1634.8s, drifts=8589
    #   b) Optuna line: Trial 0 finished with values: [0.921, 1634.77] and parameters: {'k': v, ...}

    # Parse INFO lines for trial_id -> (accuracy, runtime, drifts)
    trial_info = {}
    info_pattern = re.compile(
        r'INFO:__main__:(?:\w+\s+)?Trial\s+(\d+):\s*'
        r'accuracy=([\d.]+),\s*runtime=([\d.]+)s,\s*drifts=(\d+)'
    )
    for line in lines:
        m = info_pattern.search(line)
        if m:
            tid = int(m.group(1))
            trial_info[tid] = {
                'trial_id': tid,
                'accuracy': float(m.group(2)),
                'runtime': float(m.group(3)),
                'drifts': int(m.group(4)),
            }

    # Parse Optuna log lines for trial_id -> parameters dict
    # These lines may be split or truncated; try to extract the params dict
    trial_params = {}
    # Join all lines to handle potential line breaks within a single log entry
    full_text = '\n'.join(lines)

    # Find all "Trial N finished with values: [...] and parameters: {...}"
    optuna_pattern = re.compile(
        r'Trial\s+(\d+)\s+finished\s+with\s+values:\s*\[([^\]]*)\]\s*and\s+parameters?:\s*(\{[^}]*\})'
    )
    for m in optuna_pattern.finditer(full_text):
        tid = int(m.group(1))
        params_str = m.group(3)
        try:
            params = ast.literal_eval(params_str)
            trial_params[tid] = params
        except (ValueError, SyntaxError):
            # Try to fix common issues (truncated lines)
            pass

    # Also try a more lenient pattern that handles multi-line or truncated params
    # by looking for the params dict on the same or next line
    optuna_pattern2 = re.compile(
        r'Trial\s+(\d+)\s+finished\s+with\s+values:\s*\[([^\]]*)\]\s*and\s+para[^{]*(\{.*?\})',
        re.DOTALL
    )
    for m in optuna_pattern2.finditer(full_text):
        tid = int(m.group(1))
        if tid in trial_params:
            continue  # already parsed
        params_str = m.group(3)
        try:
            params = ast.literal_eval(params_str)
            trial_params[tid] = params
        except (ValueError, SyntaxError):
            pass

    # ── 3. Merge trial info and params ────────────────────────────────────
    trials = []
    for tid in sorted(trial_info.keys()):
        row = dict(trial_info[tid])
        if tid in trial_params:
            row.update(trial_params[tid])
        trials.append(row)

    return config, trials


def collect_param_names(trials):
    """Collect all unique parameter names across trials, preserving insertion order."""
    seen = set()
    names = []
    for t in trials:
        for k in t:
            if k not in ('trial_id', 'accuracy', 'runtime', 'drifts') and k not in seen:
                seen.add(k)
                names.append(k)
    return names


def write_csv(config, trials, output_dir):
    """Write the CSV file with trial results."""
    detector = config.get('detector', 'Unknown')
    dataset = config.get('dataset', 'Unknown')
    filename = f'{detector}_{dataset}.csv'
    filepath = os.path.join(output_dir, filename)

    param_names = collect_param_names(trials)
    fieldnames = ['trial_id', 'accuracy', 'runtime', 'drifts'] + param_names

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial)

    print(f'Wrote {filepath} ({len(trials)} trials, {len(param_names)} parameters)')
    return filepath


def write_config(config, output_dir):
    """Write the .config file with run configuration."""
    detector = config.get('detector', 'Unknown')
    dataset = config.get('dataset', 'Unknown')
    filename = f'{detector}_{dataset}.config'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        for key in ('detector', 'dataset', 'classifier', 'training_samples', 'seed'):
            if key in config:
                f.write(f'{key}={config[key]}\n')

    print(f'Wrote {filepath}')
    return filepath


def main():
    parser = ArgumentParser(description='Parse SLURM .out files from Optuna optimization runs.')
    parser.add_argument('input', nargs='+', help='One or more .out files to parse')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory (default: current dir)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filepath in args.input:
        print(f'\nParsing {filepath} ...')
        config, trials = parse_slurm_out(filepath)

        if not config.get('detector'):
            print(f'  WARNING: Could not determine detector name, skipping.')
            continue
        if not config.get('dataset'):
            print(f'  WARNING: Could not determine dataset name, skipping.')
            continue
        if not trials:
            print(f'  WARNING: No trials found, skipping.')
            continue

        print(f'  Config: {config}')
        print(f'  Trials: {len(trials)}')

        write_csv(config, trials, args.output_dir)
        write_config(config, args.output_dir)


if __name__ == '__main__':
    main()
