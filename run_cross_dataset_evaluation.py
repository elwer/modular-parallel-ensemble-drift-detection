#!/usr/bin/env python3
"""
Cross-dataset Pareto evaluation.

For each detector and each source dataset (where optimization was done),
load the Pareto-optimal configurations from results/<Detector>_<Dataset>.csv,
then evaluate each Pareto config on all *other* datasets (excluding
ForestCovertype as evaluation target). Configs that exceed 1 hour are
marked as timeout.

Designed for parallel execution: one SLURM job per (source, eval) pair.

Outputs:
  results/cross_eval/cross_eval_<source>_<eval>.csv
    Columns: detector, source_dataset, eval_dataset, pareto_index,
             accuracy, runtime, drifts, timeout, <all hyperparams>
"""

import os
import sys
import csv
import signal
import shutil
import logging
import tempfile
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml

# Suppress noisy warnings
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide',
                        category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', message=r'p-value', category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets
from detectors.bndm import BNDM
from detectors.csddm import CSDDM
from detectors.d3 import D3
from detectors.ibdd import IBDD
from detectors.ocdd import OCDD
from detectors.spll import SPLL
from detectors.udetect import UDetect
from detectors.ewdd import EWDD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSIFIER = 'HoeffdingTreeClassifier'
N_TRAINING_SAMPLES = 1600
SEED = 42
TRIAL_TIMEOUT = 3600  # 1 hour

# Datasets where optimization was performed (source datasets)
SOURCE_DATASETS = ['Electricity', 'GasSensor', 'PokerHand', 'RialtoBridgeTimelapse']

# Datasets to evaluate on — ForestCovertype is excluded as an evaluation target.
# We skip source==eval pairs later, so configs are only evaluated on *other* datasets.
EVAL_DATASETS = ['Electricity', 'GasSensor', 'PokerHand', 'RialtoBridgeTimelapse']

SINGLE_DETECTORS = ['BNDM', 'CSDDM', 'D3', 'IBDD', 'OCDD', 'SPLL', 'UDetect']
ALL_DETECTORS = SINGLE_DETECTORS + ['EWDD']


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation exceeded 1-hour time limit")


# ── Detector construction ─────────────────────────────────────────────────────

def make_detector(detector_name, params):
    """Construct a single detector instance from a params dict."""
    common = {'seed': SEED, 'recent_samples_size': int(params['recent_samples_size'])}

    if detector_name == 'BNDM':
        return BNDM(n_samples=int(params['n_samples']), const=float(params['const']),
                     threshold=float(params['threshold']),
                     max_depth=int(params['max_depth']), **common)
    elif detector_name == 'CSDDM':
        return CSDDM(n_samples=int(params['n_samples']),
                      feature_proportion=float(params['feature_proportion']),
                      n_clusters=int(params['n_clusters']),
                      confidence=float(params['confidence']), **common)
    elif detector_name == 'D3':
        return D3(n_reference_samples=int(params['n_reference_samples']),
                  recent_samples_proportion=float(params['recent_samples_proportion']),
                  threshold=float(params['threshold']), **common)
    elif detector_name == 'IBDD':
        return IBDD(n_samples=int(params['n_samples']),
                    n_consecutive_deviations=int(params['n_consecutive_deviations']),
                    n_permutations=int(params['n_permutations']),
                    update_interval=int(params['update_interval']), **common)
    elif detector_name == 'OCDD':
        return OCDD(n_samples=int(params['n_samples']),
                    threshold=float(params['threshold']), **common)
    elif detector_name == 'SPLL':
        return SPLL(n_samples=int(params['n_samples']),
                    n_clusters=int(params['n_clusters']),
                    threshold=float(params['threshold']), **common)
    elif detector_name == 'UDetect':
        dtw = params['disjoint_training_windows']
        if isinstance(dtw, str):
            dtw = dtw.strip().lower() in ('true', '1')
        return UDetect(n_windows=int(params['n_windows']),
                       n_samples=int(params['n_samples']),
                       disjoint_training_windows=dtw, **common)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def make_ewdd(params):
    """Construct an EWDD instance from a params dict. Returns (detector, config_path)."""
    dtw = params.get('udetect_disjoint_training_windows', True)
    if isinstance(dtw, str):
        dtw = dtw.strip().lower() in ('true', '1')

    config = {
        'detector_decision_criteria': params['detector_decision_criteria'],
        'ensemble_decision_criteria': params['ensemble_decision_criteria'],
        'decision_window': int(params['decision_window']),
        'suppression_window': int(params['suppression_window']),
        'verbose': False,
        'detectors': [
            {'class': 'detectors.bndm.BNDM', 'params': {
                'n_samples': int(params['bndm_n_samples']),
                'const': float(params['bndm_const']),
                'threshold': float(params['bndm_threshold']),
                'max_depth': int(params['bndm_max_depth'])}},
            {'class': 'detectors.csddm.CSDDM', 'params': {
                'n_samples': int(params['csddm_n_samples']),
                'feature_proportion': float(params['csddm_feature_proportion']),
                'n_clusters': int(params['csddm_n_clusters']),
                'confidence': float(params['csddm_confidence'])}},
            {'class': 'detectors.d3.D3', 'params': {
                'n_reference_samples': int(params['d3_n_reference_samples']),
                'recent_samples_proportion': float(params['d3_recent_samples_proportion']),
                'threshold': float(params['d3_threshold'])}},
            {'class': 'detectors.ibdd.IBDD', 'params': {
                'n_samples': int(params['ibdd_n_samples']),
                'n_consecutive_deviations': int(params['ibdd_n_consecutive_deviations']),
                'n_permutations': int(params['ibdd_n_permutations']),
                'update_interval': int(params['ibdd_update_interval'])}},
            {'class': 'detectors.ocdd.OCDD', 'params': {
                'n_samples': int(params['ocdd_n_samples']),
                'threshold': float(params['ocdd_threshold'])}},
            {'class': 'detectors.spll.SPLL', 'params': {
                'n_samples': int(params['spll_n_samples']),
                'n_clusters': int(params['spll_n_clusters']),
                'threshold': float(params['spll_threshold'])}},
            {'class': 'detectors.udetect.UDetect', 'params': {
                'n_windows': int(params['udetect_n_windows']),
                'n_samples': int(params['udetect_n_samples']),
                'disjoint_training_windows': dtw}},
        ]
    }
    fd, config_path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f)

    detector = EWDD(seed=SEED,
                    recent_samples_size=int(params['recent_samples_size']),
                    config_path=config_path)
    return detector, config_path


# ── Pareto front computation ──────────────────────────────────────────────────

def compute_pareto_front(df):
    """Compute Pareto front for maximizing accuracy and minimizing runtime.
    Returns indices of Pareto-optimal rows in df.
    """
    # Filter out inf / failed trials
    valid = df[(df['runtime'] != float('inf')) & (df['accuracy'] > 0)].copy()
    if valid.empty:
        return valid

    valid = valid.sort_values('accuracy', ascending=False).reset_index(drop=True)
    pareto_indices = []
    min_runtime = float('inf')
    for idx, row in valid.iterrows():
        if row['runtime'] <= min_runtime:
            pareto_indices.append(idx)
            min_runtime = row['runtime']

    return valid.loc[pareto_indices]


def get_classifier_path(dataset_name):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'model', CLASSIFIER, f'{CLASSIFIER}_{dataset_name}.pkl'
    )


def run_detector_on_dataset(detector, dataset_name, tmp_dir):
    """Run detector with timeout. Returns (accuracy, runtime, n_drifts, timed_out)."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT)

    try:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(directory_path=tmp_dir)
        stream = iter(dataset)

        drifts, labels, predictions, n_req_labels, runtime, peak_memory, mean_memory = \
            detector.run_stream(stream, N_TRAINING_SAMPLES, get_classifier_path(dataset_name))

        signal.alarm(0)

        correct = sum(1 for l, p in zip(labels, predictions) if l == p)
        accuracy = correct / len(labels) if labels else 0.0
        return accuracy, runtime, len(drifts), False

    except TimeoutError:
        signal.alarm(0)
        logger.warning(f"  Timeout on {dataset_name}")
        return 0.0, float('inf'), 0, True

    except Exception as e:
        signal.alarm(0)
        logger.error(f"  Failed on {dataset_name}: {e}")
        return 0.0, float('inf'), 0, True


def copy_datasets_to_tmp(dataset_names):
    """Copy dataset CSVs to /tmp for faster I/O."""
    tmp_dir = tempfile.mkdtemp(prefix='cross_eval_')
    for name in dataset_names:
        dataset_class = getattr(datasets, name)
        tmp_dataset = dataset_class()
        csv_filename = tmp_dataset.filename
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'datasets', 'files', csv_filename)
        dst = os.path.join(tmp_dir, csv_filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    logger.info(f"Copied datasets to {tmp_dir}")
    return tmp_dir


# ── Load existing results for resume ──────────────────────────────────────────

def load_completed_keys(output_csv):
    """Load set of (detector, source_dataset, eval_dataset, pareto_index) already done."""
    completed = set()
    if not os.path.exists(output_csv):
        return completed
    with open(output_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['detector'], row['source_dataset'],
                   row['eval_dataset'], int(row['pareto_index']))
            completed.add(key)
    return completed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing <Detector>_<Dataset>.csv optimization results')
    parser.add_argument('--output_dir', type=str, default='results/cross_eval',
                        help='Output directory for per-pair CSV files')
    parser.add_argument('--source_dataset', type=str, required=True,
                        help='Source dataset whose Pareto configs to evaluate')
    parser.add_argument('--eval_dataset', type=str, required=True,
                        help='Target dataset to evaluate the Pareto configs on')
    parser.add_argument('--detector', type=str, default=None,
                        help='Single detector to evaluate (default: all)')
    args = parser.parse_args()

    if args.source_dataset == args.eval_dataset:
        logger.error("source_dataset and eval_dataset must differ")
        sys.exit(1)

    detectors_to_run = [args.detector] if args.detector else ALL_DETECTORS
    source_dataset = args.source_dataset
    eval_dataset = args.eval_dataset

    # Copy only the eval dataset to tmp (source CSVs are just read, not run)
    tmp_dir = copy_datasets_to_tmp([eval_dataset])

    os.makedirs(args.output_dir, exist_ok=True)
    if args.detector:
        output_csv = os.path.join(args.output_dir,
                                  f'cross_eval_{args.detector}_{source_dataset}_{eval_dataset}.csv')
    else:
        output_csv = os.path.join(args.output_dir,
                                  f'cross_eval_{source_dataset}_{eval_dataset}.csv')

    # Load already-completed evaluations for resume
    completed = load_completed_keys(output_csv)
    if completed:
        logger.info(f"Resuming: {len(completed)} evaluations already done in {output_csv}")

    # Prepare output CSV
    header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    fieldnames = None
    if header_written:
        with open(output_csv, 'r', newline='') as f:
            fieldnames = csv.DictReader(f).fieldnames

    n_written = 0

    for detector_name in detectors_to_run:
        # Load optimization results CSV for this detector on the source dataset
        csv_path = os.path.join(args.results_dir, f'{detector_name}_{source_dataset}.csv')
        if not os.path.exists(csv_path):
            logger.info(f"Skipping {detector_name}/{source_dataset} — no CSV found")
            continue

        df = pd.read_csv(csv_path)
        if 'accuracy' not in df.columns or 'runtime' not in df.columns:
            logger.warning(f"Skipping {csv_path} — missing accuracy/runtime columns")
            continue

        # Compute Pareto front
        pareto = compute_pareto_front(df)
        if pareto.empty:
            logger.info(f"No valid Pareto points for {detector_name}/{source_dataset}")
            continue

        logger.info(f"{detector_name}/{source_dataset}: {len(pareto)} Pareto-optimal configs "
                    f"-> evaluating on {eval_dataset}")

        # Evaluate each Pareto config on the eval dataset
        for pareto_idx, (_, row) in enumerate(pareto.iterrows()):
            key = (detector_name, source_dataset, eval_dataset, pareto_idx)
            if key in completed:
                continue

            params = {k: v for k, v in row.items()
                      if k not in ('trial_id', 'accuracy', 'runtime', 'drifts',
                                   'detector', 'dataset')}

            logger.info(f"  {detector_name} pareto#{pareto_idx} on {eval_dataset}")

            try:
                if detector_name == 'EWDD':
                    det, cfg_path = make_ewdd(params)
                else:
                    det = make_detector(detector_name, params)
                    cfg_path = None

                acc, rt, nd, timed_out = run_detector_on_dataset(
                    det, eval_dataset, tmp_dir)

                if cfg_path and os.path.exists(cfg_path):
                    os.remove(cfg_path)
            except Exception as e:
                logger.error(f"    Construction failed: {e}")
                acc, rt, nd, timed_out = 0.0, float('inf'), 0, True

            out_row = {
                'detector': detector_name,
                'source_dataset': source_dataset,
                'eval_dataset': eval_dataset,
                'pareto_index': pareto_idx,
                'source_accuracy': row['accuracy'],
                'source_runtime': row['runtime'],
                'eval_accuracy': acc,
                'eval_runtime': rt,
                'eval_drifts': nd,
                'timeout': timed_out,
            }
            out_row.update(params)

            mode = 'a' if header_written else 'w'
            fnames = fieldnames if fieldnames else list(out_row.keys())
            with open(output_csv, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                if not header_written:
                    writer.writeheader()
                    header_written = True
                    fieldnames = fnames
                writer.writerow(out_row)
            n_written += 1

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Done. Wrote {n_written} new rows to {output_csv}")


if __name__ == '__main__':
    main()
