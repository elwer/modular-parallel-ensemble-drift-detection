#!/usr/bin/env python3
"""
Cross-dataset hyperparameter optimization for drift detectors.

Leave-one-out strategy across 5 datasets:
  For each detector (BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect, MOPEDDS):
  - 5 folds: optimize on 4 train datasets (1000 trials), evaluate 10 Pareto
    configs on 1 holdout dataset
  - Objective: maximize avg accuracy, minimize avg normalized runtime
    (runtime / n_samples) across the 4 train datasets

Outputs per detector per fold:
  - <Detector>_optimization_trials_fold<N>.csv  — all 1000 trials
  - <Detector>_pareto_evaluation_fold<N>.csv    — 10 selected configs
                                                   with holdout results
"""

import os
import sys
import csv
import yaml
import shutil
import logging
import tempfile
import warnings
from argparse import ArgumentParser
from itertools import combinations

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.trial import FrozenTrial, TrialState

# Suppress noisy warnings
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide',
                        category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', message=r'p-value', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from detectors.bndm import BNDM
from detectors.csddm import CSDDM
from detectors.d3 import D3
from detectors.ibdd import IBDD
from detectors.ocdd import OCDD
from detectors.spll import SPLL
from detectors.udetect import UDetect
from detectors.mopedds import MOPEDDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSIFIER = 'HoeffdingTreeClassifier'
N_TRAINING_SAMPLES = 1600
SEED = 42

ALL_DATASETS = ['Electricity', 'GasSensor', 'ForestCovertype', 'PokerHand', 'RialtoBridgeTimelapse']

# n_samples per dataset (for normalized runtime)
DATASET_N_SAMPLES = {
    'Electricity': 45_312,
    'GasSensor': 13_910,
    'ForestCovertype': 581_012,
    'PokerHand': 829_201,
    'RialtoBridgeTimelapse': 82_250,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_classifier_path(dataset_name):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model', CLASSIFIER, f'{CLASSIFIER}_{dataset_name}.pkl'
    )


def run_detector_on_dataset(detector, dataset_name, tmp_dir):
    """Run a detector on a dataset. Returns (accuracy, runtime, n_drifts)."""
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(directory_path=tmp_dir)
    stream = iter(dataset)

    drifts, labels, predictions, n_req_labels, runtime, peak_memory, mean_memory = \
        detector.run_stream(stream, N_TRAINING_SAMPLES, get_classifier_path(dataset_name))

    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    accuracy = correct / len(labels) if labels else 0.0
    return accuracy, runtime, len(drifts)


def copy_datasets_to_tmp(dataset_names):
    """Copy dataset CSVs to a unique temporary directory for faster I/O.
    Returns the path to the created temporary directory."""
    tmp_dir = tempfile.mkdtemp(prefix='cross_dataset_opt_')
    logger.info(f"Created temporary directory: {tmp_dir}")
    for name in dataset_names:
        dataset_class = getattr(datasets, name)
        tmp_dataset = dataset_class()
        csv_filename = tmp_dataset.filename
        src = os.path.join('datasets', 'files', csv_filename)
        dst = os.path.join(tmp_dir, csv_filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} -> {dst}")
    return tmp_dir


# ── Detector construction from params ─────────────────────────────────────────

def make_detector(detector_name, params):
    """Construct a detector instance from a params dict."""
    common = {'seed': SEED, 'recent_samples_size': params['recent_samples_size']}

    if detector_name == 'BNDM':
        return BNDM(n_samples=params['n_samples'], const=params['const'],
                     threshold=params['threshold'], max_depth=params['max_depth'], **common)
    elif detector_name == 'CSDDM':
        return CSDDM(n_samples=params['n_samples'], feature_proportion=params['feature_proportion'],
                      n_clusters=params['n_clusters'], confidence=params['confidence'], **common)
    elif detector_name == 'D3':
        return D3(n_reference_samples=params['n_reference_samples'],
                  recent_samples_proportion=params['recent_samples_proportion'],
                  threshold=params['threshold'], **common)
    elif detector_name == 'IBDD':
        return IBDD(n_samples=params['n_samples'],
                    n_consecutive_deviations=params['n_consecutive_deviations'],
                    n_permutations=params['n_permutations'],
                    update_interval=params['update_interval'], **common)
    elif detector_name == 'OCDD':
        return OCDD(n_samples=params['n_samples'], threshold=params['threshold'], **common)
    elif detector_name == 'SPLL':
        return SPLL(n_samples=params['n_samples'], n_clusters=params['n_clusters'],
                    threshold=params['threshold'], **common)
    elif detector_name == 'UDetect':
        return UDetect(n_windows=params['n_windows'], n_samples=params['n_samples'],
                       disjoint_training_windows=params['disjoint_training_windows'], **common)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def make_mopedds(params):
    """Construct an MOPEDDS instance from a params dict."""
    config = {
        'detector_decision_criteria': params['detector_decision_criteria'],
        'ensemble_decision_criteria': params['ensemble_decision_criteria'],
        'decision_window': params['decision_window'],
        'suppression_window': params['suppression_window'],
        'verbose': False,
        'detectors': [
            {'class': 'detectors.bndm.BNDM', 'params': {
                'n_samples': params['bndm_n_samples'], 'const': params['bndm_const'],
                'threshold': params['bndm_threshold'], 'max_depth': params['bndm_max_depth']}},
            {'class': 'detectors.csddm.CSDDM', 'params': {
                'n_samples': params['csddm_n_samples'],
                'feature_proportion': params['csddm_feature_proportion'],
                'n_clusters': params['csddm_n_clusters'], 'confidence': params['csddm_confidence']}},
            {'class': 'detectors.d3.D3', 'params': {
                'n_reference_samples': params['d3_n_reference_samples'],
                'recent_samples_proportion': params['d3_recent_samples_proportion'],
                'threshold': params['d3_threshold']}},
            {'class': 'detectors.ibdd.IBDD', 'params': {
                'n_samples': params['ibdd_n_samples'],
                'n_consecutive_deviations': params['ibdd_n_consecutive_deviations'],
                'n_permutations': params['ibdd_n_permutations'],
                'update_interval': params['ibdd_update_interval']}},
            {'class': 'detectors.ocdd.OCDD', 'params': {
                'n_samples': params['ocdd_n_samples'], 'threshold': params['ocdd_threshold']}},
            {'class': 'detectors.spll.SPLL', 'params': {
                'n_samples': params['spll_n_samples'], 'n_clusters': params['spll_n_clusters'],
                'threshold': params['spll_threshold']}},
            {'class': 'detectors.udetect.UDetect', 'params': {
                'n_windows': params['udetect_n_windows'], 'n_samples': params['udetect_n_samples'],
                'disjoint_training_windows': params['udetect_disjoint_training_windows']}},
        ]
    }
    fd, config_path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f)

    detector = MOPEDDS(seed=SEED, recent_samples_size=params['recent_samples_size'],
                    config_path=config_path)
    return detector, config_path


# ── Parameter distributions (for reconstructing trials from CSV) ─────────────

def get_param_distributions(detector_name):
    """Return a dict of Optuna distributions for the given detector."""
    dists = {'recent_samples_size': IntDistribution(50, 5000)}

    if detector_name == 'BNDM':
        dists['n_samples'] = IntDistribution(50, 500)
        dists['const'] = FloatDistribution(0.1, 10.0)
        dists['threshold'] = FloatDistribution(0.1, 0.9)
        dists['max_depth'] = IntDistribution(1, 10)
    elif detector_name == 'CSDDM':
        dists['n_samples'] = IntDistribution(50, 500)
        dists['feature_proportion'] = FloatDistribution(0.1, 1.0)
        dists['n_clusters'] = IntDistribution(2, 30)
        dists['confidence'] = CategoricalDistribution([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    elif detector_name == 'D3':
        dists['n_reference_samples'] = IntDistribution(50, 5000)
        dists['recent_samples_proportion'] = FloatDistribution(0.05, 0.5)
        dists['threshold'] = FloatDistribution(0.1, 0.9)
    elif detector_name == 'IBDD':
        dists['n_samples'] = IntDistribution(100, 2000)
        dists['n_consecutive_deviations'] = IntDistribution(1, 20)
        dists['n_permutations'] = IntDistribution(100, 1000)
        dists['update_interval'] = IntDistribution(10, 100)
    elif detector_name == 'OCDD':
        dists['n_samples'] = IntDistribution(50, 500)
        dists['threshold'] = FloatDistribution(0.1, 0.9)
    elif detector_name == 'SPLL':
        dists['n_samples'] = IntDistribution(100, 1000)
        dists['n_clusters'] = IntDistribution(2, 20)
        dists['threshold'] = FloatDistribution(0.1, 5.0)
    elif detector_name == 'UDetect':
        dists['n_windows'] = IntDistribution(5, 30)
        dists['n_samples'] = IntDistribution(20, 200)
        dists['disjoint_training_windows'] = CategoricalDistribution([True, False])
    elif detector_name == 'MOPEDDS':
        dists.update({
            'detector_decision_criteria': CategoricalDistribution(['any', 'majority', 'all']),
            'ensemble_decision_criteria': CategoricalDistribution(['any', 'majority', 'all']),
            'decision_window': IntDistribution(1, 100),
            'suppression_window': IntDistribution(0, 500),
            'bndm_n_samples': IntDistribution(50, 500),
            'bndm_const': FloatDistribution(0.1, 10.0),
            'bndm_threshold': FloatDistribution(0.1, 0.9),
            'bndm_max_depth': IntDistribution(1, 10),
            'csddm_n_samples': IntDistribution(50, 500),
            'csddm_feature_proportion': FloatDistribution(0.1, 1.0),
            'csddm_n_clusters': IntDistribution(2, 30),
            'csddm_confidence': CategoricalDistribution([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
            'd3_n_reference_samples': IntDistribution(50, 500),
            'd3_recent_samples_proportion': FloatDistribution(0.05, 0.5),
            'd3_threshold': FloatDistribution(0.1, 0.9),
            'ibdd_n_samples': IntDistribution(100, 2000),
            'ibdd_n_consecutive_deviations': IntDistribution(1, 20),
            'ibdd_n_permutations': IntDistribution(100, 1000),
            'ibdd_update_interval': IntDistribution(10, 100),
            'ocdd_n_samples': IntDistribution(50, 500),
            'ocdd_threshold': FloatDistribution(0.1, 0.9),
            'spll_n_samples': IntDistribution(100, 1000),
            'spll_n_clusters': IntDistribution(2, 20),
            'spll_threshold': FloatDistribution(0.1, 5.0),
            'udetect_n_windows': IntDistribution(5, 30),
            'udetect_n_samples': IntDistribution(20, 200),
            'udetect_disjoint_training_windows': CategoricalDistribution([True, False]),
        })
    return dists


def _cast_param(value_str, dist):
    """Cast a CSV string value to the correct Python type based on its distribution."""
    if value_str is None:
        return value_str
    if isinstance(dist, IntDistribution):
        return int(float(value_str))
    elif isinstance(dist, FloatDistribution):
        return float(value_str)
    elif isinstance(dist, CategoricalDistribution):
        # Try to match against the known choices
        for choice in dist.choices:
            if isinstance(choice, bool):
                if value_str.strip().lower() in ('true', '1'):
                    return True
                elif value_str.strip().lower() in ('false', '0'):
                    return False
            elif isinstance(choice, (int, float)):
                try:
                    if float(value_str) == float(choice):
                        return choice
                except ValueError:
                    pass
            elif str(choice) == value_str:
                return choice
        # Fallback: return as string
        return value_str
    return value_str


def _param_in_distribution(value, dist):
    """Return True if *value* is valid for the given Optuna distribution."""
    if value is None:
        return False
    if isinstance(dist, IntDistribution):
        return isinstance(value, (int, float)) and dist.low <= int(value) <= dist.high
    if isinstance(dist, FloatDistribution):
        return isinstance(value, (int, float)) and dist.low <= float(value) <= dist.high
    if isinstance(dist, CategoricalDistribution):
        return value in dist.choices
    return True


def load_existing_trials(trials_path, study, detector_name, fold_idx):
    """Load completed trials from an existing CSV into the Optuna study.

    Returns the number of trials loaded for the given fold.
    """
    if not os.path.exists(trials_path):
        return 0

    dists = get_param_distributions(detector_name)
    meta_columns = {'fold', 'train_datasets', 'test_datasets', 'trial_id',
                    'avg_accuracy', 'avg_norm_runtime'}
    n_loaded = 0
    n_skipped = 0

    with open(trials_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only load trials belonging to this fold
            if int(row['fold']) != fold_idx:
                continue

            # Reconstruct params with correct types
            params = {}
            skip = False
            for key, val in row.items():
                if key in meta_columns:
                    continue
                if key in dists:
                    casted = _cast_param(val, dists[key])
                    if not _param_in_distribution(casted, dists[key]):
                        skip = True
                        break
                    params[key] = casted
                # else: skip unknown columns

            if skip:
                n_skipped += 1
                continue

            trial = FrozenTrial(
                number=n_loaded,  # will be renumbered by study.add_trial
                state=TrialState.COMPLETE,
                values=[float(row['avg_accuracy']), float(row['avg_norm_runtime'])],
                datetime_start=None,
                datetime_complete=None,
                params=params,
                distributions=dists,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=0,
            )
            study.add_trial(trial)
            n_loaded += 1

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} trial(s) with out-of-range parameters")
    return n_loaded


# ── Objective functions ───────────────────────────────────────────────────────

def suggest_single_dd_params(trial, detector_name):
    """Suggest hyperparameters for a single drift detector."""
    params = {'recent_samples_size': trial.suggest_int('recent_samples_size', 50, 5000)}

    if detector_name == 'BNDM':
        params['n_samples'] = trial.suggest_int('n_samples', 50, 500)
        params['const'] = trial.suggest_float('const', 0.1, 10.0)
        params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
        params['max_depth'] = trial.suggest_int('max_depth', 1, 10)
    elif detector_name == 'CSDDM':
        params['n_samples'] = trial.suggest_int('n_samples', 50, 500)
        params['feature_proportion'] = trial.suggest_float('feature_proportion', 0.1, 1.0)
        params['n_clusters'] = trial.suggest_int('n_clusters', 2, 30)
        params['confidence'] = trial.suggest_categorical('confidence',
                                                          [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    elif detector_name == 'D3':
        params['n_reference_samples'] = trial.suggest_int('n_reference_samples', 50, 5000)
        params['recent_samples_proportion'] = trial.suggest_float('recent_samples_proportion', 0.05, 0.5)
        params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
    elif detector_name == 'IBDD':
        params['n_samples'] = trial.suggest_int('n_samples', 100, 2000)
        params['n_consecutive_deviations'] = trial.suggest_int('n_consecutive_deviations', 1, 20)
        params['n_permutations'] = trial.suggest_int('n_permutations', 100, 1000)
        params['update_interval'] = trial.suggest_int('update_interval', 10, 100)
    elif detector_name == 'OCDD':
        params['n_samples'] = trial.suggest_int('n_samples', 50, 500)
        params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
    elif detector_name == 'SPLL':
        params['n_samples'] = trial.suggest_int('n_samples', 100, 1000)
        params['n_clusters'] = trial.suggest_int('n_clusters', 2, 20)
        params['threshold'] = trial.suggest_float('threshold', 0.1, 5.0)
    elif detector_name == 'UDetect':
        params['n_windows'] = trial.suggest_int('n_windows', 5, 30)
        params['n_samples'] = trial.suggest_int('n_samples', 20, 200)
        params['disjoint_training_windows'] = trial.suggest_categorical(
            'disjoint_training_windows', [True, False])
    return params


def suggest_mopedds_params(trial):
    """Suggest hyperparameters for MOPEDDS."""
    params = {
        'recent_samples_size': trial.suggest_int('recent_samples_size', 50, 5000),
        'detector_decision_criteria': trial.suggest_categorical(
            'detector_decision_criteria', ['any', 'majority', 'all']),
        'ensemble_decision_criteria': trial.suggest_categorical(
            'ensemble_decision_criteria', ['any', 'majority', 'all']),
        'decision_window': trial.suggest_int('decision_window', 1, 100),
        'suppression_window': trial.suggest_int('suppression_window', 0, 500),
        'bndm_n_samples': trial.suggest_int('bndm_n_samples', 50, 500),
        'bndm_const': trial.suggest_float('bndm_const', 0.1, 10.0),
        'bndm_threshold': trial.suggest_float('bndm_threshold', 0.1, 0.9),
        'bndm_max_depth': trial.suggest_int('bndm_max_depth', 1, 10),
        'csddm_n_samples': trial.suggest_int('csddm_n_samples', 50, 500),
        'csddm_feature_proportion': trial.suggest_float('csddm_feature_proportion', 0.1, 1.0),
        'csddm_n_clusters': trial.suggest_int('csddm_n_clusters', 2, 30),
        'csddm_confidence': trial.suggest_categorical(
            'csddm_confidence', [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        'd3_n_reference_samples': trial.suggest_int('d3_n_reference_samples', 50, 500),
        'd3_recent_samples_proportion': trial.suggest_float('d3_recent_samples_proportion', 0.05, 0.5),
        'd3_threshold': trial.suggest_float('d3_threshold', 0.1, 0.9),
        'ibdd_n_samples': trial.suggest_int('ibdd_n_samples', 100, 2000),
        'ibdd_n_consecutive_deviations': trial.suggest_int('ibdd_n_consecutive_deviations', 1, 20),
        'ibdd_n_permutations': trial.suggest_int('ibdd_n_permutations', 100, 1000),
        'ibdd_update_interval': trial.suggest_int('ibdd_update_interval', 10, 100),
        'ocdd_n_samples': trial.suggest_int('ocdd_n_samples', 50, 500),
        'ocdd_threshold': trial.suggest_float('ocdd_threshold', 0.1, 0.9),
        'spll_n_samples': trial.suggest_int('spll_n_samples', 100, 1000),
        'spll_n_clusters': trial.suggest_int('spll_n_clusters', 2, 20),
        'spll_threshold': trial.suggest_float('spll_threshold', 0.1, 5.0),
        'udetect_n_windows': trial.suggest_int('udetect_n_windows', 5, 30),
        'udetect_n_samples': trial.suggest_int('udetect_n_samples', 20, 200),
        'udetect_disjoint_training_windows': trial.suggest_categorical(
            'udetect_disjoint_training_windows', [True, False]),
    }
    return params


def evaluate_on_datasets(detector_name, params, dataset_names, tmp_dir):
    """Run a detector config on multiple datasets.
    Returns dict: {dataset: (accuracy, runtime, n_drifts)} and
    (avg_accuracy, avg_normalized_runtime).
    """
    results = {}
    accuracies = []
    norm_runtimes = []

    for ds in dataset_names:
        try:
            if detector_name == 'MOPEDDS':
                detector, config_path = make_mopedds(params)
            else:
                detector = make_detector(detector_name, params)
                config_path = None

            acc, rt, nd = run_detector_on_dataset(detector, ds, tmp_dir)
            results[ds] = (acc, rt, nd)
            accuracies.append(acc)
            norm_runtimes.append(rt / DATASET_N_SAMPLES[ds])

            if config_path and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            logger.error(f"  {detector_name} failed on {ds}: {e}")
            results[ds] = (0.0, float('inf'), 0)
            accuracies.append(0.0)
            norm_runtimes.append(float('inf'))

    avg_acc = np.mean(accuracies)
    avg_norm_rt = np.mean(norm_runtimes)
    return results, avg_acc, avg_norm_rt


def make_objective(detector_name, train_datasets, tmp_dir):
    """Create an Optuna objective that evaluates across train_datasets."""

    def objective(trial):
        if detector_name == 'MOPEDDS':
            params = suggest_mopedds_params(trial)
        else:
            params = suggest_single_dd_params(trial, detector_name)

        _, avg_acc, avg_norm_rt = evaluate_on_datasets(detector_name, params, train_datasets, tmp_dir)

        logger.info(f"{detector_name} Trial {trial.number}: "
                     f"avg_accuracy={avg_acc:.4f}, avg_norm_runtime={avg_norm_rt:.6f}")
        return avg_acc, avg_norm_rt

    return objective


# ── Pareto front utilities ────────────────────────────────────────────────────

def compute_pareto_front(trials):
    """Return Pareto-optimal trials (maximize values[0], minimize values[1])."""
    sorted_trials = sorted(trials, key=lambda t: t.values[0], reverse=True)
    pareto = []
    min_rt = float('inf')
    for t in sorted_trials:
        if t.values[1] <= min_rt:
            pareto.append(t)
            min_rt = t.values[1]
    return pareto


def select_pareto_configs(pareto_trials, n=10):
    """Select n evenly-spaced configs from the Pareto front.
    Balances between accuracy-oriented and runtime-oriented configs.
    """
    if len(pareto_trials) <= n:
        return pareto_trials

    # Sort by accuracy ascending
    sorted_pareto = sorted(pareto_trials, key=lambda t: t.values[0])
    indices = np.linspace(0, len(sorted_pareto) - 1, n, dtype=int)
    # Use unique indices (in case of rounding duplicates)
    indices = list(dict.fromkeys(indices))
    selected = [sorted_pareto[i] for i in indices]

    # If we lost some due to dedup, fill from remaining
    if len(selected) < n:
        remaining = [t for t in sorted_pareto if t not in selected]
        extra_indices = np.linspace(0, len(remaining) - 1, n - len(selected), dtype=int)
        for i in dict.fromkeys(extra_indices):
            selected.append(remaining[i])

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def _count_existing_eval_rows(eval_path, fold_idx):
    """Count how many evaluation rows already exist for a given fold."""
    if not os.path.exists(eval_path):
        return 0
    count = 0
    with open(eval_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['fold']) == fold_idx:
                count += 1
    return count


def run_for_detector(detector_name, n_trials, output_dir, fold_idx=None):
    """Run cross-dataset optimization for one detector.
    
    If fold_idx is given, run only that fold. Otherwise run all 5 folds.
    Resumes from existing result CSVs: loads prior trials into Optuna so the
    sampler can learn from them, then only runs the remaining trials.  New
    rows are appended to the same files — nothing is overwritten.
    """
    logger.info(f"\n{'='*80}\nStarting cross-dataset optimization for {detector_name}\n{'='*80}")

    tmp_dir = copy_datasets_to_tmp(ALL_DATASETS)

    # Leave-one-out: 5 folds, each holds out 1 dataset
    test_combos = [(ds,) for ds in ALL_DATASETS]

    if fold_idx is not None:
        folds_to_run = [(fold_idx, test_combos[fold_idx])]
    else:
        folds_to_run = list(enumerate(test_combos))

    os.makedirs(output_dir, exist_ok=True)
    fold_suffix = f'_fold{fold_idx}' if fold_idx is not None else ''

    # Prepare trials CSV (write header once, append rows continuously)
    trials_path = os.path.join(output_dir, f'{detector_name}_optimization_trials{fold_suffix}.csv')
    # If the file already exists (with content), we must append, not overwrite
    trials_header_written = os.path.exists(trials_path) and os.path.getsize(trials_path) > 0
    # Read existing header so appended rows use the same column order
    trials_fieldnames = None
    if trials_header_written:
        with open(trials_path, 'r', newline='') as f:
            trials_fieldnames = csv.DictReader(f).fieldnames

    # Prepare evaluation CSV
    eval_path = os.path.join(output_dir, f'{detector_name}_pareto_evaluation{fold_suffix}.csv')
    eval_header_written = os.path.exists(eval_path) and os.path.getsize(eval_path) > 0
    eval_fieldnames = None
    if eval_header_written:
        with open(eval_path, 'r', newline='') as f:
            eval_fieldnames = csv.DictReader(f).fieldnames

    n_trials_written = 0
    n_eval_written = 0

    for fi, test_datasets in folds_to_run:
        train_datasets = [d for d in ALL_DATASETS if d not in test_datasets]
        test_datasets = list(test_datasets)

        logger.info(f"\n--- Fold {fi}: train={train_datasets}, test={test_datasets} ---")

        # ── Optimization ──────────────────────────────────────────────────
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(
            study_name=f'{detector_name}_fold{fi}',
            directions=['maximize', 'minimize'],
            sampler=sampler,
        )

        # Load previously completed trials from CSV into the study so the
        # sampler benefits from prior results.
        n_existing = load_existing_trials(trials_path, study, detector_name, fi)
        remaining_trials = max(0, n_trials - n_existing)

        if n_existing > 0:
            logger.info(f"  Loaded {n_existing} existing trials for fold {fi} from {trials_path}")

        if remaining_trials == 0:
            logger.info(f"  Fold {fi} already has {n_existing}/{n_trials} trials — skipping optimization")
        else:
            logger.info(f"  Running {remaining_trials} new trials (existing: {n_existing}, target: {n_trials})")

            # Callback to write each completed trial to CSV immediately
            def trial_callback(study, trial, _fi=fi, _train=train_datasets, _test=test_datasets):
                nonlocal trials_header_written, trials_fieldnames, n_trials_written
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    return
                # Only write newly computed trials (skip the ones loaded from CSV)
                if trial.number < n_existing:
                    return
                row = {
                    'fold': _fi,
                    'train_datasets': ';'.join(_train),
                    'test_datasets': ';'.join(_test),
                    'trial_id': trial.number,
                    'avg_accuracy': trial.values[0],
                    'avg_norm_runtime': trial.values[1],
                }
                row.update(trial.params)
                mode = 'a' if trials_header_written else 'w'
                fieldnames = trials_fieldnames if trials_fieldnames else list(row.keys())
                with open(trials_path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not trials_header_written:
                        writer.writeheader()
                        trials_header_written = True
                        trials_fieldnames = fieldnames
                    writer.writerow(row)
                n_trials_written += 1

            objective = make_objective(detector_name, train_datasets, tmp_dir)
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True,
                           callbacks=[trial_callback])

            logger.info(f"Wrote {n_trials_written} new trial rows so far to {trials_path}")

        # ── Pareto selection ──────────────────────────────────────────────
        # Check if evaluation for this fold was already done
        existing_eval_rows = _count_existing_eval_rows(eval_path, fi)
        if existing_eval_rows > 0:
            logger.info(f"  Fold {fi} already has {existing_eval_rows} evaluation rows — skipping Pareto evaluation")
            continue

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pareto = compute_pareto_front(completed)
        selected = select_pareto_configs(pareto, n=10)

        logger.info(f"  Pareto front: {len(pareto)} trials, selected {len(selected)} configs")

        # ── Evaluate selected configs on holdout datasets ─────────────────
        for sel_idx, trial in enumerate(selected):
            params = trial.params
            logger.info(f"  Evaluating config {sel_idx} (trial {trial.number}) on {test_datasets}")

            for ds in test_datasets:
                try:
                    if detector_name == 'MOPEDDS':
                        det, cfg_path = make_mopedds(params)
                    else:
                        det = make_detector(detector_name, params)
                        cfg_path = None

                    acc, rt, nd = run_detector_on_dataset(det, ds, tmp_dir)

                    if cfg_path and os.path.exists(cfg_path):
                        os.remove(cfg_path)
                except Exception as e:
                    logger.error(f"    Failed on {ds}: {e}")
                    acc, rt, nd = 0.0, float('inf'), 0

                eval_row = {
                    'fold': fi,
                    'train_datasets': ';'.join(train_datasets),
                    'test_datasets': ';'.join(test_datasets),
                    'config_index': sel_idx,
                    'original_trial_id': trial.number,
                    'train_avg_accuracy': trial.values[0],
                    'train_avg_norm_runtime': trial.values[1],
                    'eval_dataset': ds,
                    'eval_accuracy': acc,
                    'eval_runtime': rt,
                    'eval_norm_runtime': rt / DATASET_N_SAMPLES[ds],
                    'eval_drifts': nd,
                }
                eval_row.update(params)

                # Write evaluation row immediately (always append)
                mode = 'a' if eval_header_written else 'w'
                fieldnames = eval_fieldnames if eval_fieldnames else list(eval_row.keys())
                with open(eval_path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not eval_header_written:
                        writer.writeheader()
                        eval_header_written = True
                        eval_fieldnames = fieldnames
                    writer.writerow(eval_row)
                n_eval_written += 1

    # Clean up temporary directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Removed temporary directory: {tmp_dir}")

    logger.info(f"Wrote {trials_path} ({n_trials_written} new rows)")
    logger.info(f"Wrote {eval_path} ({n_eval_written} new rows)")


def main():
    parser = ArgumentParser(description='Cross-dataset hyperparameter optimization for drift detectors.')
    parser.add_argument('--detector', type=str, required=True,
                        choices=['BNDM', 'CSDDM', 'D3', 'IBDD', 'OCDD', 'SPLL', 'UDetect', 'MOPEDDS'],
                        help='Which detector to optimize')
    parser.add_argument('--n_trials', type=int, default=1000,
                        help='Number of Optuna trials per fold (default: 1000)')
    parser.add_argument('--fold', type=int, default=None, choices=range(5),
                        help='Run only this fold index (0-4). If omitted, run all 5 folds.')
    parser.add_argument('--output_dir', type=str, default='optimization/cross_dataset_results',
                        help='Output directory for CSV files')
    args = parser.parse_args()

    test_combos = [(ds,) for ds in ALL_DATASETS]

    print("=" * 80)
    print("Cross-Dataset Hyperparameter Optimization (Leave-One-Out)")
    print("=" * 80)
    print(f"Detector: {args.detector}")
    print(f"Trials per fold: {args.n_trials}")
    print(f"Datasets: {ALL_DATASETS}")
    if args.fold is not None:
        train = [d for d in ALL_DATASETS if d not in test_combos[args.fold]]
        test = list(test_combos[args.fold])
        print(f"Fold: {args.fold} (train={train}, test={test})")
    else:
        print(f"Folds: all 5 (leave-one-out)")
    print(f"Classifier: {CLASSIFIER}")
    print(f"Training samples: {N_TRAINING_SAMPLES}")
    print(f"Seed: {SEED}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    run_for_detector(args.detector, args.n_trials, args.output_dir, fold_idx=args.fold)

    print("\nDone.")


if __name__ == '__main__':
    main()
