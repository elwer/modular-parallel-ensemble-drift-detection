"""
MOPEDDS Pre-Selected Ensemble Optimization using Optuna.

Instead of optimizing all ~28 hyperparameters jointly, this script:
  1. Reads the single-DD optimization results from results/<DD>_<Dataset>.csv
  2. Extracts the Pareto-optimal configurations for each detector
  3. Uses Optuna to search only over:
       - Which pre-selected config to use per detector (index into Pareto front)
       - Whether to include each detector at all
       - Ensemble-level parameters (decision criteria, windows, recent_samples_size)

This drastically reduces the search space and leverages already-found good
configurations for each ensemble member.

Outputs:
  - results/MOPEDDS_<Dataset>_PreSelected.csv   — all trial results
  - results/MOPEDDS_<Dataset>_PreSelected.config — config metadata
"""

import os
import sys
import csv
import datetime
import signal
import yaml
import shutil
import tempfile
import logging
import warnings
from argparse import ArgumentParser

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
from detectors.mopedds import MOPEDDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed parameters
DATASET = None  # Set from CLI argument
CLASSIFIER = 'HoeffdingTreeClassifier'
N_TRAINING_SAMPLES = 1600
SEED = 42
TRIAL_TIMEOUT = 3600  # 60 minutes per trial

# All single drift detectors that can be ensemble members
SINGLE_DD_NAMES = ['BNDM', 'CSDDM', 'D3', 'IBDD', 'OCDD', 'SPLL', 'UDetect']

# Mapping from detector name to its class path (for MOPEDDS config YAML)
DD_CLASS_PATHS = {
    'BNDM': 'detectors.bndm.BNDM',
    'CSDDM': 'detectors.csddm.CSDDM',
    'D3': 'detectors.d3.D3',
    'IBDD': 'detectors.ibdd.IBDD',
    'OCDD': 'detectors.ocdd.OCDD',
    'SPLL': 'detectors.spll.SPLL',
    'UDetect': 'detectors.udetect.UDetect',
}

# Mapping from detector name to the parameter names in its CSV (excluding meta columns)
DD_PARAM_NAMES = {
    'BNDM': ['n_samples', 'const', 'threshold', 'max_depth'],
    'CSDDM': ['n_samples', 'feature_proportion', 'n_clusters', 'confidence'],
    'D3': ['n_reference_samples', 'recent_samples_proportion', 'threshold'],
    'IBDD': ['n_samples', 'n_consecutive_deviations', 'n_permutations', 'update_interval'],
    'OCDD': ['n_samples', 'threshold'],
    'SPLL': ['n_samples', 'n_clusters', 'threshold'],
    'UDetect': ['n_windows', 'n_samples', 'disjoint_training_windows'],
}

# Types for each parameter (for casting from CSV strings)
DD_PARAM_TYPES = {
    'n_samples': int,
    'const': float,
    'threshold': float,
    'max_depth': int,
    'feature_proportion': float,
    'n_clusters': int,
    'confidence': float,
    'n_reference_samples': int,
    'recent_samples_proportion': float,
    'n_consecutive_deviations': int,
    'n_permutations': int,
    'update_interval': int,
    'n_windows': int,
    'disjoint_training_windows': lambda x: x.strip().lower() in ('true', '1') if isinstance(x, str) else bool(x),
}


def _timeout_handler(signum, frame):
    raise TimeoutError("Trial exceeded 60-minute time limit")


# ── Loading single DD results ────────────────────────────────────────────────

def _cast_dd_param(name, value_str):
    """Cast a single-DD CSV parameter value to its correct Python type."""
    cast_fn = DD_PARAM_TYPES.get(name, str)
    if callable(cast_fn) and cast_fn in (int, float, str):
        return cast_fn(float(value_str)) if cast_fn == int else cast_fn(value_str)
    elif callable(cast_fn):
        return cast_fn(value_str)
    return value_str


def compute_pareto_front(rows):
    """Given a list of dicts with 'accuracy' and 'runtime', return Pareto-optimal rows.

    Pareto: maximize accuracy, minimize runtime.
    """
    # Sort by accuracy descending
    sorted_rows = sorted(rows, key=lambda r: r['accuracy'], reverse=True)
    pareto = []
    min_rt = float('inf')
    for r in sorted_rows:
        if r['runtime'] <= min_rt:
            pareto.append(r)
            min_rt = r['runtime']
    return pareto


def load_single_dd_candidates(results_dir, dataset_name):
    """Load Pareto-optimal configs from single-DD CSVs.

    Returns a dict: {detector_name: [list of param dicts]}.
    Each param dict contains only the detector-specific parameters (no meta).
    """
    candidates = {}

    for dd_name in SINGLE_DD_NAMES:
        csv_path = os.path.join(results_dir, f'{dd_name}_{dataset_name}.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"No results found for {dd_name} on {dataset_name}: {csv_path}")
            continue

        # Read all completed trials
        rows = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row['accuracy'])
                    rt = float(row['runtime'])
                except (ValueError, KeyError):
                    continue
                if rt == float('inf') or acc == 0.0:
                    continue

                # Extract detector-specific params
                params = {}
                for pname in DD_PARAM_NAMES[dd_name]:
                    if pname in row:
                        params[pname] = _cast_dd_param(pname, row[pname])
                    else:
                        break
                else:
                    rows.append({
                        'accuracy': acc,
                        'runtime': rt,
                        'params': params,
                    })

        if not rows:
            logger.warning(f"No valid trials found for {dd_name} on {dataset_name}")
            continue

        # Compute Pareto front
        pareto = compute_pareto_front(rows)
        logger.info(f"{dd_name}: {len(rows)} trials, {len(pareto)} Pareto-optimal")

        # Use all Pareto-optimal configs (no accuracy filter, no cap)
        selected = sorted(pareto, key=lambda r: r['accuracy'])

        candidates[dd_name] = [s['params'] for s in selected]
        logger.info(f"  Selected {len(candidates[dd_name])} candidate configs for {dd_name}")

    return candidates


# ── MOPEDDS config creation from pre-selected params ────────────────────────────

def create_mopedds_config_from_candidates(ensemble_params, detector_configs):
    """Create a temporary MOPEDDS YAML config file.

    Args:
        ensemble_params: dict with decision_criteria, decision_window, suppression_window
        detector_configs: list of (dd_name, params_dict) for included detectors
    """
    config = {
        'detector_decision_criteria': ensemble_params['detector_decision_criteria'],
        'ensemble_decision_criteria': ensemble_params['ensemble_decision_criteria'],
        'decision_window': ensemble_params['decision_window'],
        'suppression_window': ensemble_params['suppression_window'],
        'verbose': False,
        'detectors': [],
    }

    for dd_name, params in detector_configs:
        config['detectors'].append({
            'class': DD_CLASS_PATHS[dd_name],
            'params': params,
        })

    fd, config_path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f)

    return config_path


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(candidates, optimize_detector_selection=False):
    """Create the Optuna objective using pre-selected candidates.

    The search space consists of:
      - recent_samples_size (shared)
      - ensemble-level params (2x decision criteria, decision_window, suppression_window)
      - per detector: config_idx_<dd> (int index into candidates)
      - if optimize_detector_selection: include_<dd> (bool) per detector

    If optimize_detector_selection is False (default), all available detectors
    are always included in the ensemble.
    """

    def objective(trial):
        # Ensemble-level hyperparameters
        recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)

        ensemble_params = {
            'detector_decision_criteria': trial.suggest_categorical(
                'detector_decision_criteria', ['any', 'majority', 'all']),
            'ensemble_decision_criteria': trial.suggest_categorical(
                'ensemble_decision_criteria', ['any', 'majority', 'all']),
            'decision_window': trial.suggest_int('decision_window', 1, 100),
            'suppression_window': trial.suggest_int('suppression_window', 0, 500),
        }

        # Select which config to use for each detector
        detector_configs = []
        for dd_name in SINGLE_DD_NAMES:
            if dd_name not in candidates:
                continue

            n_cands = len(candidates[dd_name])

            if optimize_detector_selection:
                include = trial.suggest_categorical(f'include_{dd_name}', [True, False])
            else:
                include = True

            if include:
                if n_cands == 1:
                    config_idx = 0
                else:
                    config_idx = trial.suggest_int(f'config_idx_{dd_name}', 0, n_cands - 1)
                detector_configs.append((dd_name, candidates[dd_name][config_idx]))
            else:
                # Still suggest config_idx so Optuna has consistent parameter space
                if n_cands > 1:
                    trial.suggest_int(f'config_idx_{dd_name}', 0, n_cands - 1)

        # Need at least 2 detectors for a meaningful ensemble
        if len(detector_configs) < 2:
            return 0.0, float('inf')

        # Create MOPEDDS config and run
        config_path = create_mopedds_config_from_candidates(ensemble_params, detector_configs)

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TRIAL_TIMEOUT)

            dataset_class = getattr(datasets, DATASET)
            dataset = dataset_class(directory_path='/tmp')
            stream = iter(dataset)

            detector = MOPEDDS(
                seed=SEED,
                recent_samples_size=recent_samples_size,
                config_path=config_path,
            )

            classifier_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl',
            )

            drifts, labels, predictions, n_req_labels, runtime, peak_memory, mean_memory = \
                detector.run_stream(stream, N_TRAINING_SAMPLES, classifier_path)

            signal.alarm(0)

            correct = sum(1 for l, p in zip(labels, predictions) if l == p)
            accuracy = correct / len(labels) if labels else 0.0

            logger.info(f"Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, "
                        f"drifts={len(drifts)}, detectors={[d[0] for d in detector_configs]}")
            trial.set_user_attr('drifts', len(drifts))
            trial.set_user_attr('n_detectors', len(detector_configs))
            trial.set_user_attr('detectors_used', ','.join(d[0] for d in detector_configs))

            return accuracy, runtime

        except TimeoutError:
            logger.warning(f"Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
            trial.set_user_attr('drifts', -1)
            return 0.0, float('inf')

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('drifts', -1)
            return 0.0, float('inf')

        finally:
            if os.path.exists(config_path):
                os.remove(config_path)

    return objective


# ── Trial CSV persistence ─────────────────────────────────────────────────────

def get_preselected_param_distributions(candidates, optimize_detector_selection=False):
    """Build the Optuna distribution dict for the pre-selected search space."""
    dists = {
        'recent_samples_size': IntDistribution(50, 5000),
        'detector_decision_criteria': CategoricalDistribution(['any', 'majority', 'all']),
        'ensemble_decision_criteria': CategoricalDistribution(['any', 'majority', 'all']),
        'decision_window': IntDistribution(1, 100),
        'suppression_window': IntDistribution(0, 500),
    }

    for dd_name in SINGLE_DD_NAMES:
        if dd_name not in candidates:
            continue
        if optimize_detector_selection:
            dists[f'include_{dd_name}'] = CategoricalDistribution([True, False])
        n_cands = len(candidates[dd_name])
        if n_cands > 1:
            dists[f'config_idx_{dd_name}'] = IntDistribution(0, n_cands - 1)

    return dists


def _cast_param(value_str, dist):
    """Cast a CSV string value to the correct Python type based on its distribution."""
    if value_str is None or value_str == '':
        return value_str
    if isinstance(dist, IntDistribution):
        try:
            return int(float(value_str))
        except (ValueError, TypeError):
            return value_str
    elif isinstance(dist, FloatDistribution):
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return value_str
    elif isinstance(dist, CategoricalDistribution):
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


def load_existing_trials(csv_path, study, candidates, optimize_detector_selection=False):
    """Load completed trials from an existing PreSelected CSV into the Optuna study."""
    if not os.path.exists(csv_path):
        return 0

    dists = get_preselected_param_distributions(candidates, optimize_detector_selection=optimize_detector_selection)
    meta_columns = {'trial_id', 'accuracy', 'runtime', 'drifts', 'n_detectors', 'detectors_used'}
    n_loaded = 0
    n_skipped = 0

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
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

            if skip:
                n_skipped += 1
                continue

            # Build per-row distributions containing only keys present in params,
            # because some config_idx_* keys may be absent when a detector had
            # only 1 candidate (Optuna requires params and distributions to match).
            row_dists = {k: dists[k] for k in params if k in dists}

            trial = FrozenTrial(
                number=n_loaded,
                state=TrialState.COMPLETE,
                value=None,
                values=[float(row['accuracy']), float(row['runtime'])],
                datetime_start=datetime.datetime.now(),
                datetime_complete=datetime.datetime.now(),
                params=params,
                distributions=row_dists,
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


# ── Config file generation ────────────────────────────────────────────────────

def write_best_config(trial, candidates, output_path):
    """Write the MOPEDDS .config YAML for the best trial."""
    params = trial.params

    detector_configs = []
    for dd_name in SINGLE_DD_NAMES:
        if dd_name not in candidates:
            continue
        include_key = f'include_{dd_name}'
        idx_key = f'config_idx_{dd_name}'
        if params.get(include_key, True):  # default True = always include
            n_cands = len(candidates[dd_name])
            config_idx = params.get(idx_key, 0) if n_cands > 1 else 0
            detector_configs.append((dd_name, candidates[dd_name][config_idx]))

    config = {
        'detector_decision_criteria': params['detector_decision_criteria'],
        'ensemble_decision_criteria': params['ensemble_decision_criteria'],
        'decision_window': params['decision_window'],
        'suppression_window': params['suppression_window'],
        'verbose': False,
        'detectors': [],
    }
    for dd_name, dd_params in detector_configs:
        config['detectors'].append({
            'class': DD_CLASS_PATHS[dd_name],
            'params': dd_params,
        })

    with open(output_path, 'w') as f:
        f.write(f"# MOPEDDS Pre-Selected Configuration\n")
        f.write(f"# Generated from single-DD Pareto-optimal configs\n")
        f.write(f"# Accuracy: {trial.values[0]:.4f}, Runtime: {trial.values[1]:.1f}s\n")
        f.write(f"# recent_samples_size: {params['recent_samples_size']}\n")
        f.write(f"# Included detectors: {[d[0] for d in detector_configs]}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Wrote config to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description='MOPEDDS optimization with pre-selected single-DD configs.')
    parser.add_argument('--n_trials', type=int, default=500,
                        help='Number of optimization trials (default: 500)')
    parser.add_argument('--n_candidates', type=int, default=None,
                        help='(deprecated, ignored) All Pareto-optimal candidates are now used')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for the entire optimization')
    parser.add_argument('--dataset', type=str, default='Electricity',
                        help='Dataset class name (e.g., Electricity, ForestCovertype)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing single-DD CSV results')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for PreSelected CSV and config')
    parser.add_argument('--optimize_selection', action='store_true', default=False,
                        help='Also optimize which detectors to include (disabled by default: all detectors used)')
    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset

    # ── Step 1: Load pre-selected candidates from single-DD results ──────
    logger.info(f"Loading single-DD candidates from {args.results_dir} for {DATASET}...")
    candidates = load_single_dd_candidates(args.results_dir, DATASET)

    if len(candidates) < 2:
        logger.error(f"Need at least 2 detectors with results, found {len(candidates)}: "
                     f"{list(candidates.keys())}. Run single-DD optimization first.")
        sys.exit(1)

    total_configs = sum(len(v) for v in candidates.values())
    logger.info(f"Loaded {total_configs} candidate configs across {len(candidates)} detectors")

    # ── Step 2: Copy dataset to /tmp for faster I/O ──────────────────────
    dataset_class = getattr(datasets, DATASET)
    tmp_dataset = dataset_class()
    csv_filename = tmp_dataset.filename
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'datasets', 'files', csv_filename)
    tmp_path = os.path.join('/tmp', csv_filename)
    shutil.copy2(src_path, tmp_path)
    logger.info(f"Copied {src_path} -> {tmp_path}")

    # ── Step 3: Set up Optuna study ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(args.output_dir, f'MOPEDDS_{DATASET}_PreSelected.csv')
    csv_header_written = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    csv_fieldnames = None
    if csv_header_written:
        with open(results_csv, 'r', newline='') as f:
            csv_fieldnames = csv.DictReader(f).fieldnames
    n_rows_written = 0

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=f'mopedds_preselected_{DATASET}',
        directions=['maximize', 'minimize'],
        sampler=sampler,
    )

    # Load previously completed trials
    n_existing = load_existing_trials(results_csv, study, candidates,
                                       optimize_detector_selection=args.optimize_selection)
    remaining_trials = max(0, args.n_trials - n_existing)

    print("=" * 80)
    print("MOPEDDS Pre-Selected Ensemble Optimization")
    print("=" * 80)
    print(f"Dataset: {DATASET}")
    print(f"Classifier: {CLASSIFIER}")
    print(f"Training samples: {N_TRAINING_SAMPLES}")
    print(f"Seed: {SEED}")
    print(f"Max trials: {args.n_trials}")
    print(f"Existing trials: {n_existing}")
    print(f"Remaining trials: {remaining_trials}")
    print(f"Optimize detector selection: {args.optimize_selection}")
    print(f"Candidates per detector: all Pareto-optimal")
    print(f"Available detectors: {list(candidates.keys())}")
    for dd_name, cands in candidates.items():
        print(f"  {dd_name}: {len(cands)} candidate configs")
    print(f"Output CSV: {results_csv}")
    print("=" * 80)
    print()

    if remaining_trials == 0:
        logger.info(f"Already have {n_existing}/{args.n_trials} trials — skipping optimization")
    else:
        if n_existing > 0:
            logger.info(f"Loaded {n_existing} existing trials from {results_csv}")
        logger.info(f"Running {remaining_trials} new trials")

        # Callback to write each completed trial to CSV immediately
        def trial_callback(study, trial):
            nonlocal csv_header_written, csv_fieldnames, n_rows_written
            if trial.state != TrialState.COMPLETE:
                return
            if trial.number < n_existing:
                return
            row = {
                'trial_id': trial.number,
                'accuracy': trial.values[0],
                'runtime': trial.values[1],
                'drifts': trial.user_attrs.get('drifts', ''),
                'n_detectors': trial.user_attrs.get('n_detectors', ''),
                'detectors_used': trial.user_attrs.get('detectors_used', ''),
            }
            row.update(trial.params)

            if not csv_header_written:
                csv_fieldnames = list(row.keys())
                with open(results_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
                csv_header_written = True
            else:
                new_fields = [k for k in row if k not in csv_fieldnames]
                if new_fields:
                    # Expand header: rewrite entire CSV with new columns
                    csv_fieldnames = csv_fieldnames + new_fields
                    existing_rows = []
                    with open(results_csv, 'r', newline='') as f:
                        existing_rows = list(csv.DictReader(f))
                    with open(results_csv, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                        writer.writeheader()
                        writer.writerows(existing_rows)
                        writer.writerow(row)
                else:
                    with open(results_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                        writer.writerow(row)
            n_rows_written += 1

        objective = make_objective(candidates, optimize_detector_selection=args.optimize_selection)
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            show_progress_bar=True,
            callbacks=[trial_callback],
        )

        logger.info(f"Wrote {n_rows_written} new rows to {results_csv}")

    # ── Step 4: Print results and write best config ──────────────────────
    pareto_trials = study.best_trials
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print(f"\nPareto front: {len(pareto_trials)} trials")
    for t in sorted(pareto_trials, key=lambda t: t.values[0], reverse=True):
        detectors_str = t.user_attrs.get('detectors_used', '?')
        print(f"  Trial {t.number}: accuracy={t.values[0]:.4f}, runtime={t.values[1]:.1f}s, "
              f"detectors=[{detectors_str}]")

    # Write config for best-accuracy Pareto trial
    if pareto_trials:
        best_trial = max(pareto_trials, key=lambda t: t.values[0])
        config_path = os.path.join(args.output_dir, f'MOPEDDS_{DATASET}_PreSelected.config')
        write_best_config(best_trial, candidates, config_path)

        print(f"\nBest config written to: {config_path}")
        print(f"  Accuracy: {best_trial.values[0]:.4f}")
        print(f"  Runtime:  {best_trial.values[1]:.1f}s")
        print(f"  recent_samples_size: {best_trial.params['recent_samples_size']}")

        # Also write fastest Pareto config if different
        fastest_trial = min(pareto_trials, key=lambda t: t.values[1])
        if fastest_trial.number != best_trial.number:
            fast_config_path = os.path.join(args.output_dir,
                                            f'MOPEDDS_{DATASET}_PreSelected_Fast.config')
            write_best_config(fastest_trial, candidates, fast_config_path)
            print(f"\nFastest config written to: {fast_config_path}")
            print(f"  Accuracy: {fastest_trial.values[0]:.4f}")
            print(f"  Runtime:  {fastest_trial.values[1]:.1f}s")

    # Clean up /tmp copy
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        logger.info(f"Removed {tmp_path}")


if __name__ == '__main__':
    main()
