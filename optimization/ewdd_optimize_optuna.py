"""
EWDD Hyperparameter Optimization using Optuna.

This script defines the search space for all EWDD parameters and launches
the optimization process using Optuna with TPE sampler.

Detectors: BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect
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

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.trial import FrozenTrial, TrialState

# Suppress scipy warnings
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide',
                        category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', message=r'p-value', category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from detectors.ewdd import EWDD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed parameters
DATASET = None  # Set from CLI argument
CLASSIFIER = 'HoeffdingTreeClassifier'
N_TRAINING_SAMPLES = 1600
SEED = 42
TRIAL_TIMEOUT = 3600  # 60 minutes per trial


def _timeout_handler(signum, frame):
    raise TimeoutError("Trial exceeded 60-minute time limit")


def create_ewdd_config(params: dict) -> str:
    """Create a temporary EWDD config file from hyperparameters."""
    config = {
        'detector_decision_criteria': params['detector_decision_criteria'],
        'ensemble_decision_criteria': params['ensemble_decision_criteria'],
        'decision_window': params['decision_window'],
        'suppression_window': params['suppression_window'],
        'verbose': False,
        'detectors': [
            {
                'class': 'detectors.bndm.BNDM',
                'params': {
                    'n_samples': params['bndm_n_samples'],
                    'const': params['bndm_const'],
                    'threshold': params['bndm_threshold'],
                    'max_depth': params['bndm_max_depth'],
                }
            },
            {
                'class': 'detectors.csddm.CSDDM',
                'params': {
                    'n_samples': params['csddm_n_samples'],
                    'feature_proportion': params['csddm_feature_proportion'],
                    'n_clusters': params['csddm_n_clusters'],
                    'confidence': params['csddm_confidence'],
                }
            },
            {
                'class': 'detectors.d3.D3',
                'params': {
                    'n_reference_samples': params['d3_n_reference_samples'],
                    'recent_samples_proportion': params['d3_recent_samples_proportion'],
                    'threshold': params['d3_threshold'],
                }
            },
            {
                'class': 'detectors.ibdd.IBDD',
                'params': {
                    'n_samples': params['ibdd_n_samples'],
                    'n_consecutive_deviations': params['ibdd_n_consecutive_deviations'],
                    'n_permutations': params['ibdd_n_permutations'],
                    'update_interval': params['ibdd_update_interval'],
                }
            },
            {
                'class': 'detectors.ocdd.OCDD',
                'params': {
                    'n_samples': params['ocdd_n_samples'],
                    'threshold': params['ocdd_threshold'],
                }
            },
            {
                'class': 'detectors.spll.SPLL',
                'params': {
                    'n_samples': params['spll_n_samples'],
                    'n_clusters': params['spll_n_clusters'],
                    'threshold': params['spll_threshold'],
                }
            },
            {
                'class': 'detectors.udetect.UDetect',
                'params': {
                    'n_windows': params['udetect_n_windows'],
                    'n_samples': params['udetect_n_samples'],
                    'disjoint_training_windows': params['udetect_disjoint_training_windows'],
                }
            }
        ]
    }
    
    # Write to temporary file
    fd, config_path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def objective(trial: optuna.Trial):
    """Optuna objective function for EWDD optimization. Returns (accuracy, runtime)."""
    
    # Sample recent_samples_size (shared across all detectors)
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    
    # Sample hyperparameters
    params = {
        # EWDD ensemble parameters
        'detector_decision_criteria': trial.suggest_categorical(
            'detector_decision_criteria', ['any', 'majority', 'all']),
        'ensemble_decision_criteria': trial.suggest_categorical(
            'ensemble_decision_criteria', ['any', 'majority', 'all']),
        'decision_window': trial.suggest_int('decision_window', 1, 100),
        'suppression_window': trial.suggest_int('suppression_window', 0, 500),
        
        # BNDM parameters
        'bndm_n_samples': trial.suggest_int('bndm_n_samples', 50, 500),
        'bndm_const': trial.suggest_float('bndm_const', 0.1, 10.0),
        'bndm_threshold': trial.suggest_float('bndm_threshold', 0.1, 0.9),
        'bndm_max_depth': trial.suggest_int('bndm_max_depth', 1, 10),
        
        # CSDDM parameters
        'csddm_n_samples': trial.suggest_int('csddm_n_samples', 50, 500),
        'csddm_feature_proportion': trial.suggest_float('csddm_feature_proportion', 0.1, 1.0),
        'csddm_n_clusters': trial.suggest_int('csddm_n_clusters', 2, 30),
        'csddm_confidence': trial.suggest_categorical(
            'csddm_confidence', [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        
        # D3 parameters
        'd3_n_reference_samples': trial.suggest_int('d3_n_reference_samples', 50, 500),
        'd3_recent_samples_proportion': trial.suggest_float('d3_recent_samples_proportion', 0.05, 0.5),
        'd3_threshold': trial.suggest_float('d3_threshold', 0.1, 0.9),
        
        # IBDD parameters
        'ibdd_n_samples': trial.suggest_int('ibdd_n_samples', 100, 2000),
        'ibdd_n_consecutive_deviations': trial.suggest_int('ibdd_n_consecutive_deviations', 1, 20),
        'ibdd_n_permutations': trial.suggest_int('ibdd_n_permutations', 100, 1000),
        'ibdd_update_interval': trial.suggest_int('ibdd_update_interval', 10, 100),
        
        # OCDD parameters
        'ocdd_n_samples': trial.suggest_int('ocdd_n_samples', 50, 500),
        'ocdd_threshold': trial.suggest_float('ocdd_threshold', 0.1, 0.9),
        
        # SPLL parameters
        'spll_n_samples': trial.suggest_int('spll_n_samples', 100, 1000),
        'spll_n_clusters': trial.suggest_int('spll_n_clusters', 2, 20),
        'spll_threshold': trial.suggest_float('spll_threshold', 0.1, 5.0),
        
        # UDetect parameters
        'udetect_n_windows': trial.suggest_int('udetect_n_windows', 5, 30),
        'udetect_n_samples': trial.suggest_int('udetect_n_samples', 20, 200),
        'udetect_disjoint_training_windows': trial.suggest_categorical(
            'udetect_disjoint_training_windows', [True, False]),
    }
    
    # Create config file
    config_path = create_ewdd_config(params)
    
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TRIAL_TIMEOUT)
        
        # Load dataset
        dataset_class = getattr(datasets, DATASET)
        dataset = dataset_class(directory_path='/tmp')
        stream = iter(dataset)
        
        # Create EWDD detector
        detector = EWDD(
            seed=SEED,
            recent_samples_size=recent_samples_size,
            config_path=config_path
        )
        
        # Build classifier path
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model',
            CLASSIFIER,
            f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        # Run stream
        drifts, labels, predictions, n_req_labels, runtime, peak_memory, mean_memory = \
            detector.run_stream(stream, N_TRAINING_SAMPLES, classifier_path)
        
        signal.alarm(0)  # Cancel the alarm
        
        # Calculate accuracy
        correct = sum(1 for l, p in zip(labels, predictions) if l == p)
        accuracy = correct / len(labels) if labels else 0.0
        
        # Log intermediate results
        logger.info(f"Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={len(drifts)}")
        trial.set_user_attr('drifts', len(drifts))
        
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
        # Clean up temp config file
        if os.path.exists(config_path):
            os.remove(config_path)


# ── Parameter distributions (for reconstructing trials from CSV) ─────────────

def get_ewdd_param_distributions():
    """Return a dict of Optuna distributions for EWDD parameters."""
    return {
        'recent_samples_size': IntDistribution(50, 5000),
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
    }


def _cast_param(value_str, dist):
    """Cast a CSV string value to the correct Python type based on its distribution."""
    if value_str is None:
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


def load_existing_trials(csv_path, study):
    """Load completed EWDD trials from an existing CSV into the Optuna study.

    Returns (n_loaded, n_successful) where n_successful counts trials with
    finite runtime (i.e. not 'inf').
    """
    if not os.path.exists(csv_path):
        return 0, 0

    dists = get_ewdd_param_distributions()
    meta_columns = {'dataset', 'trial_id', 'accuracy', 'runtime', 'drifts'}
    n_loaded = 0
    n_successful = 0
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

            runtime_val = float(row['runtime'])
            trial = FrozenTrial(
                number=n_loaded,
                state=TrialState.COMPLETE,
                value=None,
                values=[float(row['accuracy']), runtime_val],
                datetime_start=datetime.datetime.now(),
                datetime_complete=datetime.datetime.now(),
                params=params,
                distributions=dists,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=0,
            )
            study.add_trial(trial)
            n_loaded += 1
            if runtime_val != float('inf'):
                n_successful += 1

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} trial(s) with out-of-range parameters")
    return n_loaded, n_successful


def main():
    parser = ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Target number of successful (non-inf runtime) runs')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--study_name', type=str, default='ewdd_optimization',
                        help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL for distributed optimization (e.g., sqlite:///ewdd.db)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for the entire optimization')
    parser.add_argument('--dataset', type=str, default='Electricity',
                        help='Dataset class name (e.g., Electricity, RialtoBridgeTimelapse, PokerHand)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for CSV result files')
    args = parser.parse_args()
    
    global DATASET
    DATASET = args.dataset
    
    # Copy dataset CSV to /tmp for faster I/O
    dataset_class = getattr(datasets, DATASET)
    tmp_dataset = dataset_class()
    csv_filename = tmp_dataset.filename
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'files', csv_filename)
    tmp_path = os.path.join('/tmp', csv_filename)
    shutil.copy2(src_path, tmp_path)
    logger.info(f"Copied {src_path} -> {tmp_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(args.output_dir, f'EWDD_{DATASET}.csv')
    csv_header_written = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    # Read existing header so appended rows use the same column order
    csv_fieldnames = None
    if csv_header_written:
        with open(results_csv, 'r', newline='') as f:
            csv_fieldnames = csv.DictReader(f).fieldnames
    n_rows_written = 0
    
    # Create or load study
    sampler = TPESampler(seed=SEED)
    
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            directions=['maximize', 'minimize'],
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            directions=['maximize', 'minimize'],
            sampler=sampler,
        )
    
    # Load previously completed trials from CSV so the sampler can
    # learn from them, then only run the remaining successful runs.
    n_existing, n_existing_successful = load_existing_trials(results_csv, study)
    remaining_successful = max(0, args.n_trials - n_existing_successful)
    print(f"\n>>> {n_existing_successful} / {args.n_trials} successful runs "
          f"({n_existing} total trials) <<<\n")
    
    print("=" * 80)
    print("EWDD Hyperparameter Optimization using Optuna")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Target successful runs: {args.n_trials}")
    print(f"Existing trials: {n_existing} ({n_existing_successful} successful)")
    print(f"Remaining successful runs needed: {remaining_successful}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Timeout: {args.timeout}s" if args.timeout else "Timeout: None")
    print(f"Storage: {args.storage}" if args.storage else "Storage: In-memory")
    print(f"Output: {results_csv}")
    print(f"Fixed parameters:")
    print(f"  - Dataset: {DATASET}")
    print(f"  - Classifier: {CLASSIFIER}")
    print(f"  - Training samples: {N_TRAINING_SAMPLES}")
    print(f"  - Seed: {SEED}")
    print(f"  - Recent samples size: optimized (50-5000)")
    print("=" * 80)
    print()
    
    if remaining_successful == 0:
        logger.info(f"Already have {n_existing_successful}/{args.n_trials} successful runs — skipping optimization")
    else:
        if n_existing > 0:
            logger.info(f"Loaded {n_existing} existing trials ({n_existing_successful} successful) from {results_csv}")
        logger.info(f"Need {remaining_successful} more successful runs")
        
        # Callback to write each completed trial to CSV immediately
        # and track successful (non-inf runtime) runs
        _n_new_successful = 0
        _n_new_trials = 0
        def trial_callback(study, trial):
            nonlocal csv_header_written, csv_fieldnames, n_rows_written
            nonlocal _n_new_successful, _n_new_trials
            if trial.state != TrialState.COMPLETE:
                return
            if trial.number < n_existing:
                return
            _n_new_trials += 1
            if trial.values[1] != float('inf'):
                _n_new_successful += 1
            row = {
                'trial_id': trial.number,
                'accuracy': trial.values[0],
                'runtime': trial.values[1],
                'drifts': trial.user_attrs.get('drifts', ''),
            }
            row.update(trial.params)
            mode = 'a' if csv_header_written else 'w'
            fieldnames = csv_fieldnames if csv_fieldnames else list(row.keys())
            with open(results_csv, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not csv_header_written:
                    writer.writeheader()
                    csv_header_written = True
                    csv_fieldnames = fieldnames
                writer.writerow(row)
            n_rows_written += 1
        
        # Run trials in batches until we have enough successful runs
        BATCH_SIZE = 10
        while _n_new_successful < remaining_successful:
            batch = min(BATCH_SIZE, remaining_successful - _n_new_successful)
            study.optimize(
                objective,
                n_trials=batch,
                n_jobs=args.n_jobs,
                timeout=args.timeout,
                show_progress_bar=False,
                callbacks=[trial_callback],
            )
            logger.info(f"EWDD: {n_existing_successful + _n_new_successful}/{args.n_trials} "
                        f"successful runs ({n_existing + _n_new_trials} total trials)")
        
        logger.info(f"EWDD: Finished — {_n_new_successful} new successful runs "
                    f"out of {_n_new_trials} total new trials")
        logger.info(f"Wrote {n_rows_written} new rows to {results_csv}")
    
    # Print results
    pareto_trials = study.best_trials
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print(f"\nPareto front: {len(pareto_trials)} trials")
    for t in sorted(pareto_trials, key=lambda t: t.values[0], reverse=True):
        print(f"  Trial {t.number}: accuracy={t.values[0]:.4f}, runtime={t.values[1]:.1f}s")
    
    # Generate YAML config for the highest-accuracy Pareto trial
    best_trial = max(pareto_trials, key=lambda t: t.values[0])
    best = best_trial.params
    print("\n" + "=" * 80)
    print(f"Best accuracy configuration as ewdd.config YAML (Trial {best_trial.number}: accuracy={best_trial.values[0]:.4f}, runtime={best_trial.values[1]:.1f}s):")
    print("=" * 80)
    print(f"""
# recent_samples_size: {best.get('recent_samples_size', 2424)}

detector_decision_criteria: {best.get('detector_decision_criteria', 'any')}
ensemble_decision_criteria: {best.get('ensemble_decision_criteria', 'any')}
decision_window: {best.get('decision_window', 10)}
suppression_window: {best.get('suppression_window', 10)}
verbose: false

detectors:
  - class: detectors.bndm.BNDM
    params:
      n_samples: {best.get('bndm_n_samples', 50)}
      const: {best.get('bndm_const', 1.0):.6f}
      threshold: {best.get('bndm_threshold', 0.5):.6f}
      max_depth: {best.get('bndm_max_depth', 3)}
  - class: detectors.csddm.CSDDM
    params:
      n_samples: {best.get('csddm_n_samples', 141)}
      feature_proportion: {best.get('csddm_feature_proportion', 0.3):.6f}
      n_clusters: {best.get('csddm_n_clusters', 15)}
      confidence: {best.get('csddm_confidence', 0.005):.6f}
  - class: detectors.d3.D3
    params:
      n_reference_samples: {best.get('d3_n_reference_samples', 163)}
      recent_samples_proportion: {best.get('d3_recent_samples_proportion', 0.11):.6f}
      threshold: {best.get('d3_threshold', 0.27):.6f}
  - class: detectors.ibdd.IBDD
    params:
      n_samples: {best.get('ibdd_n_samples', 1361)}
      n_consecutive_deviations: {best.get('ibdd_n_consecutive_deviations', 8)}
      n_permutations: {best.get('ibdd_n_permutations', 440)}
      update_interval: {best.get('ibdd_update_interval', 39)}
  - class: detectors.ocdd.OCDD
    params:
      n_samples: {best.get('ocdd_n_samples', 100)}
      threshold: {best.get('ocdd_threshold', 0.3):.6f}
  - class: detectors.spll.SPLL
    params:
      n_samples: {best.get('spll_n_samples', 500)}
      n_clusters: {best.get('spll_n_clusters', 5)}
      threshold: {best.get('spll_threshold', 0.5):.6f}
  - class: detectors.udetect.UDetect
    params:
      n_windows: {best.get('udetect_n_windows', 10)}
      n_samples: {best.get('udetect_n_samples', 50)}
      disjoint_training_windows: {str(best.get('udetect_disjoint_training_windows', True)).lower()}
""")
    
    # Also print the fastest Pareto trial if different
    fastest_trial = min(pareto_trials, key=lambda t: t.values[1])
    if fastest_trial.number != best_trial.number:
        best_f = fastest_trial.params
        print("\n" + "=" * 80)
        print(f"Fastest configuration as ewdd.config YAML (Trial {fastest_trial.number}: accuracy={fastest_trial.values[0]:.4f}, runtime={fastest_trial.values[1]:.1f}s):")
        print("=" * 80)
        print(f"""
# recent_samples_size: {best_f.get('recent_samples_size', 2424)}

detector_decision_criteria: {best_f.get('detector_decision_criteria', 'any')}
ensemble_decision_criteria: {best_f.get('ensemble_decision_criteria', 'any')}
decision_window: {best_f.get('decision_window', 10)}
suppression_window: {best_f.get('suppression_window', 10)}
verbose: false

detectors:
  - class: detectors.bndm.BNDM
    params:
      n_samples: {best_f.get('bndm_n_samples', 50)}
      const: {best_f.get('bndm_const', 1.0):.6f}
      threshold: {best_f.get('bndm_threshold', 0.5):.6f}
      max_depth: {best_f.get('bndm_max_depth', 3)}
  - class: detectors.csddm.CSDDM
    params:
      n_samples: {best_f.get('csddm_n_samples', 141)}
      feature_proportion: {best_f.get('csddm_feature_proportion', 0.3):.6f}
      n_clusters: {best_f.get('csddm_n_clusters', 15)}
      confidence: {best_f.get('csddm_confidence', 0.005):.6f}
  - class: detectors.d3.D3
    params:
      n_reference_samples: {best_f.get('d3_n_reference_samples', 163)}
      recent_samples_proportion: {best_f.get('d3_recent_samples_proportion', 0.11):.6f}
      threshold: {best_f.get('d3_threshold', 0.27):.6f}
  - class: detectors.ibdd.IBDD
    params:
      n_samples: {best_f.get('ibdd_n_samples', 1361)}
      n_consecutive_deviations: {best_f.get('ibdd_n_consecutive_deviations', 8)}
      n_permutations: {best_f.get('ibdd_n_permutations', 440)}
      update_interval: {best_f.get('ibdd_update_interval', 39)}
  - class: detectors.ocdd.OCDD
    params:
      n_samples: {best_f.get('ocdd_n_samples', 100)}
      threshold: {best_f.get('ocdd_threshold', 0.3):.6f}
  - class: detectors.spll.SPLL
    params:
      n_samples: {best_f.get('spll_n_samples', 500)}
      n_clusters: {best_f.get('spll_n_clusters', 5)}
      threshold: {best_f.get('spll_threshold', 0.5):.6f}
  - class: detectors.udetect.UDetect
    params:
      n_windows: {best_f.get('udetect_n_windows', 10)}
      n_samples: {best_f.get('udetect_n_samples', 50)}
      disjoint_training_windows: {str(best_f.get('udetect_disjoint_training_windows', True)).lower()}
""")
    
    # Clean up /tmp copy
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        logger.info(f"Removed {tmp_path}")


if __name__ == '__main__':
    main()
