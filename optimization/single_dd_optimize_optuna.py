"""
Single Drift Detector Hyperparameter Optimization using Optuna.

This script optimizes each individual drift detector (BNDM, CSDDM, D3, IBDD,
OCDD, SPLL, UDetect) separately using the same number of trials, allowing
comparison with MOPEDDS.
"""

import os
import sys
import csv
import datetime
import signal
import shutil
import logging
import warnings
from argparse import ArgumentParser

import optuna

# Suppress scipy RuntimeWarning for kurtosis calculation with small samples
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide',
                        category=RuntimeWarning, module='scipy')
# Suppress scipy p-value warnings from anderson_ksamp
warnings.filterwarnings('ignore', message=r'p-value', category=UserWarning)
from optuna.samplers import TPESampler
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.trial import FrozenTrial, TrialState

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


def run_detector(detector, classifier_path: str):
    """Run a detector on the configured dataset and return accuracy, runtime, and drift count."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT)
    
    dataset_class = getattr(datasets, DATASET)
    dataset = dataset_class(directory_path='/tmp')
    stream = iter(dataset)
    
    drifts, labels, predictions, n_req_labels, runtime, peak_memory, mean_memory = \
        detector.run_stream(stream, N_TRAINING_SAMPLES, classifier_path)
    
    signal.alarm(0)  # Cancel the alarm
    
    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    accuracy = correct / len(labels) if labels else 0.0
    
    return accuracy, runtime, len(drifts)


def objective_bndm(trial: optuna.Trial) -> float:
    """Optuna objective function for BNDM optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_samples': trial.suggest_int('n_samples', 50, 500),
        'const': trial.suggest_float('const', 0.1, 10.0),
        'threshold': trial.suggest_float('threshold', 0.1, 0.9),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
    }
    
    try:
        detector = BNDM(
            n_samples=params['n_samples'],
            const=params['const'],
            threshold=params['threshold'],
            max_depth=params['max_depth'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"BNDM Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"BNDM Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"BNDM Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_csddm(trial: optuna.Trial) -> float:
    """Optuna objective function for CSDDM optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_samples': trial.suggest_int('n_samples', 50, 500),
        'feature_proportion': trial.suggest_float('feature_proportion', 0.1, 1.0),
        'n_clusters': trial.suggest_int('n_clusters', 2, 30),
        'confidence': trial.suggest_categorical(
            'confidence', [0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
    }
    
    try:
        detector = CSDDM(
            n_samples=params['n_samples'],
            feature_proportion=params['feature_proportion'],
            n_clusters=params['n_clusters'],
            confidence=params['confidence'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"CSDDM Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"CSDDM Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"CSDDM Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_d3(trial: optuna.Trial) -> float:
    """Optuna objective function for D3 optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_reference_samples': trial.suggest_int('n_reference_samples', 50, 5000),
        'recent_samples_proportion': trial.suggest_float('recent_samples_proportion', 0.05, 0.5),
        'threshold': trial.suggest_float('threshold', 0.1, 0.9),
    }
    
    try:
        detector = D3(
            n_reference_samples=params['n_reference_samples'],
            recent_samples_proportion=params['recent_samples_proportion'],
            threshold=params['threshold'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"D3 Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"D3 Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"D3 Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_ibdd(trial: optuna.Trial) -> float:
    """Optuna objective function for IBDD optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_samples': trial.suggest_int('n_samples', 100, 2000),
        'n_consecutive_deviations': trial.suggest_int('n_consecutive_deviations', 1, 20),
        'n_permutations': trial.suggest_int('n_permutations', 100, 1000),
        'update_interval': trial.suggest_int('update_interval', 10, 100),
    }
    
    try:
        detector = IBDD(
            n_samples=params['n_samples'],
            n_consecutive_deviations=params['n_consecutive_deviations'],
            n_permutations=params['n_permutations'],
            update_interval=params['update_interval'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"IBDD Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"IBDD Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"IBDD Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_ocdd(trial: optuna.Trial) -> float:
    """Optuna objective function for OCDD optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_samples': trial.suggest_int('n_samples', 50, 500),
        'threshold': trial.suggest_float('threshold', 0.1, 0.9),
    }
    
    try:
        detector = OCDD(
            n_samples=params['n_samples'],
            threshold=params['threshold'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"OCDD Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"OCDD Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"OCDD Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_spll(trial: optuna.Trial) -> float:
    """Optuna objective function for SPLL optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_samples': trial.suggest_int('n_samples', 100, 1000),
        'n_clusters': trial.suggest_int('n_clusters', 2, 20),
        'threshold': trial.suggest_float('threshold', 0.1, 5.0),
    }
    
    try:
        detector = SPLL(
            n_samples=params['n_samples'],
            n_clusters=params['n_clusters'],
            threshold=params['threshold'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"SPLL Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"SPLL Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"SPLL Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


def objective_udetect(trial: optuna.Trial) -> float:
    """Optuna objective function for UDetect optimization."""
    recent_samples_size = trial.suggest_int('recent_samples_size', 50, 5000)
    params = {
        'n_windows': trial.suggest_int('n_windows', 5, 30),
        'n_samples': trial.suggest_int('n_samples', 20, 200),
        'disjoint_training_windows': trial.suggest_categorical(
            'disjoint_training_windows', [True, False]),
    }
    
    try:
        detector = UDetect(
            n_windows=params['n_windows'],
            n_samples=params['n_samples'],
            disjoint_training_windows=params['disjoint_training_windows'],
            seed=SEED,
            recent_samples_size=recent_samples_size,
        )
        
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model', CLASSIFIER, f'{CLASSIFIER}_{DATASET}.pkl'
        )
        
        accuracy, runtime, n_drifts = run_detector(detector, classifier_path)
        logger.info(f"UDetect Trial {trial.number}: accuracy={accuracy:.4f}, runtime={runtime:.1f}s, drifts={n_drifts}")
        trial.set_user_attr('drifts', n_drifts)
        return accuracy, runtime
        
    except TimeoutError:
        logger.warning(f"UDetect Trial {trial.number} timed out after {TRIAL_TIMEOUT}s — skipping")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')
    except Exception as e:
        logger.error(f"UDetect Trial {trial.number} failed: {e}")
        trial.set_user_attr('drifts', -1)
        return 0.0, float('inf')


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
    return dists


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


def load_existing_trials(csv_path, study, detector_name):
    """Load completed trials from an existing CSV into the Optuna study.

    Returns (n_loaded, n_successful) where n_successful counts trials with
    finite runtime (i.e. not 'inf').
    """
    if not os.path.exists(csv_path):
        return 0, 0

    dists = get_param_distributions(detector_name)
    meta_columns = {'trial_id', 'accuracy', 'runtime', 'drifts', 'detector', 'dataset'}
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


DETECTORS = {
    'BNDM': {
        'objective': objective_bndm,
        'params': ['recent_samples_size', 'n_samples', 'const', 'threshold', 'max_depth'],
    },
    'CSDDM': {
        'objective': objective_csddm,
        'params': ['recent_samples_size', 'n_samples', 'feature_proportion', 'n_clusters', 'confidence'],
    },
    'D3': {
        'objective': objective_d3,
        'params': ['recent_samples_size', 'n_reference_samples', 'recent_samples_proportion', 'threshold'],
    },
    'IBDD': {
        'objective': objective_ibdd,
        'params': ['recent_samples_size', 'n_samples', 'n_consecutive_deviations', 'n_permutations', 'update_interval'],
    },
    'OCDD': {
        'objective': objective_ocdd,
        'params': ['recent_samples_size', 'n_samples', 'threshold'],
    },
    'SPLL': {
        'objective': objective_spll,
        'params': ['recent_samples_size', 'n_samples', 'n_clusters', 'threshold'],
    },
    'UDetect': {
        'objective': objective_udetect,
        'params': ['recent_samples_size', 'n_windows', 'n_samples', 'disjoint_training_windows'],
    },
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Target number of successful (non-inf runtime) runs per detector')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL (e.g., sqlite:///single_dd.db)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds per detector')
    parser.add_argument('--detectors', type=str, nargs='+', 
                        default=['BNDM', 'CSDDM', 'D3', 'IBDD', 'OCDD', 'SPLL', 'UDetect'],
                        choices=['BNDM', 'CSDDM', 'D3', 'IBDD', 'OCDD', 'SPLL', 'UDetect'],
                        help='Which detectors to optimize')
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
    n_rows_written = 0
    
    print("=" * 80)
    print("Single Drift Detector Hyperparameter Optimization")
    print("=" * 80)
    print(f"Target successful runs per detector: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Detectors to optimize: {args.detectors}")
    print(f"Output directory: {args.output_dir}")
    print(f"Fixed parameters:")
    print(f"  - Dataset: {DATASET}")
    print(f"  - Classifier: {CLASSIFIER}")
    print(f"  - Training samples: {N_TRAINING_SAMPLES}")
    print(f"  - Seed: {SEED}")
    print(f"  - Recent samples size: optimized (50-5000)")
    print("=" * 80)
    print()
    
    results = {}
    
    for detector_name in args.detectors:
        print("\n" + "=" * 80)
        print(f"Optimizing {detector_name}")
        print("=" * 80)
        
        sampler = TPESampler(seed=SEED)
        study_name = f'{detector_name.lower()}_optimization'
        
        if args.storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=args.storage,
                load_if_exists=True,
                directions=['maximize', 'minimize'],
                sampler=sampler,
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                directions=['maximize', 'minimize'],
                sampler=sampler,
            )
        
        # Load previously completed trials from CSV so the sampler can
        # learn from them, then only run the remaining successful runs.
        results_csv = os.path.join(args.output_dir, f'{detector_name}_{DATASET}.csv')
        n_existing, n_existing_successful = load_existing_trials(results_csv, study, detector_name)
        remaining_successful = max(0, args.n_trials - n_existing_successful)
        print(f"\n>>> {detector_name}: {n_existing_successful} / {args.n_trials} successful runs "
              f"({n_existing} total trials) <<<\n")
        
        if n_existing > 0:
            logger.info(f"Loaded {n_existing} existing trials ({n_existing_successful} successful) "
                        f"for {detector_name} from {results_csv}")
        
        if remaining_successful == 0:
            logger.info(f"{detector_name} already has {n_existing_successful}/{args.n_trials} "
                        f"successful runs — skipping")
        else:
            logger.info(f"Need {remaining_successful} more successful runs for {detector_name} "
                        f"(existing: {n_existing_successful} successful / {n_existing} total, "
                        f"target: {args.n_trials} successful)")
            
            # Callback to write each completed trial to CSV immediately
            # and track successful (non-inf runtime) runs
            _results_csv = results_csv
            _csv_header_written = os.path.exists(_results_csv) and os.path.getsize(_results_csv) > 0
            # Read existing header so appended rows use the same column order
            _csv_fieldnames = None
            if _csv_header_written:
                with open(_results_csv, 'r', newline='') as f:
                    _csv_fieldnames = csv.DictReader(f).fieldnames
            _n_new_successful = 0
            _n_new_trials = 0
            def trial_callback(study, trial, _det=detector_name, _n_existing=n_existing,
                               _csv=_results_csv):
                nonlocal _csv_header_written, _csv_fieldnames, n_rows_written
                nonlocal _n_new_successful, _n_new_trials
                if trial.state != TrialState.COMPLETE:
                    return
                if trial.number < _n_existing:
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
                mode = 'a' if _csv_header_written else 'w'
                fieldnames = _csv_fieldnames if _csv_fieldnames else list(row.keys())
                with open(_csv, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not _csv_header_written:
                        writer.writeheader()
                        _csv_header_written = True
                        _csv_fieldnames = fieldnames
                    writer.writerow(row)
                n_rows_written += 1
            
            # Run trials in batches until we have enough successful runs
            BATCH_SIZE = 10
            while _n_new_successful < remaining_successful:
                batch = min(BATCH_SIZE, remaining_successful - _n_new_successful)
                study.optimize(
                    DETECTORS[detector_name]['objective'],
                    n_trials=batch,
                    n_jobs=args.n_jobs,
                    timeout=args.timeout,
                    show_progress_bar=False,
                    callbacks=[trial_callback],
                )
                logger.info(f"{detector_name}: {n_existing_successful + _n_new_successful}/{args.n_trials} "
                            f"successful runs ({n_existing + _n_new_trials} total trials)")
            
            logger.info(f"{detector_name}: Finished — {_n_new_successful} new successful runs "
                        f"out of {_n_new_trials} total new trials")
        
        pareto_trials = study.best_trials
        results[detector_name] = {
            'pareto_trials': pareto_trials,
        }
        
        print(f"\n{detector_name} Pareto front: {len(pareto_trials)} trials")
        for t in sorted(pareto_trials, key=lambda t: t.values[0], reverse=True):
            print(f"  Trial {t.number}: accuracy={t.values[0]:.4f}, runtime={t.values[1]:.1f}s")
            for key, value in t.params.items():
                print(f"    {key}: {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY (Pareto Front)")
    print("=" * 80)
    print(f"\n{'Detector':<10} {'# Pareto Trials':<18} {'Best Accuracy':<16} {'Best Runtime':<14}")
    print("-" * 60)
    
    for detector_name, result in results.items():
        pareto = result['pareto_trials']
        best_acc = max(t.values[0] for t in pareto)
        best_rt = min(t.values[1] for t in pareto)
        print(f"{detector_name:<10} {len(pareto):<18} {best_acc:.4f}           {best_rt:.1f}s")
    
    # Print best configs (highest accuracy from each Pareto front)
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS (highest accuracy on Pareto front)")
    print("=" * 80)
    
    for detector_name, result in results.items():
        best_trial = max(result['pareto_trials'], key=lambda t: t.values[0])
        print(f"\n# {detector_name} (Trial {best_trial.number}: accuracy={best_trial.values[0]:.4f}, runtime={best_trial.values[1]:.1f}s)")
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    logger.info(f"Wrote {n_rows_written} new rows to {args.output_dir}")
    
    # Clean up /tmp copy
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        logger.info(f"Removed {tmp_path}")


if __name__ == '__main__':
    main()
