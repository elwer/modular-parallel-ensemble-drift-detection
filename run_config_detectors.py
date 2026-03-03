#!/usr/bin/env python3
"""
Run Drift Detectors from Configuration File

This script runs all drift detectors defined in a configuration file on a given dataset.

Usage:
    python run_config_detectors.py <Dataset> <ConfigPath> <RecentSamplesSize> <TrainSamples> <Accuracy> <Runtime> <ReqLabels>

Arguments:
    Dataset           - Dataset class name or expression (e.g., 'Electricity' or 'SineClustersPre()')
    ConfigPath        - Path to the YAML configuration file containing detector definitions
    RecentSamplesSize - Number of recent samples to use for drift detection (int)
    TrainSamples      - Number of training samples (int)
    Accuracy          - Boolean flag (True/False/1/0) to output accuracy metric
    Runtime           - Boolean flag (True/False/1/0) to output runtime metric
    ReqLabels         - Boolean flag (True/False/1/0) to output requested labels metric

Examples:
    python run_config_detectors.py Electricity detectors/ewdd/configs/ewdd.config 500 1600 True True False
    python run_config_detectors.py SineClustersPre detectors/ewdd/configs/ewdd.config 1000 1600 True False True
"""

import os
import sys
import ast
import yaml
import warnings
import importlib

warnings.filterwarnings("ignore")

from metrics.metrics import get_metrics
from datasets import *


def parse_expression(expr_str):
    """
    Parses a string like 'ClassName(arg=value, ...)' or 'ClassName()'
    and returns (class name, instantiated object).
    """
    try:
        node = ast.parse(expr_str, mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            class_name = node.func.id
            obj = eval(expr_str, globals())
            return class_name, obj
        elif isinstance(node, ast.Name):
            class_name = node.id
            obj = eval(expr_str + "()", globals())
            return class_name, obj
        else:
            raise ValueError(f"Invalid expression format: {expr_str}")
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expr_str}': {e}")


def load_config(config_path):
    """
    Load and parse the YAML configuration file.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_detector_class(class_path):
    """
    Dynamically import and return a detector class from its full path.
    E.g., 'detectors.csddm.CSDDM' -> CSDDM class
    """
    parts = class_path.rsplit('.', 1)
    if len(parts) == 2:
        module_path, class_name = parts
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Invalid class path: {class_path}")


def instantiate_detector(detector_config, recent_samples_size):
    """
    Instantiate a detector from its configuration, injecting recent_samples_size.
    """
    class_path = detector_config['class']
    params = detector_config.get('params', {}).copy()
    
    params['recent_samples_size'] = recent_samples_size
    
    detector_class = get_detector_class(class_path)
    return detector_class(**params)


def run_detector(detector, detector_name, stream, n_training_samples, classifier_path):
    """
    Run a single detector on the stream and return results.
    """
    (drifts, labels, predictions, n_req_labels, runtime, peak_memory,
     mean_memory) = detector.run_stream(stream, n_training_samples, classifier_path)
    
    metrics = get_metrics(stream, drifts, labels, predictions, n_req_labels,
                          n_training_samples)
    
    # Reconstruct drift_responses from drifts list
    # drift_responses[i] = True if sample (n_training_samples + i) is in drifts
    num_samples = len(labels)
    drift_set = set(drifts)
    drift_responses = [((n_training_samples + i) in drift_set) for i in range(num_samples)]
    
    return {
        'detector_name': detector_name,
        'drifts': drifts,
        'labels': labels,
        'predictions': predictions,
        'n_req_labels': n_req_labels,
        'runtime': runtime,
        'peak_memory': peak_memory,
        'mean_memory': mean_memory,
        'metrics': metrics,
        'drift_responses': drift_responses
    }


if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("Usage: python run_config_detectors.py <Dataset> <ConfigPath> <RecentSamplesSize> <TrainSamples> <Accuracy> <Runtime> <ReqLabels>")
        sys.exit(1)
    
    dataset_expr = sys.argv[1]
    config_path = sys.argv[2]
    recent_samples_size = int(sys.argv[3])
    n_training_samples = int(sys.argv[4])
    benchmark_accuracy = bool(eval(sys.argv[5]))
    benchmark_runtime = bool(eval(sys.argv[6]))
    benchmark_reqlabels = bool(eval(sys.argv[7]))
    
    config = load_config(config_path)
    
    dataset_string, stream = parse_expression(dataset_expr)
    
    classifier_string = "HoeffdingTreeClassifier"
    classifier_path = (os.getcwd() + "/model/" + classifier_string + "/" +
                       classifier_string + "_" + dataset_string + ".pkl")
    
    detectors_config = config.get('detectors', [])
    
    print(f"=" * 80)
    print(f"Running {len(detectors_config)} detectors from config: {config_path}")
    print(f"Dataset: {dataset_string}")
    print(f"Recent Samples Size: {recent_samples_size}")
    print(f"Training Samples: {n_training_samples}")
    print(f"=" * 80)
    
    results = []
    for detector_config in detectors_config:
        detector_class_path = detector_config['class']
        detector_name = detector_class_path.rsplit('.', 1)[-1]
        
        print(f"\n--- Running {detector_name} ---")
        print(f"Parameters: {detector_config.get('params', {})}")
        
        try:
            detector = instantiate_detector(detector_config, recent_samples_size)
            
            if detector_name == "CDLEEDS":
                sys.setrecursionlimit(10000)
            
            _, stream = parse_expression(dataset_expr)
            
            result = run_detector(detector, detector_name, stream, 
                                  n_training_samples, classifier_path)
            results.append(result)
            
            print(f"General Info: detected drift points: {result['drifts']}")
            
            true_false_predictions = [int(l == p) for l, p in 
                                      zip(result['labels'], result['predictions'])]
            print(f"General Info: True/False Predictions: {true_false_predictions}")
            
            if benchmark_accuracy:
                print(f"ACCURACY: {result['metrics'].accuracy:.2f}")
            if benchmark_runtime:
                print(f"RUNTIME: {result['runtime']:.0f}")
            if benchmark_reqlabels:
                print(f"REQLABELS: {result['metrics'].portion_req_labels:.2f}")
            
            print(f"OO-Info: runtime: {result['runtime']:.0f}")
            print(f"OO-Info: peak_memory: {result['peak_memory']}")
            print(f"OO-Info: mean_memory: {result['mean_memory']}")
            print(f"OO-Info: accuracy: {result['metrics'].accuracy:.2f}")
            print(f"OO-Info: portion_req_label: {result['metrics'].portion_req_labels:.2f}")
            print(f"OO-Info: lpd: {result['metrics'].lpd:.2f}")
            
        except Exception as e:
            print(f"Error running {detector_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print(f"Summary: Ran {len(results)} detectors successfully")
    print(f"{'=' * 80}")
    
    # Compare drift responses across detectors
    if len(results) >= 2:
        print(f"\n{'=' * 80}")
        print("Drift Response Comparison")
        print(f"{'=' * 80}")
        
        # Get all drift response lists
        detector_names = [r['detector_name'] for r in results]
        drift_responses_list = [r['drift_responses'] for r in results]
        
        # Check lengths
        lengths = [len(dr) for dr in drift_responses_list]
        print(f"\nResponse list lengths: {dict(zip(detector_names, lengths))}")
        
        # Pairwise comparison
        print(f"\nPairwise Differences:")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                name_i = detector_names[i]
                name_j = detector_names[j]
                resp_i = drift_responses_list[i]
                resp_j = drift_responses_list[j]
                
                # Compare up to the minimum length
                min_len = min(len(resp_i), len(resp_j))
                differences = sum(1 for k in range(min_len) if resp_i[k] != resp_j[k])
                agreement = min_len - differences
                agreement_pct = (agreement / min_len * 100) if min_len > 0 else 0
                
                print(f"  {name_i} vs {name_j}:")
                print(f"    - Compared samples: {min_len}")
                print(f"    - Differences: {differences}")
                print(f"    - Agreement: {agreement} ({agreement_pct:.2f}%)")
                
                # Show where differences occur (first 10)
                diff_indices = [k for k in range(min_len) if resp_i[k] != resp_j[k]][:10]
                if diff_indices:
                    print(f"    - First diff indices (up to 10): {diff_indices}")
        
        # Summary: samples where all detectors agree vs disagree
        min_len = min(lengths)
        all_agree = 0
        any_disagree = 0
        for k in range(min_len):
            responses_at_k = [drift_responses_list[i][k] for i in range(len(results))]
            if all(r == responses_at_k[0] for r in responses_at_k):
                all_agree += 1
            else:
                any_disagree += 1
        
        print(f"\nOverall Agreement (across all {len(results)} detectors):")
        print(f"  - Samples where ALL agree: {all_agree} ({all_agree/min_len*100:.2f}%)")
        print(f"  - Samples where ANY disagree: {any_disagree} ({any_disagree/min_len*100:.2f}%)")
        
        # Calculate "best possible" accuracy - taking the best prediction at each sample
        print(f"\n--- Best Possible Accuracy (Oracle) ---")
        all_labels = [r['labels'] for r in results]
        all_predictions = [r['predictions'] for r in results]
        
        # Verify all have same labels (sanity check)
        min_samples = min(len(l) for l in all_labels)
        
        # For each sample, check if ANY detector got it right
        best_correct = 0
        for k in range(min_samples):
            label = all_labels[0][k]  # Labels should be the same across all
            # Check if any detector predicted correctly at this sample
            any_correct = any(all_predictions[i][k] == label for i in range(len(results)))
            if any_correct:
                best_correct += 1
        
        best_accuracy = best_correct / min_samples if min_samples > 0 else 0
        
        # Also show individual accuracies for comparison
        print(f"Individual detector accuracies:")
        for r in results:
            individual_correct = sum(1 for l, p in zip(r['labels'], r['predictions']) if l == p)
            individual_acc = individual_correct / len(r['labels']) if r['labels'] else 0
            print(f"  - {r['detector_name']}: {individual_acc:.4f} ({individual_correct}/{len(r['labels'])})")
        
        print(f"\nBest possible (oracle) accuracy: {best_accuracy:.4f} ({best_correct}/{min_samples})")
        print(f"  (Taking the correct prediction whenever ANY detector is right)")
        print(f"{'=' * 80}")
