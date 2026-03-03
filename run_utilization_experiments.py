#!/usr/bin/env python3
"""
Run CPU utilization experiments for drift detectors.

Reads detector configurations from CSV files in utilization_experiments/lukats_configs/<dataset>/,
runs each unique configuration, and records CPU utilization every N seconds.

Usage:
    python run_utilization_experiments.py <Cores> <Accuracy> <Runtime> <ReqLabels> <Dataset> \
        <TrainSamples> <Classifier> <extra_param_key> <extra_param_value> ...

Example:
    python run_utilization_experiments.py 4 True True False Electricity 1600 \
        HoeffdingTreeClassifier recent_samples_size 1000
"""

import os
import sys
import ast
import csv
import time
import warnings

warnings.filterwarnings("ignore")

from optimization.classifiers import *
from datasets import *
from detectors import *
import detectors.base

csv.field_size_limit(sys.maxsize)

DETECTORS = ["bndm", "csddm", "d3", "ibdd", "ocdd", "spll", "udetect"]
DETECTORS = ["ocdd", "spll", "udetect", "ibdd"]
DETECTORS = ["ibdd"]
DETECTOR_CLASSES = {
    #"bndm": BNDM,
    #"csddm": CSDDM,
    #"d3": D3,
    #"ibdd": IBDD,
    #"ocdd": OCDD,
    #"spll": SPLL,
    #"udetect": UDetect,
    "ibdd": IBDD,
}

SKIP_COLUMNS = {"seed", "lpd", "acc", "f1", "drifts"}

SAMPLING_INTERVAL = 2.0  # seconds


def should_skip_column(col_name):
    """Check if a column should be skipped based on prefix matching."""
    col_lower = col_name.lower().strip()
    for skip in SKIP_COLUMNS:
        if col_lower == skip or col_lower.startswith(skip + " ") or col_lower.startswith(skip + "("):
            return True
    return False


def parse_value(value_str):
    """Parse a string value into the appropriate Python type."""
    value_str = value_str.strip()
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        pass
    return value_str


def load_unique_configs(csv_path):
    """Load unique configurations from a CSV file, skipping irrelevant columns."""
    configs = []
    seen = set()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if not should_skip_column(c)]

        for row in reader:
            param_dict = {}
            for col in param_cols:
                param_dict[col] = parse_value(row[col])

            # Create a hashable key for deduplication
            key = tuple(sorted((k, str(v)) for k, v in param_dict.items()))
            if key not in seen:
                seen.add(key)
                configs.append(param_dict)

    return configs


def parse_expression(expr_str):
    """Parse a class name or expression and return (name, instance)."""
    try:
        node = ast.parse(expr_str, mode="eval").body
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



def write_config_file(filepath, configs_with_ids):
    """Write configurations with their IDs to a config file."""
    if not configs_with_ids:
        return
    param_keys = list(configs_with_ids[0][1].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + param_keys)
        for config_id, params in configs_with_ids:
            writer.writerow([config_id] + [params[k] for k in param_keys])


def run_experiment(detector_name, config, extra_params, dataset_expr,
                   n_training_samples, classifier_path):
    """Instantiate a detector with the given config and run it on the dataset."""
    detector_cls = DETECTOR_CLASSES[detector_name]

    # Merge config with extra params (extra params like recent_samples_size)
    params = dict(config)
    params.update(extra_params)

    detector = detector_cls(**params)

    # Re-parse dataset for each run (streams are consumed)
    _, stream = parse_expression(dataset_expr)

    # Run the detector
    result = detector.run_stream(stream, n_training_samples, classifier_path)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python run_utilization_experiments.py <Cores> <Accuracy> <Runtime> <ReqLabels> "
              "<Dataset> <TrainSamples> <Classifier> [extra_param_key extra_param_value ...]")
        sys.exit(1)

    num_cores = int(sys.argv[1])
    benchmark_accuracy = bool(eval(sys.argv[2]))
    benchmark_runtime = bool(eval(sys.argv[3]))
    benchmark_reqlabels = bool(eval(sys.argv[4]))
    dataset_expr = sys.argv[5]
    n_training_samples = int(sys.argv[6])
    classifier_expr = sys.argv[7]

    # Parse extra key-value params (e.g. recent_samples_size 1000)
    extra_params = {}
    i = 8
    while i + 1 < len(sys.argv):
        key = sys.argv[i]
        val = parse_value(sys.argv[i + 1])
        extra_params[key] = val
        i += 2

    dataset_string, _ = parse_expression(dataset_expr)
    classifier_string, _ = parse_expression(classifier_expr)
    classifier_path = os.path.join(
        os.getcwd(), "model", classifier_string,
        f"{classifier_string}_{dataset_string}.pkl"
    )

    config_dir = os.path.join("utilization_experiments", "lukats_configs")
    output_dir = os.path.join("utilization_experiments", f"{num_cores}_cores")
    os.makedirs(output_dir, exist_ok=True)

    # Find config subdirectory - look for any available subdirectory
    # (configs may be stored under a different dataset name)
    available_dirs = [d for d in os.listdir(config_dir)
                      if os.path.isdir(os.path.join(config_dir, d))]
    if not available_dirs:
        print(f"Error: No config subdirectories found in {config_dir}")
        sys.exit(1)

    # Use the first available directory (or match dataset name if possible)
    config_subdir = None
    for d in available_dirs:
        if d.lower() == dataset_string.lower():
            config_subdir = d
            break
    if config_subdir is None:
        config_subdir = available_dirs[0]
        print(f"Warning: No config directory matching '{dataset_string}', "
              f"using '{config_subdir}' instead.")

    config_base = os.path.join(config_dir, config_subdir)

    from jumper_extension.core.service import build_perfmonitor_service

    print(f"{'=' * 80}")
    print(f"CPU Utilization Experiment")
    print(f"  Cores: {num_cores}")
    print(f"  Dataset: {dataset_string}")
    print(f"  Classifier: {classifier_string}")
    print(f"  Training samples: {n_training_samples}")
    print(f"  Extra params: {extra_params}")
    print(f"  Config directory: {config_base}")
    print(f"  Output directory: {output_dir}")
    print(f"  Sampling interval: {SAMPLING_INTERVAL}s")
    print(f"{'=' * 80}")

    for dd_name in DETECTORS:
        csv_path = os.path.join(config_base, f"{dd_name}.csv")
        if not os.path.exists(csv_path):
            print(f"\nSkipping {dd_name}: config file not found at {csv_path}")
            continue

        configs = load_unique_configs(csv_path)
        print(f"\n{'=' * 80}")
        print(f"Detector: {dd_name.upper()} — {len(configs)} unique configurations")
        print(f"{'=' * 80}")

        configs_with_ids = []
        for config_id, config in enumerate(configs):
            configs_with_ids.append((config_id, config))

        # Write config file
        config_file = os.path.join(output_dir,
                                   f"{dataset_string}_{dd_name}.config")
        write_config_file(config_file, configs_with_ids)
        print(f"Config file written: {config_file}")

        for config_id, config in configs_with_ids:
            print(f"\n  --- Config ID {config_id}: {config} ---")

            result_file = os.path.join(
                output_dir, f"{dataset_string}_{dd_name}_{config_id}.csv"
            )

            service = build_perfmonitor_service()
            service.start_monitoring(SAMPLING_INTERVAL)

            try:
                with service.monitored():
                    result = run_experiment(
                        dd_name, config, extra_params,
                        dataset_expr, n_training_samples, classifier_path
                    )
                drifts, labels, predictions, n_req_labels, runtime, peak_mem, mean_mem = result
                print(f"  Runtime: {runtime:.1f}s, Drifts: {len(drifts)}, "
                      f"Peak mem: {peak_mem:.1f}MB")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                service.export_perfdata(file=result_file, level="slurm")
                service.stop_monitoring()

            print(f"  CPU utilization written: {result_file}")

    print(f"\n{'=' * 80}")
    print("All experiments completed.")
    print(f"{'=' * 80}")
