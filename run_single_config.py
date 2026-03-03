#!/usr/bin/env python3
"""
Run a single detector configuration and record the runtime.

Usage:
    python run_single_config.py <detector> <config_id> <cores> <run_id> <dataset_expr> \
        <train_samples> <classifier_expr> [extra_key extra_val ...]

Output is written to: runtime_experiments/<detector>_<config_id>/<cores>_cores/run_<run_id>.csv
The CSV contains columns: detector, config_id, cores, run_id, runtime_s
"""

import os
import sys
import ast
import csv
import warnings

warnings.filterwarnings("ignore")

from optimization.classifiers import *
from datasets import *
from detectors import *
import detectors.base

csv.field_size_limit(sys.maxsize)

DETECTOR_CLASSES = {
    "bndm": BNDM,
    "csddm": CSDDM,
    "d3": D3,
    "ibdd": IBDD,
    "ocdd": OCDD,
    "spll": SPLL,
    "udetect": UDetect,
}

SKIP_COLUMNS = {"seed", "lpd", "acc", "f1", "drifts"}


def should_skip_column(col_name):
    col_lower = col_name.lower().strip()
    for skip in SKIP_COLUMNS:
        if col_lower == skip or col_lower.startswith(skip + " ") or col_lower.startswith(skip + "("):
            return True
    return False


def parse_value(value_str):
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
    configs = []
    seen = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if not should_skip_column(c)]
        for row in reader:
            param_dict = {}
            for col in param_cols:
                param_dict[col] = parse_value(row[col])
            key = tuple(sorted((k, str(v)) for k, v in param_dict.items()))
            if key not in seen:
                seen.add(key)
                configs.append(param_dict)
    return configs


def parse_expression(expr_str):
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


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python run_single_config.py <detector> <config_id> <cores> <run_id> "
              "<dataset_expr> <train_samples> <classifier_expr> [extra_key extra_val ...]")
        sys.exit(1)

    detector_name = sys.argv[1].lower()
    config_id = int(sys.argv[2])
    num_cores = int(sys.argv[3])
    run_id = int(sys.argv[4])
    dataset_expr = sys.argv[5]
    n_training_samples = int(sys.argv[6])
    classifier_expr = sys.argv[7]

    extra_params = {}
    i = 8
    while i + 1 < len(sys.argv):
        extra_params[sys.argv[i]] = parse_value(sys.argv[i + 1])
        i += 2

    dataset_string, _ = parse_expression(dataset_expr)
    classifier_string, _ = parse_expression(classifier_expr)
    classifier_path = os.path.join(
        os.getcwd(), "model", classifier_string,
        f"{classifier_string}_{dataset_string}.pkl"
    )

    # Load configs from the lukats_configs directory
    config_dir = os.path.join("utilization_experiments", "lukats_configs")
    available_dirs = [d for d in os.listdir(config_dir)
                      if os.path.isdir(os.path.join(config_dir, d))]
    config_subdir = None
    for d in available_dirs:
        if d.lower() == dataset_string.lower():
            config_subdir = d
            break
    if config_subdir is None:
        config_subdir = available_dirs[0]

    csv_path = os.path.join(config_dir, config_subdir, f"{detector_name}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Config file not found: {csv_path}")
        sys.exit(1)

    configs = load_unique_configs(csv_path)
    if config_id >= len(configs):
        print(f"Error: config_id {config_id} out of range (only {len(configs)} configs)")
        sys.exit(1)

    config = configs[config_id]
    params = dict(config)
    params.update(extra_params)

    print(f"Detector: {detector_name.upper()}, Config ID: {config_id}, "
          f"Cores: {num_cores}, Run: {run_id}")
    print(f"Params: {params}")

    detector_cls = DETECTOR_CLASSES[detector_name]
    detector = detector_cls(**params)

    _, stream = parse_expression(dataset_expr)

    import time
    t0 = time.perf_counter()
    result = detector.run_stream(stream, n_training_samples, classifier_path)
    runtime = time.perf_counter() - t0

    drifts, labels, predictions, n_req_labels, rt_reported, peak_mem, mean_mem = result
    print(f"Runtime: {runtime:.2f}s (reported: {rt_reported:.2f}s), Drifts: {len(drifts)}")

    # Write result
    output_dir = os.path.join("runtime_experiments",
                              f"{detector_name}_{config_id}",
                              f"{num_cores}_cores")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"run_{run_id}.csv")

    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "config_id", "cores", "run_id", "runtime_s"])
        writer.writerow([detector_name, config_id, num_cores, run_id, f"{runtime:.4f}"])

    print(f"Result written: {result_file}")
