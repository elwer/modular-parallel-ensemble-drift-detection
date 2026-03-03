#!/usr/bin/env python3
"""
Test script that runs each drift detector defined in ewdd.config separately
and prints the accuracy and runtime for each.

Usage:
    python test_ewdd_detectors.py <recent_samples_size>
    
Example:
    python test_ewdd_detectors.py 2424
"""
import argparse

import os
import sys
import subprocess
import re
from pathlib import Path

# Use the same yaml import as ewdd.py
try:
    import yaml
except ImportError:
    # Fallback: parse YAML manually for simple configs
    yaml = None

# Path to ewdd config
CONFIG_PATH = Path(__file__).parent / "detectors" / "ewdd" / "configs" / "ewdd.config"


def get_base_cmd(recent_samples_size: int) -> list:
    """Build base command with recent_samples_size parameter."""
    return [
        sys.executable, "main.py",
        "True", "True", "False",  # Accuracy, Runtime, ReqLabels
        "Electricity",
        "1600",
        "HoeffdingTreeClassifier"
    ]


def load_detectors_from_config(config_path: str) -> list:
    """Load detector configurations from ewdd.config."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    if yaml is not None:
        config = yaml.safe_load(content)
        detectors = []
        for detector_config in config.get('detectors', []):
            detector_class = detector_config['class']
            params = detector_config.get('params', {})
            detectors.append({
                'class': detector_class,
                'params': params
            })
        return detectors
    else:
        # Simple manual parsing for YAML without the library
        detectors = []
        lines = content.split('\n')
        current_detector = None
        in_params = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                continue
            
            if '- class:' in line:
                if current_detector:
                    detectors.append(current_detector)
                class_name = stripped.split('class:')[1].strip()
                current_detector = {'class': class_name, 'params': {}}
                in_params = False
            elif 'params:' in stripped and current_detector:
                in_params = True
            elif in_params and current_detector and ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                current_detector['params'][key] = value
        
        if current_detector:
            detectors.append(current_detector)
        
        return detectors


def build_detector_command(detector_config: dict) -> list:
    """Build command line arguments for a detector."""
    # Extract class name (e.g., 'detectors.ibdd.IBDD' -> 'IBDD')
    class_path = detector_config['class']
    class_name = class_path.split('.')[-1]
    
    # Build parameter arguments
    params = detector_config.get('params', {})
    param_args = []
    for key, value in params.items():
        param_args.append(str(key))
        param_args.append(str(value))
    
    return [class_name] + param_args


def parse_output(output: str) -> dict:
    """Parse the output to extract accuracy and runtime."""
    result = {
        'accuracy': None,
        'runtime': None
    }
    
    for line in output.split('\n'):
        if line.startswith('ACCURACY:'):
            match = re.search(r'ACCURACY:\s*([\d.]+)', line)
            if match:
                result['accuracy'] = float(match.group(1))
        elif line.startswith('RUNTIME:'):
            match = re.search(r'RUNTIME:\s*([\d.]+)', line)
            if match:
                result['runtime'] = float(match.group(1))
    
    return result


def run_detector(detector_config: dict, recent_samples_size: int) -> dict:
    """Run a single detector and return results."""
    detector_args = build_detector_command(detector_config)
    # Add recent_samples_size to detector params
    detector_args.extend(["recent_samples_size", str(recent_samples_size)])
    cmd = get_base_cmd(recent_samples_size) + detector_args
    
    class_name = detector_config['class'].split('.')[-1]
    
    print(f"\n{'='*60}")
    print(f"Running: {class_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=600  # 10 minute timeout
        )
        
        output = result.stdout
        if result.returncode != 0:
            print(f"Error running {class_name}:")
            print(result.stderr)
            return {
                'detector': class_name,
                'accuracy': None,
                'runtime': None,
                'error': result.stderr
            }
        
        parsed = parse_output(output)
        return {
            'detector': class_name,
            'accuracy': parsed['accuracy'],
            'runtime': parsed['runtime'],
            'error': None
        }
        
    except subprocess.TimeoutExpired:
        return {
            'detector': class_name,
            'accuracy': None,
            'runtime': None,
            'error': 'Timeout (>600s)'
        }
    except Exception as e:
        return {
            'detector': class_name,
            'accuracy': None,
            'runtime': None,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Test each drift detector from ewdd.config separately')
    parser.add_argument('recent_samples_size', type=int, help='Recent samples size parameter for detectors')
    args = parser.parse_args()
    
    recent_samples_size = args.recent_samples_size
    
    print("=" * 60)
    print("EWDD Detector Individual Test")
    print("=" * 60)
    print(f"Config: {CONFIG_PATH}")
    print(f"Dataset: Electricity")
    print(f"Training samples: 1600")
    print(f"Classifier: HoeffdingTreeClassifier")
    print(f"Recent samples size: {recent_samples_size}")
    
    # Load detectors from config
    detectors = load_detectors_from_config(CONFIG_PATH)
    print(f"\nFound {len(detectors)} detectors in config")
    
    # Run each detector
    results = []
    for detector_config in detectors:
        result = run_detector(detector_config, recent_samples_size)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Detector':<15} {'Accuracy':<12} {'Runtime (ms)':<15} {'Status'}")
    print("-" * 60)
    
    for r in results:
        accuracy_str = f"{r['accuracy']:.2f}" if r['accuracy'] is not None else "N/A"
        runtime_str = f"{r['runtime']:.0f}" if r['runtime'] is not None else "N/A"
        status = "OK" if r['error'] is None else f"ERROR: {r['error'][:30]}"
        print(f"{r['detector']:<15} {accuracy_str:<12} {runtime_str:<15} {status}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
