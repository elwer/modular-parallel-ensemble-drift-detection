import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("expert", "generalist"), required=True)
    parser.add_argument("--detector-type", required=True)
    parser.add_argument("--fold-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optuna-storage", required=True)
    parser.add_argument("--stream-length", type=int, default=8000)
    parser.add_argument("--base-stream-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--expert-trials", type=int, default=100)
    parser.add_argument("--deployment-trials", type=int, default=100)
    parser.add_argument("--per-trial-timeout", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=7)
    args = parser.parse_args()

    config = json.loads(Path(args.fold_config).read_text())
    common = [
        "--optuna-storage", args.optuna_storage,
        "--n-streams", str(config["n_streams"]),
        "--base-stream-seed", str(args.base_stream_seed),
        "--stream-seeds", ",".join(map(str, config["stream_seeds"])),
        "--drift-frequencies", ",".join(map(str, config["drift_frequencies"])),
        "--stream-length", str(args.stream_length),
        "--eval-stream-indices", ",".join(map(str, config["eval_indices"])),
        "--generators", ",".join(config["generators"]),
        "--profiles", json.dumps(config["profiles"], separators=(",", ":")),
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
    ]
    if args.mode == "expert":
        script = Path(__file__).with_name("expert_optimize.py")
        command = [
            sys.executable, str(script), *common,
            "--detector-type", args.detector_type,
            "--n-trials-expert", str(args.expert_trials),
            "--n-trials-deployment", str(args.deployment_trials),
            "--per-trial-timeout", str(args.per_trial_timeout),
            "--n-jobs", str(args.n_jobs),
        ]
    else:
        script = Path(__file__).with_name("generalist_optimize.py")
        command = [
            sys.executable, str(script), *common,
            "--detector-type", args.detector_type,
            "--n-trials", str(args.expert_trials * len(config["profiles"]) + args.deployment_trials),
            "--per-trial-timeout", str(args.per_trial_timeout),
        ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
