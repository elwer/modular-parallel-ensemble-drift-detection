import argparse
import json
from pathlib import Path


PATTERNS = [
    ("SineClusters", 200),
    ("WaveformDrift2", 400),
    ("SineClusters", 500),
    ("WaveformDrift2", 750),
    ("SineClusters", 1000),
    ("WaveformDrift2", 1250),
    ("SineClusters", 1500),
    ("WaveformDrift2", 2000),
    ("SineClusters", 2500),
    ("WaveformDrift2", 3000),
]


def build_configs(output_dir, repeats, base_seed):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generators = []
    frequencies = []
    seeds = []
    for repeat in range(repeats):
        for index, (generator, frequency) in enumerate(PATTERNS):
            generators.append(generator)
            frequencies.append(frequency)
            seeds.append(base_seed + repeat * 10000 + index)

    profiles = [
        {
            "name": f"{generator.lower()}_{frequency}",
            "generator_filter": generator,
            "drift_freq_min": frequency,
            "drift_freq_max": frequency,
        }
        for generator, frequency in PATTERNS
    ]
    pattern_folds = [(2 * i, 2 * i + 1) for i in range(5)]
    fold = 0
    for eval_repeat in range(repeats):
        for held_out_patterns in pattern_folds:
            offset = eval_repeat * len(PATTERNS)
            eval_indices = [offset + index for index in held_out_patterns]
            eval_set = set(eval_indices)
            train_indices = [i for i in range(len(generators)) if i not in eval_set]
            config = {
                "fold": fold,
                "n_streams": len(generators),
                "generators": generators,
                "drift_frequencies": frequencies,
                "stream_seeds": seeds,
                "train_indices": train_indices,
                "eval_indices": eval_indices,
                "profiles": profiles,
            }
            (output / f"fold_{fold}.json").write_text(json.dumps(config, indent=2))
            fold += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    build_configs(args.output_dir, args.repeats, args.base_seed)


if __name__ == "__main__":
    main()
