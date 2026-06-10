#!/usr/bin/env python3
"""
Generate ensemble scalability YAML configs for ``main_synthetic.py``.

Creates ``scalability_configs/seed{N}/seed{N}_{K}.yaml`` for
N in 1..--n-seeds and K in 2,4,8,16,32,64,128, where each
``seed{N}_{K}.yaml`` lists the first ``K`` detectors of a randomly
generated 128-detector pool for that seed.

Candidates: BNDM, CSDDM, D3, IBDD, OCDD, SPLL, UDetect.

Pool construction guarantees that the prefix of size:
    8   contains each candidate DD at least 1 time
    16  contains each candidate DD at least 2 times
    32  contains each candidate DD at least 4 times
    64  contains each candidate DD at least 8 times
    128 contains each candidate DD at least 16 times

Order within each prefix-extension block is randomized, and each
detector's hyperparameters are sampled at random from the same
parameter spaces used by the Optuna single-DD optimization
(``optimization/single_dd_optimize_optuna.py``).

Usage:
    python generate_scalability_configs.py [--n-seeds 10] \
        [--out-dir scalability_configs] [--master-seed 0]
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml


CANDIDATES: List[str] = ["BNDM", "CSDDM", "D3", "IBDD", "OCDD", "SPLL", "UDetect"]

CLASS_PATH: Dict[str, str] = {
    "BNDM": "detectors.bndm.BNDM",
    "CSDDM": "detectors.csddm.CSDDM",
    "D3": "detectors.d3.D3",
    "IBDD": "detectors.ibdd.IBDD",
    "OCDD": "detectors.ocdd.OCDD",
    "SPLL": "detectors.spll.SPLL",
    "UDetect": "detectors.udetect.UDetect",
}

ENSEMBLE_SIZES: List[int] = [2, 4, 8, 16, 32, 64, 128]
# Minimum count per candidate required at each prefix length (None = no constraint).
MIN_PER_DD_AT_PREFIX: Dict[int, int] = {2: 0, 4: 0, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16}


# ---------------------------------------------------------------------------
# Random parameter samplers (mirroring single_dd_optimize_optuna.py ranges).
# ---------------------------------------------------------------------------

def _sample_bndm(rng: random.Random) -> Dict:
    return {
        "n_samples": rng.randint(50, 500),
        "const": rng.uniform(0.1, 10.0),
        "threshold": rng.uniform(0.1, 0.9),
        "max_depth": rng.randint(1, 10),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_csddm(rng: random.Random) -> Dict:
    return {
        "n_samples": rng.randint(50, 500),
        "feature_proportion": rng.uniform(0.1, 1.0),
        "n_clusters": rng.randint(2, 30),
        "confidence": rng.choice([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_d3(rng: random.Random) -> Dict:
    return {
        "n_reference_samples": rng.randint(50, 5000),
        "recent_samples_proportion": rng.uniform(0.05, 0.5),
        "threshold": rng.uniform(0.1, 0.9),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_ibdd(rng: random.Random) -> Dict:
    return {
        "n_samples": rng.randint(100, 2000),
        "n_consecutive_deviations": rng.randint(1, 20),
        "n_permutations": rng.randint(100, 1000),
        "update_interval": rng.randint(10, 100),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_ocdd(rng: random.Random) -> Dict:
    return {
        "n_samples": rng.randint(50, 500),
        "threshold": rng.uniform(0.1, 0.9),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_spll(rng: random.Random) -> Dict:
    return {
        "n_samples": rng.randint(100, 1000),
        "n_clusters": rng.randint(2, 20),
        "threshold": rng.uniform(0.1, 5.0),
        "recent_samples_size": rng.randint(50, 5000),
    }


def _sample_udetect(rng: random.Random) -> Dict:
    return {
        "n_windows": rng.randint(5, 30),
        "n_samples": rng.randint(20, 200),
        "disjoint_training_windows": rng.choice([True, False]),
        "recent_samples_size": rng.randint(50, 5000),
    }


SAMPLERS: Dict[str, Callable[[random.Random], Dict]] = {
    "BNDM": _sample_bndm,
    "CSDDM": _sample_csddm,
    "D3": _sample_d3,
    "IBDD": _sample_ibdd,
    "OCDD": _sample_ocdd,
    "SPLL": _sample_spll,
    "UDetect": _sample_udetect,
}


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _build_chunk(rng: random.Random,
                 chunk_size: int,
                 current_counts: Dict[str, int],
                 target_min_after_chunk: int) -> List[str]:
    """Build a chunk of detector names that, appended to the current pool,
    raises every candidate's count to at least ``target_min_after_chunk``.
    Remaining slots are filled uniformly at random across candidates."""
    needs: List[str] = []
    for dd in CANDIDATES:
        deficit = max(0, target_min_after_chunk - current_counts.get(dd, 0))
        needs.extend([dd] * deficit)

    if len(needs) > chunk_size:
        raise ValueError(
            f"Cannot satisfy minimum {target_min_after_chunk} per DD inside "
            f"chunk of size {chunk_size}: needs {len(needs)} mandatory slots."
        )

    fillers = [rng.choice(CANDIDATES) for _ in range(chunk_size - len(needs))]
    chunk = needs + fillers
    rng.shuffle(chunk)
    return chunk


def build_pool(rng: random.Random) -> List[str]:
    """Construct an ordered 128-element list of detector names that satisfies
    the prefix-coverage constraints for sizes 8/16/32/64/128."""
    pool: List[str] = []
    counts: Dict[str, int] = {dd: 0 for dd in CANDIDATES}

    # Each tuple: (chunk_size, target_min_after_chunk).
    schedule: List[Tuple[int, int]] = [
        (8, MIN_PER_DD_AT_PREFIX[8]),
        (8, MIN_PER_DD_AT_PREFIX[16]),
        (16, MIN_PER_DD_AT_PREFIX[32]),
        (32, MIN_PER_DD_AT_PREFIX[64]),
        (64, MIN_PER_DD_AT_PREFIX[128]),
    ]

    for chunk_size, target_min in schedule:
        chunk = _build_chunk(rng, chunk_size, counts, target_min)
        for dd in chunk:
            counts[dd] += 1
        pool.extend(chunk)

    assert len(pool) == 128, len(pool)
    _validate_pool(pool)
    return pool


def _validate_pool(pool: List[str]) -> None:
    for size, min_each in MIN_PER_DD_AT_PREFIX.items():
        if min_each <= 0:
            continue
        prefix_counts = {dd: 0 for dd in CANDIDATES}
        for dd in pool[:size]:
            prefix_counts[dd] += 1
        for dd in CANDIDATES:
            if prefix_counts[dd] < min_each:
                raise AssertionError(
                    f"Prefix size {size}: {dd} appears {prefix_counts[dd]} "
                    f"times, expected >= {min_each}"
                )


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

def _materialize_pool(pool_names: List[str], rng: random.Random) -> List[Dict]:
    """Sample params for each detector slot and produce list-of-dict pool."""
    entries: List[Dict] = []
    for name in pool_names:
        params = SAMPLERS[name](rng)
        entries.append({"class": CLASS_PATH[name], "params": params})
    return entries


def _build_config_doc(seed: int, size: int, entries: List[Dict]) -> Dict:
    return {
        "seed": seed,
        "ensemble_size": size,
        "detectors": entries[:size],
    }


def write_seed_configs(seed: int, out_root: Path) -> None:
    rng = random.Random(seed)

    pool_names = build_pool(rng)
    pool_entries = _materialize_pool(pool_names, rng)

    seed_dir = out_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for size in ENSEMBLE_SIZES:
        doc = _build_config_doc(seed, size, pool_entries)
        out_path = seed_dir / f"seed{seed}_{size}.yaml"
        with out_path.open("w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Number of seeds to generate (default 10).")
    p.add_argument("--out-dir", default="scalability_configs",
                   help="Output root directory (default scalability_configs).")
    p.add_argument("--master-seed", type=int, default=0,
                   help="Master seed offset; seeds used are "
                        "master_seed+1 .. master_seed+n_seeds.")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.n_seeds + 1):
        seed = args.master_seed + i
        print(f"\n=== seed {seed} ===")
        write_seed_configs(seed, out_root)

    print(f"\nDone. Wrote configs under {out_root.resolve()}")


if __name__ == "__main__":
    main()
