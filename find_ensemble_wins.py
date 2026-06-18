#!/usr/bin/env python3
"""
Search all synthetic-result .log files for configurations where the ensemble
beats every single member on the F1 score:

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2*TP / (2*TP + FP + FN)

Usage:
    python find_ensemble_wins.py [results_dir] [--min-margin M] [--top K]

Arguments:
    results_dir     Root directory containing <dataset>/<...>.log files
                    (default: synthetic_results).
    --min-margin M  Require ensemble F1 to exceed the best member's F1 by at
                    least M (default: 0, i.e. strictly higher is enough).
    --top K         Only print the K best runs by ensemble F1 (default: all).
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILENAME_RE = re.compile(
    r"seed(?P<seed>\d+)"
    r"(?:_det-(?P<detector_criterion>any|all|majority)"
    r"_ens-(?P<ensemble_criterion>any|all|majority)"
    r"_dw(?P<decision_window>\d+)"
    r"_sw(?P<suppression_window>\d+))?\.log$"
)
ENSEMBLE_RE = re.compile(
    r"^ENSEMBLE_N=(?P<N>\d+): "
    r"TP=(?P<TP>\d+) FP=(?P<FP>\d+) FN=(?P<FN>\d+) DELAY=(?P<delay>[\d.]+|nan)",
    re.MULTILINE,
)
MEMBER_RE = re.compile(
    r"^MEMBER_N=(?P<N>\d+) idx=(?P<idx>\d+) name=(?P<name>\S+): "
    r"TP=(?P<TP>\d+) FP=(?P<FP>\d+) FN=(?P<FN>\d+) DELAY=(?P<delay>[\d.]+|nan)",
    re.MULTILINE,
)


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    delay: float
    name: str = ""

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / denom) if denom > 0 else 0.0


@dataclass
class Run:
    log_path: Path
    dataset: str
    params: Dict[str, Optional[str]]
    n: int
    ensemble: Metrics
    members: List[Metrics] = field(default_factory=list)

    def describe(self) -> str:
        p = self.params
        parts = [f"dataset={self.dataset}", f"seed={p.get('seed')}"]
        for key, label in [("detector_criterion", "det"),
                           ("ensemble_criterion", "ens"),
                           ("decision_window", "dw"),
                           ("suppression_window", "sw")]:
            if p.get(key) is not None:
                parts.append(f"{label}={p[key]}")
        parts.append(f"N={self.n}")
        return " ".join(parts)


def parse_log(log_path: Path) -> List[Run]:
    m = FILENAME_RE.search(log_path.name)
    if m is None:
        print(f"[skip] unparsable file name: {log_path}", file=sys.stderr)
        return []
    params = m.groupdict()
    text = log_path.read_text(errors="replace")

    ensembles: Dict[int, Metrics] = {}
    for em in ENSEMBLE_RE.finditer(text):
        ensembles[int(em["N"])] = Metrics(
            tp=int(em["TP"]), fp=int(em["FP"]), fn=int(em["FN"]),
            delay=float(em["delay"]),
        )

    members: Dict[int, List[Metrics]] = {}
    for mm in MEMBER_RE.finditer(text):
        members.setdefault(int(mm["N"]), []).append(Metrics(
            tp=int(mm["TP"]), fp=int(mm["FP"]), fn=int(mm["FN"]),
            delay=float(mm["delay"]), name=mm["name"],
        ))

    runs = []
    for n, ens in sorted(ensembles.items()):
        runs.append(Run(
            log_path=log_path,
            dataset=log_path.parent.name,
            params=params,
            n=n,
            ensemble=ens,
            members=members.get(n, []),
        ))
    return runs


def ensemble_beats_all_members(run: Run, min_margin: float) -> bool:
    """True iff the ensemble F1 exceeds every member's F1 by > min_margin."""
    if not run.members:
        return False
    best_member_f1 = max(m.f1 for m in run.members)
    return run.ensemble.f1 > best_member_f1 + min_margin


def fmt_delay(delay: float) -> str:
    return f"{delay:.2f}" if not math.isnan(delay) else "nan"


def main(argv) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default="synthetic_results")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Minimum F1 advantage over the best member.")
    parser.add_argument("--top", type=int, default=None,
                        help="Only print the K best runs by F1 delta over the best member.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    log_files = sorted(results_dir.glob("*/*.log"))
    if not log_files:
        print(f"No .log files found under {results_dir}/", file=sys.stderr)
        return 1

    runs: List[Run] = []
    for log_path in log_files:
        runs.extend(parse_log(log_path))

    wins = [r for r in runs if ensemble_beats_all_members(r, args.min_margin)]

    print(f"Scanned {len(log_files)} log files, {len(runs)} (config, N) runs.")
    print(f"{len(wins)} runs where the ensemble F1 beats every member's F1"
          + (f" by > {args.min_margin}" if args.min_margin > 0 else "")
          + ".\n")

    # Largest F1 delta over the best member first.
    wins.sort(key=lambda r: -(r.ensemble.f1 - max(m.f1 for m in r.members)))
    if args.top is not None:
        wins = wins[:args.top]

    for run in wins:
        ens = run.ensemble
        best_member_f1 = max(m.f1 for m in run.members)
        print(f"WIN  {run.describe()}")
        print(f"     ensemble:  F1={ens.f1:.4f}  TP={ens.tp:<4d} FP={ens.fp:<4d} "
              f"FN={ens.fn:<4d} delay={fmt_delay(ens.delay)}")
        print(f"     margin over best member: +{ens.f1 - best_member_f1:.4f} F1")
        print(f"     members ({len(run.members)}):")
        for m in sorted(run.members, key=lambda m: -m.f1):
            print(f"       {m.name:<30s} F1={m.f1:.4f}  TP={m.tp:<4d} "
                  f"FP={m.fp:<4d} FN={m.fn:<4d} delay={fmt_delay(m.delay)}")
        print(f"     log: {run.log_path}\n")

    if not wins:
        print("No configuration found where the ensemble F1 exceeds all "
              "members. Try lowering --min-margin (or check the logs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
