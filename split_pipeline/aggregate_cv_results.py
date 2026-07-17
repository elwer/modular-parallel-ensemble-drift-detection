import argparse
import csv
import json
import statistics
from pathlib import Path


DETECTORS = ("BNDM", "CSDDM", "D3", "IBDD", "OCDD", "SPLL", "UDetect")


def read_metric(path, metric):
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or metric not in rows[0]:
        return None
    return float(rows[0][metric])


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    rows = []
    for fold_dir in sorted(root.glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        for detector in DETECTORS:
            expert_dir = fold_dir / detector / "expert"
            generalist_dir = fold_dir / detector / "generalist"
            rows.append({
                "fold": fold,
                "detector_type": detector,
                "expert_fixed_macro_f1": read_metric(
                    expert_dir / f"phase1_per_dd_{detector.lower()}.csv", "macro_f1"),
                "expert_optimized_macro_f1": read_metric(
                    expert_dir / f"phase2_per_dd_{detector.lower()}.csv", "macro_f1"),
                "generalist_macro_f1": read_metric(
                    generalist_dir / f"generalists_eval_{detector.lower()}.csv", "macro_f1"),
            })

    summaries = {}
    for detector in DETECTORS:
        detector_rows = [row for row in rows if row["detector_type"] == detector]
        summaries[detector] = {
            "expert_fixed": summarize([r["expert_fixed_macro_f1"] for r in detector_rows]),
            "expert_optimized": summarize([r["expert_optimized_macro_f1"] for r in detector_rows]),
            "generalist": summarize([r["generalist_macro_f1"] for r in detector_rows]),
        }
    summaries["all_detectors"] = {
        "expert_optimized": summarize([r["expert_optimized_macro_f1"] for r in rows]),
        "generalist": summarize([r["generalist_macro_f1"] for r in rows]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=2))

    print("detector       expert_mean  expert_std  generalist_mean  generalist_std")
    for detector in DETECTORS:
        expert = summaries[detector]["expert_optimized"]
        generalist = summaries[detector]["generalist"]
        print(f"{detector:<14} {expert['mean']!s:>11} {expert['std']!s:>11} "
              f"{generalist['mean']!s:>16} {generalist['std']!s:>15}")


if __name__ == "__main__":
    main()
