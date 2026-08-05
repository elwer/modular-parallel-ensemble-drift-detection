#!/usr/bin/env python3
"""Phase 3: Jointly optimize 2-DD ensembles to beat best single DD.

Loads existing single-detector results from run_overnight_benchmark.py,
then uses Optuna to jointly optimize pairs of detectors with shared
voting criterion and decision window, maximizing macro F1 across all
frequencies.

Also evaluates a "cross-DD best expert per freq" using held-out eval F1
instead of train F1 for selection.
"""
import sys
import json
import time
import os
import itertools
import multiprocessing as mp
import optuna
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _suggest_detector_params, _instantiate,
    build_stream, MAX_WINDOW_FRACTION, _cap, _f1_from_counts
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections

# --- Config (must match run_overnight_benchmark.py) ---
GENERATOR = "SineClusters"
DRIFT_FREQS = [200, 500, 1000, 2000]
STREAM_LENGTH = 4000
SEED = 1337
TOLERANCES = [f // 10 for f in DRIFT_FREQS]
TRAIN_SEEDS = [42]
# Use different eval seeds for ensemble selection (avoid overfitting to eval seeds)
ENSEMBLE_TRAIN_SEEDS = [42, 43]  # for ensemble optimization
ENSEMBLE_EVAL_SEEDS = [45, 46]   # for final evaluation
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 15
N_ENSEMBLE_TRIALS = 30

# Pairs to try (focused on complementary strengths)
ENSEMBLE_PAIRS = [
    ("UDetect", "SPLL"),      # UDetect strong at 200,500; SPLL strong at 1000,2000
    ("UDetect", "CSDDM"),     # UDetect strong at 200,500; CSDDM strong at 1000
    ("UDetect", "BNDM"),      # UDetect + BNDM (decent at 1000)
    ("SPLL", "CSDDM"),        # Both strong at 1000
    ("UDetect", "OCDD"),      # UDetect + OCDD (OCDD decent at 500)
]


def _ensemble_worker(queue, slot_specs, ensemble_criterion, decision_window,
                     freq, stream_seed, s_idx, tolerance, suppression,
                     stream_length, generator):
    try:
        stream = build_stream(generator, freq, stream_length, stream_seed)
        known = list(stream.drifts)

        detectors = []
        names = []
        for i, (kind, params) in enumerate(slot_specs):
            n_samples_key = "n_samples" if "n_samples" in params else "n_reference_samples"
            recent_size = params.get(n_samples_key, 200)
            det = _instantiate(kind, params, seed=SEED + i + 1000 * s_idx,
                               recent_samples_size=recent_size)
            detectors.append(det)
            names.append(f"[{i:03d}]{kind}")

        _, raw_ensemble = run_ensemble(
            stream, detectors, names,
            detector_criterion="any",
            ensemble_criterion=ensemble_criterion,
            decision_window=decision_window,
        )
        dets = apply_suppression(raw_ensemble, suppression)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        queue.put(("ok", f1))
    except Exception as e:
        queue.put(("error", str(e)))


def run_ensemble_eval(slot_specs, ensemble_criterion, decision_window,
                      freq, stream_seed, s_idx):
    """Run ensemble on one stream with hard timeout, return F1."""
    tol = freq // 10
    supp = freq // 2
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_ensemble_worker,
                       args=(queue, slot_specs, ensemble_criterion, decision_window,
                             freq, stream_seed, s_idx, tol, supp,
                             STREAM_LENGTH, GENERATOR))
    proc.start()
    proc.join(PER_STREAM_TIMEOUT * 3)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return 0.0
    try:
        result = queue.get_nowait()
        return result[1] if result[0] == "ok" else 0.0
    except:
        return 0.0


def make_ensemble_objective(dd1, dd2):
    """Optuna objective: jointly optimize params for dd1+dd2 ensemble."""
    max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)

    def objective(trial):
        # Suggest params for both detectors
        params1 = _suggest_detector_params(trial, "d1_", dd1, max_window=max_window)
        params2 = _suggest_detector_params(trial, "d2_", dd2, max_window=max_window)

        # Suggest ensemble config
        criterion = trial.suggest_categorical("criterion", ["any", "majority", "all"])
        decision_window = trial.suggest_int("decision_window", 1, 5)

        f1s = []
        for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
            for s_idx, seed in enumerate(ENSEMBLE_TRAIN_SEEDS):
                slot_specs = [(dd1, params1), (dd2, params2)]
                f1 = run_ensemble_eval(slot_specs, criterion, decision_window,
                                       freq, seed, s_idx)
                f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def eval_ensemble_final(dd1, params1, dd2, params2, criterion, decision_window):
    """Evaluate ensemble on held-out eval seeds, per freq."""
    per_freq = {}
    for freq in DRIFT_FREQS:
        f1s = []
        for i, seed in enumerate(ENSEMBLE_EVAL_SEEDS):
            slot_specs = [(dd1, params1), (dd2, params2)]
            f1 = run_ensemble_eval(slot_specs, criterion, decision_window,
                                   freq, seed, 600 + i)
            f1s.append(f1)
        per_freq[freq] = sum(f1s) / len(f1s)
    macro = sum(per_freq.values()) / len(per_freq)
    return macro, per_freq


def cross_dd_best_expert_by_eval(results):
    """Pick best expert per freq using held-out eval F1 (not train F1)."""
    print(f"\n  --- Cross-DD best expert by eval F1 ---", flush=True)
    per_freq = {}
    for freq in DRIFT_FREQS:
        best_dd = None
        best_f1 = -1
        for dd in DETECTORS:
            det = results["detectors"].get(dd, {})
            if "error" in det or not det:
                continue
            exp = det.get("experts", {}).get(str(freq))
            if not exp:
                continue
            # Re-evaluate on ensemble eval seeds
            params = exp["params"]
            f1s = []
            for i, seed in enumerate(ENSEMBLE_EVAL_SEEDS):
                from run_overnight_benchmark import run_single
                f1 = run_single(dd, params, freq, seed, 500 + i,
                                freq // 10, freq // 2)
                f1s.append(f1)
            mean_f1 = sum(f1s) / len(f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_dd = dd
        if best_dd:
            per_freq[freq] = {"detector": best_dd, "f1": best_f1}
            print(f"    freq={freq}: {best_dd} F1={best_f1:.4f}", flush=True)
    macro = sum(v["f1"] for v in per_freq.values()) / len(per_freq)
    print(f"  Cross-DD best-expert-by-eval macro F1: {macro:.4f}", flush=True)
    return {"macro_f1": macro, "detail": per_freq}


def main():
    # Load existing results
    results_path = f"{OUTPUT_DIR}/results_final.json"
    if not os.path.exists(results_path):
        results_path = f"{OUTPUT_DIR}/results_partial.json"
    print(f"Loading results from {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    # Print current best single DD
    best_gen = -1
    best_gen_dd = None
    best_exp = -1
    best_exp_dd = None
    for dd in DETECTORS:
        det = results["detectors"].get(dd, {})
        if "error" in det or not det:
            continue
        gen = det.get("generalist", {}).get("eval_macro_f1", 0)
        exp = det.get("expert_macro_f1", 0)
        if gen > best_gen:
            best_gen = gen
            best_gen_dd = dd
        if exp > best_exp:
            best_exp = exp
            best_exp_dd = dd
    print(f"Best single generalist: {best_gen_dd} = {best_gen:.4f}")
    print(f"Best single expert: {best_exp_dd} = {best_exp:.4f}")
    print(f"Target: beat {max(best_gen, best_exp):.4f}")

    start_time = time.time()

    # Phase 3a: Cross-DD best expert by eval F1
    cross_eval = cross_dd_best_expert_by_eval(results)

    # Phase 3b: Jointly optimize 2-DD ensembles
    print(f"\n{'='*70}")
    print(f"Phase 3: Joint 2-DD Ensemble Optimization")
    print(f"{'='*70}", flush=True)

    best_ensemble_macro = -1
    best_ensemble_config = None
    ensemble_results = {}

    for dd1, dd2 in ENSEMBLE_PAIRS:
        pair_name = f"{dd1}+{dd2}"
        print(f"\n  --- Optimizing {pair_name} ---", flush=True)
        t0 = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_ensemble_objective(dd1, dd2),
                       n_trials=N_ENSEMBLE_TRIALS, show_progress_bar=True)

        best = study.best_trial
        print(f"  Train macro F1: {best.value:.4f}")
        print(f"  Params: {best.params}")
        print(f"  Time: {time.time() - t0:.1f}s", flush=True)

        # Extract params
        params1 = {k[3:]: v for k, v in best.params.items() if k.startswith("d1_")}
        params2 = {k[3:]: v for k, v in best.params.items() if k.startswith("d2_")}
        criterion = best.params["criterion"]
        decision_window = best.params["decision_window"]

        # Evaluate on held-out seeds
        eval_macro, eval_per_freq = eval_ensemble_final(
            dd1, params1, dd2, params2, criterion, decision_window)
        print(f"  Eval macro F1: {eval_macro:.4f}")
        for f in DRIFT_FREQS:
            print(f"    freq={f}: F1={eval_per_freq[f]:.4f}", flush=True)

        ensemble_results[pair_name] = {
            "train_f1": best.value,
            "eval_macro_f1": eval_macro,
            "eval_per_freq": eval_per_freq,
            "params_d1": params1,
            "params_d2": params2,
            "criterion": criterion,
            "decision_window": decision_window,
        }

        if eval_macro > best_ensemble_macro:
            best_ensemble_macro = eval_macro
            best_ensemble_config = pair_name
            print(f"  *** NEW BEST ENSEMBLE: {pair_name} = {eval_macro:.4f} ***", flush=True)

    # Final summary
    print(f"\n{'='*70}")
    print(f"ENSEMBLE OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Best single generalist: {best_gen_dd} = {best_gen:.4f}")
    print(f"Best single expert: {best_exp_dd} = {best_exp:.4f}")
    print(f"Cross-DD best-expert-by-eval: {cross_eval['macro_f1']:.4f}")
    print(f"\nJoint ensembles:")
    for pair_name, info in ensemble_results.items():
        marker = " ***" if pair_name == best_ensemble_config else ""
        print(f"  {pair_name}: train={info['train_f1']:.4f}, eval={info['eval_macro_f1']:.4f} ({info['criterion']}, dw={info['decision_window']}){marker}")

    all_candidates = {
        "best_single_gen": best_gen,
        "best_single_expert": best_exp,
        "cross_dd_best_by_eval": cross_eval["macro_f1"],
    }
    for pair_name, info in ensemble_results.items():
        all_candidates[f"ensemble_{pair_name}"] = info["eval_macro_f1"]

    winner = max(all_candidates, key=all_candidates.get)
    print(f"\n*** OVERALL WINNER: {winner} = {all_candidates[winner]:.4f} ***")

    if all_candidates[winner] > max(best_gen, best_exp):
        print("*** ENSEMBLE BEATS ALL SINGLE DDS! ***")
    else:
        print("*** Single DD still wins. Consider larger search. ***")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Save
    output = {
        "cross_dd_best_by_eval": cross_eval,
        "joint_ensembles": ensemble_results,
        "best_ensemble": best_ensemble_config,
        "best_ensemble_f1": best_ensemble_macro,
        "all_candidates": all_candidates,
        "winner": winner,
    }
    with open(f"{OUTPUT_DIR}/ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {OUTPUT_DIR}/ensemble_results.json")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
