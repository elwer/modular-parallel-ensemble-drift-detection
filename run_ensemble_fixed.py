#!/usr/bin/env python3
"""Ensemble benchmark with FIXED single-detector params.

Uses the already-optimized single detector parameters from Phase 1 of
run_ensemble_benchmark.py and only searches over voting criterion and
decision window. This gives the ensemble a fair chance since the
individual detectors are already well-tuned.
"""
import sys
import json
import time
import os
import multiprocessing as mp
import optuna
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _instantiate, _f1_from_counts,
    MAX_WINDOW_FRACTION,
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections
from optimization.synthetic_f1_multistream_optimize_optuna import GENERATORS as BASE_GENERATORS
from datasets.sineclusters_hard import SineClustersHard
from datasets.waveform_hard import WaveformDriftHard

GENERATORS = dict(BASE_GENERATORS)
GENERATORS["SineClustersHard"] = SineClustersHard
GENERATORS["WaveformDriftHard"] = WaveformDriftHard

SCENARIO_GENERATORS = ["SineClustersHard", "WaveformDriftHard"]
DRIFT_FREQS = [100, 200, 500, 1000, 2000]
STREAM_LENGTH = 5000
SEED = 1337
TOLERANCE = 100
TIGHT_TOLERANCE = 20
SUPPRESSION = 50
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 90
N_TRIALS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_stream_local(generator_name, drift_frequency, stream_length, seed):
    cls = GENERATORS[generator_name]
    return cls(drift_frequency=drift_frequency,
               stream_length=stream_length,
               seed=seed)


def _ensemble_worker(queue, slot_specs, ensemble_criterion, decision_window,
                     freq, stream_seed, s_idx, tolerance, suppression, generator):
    try:
        stream = build_stream_local(generator, freq, STREAM_LENGTH, stream_seed)
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
        queue.put(("ok", f1, tp, fp, fn))
    except Exception as e:
        queue.put(("error", str(e)))


def eval_ensemble(slot_specs, ensemble_criterion, decision_window,
                  generator, freqs, seeds,
                  tolerance=TOLERANCE, suppression=SUPPRESSION):
    results = {}
    for freq in freqs:
        f1s = []
        for i, seed in enumerate(seeds):
            ctx = mp.get_context("fork")
            queue = ctx.Queue()
            proc = ctx.Process(target=_ensemble_worker,
                               args=(queue, slot_specs, ensemble_criterion,
                                     decision_window, freq, seed, i,
                                     tolerance, suppression, generator))
            proc.start()
            proc.join(PER_STREAM_TIMEOUT)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                f1s.append(0.0)
            else:
                try:
                    result = queue.get_nowait()
                    if result[0] == "ok":
                        f1s.append(result[1])
                    else:
                        f1s.append(0.0)
                except:
                    f1s.append(0.0)
        results[freq] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def macro_f1(per_freq, freqs):
    return sum(per_freq[f] for f in freqs) / len(freqs)


def main():
    print("=" * 70)
    print("Ensemble Fixed-Params Benchmark")
    print("=" * 70)

    # Load optimized single-detector params
    partial_path = f"{OUTPUT_DIR}/ensemble_benchmark_partial.json"
    if not os.path.exists(partial_path):
        print(f"ERROR: {partial_path} not found. Run run_ensemble_benchmark.py first.")
        return

    all_results = json.load(open(partial_path))

    slot_specs = []
    for kind in DETECTORS:
        key = f"single_{kind}"
        if key not in all_results or "error" in all_results.get(key, {}):
            print(f"WARNING: {key} not found or has error, skipping")
            continue
        params = all_results[key]["params"]
        slot_specs.append((kind, params))
        print(f"  {kind}: loaded params {params}")

    if len(slot_specs) < 2:
        print("ERROR: Not enough detectors with optimized params.")
        return

    print(f"\nLoaded {len(slot_specs)} detector configs")
    print(f"Optimizing only criterion + decision_window ({N_TRIALS} trials)")
    print(flush=True)

    # Optimize criterion + decision_window only
    def objective(trial):
        criterion = trial.suggest_categorical("criterion", ["any", "majority", "all"])
        decision_window = trial.suggest_int("decision_window", 1, 10)

        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                for s_idx, seed in enumerate(TRAIN_SEEDS):
                    ctx = mp.get_context("fork")
                    queue = ctx.Queue()
                    proc = ctx.Process(target=_ensemble_worker,
                                       args=(queue, slot_specs, criterion,
                                             decision_window, freq, seed, s_idx,
                                             TOLERANCE, SUPPRESSION, generator))
                    proc.start()
                    proc.join(PER_STREAM_TIMEOUT)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(5)
                        if proc.is_alive():
                            proc.kill()
                        f1s.append(0.0)
                    else:
                        try:
                            result = queue.get_nowait()
                            f1s.append(result[1] if result[0] == "ok" else 0.0)
                        except:
                            f1s.append(0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_trial
    best_criterion = best.params["criterion"]
    best_dw = best.params["decision_window"]
    print(f"\nBest criterion: {best_criterion}, dw: {best_dw}, train F1: {best.value:.4f}")
    print(f"Optimization time: {time.time() - t0:.1f}s")

    # Evaluate all criteria with best_dw on held-out seeds
    results = {}
    for crit in ["any", "majority", "all"]:
        print(f"\n  Evaluating criterion='{crit}', dw={best_dw} (tol={TOLERANCE})...", flush=True)
        sc_ev = eval_ensemble(slot_specs, crit, best_dw,
                              SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                              tolerance=TOLERANCE, suppression=SUPPRESSION)
        wf_ev = eval_ensemble(slot_specs, crit, best_dw,
                              SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                              tolerance=TOLERANCE, suppression=SUPPRESSION)
        sc_m = macro_f1(sc_ev, DRIFT_FREQS)
        wf_m = macro_f1(wf_ev, DRIFT_FREQS)
        comb = (sc_m + wf_m) / 2
        print(f"  SC: {sc_m:.4f}, WF: {wf_m:.4f}, Combined: {comb:.4f}")

        # Tight tolerance
        print(f"  Evaluating criterion='{crit}', dw={best_dw} (tol={TIGHT_TOLERANCE})...", flush=True)
        sc_tight = eval_ensemble(slot_specs, crit, best_dw,
                                 SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                 tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
        wf_tight = eval_ensemble(slot_specs, crit, best_dw,
                                 SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                 tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
        sc_t = macro_f1(sc_tight, DRIFT_FREQS)
        wf_t = macro_f1(wf_tight, DRIFT_FREQS)
        comb_t = (sc_t + wf_t) / 2
        print(f"  SC tight: {sc_t:.4f}, WF tight: {wf_t:.4f}, Combined tight: {comb_t:.4f}")

        results[f"ensemble_{crit}"] = {
            "criterion": crit,
            "decision_window": best_dw,
            "sc_f1": sc_m, "wf_f1": wf_m, "combined_f1": comb,
            "sc_f1_tight": sc_t, "wf_f1_tight": wf_t, "combined_f1_tight": comb_t,
            "sc_detail": {str(f): sc_ev[f] for f in DRIFT_FREQS},
            "wf_detail": {str(f): wf_ev[f] for f in DRIFT_FREQS},
            "sc_detail_tight": {str(f): sc_tight[f] for f in DRIFT_FREQS},
            "wf_detail_tight": {str(f): wf_tight[f] for f in DRIFT_FREQS},
        }

    # Also try different decision windows with best criterion
    print(f"\n  Trying different decision windows with criterion='{best_criterion}'...", flush=True)
    for dw in [1, 3, 5, 7, 10, 15, 20]:
        if dw == best_dw:
            continue
        sc_ev = eval_ensemble(slot_specs, best_criterion, dw,
                              SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                              tolerance=TOLERANCE, suppression=SUPPRESSION)
        wf_ev = eval_ensemble(slot_specs, best_criterion, dw,
                              SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                              tolerance=TOLERANCE, suppression=SUPPRESSION)
        sc_m = macro_f1(sc_ev, DRIFT_FREQS)
        wf_m = macro_f1(wf_ev, DRIFT_FREQS)
        comb = (sc_m + wf_m) / 2
        print(f"  dw={dw}: SC: {sc_m:.4f}, WF: {wf_m:.4f}, Combined: {comb:.4f}")
        results[f"ensemble_{best_criterion}_dw{dw}"] = {
            "criterion": best_criterion,
            "decision_window": dw,
            "sc_f1": sc_m, "wf_f1": wf_m, "combined_f1": comb,
        }

    # Save results
    output_path = f"{OUTPUT_DIR}/ensemble_fixed_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON: Single detectors vs Ensemble (fixed params)")
    print(f"{'='*70}")

    print(f"\n--- Standard tolerance (tol={TOLERANCE}) ---")
    print(f"{'Detector':<20} {'SC F1':>8} {'WF F1':>8} {'Combined':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10}")

    best_single = -1
    best_single_name = ""
    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        comb = d.get("combined_f1", 0)
        print(f"{kind:<20} {d.get('sc_f1',0):>8.4f} {d.get('wf_f1',0):>8.4f} {comb:>10.4f}")
        if comb > best_single:
            best_single = comb
            best_single_name = kind

    best_ensemble = -1
    best_ensemble_label = ""
    for key, d in results.items():
        if "combined_f1" not in d:
            continue
        comb = d["combined_f1"]
        label = f"ens-{d['criterion']}-dw{d['decision_window']}"
        print(f"{label:<20} {d['sc_f1']:>8.4f} {d['wf_f1']:>8.4f} {comb:>10.4f}")
        if comb > best_ensemble:
            best_ensemble = comb
            best_ensemble_label = label

    print(f"\n  Best single: {best_single_name} = {best_single:.4f}")
    print(f"  Best ensemble: {best_ensemble_label} = {best_ensemble:.4f}")
    if best_ensemble > best_single:
        print(f"  *** ENSEMBLE WINS by {best_ensemble - best_single:.4f} ***")
    else:
        print(f"  Single detector wins by {best_single - best_ensemble:.4f}")

    # Tight tolerance comparison
    print(f"\n--- Tight tolerance (tol={TIGHT_TOLERANCE}) ---")
    print(f"{'Detector':<20} {'SC F1':>8} {'WF F1':>8} {'Combined':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10}")

    best_single_tight = -1
    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        comb = d.get("combined_f1_tight", 0)
        print(f"{kind:<20} {d.get('sc_f1_tight',0):>8.4f} {d.get('wf_f1_tight',0):>8.4f} {comb:>10.4f}")
        if comb > best_single_tight:
            best_single_tight = comb

    best_ensemble_tight = -1
    for key, d in results.items():
        if "combined_f1_tight" not in d:
            continue
        comb = d["combined_f1_tight"]
        label = f"ens-{d['criterion']}-dw{d['decision_window']}"
        print(f"{label:<20} {d.get('sc_f1_tight',0):>8.4f} {d.get('wf_f1_tight',0):>8.4f} {comb:>10.4f}")
        if comb > best_ensemble_tight:
            best_ensemble_tight = comb

    print(f"\n  Best single (tight): {best_single_tight:.4f}")
    print(f"  Best ensemble (tight): {best_ensemble_tight:.4f}")
    if best_ensemble_tight > best_single_tight:
        print(f"  *** ENSEMBLE WINS (tight) by {best_ensemble_tight - best_single_tight:.4f} ***")
    else:
        print(f"  Single detector wins (tight) by {best_single_tight - best_ensemble_tight:.4f}")

    print(f"\nResults saved to {output_path}")
    print(f"Total time: {time.time() - t0:.0f}s ({(time.time() - t0)/3600:.1f}h)")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
