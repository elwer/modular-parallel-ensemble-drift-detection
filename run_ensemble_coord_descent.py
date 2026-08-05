#!/usr/bin/env python3
"""Coordinate descent ensemble optimization.

Instead of optimizing all 7 detectors jointly (30+ dimensions, intractable),
optimize one detector at a time while keeping the others fixed. Each
sub-problem has only ~6 dimensions (4 detector params + k + dw), which
Optuna can handle in 10-15 trials.

Strategy:
  1. Fix SPLL params (best generalist, F1=0.87 — the anchor)
  2. Initialize other 6 detectors with generalist params
  3. Round 1: For each non-SPLL detector, optimize its params + k + dw
  4. Round 2: Repeat with updated params
  5. Evaluate final ensemble on held-out seeds
  6. Compare to best single generalist
"""
import sys
import json
import time
import os
import copy
import multiprocessing as mp
import optuna
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _suggest_detector_params, _instantiate, _f1_from_counts,
    MAX_WINDOW_FRACTION, _cap, CLASS_PATH,
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections
from optimization.synthetic_f1_multistream_optimize_optuna import GENERATORS as BASE_GENERATORS
from datasets.sineclusters_hard import SineClustersHard
from datasets.waveform_hard import WaveformDriftHard

GENERATORS = dict(BASE_GENERATORS)
GENERATORS["SineClustersHard"] = SineClustersHard
GENERATORS["WaveformDriftHard"] = WaveformDriftHard

SCENARIO_GENERATORS = ["SineClustersHard", "WaveformDriftHard"]
ALL_FREQS = [100, 200, 500, 1000, 2000]
TRAIN_FREQS = [100, 200, 500, 1000, 2000]  # all frequencies, same as generalist
STREAM_LENGTH = 5000
SEED = 1337
TOLERANCE = 100
TIGHT_TOLERANCE = 20
SUPPRESSION = 50
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
ANCHOR = "SPLL"  # fix this detector's params
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 90
N_TRIALS_PER_DET = 12
N_ROUNDS = 2
MAX_WINDOW = int(min(ALL_FREQS) * MAX_WINDOW_FRACTION)  # 50

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
                  generators, freqs, seeds,
                  tolerance=TOLERANCE, suppression=SUPPRESSION):
    """Evaluate ensemble on given generators/freqs/seeds. Returns macro F1.
    Runs all stream evaluations in parallel."""
    ctx = mp.get_context("fork")
    tasks = []
    for generator in generators:
        for freq in freqs:
            for s_idx, seed in enumerate(seeds):
                tasks.append((generator, freq, seed, s_idx))

    procs = []
    queues = []
    for generator, freq, seed, s_idx in tasks:
        q = ctx.Queue()
        p = ctx.Process(target=_ensemble_worker,
                        args=(q, slot_specs, ensemble_criterion,
                              decision_window, freq, seed, s_idx,
                              tolerance, suppression, generator))
        p.start()
        procs.append(p)
        queues.append(q)

    all_f1s = []
    for p, q in zip(procs, queues):
        p.join(PER_STREAM_TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
            all_f1s.append(0.0)
        else:
            try:
                result = q.get_nowait()
                all_f1s.append(result[1] if result[0] == "ok" else 0.0)
            except:
                all_f1s.append(0.0)
    return sum(all_f1s) / len(all_f1s) if all_f1s else 0.0


def _single_worker(queue, kind, params, freq, stream_seed, s_idx,
                   tolerance, suppression, generator):
    try:
        stream = build_stream_local(generator, freq, STREAM_LENGTH, stream_seed)
        known = list(stream.drifts)
        n_samples_key = "n_samples" if "n_samples" in params else "n_reference_samples"
        recent_size = params.get(n_samples_key, 200)
        det = _instantiate(kind, params, seed=SEED + s_idx * 1000,
                           recent_samples_size=recent_size)
        detections = []
        for i, (x, _y) in enumerate(stream):
            try:
                if det.update(x):
                    detections.append(i)
            except Exception:
                pass
        dets = apply_suppression(detections, suppression)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        queue.put(("ok", f1, tp, fp, fn))
    except Exception as e:
        queue.put(("error", str(e)))


def eval_single(kind, params, generators, freqs, seeds,
                tolerance=TOLERANCE, suppression=SUPPRESSION):
    all_f1s = []
    for generator in generators:
        for freq in freqs:
            for s_idx, seed in enumerate(seeds):
                ctx = mp.get_context("fork")
                queue = ctx.Queue()
                proc = ctx.Process(target=_single_worker,
                                   args=(queue, kind, params, freq, seed, s_idx,
                                         tolerance, suppression, generator))
                proc.start()
                proc.join(PER_STREAM_TIMEOUT)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                    if proc.is_alive():
                        proc.kill()
                    all_f1s.append(0.0)
                else:
                    try:
                        result = queue.get_nowait()
                        all_f1s.append(result[1] if result[0] == "ok" else 0.0)
                    except:
                        all_f1s.append(0.0)
    return sum(all_f1s) / len(all_f1s) if all_f1s else 0.0


def main():
    print("=" * 70)
    print("Coordinate Descent Ensemble Optimization")
    print("=" * 70)

    # Load generalist params
    partial_path = f"{OUTPUT_DIR}/ensemble_benchmark_partial.json"
    if not os.path.exists(partial_path):
        print(f"ERROR: {partial_path} not found.")
        return

    all_results = json.load(open(partial_path))

    # Initialize slot_specs with generalist params
    slot_specs = []
    for kind in DETECTORS:
        key = f"single_{kind}"
        if key not in all_results or "error" in all_results.get(key, {}):
            print(f"WARNING: {key} not found or has error")
            continue
        params = all_results[key]["params"]
        slot_specs.append((kind, params))
        print(f"  {kind}: {params}")

    print(f"\nLoaded {len(slot_specs)} detector configs")

    # Evaluate initial ensemble (generalist params, majority voting)
    print(f"\n--- Initial ensemble (generalist params, at_least_4, dw=16) ---")
    init_f1 = eval_ensemble(slot_specs, "at_least_4", 16,
                            SCENARIO_GENERATORS, TRAIN_FREQS, TRAIN_SEEDS)
    print(f"  Train F1: {init_f1:.4f}", flush=True)

    # Also evaluate best single generalist for reference
    best_single_f1 = max(
        all_results.get(f"single_{k}", {}).get("combined_f1", 0)
        for k in DETECTORS
    )
    print(f"  Best single generalist F1: {best_single_f1:.4f}")

    # ---- Phase 1: Optimize k/dw on generalist params ----
    print(f"\n{'='*70}")
    print(f"Phase 1: Optimize k/dw on generalist params")
    print(f"{'='*70}", flush=True)

    def kd_objective(trial):
        k = trial.suggest_int("k", 1, len(slot_specs))
        dw = trial.suggest_int("decision_window", 1, 20)
        criterion = f"at_least_{k}"
        return eval_ensemble(slot_specs, criterion, dw,
                             SCENARIO_GENERATORS, TRAIN_FREQS, TRAIN_SEEDS)

    kd_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    kd_study.optimize(kd_objective, n_trials=15, show_progress_bar=True)
    best_k = kd_study.best_trial.params["k"]
    best_dw = kd_study.best_trial.params["decision_window"]
    best_criterion = f"at_least_{best_k}"
    current_best_f1 = kd_study.best_value
    print(f"\n  Best k={best_k}, dw={best_dw}, F1={current_best_f1:.4f}", flush=True)

    # ---- Phase 2: Coordinate descent on detector params (k/dw fixed) ----
    for round_idx in range(N_ROUNDS):
        print(f"\n{'='*70}")
        print(f"Round {round_idx + 1}/{N_ROUNDS} (k={best_k}, dw={best_dw})")
        print(f"{'='*70}", flush=True)

        for det_idx, (kind, current_params) in enumerate(slot_specs):
            if kind == ANCHOR:
                print(f"\n  [{det_idx}] {kind}: FIXED (anchor)")
                continue

            print(f"\n  [{det_idx}] Optimizing {kind} (k={best_k}, dw={best_dw})...", flush=True)
            t0 = time.time()

            fixed_specs = list(slot_specs)

            def objective(trial):
                new_params = _suggest_detector_params(
                    trial, f"r{round_idx}_d{det_idx}_", kind,
                    max_window=MAX_WINDOW)
                trial_specs = list(fixed_specs)
                trial_specs[det_idx] = (kind, new_params)
                return eval_ensemble(trial_specs, best_criterion, best_dw,
                                    SCENARIO_GENERATORS, TRAIN_FREQS, TRAIN_SEEDS)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED + det_idx + 100 * round_idx))
            study.optimize(objective, n_trials=N_TRIALS_PER_DET,
                           show_progress_bar=True)

            best = study.best_trial
            best_params = _suggest_detector_params(
                best, f"r{round_idx}_d{det_idx}_", kind, max_window=MAX_WINDOW)

            # Only accept if better than current
            if best.value > current_best_f1:
                print(f"  {kind}: F1={best.value:.4f} > {current_best_f1:.4f} ACCEPTED")
                print(f"  New params: {best_params}")
                slot_specs[det_idx] = (kind, best_params)
                current_best_f1 = best.value
            else:
                print(f"  {kind}: F1={best.value:.4f} <= {current_best_f1:.4f} REJECTED (keeping generalist)")

            print(f"  Time: {time.time() - t0:.1f}s", flush=True)

            # Save partial results
            partial = {
                "round": round_idx,
                "detector": kind,
                "best_k": best_k,
                "best_dw": best_dw,
                "current_best_f1": current_best_f1,
                "slot_specs": [(k, p) for k, p in slot_specs],
            }
            with open(f"{OUTPUT_DIR}/coord_descent_partial.json", "w") as f:
                json.dump(partial, f, indent=2, default=str)

        # ---- Phase 2.5: Re-optimize k/dw after each round ----
        print(f"\n  Re-optimizing k/dw after round {round_idx + 1}...", flush=True)
        def kd_objective2(trial):
            k = trial.suggest_int("k", 1, len(slot_specs))
            dw = trial.suggest_int("decision_window", 1, 20)
            criterion = f"at_least_{k}"
            return eval_ensemble(slot_specs, criterion, dw,
                                SCENARIO_GENERATORS, TRAIN_FREQS, TRAIN_SEEDS)
        kd_study2 = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + 999 + round_idx))
        kd_study2.optimize(kd_objective2, n_trials=10, show_progress_bar=True)
        if kd_study2.best_value > current_best_f1:
            best_k = kd_study2.best_trial.params["k"]
            best_dw = kd_study2.best_trial.params["decision_window"]
            best_criterion = f"at_least_{best_k}"
            current_best_f1 = kd_study2.best_value
            print(f"  Updated k={best_k}, dw={best_dw}, F1={current_best_f1:.4f}", flush=True)
        else:
            print(f"  k/dw unchanged (best={kd_study2.best_value:.4f} <= {current_best_f1:.4f})", flush=True)

    # ---- Final evaluation ----
    print(f"\n{'='*70}")
    print(f"Final Evaluation")
    print(f"{'='*70}")

    best_criterion = f"at_least_{best_k}"
    print(f"\nBest criterion: {best_criterion}, dw={best_dw}")
    print(f"\nFinal detector params:")
    for kind, params in slot_specs:
        print(f"  {kind}: {params}")

    # Evaluate on all freqs, eval seeds, both tolerances
    print(f"\n--- Standard tolerance ({TOLERANCE}) ---")
    sc_results = {}
    wf_results = {}
    for freq in ALL_FREQS:
        sc_f1s = []
        wf_f1s = []
        for s_idx, seed in enumerate(EVAL_SEEDS):
            ctx = mp.get_context("fork")
            q = ctx.Queue()
            p = ctx.Process(target=_ensemble_worker,
                            args=(q, slot_specs, best_criterion, best_dw,
                                  freq, seed, s_idx, TOLERANCE, SUPPRESSION,
                                  SCENARIO_GENERATORS[0]))
            p.start()
            p.join(PER_STREAM_TIMEOUT)
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill()
                sc_f1s.append(0.0)
            else:
                try:
                    r = q.get_nowait()
                    sc_f1s.append(r[1] if r[0] == "ok" else 0.0)
                except:
                    sc_f1s.append(0.0)

            ctx2 = mp.get_context("fork")
            q2 = ctx2.Queue()
            p2 = ctx2.Process(target=_ensemble_worker,
                             args=(q2, slot_specs, best_criterion, best_dw,
                                   freq, seed, s_idx, TOLERANCE, SUPPRESSION,
                                   SCENARIO_GENERATORS[1]))
            p2.start()
            p2.join(PER_STREAM_TIMEOUT)
            if p2.is_alive():
                p2.terminate(); p2.join(5)
                if p2.is_alive(): p2.kill()
                wf_f1s.append(0.0)
            else:
                try:
                    r2 = q2.get_nowait()
                    wf_f1s.append(r2[1] if r2[0] == "ok" else 0.0)
                except:
                    wf_f1s.append(0.0)
        sc_results[freq] = sum(sc_f1s) / len(sc_f1s)
        wf_results[freq] = sum(wf_f1s) / len(wf_f1s)
        print(f"  freq={freq}: SC={sc_results[freq]:.4f} WF={wf_results[freq]:.4f}")

    sc_macro = sum(sc_results.values()) / len(sc_results)
    wf_macro = sum(wf_results.values()) / len(wf_results)
    combined = (sc_macro + wf_macro) / 2
    print(f"\n  SC macro: {sc_macro:.4f}")
    print(f"  WF macro: {wf_macro:.4f}")
    print(f"  Combined: {combined:.4f}")

    # Tight tolerance
    print(f"\n--- Tight tolerance ({TIGHT_TOLERANCE}) ---")
    sc_tight = {}
    wf_tight = {}
    for freq in ALL_FREQS:
        sc_f1s = []
        wf_f1s = []
        for s_idx, seed in enumerate(EVAL_SEEDS):
            ctx = mp.get_context("fork")
            q = ctx.Queue()
            p = ctx.Process(target=_ensemble_worker,
                            args=(q, slot_specs, best_criterion, best_dw,
                                  freq, seed, s_idx, TIGHT_TOLERANCE, SUPPRESSION,
                                  SCENARIO_GENERATORS[0]))
            p.start()
            p.join(PER_STREAM_TIMEOUT)
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill()
                sc_f1s.append(0.0)
            else:
                try:
                    r = q.get_nowait()
                    sc_f1s.append(r[1] if r[0] == "ok" else 0.0)
                except:
                    sc_f1s.append(0.0)

            ctx2 = mp.get_context("fork")
            q2 = ctx2.Queue()
            p2 = ctx2.Process(target=_ensemble_worker,
                             args=(q2, slot_specs, best_criterion, best_dw,
                                   freq, seed, s_idx, TIGHT_TOLERANCE, SUPPRESSION,
                                   SCENARIO_GENERATORS[1]))
            p2.start()
            p2.join(PER_STREAM_TIMEOUT)
            if p2.is_alive():
                p2.terminate(); p2.join(5)
                if p2.is_alive(): p2.kill()
                wf_f1s.append(0.0)
            else:
                try:
                    r2 = q2.get_nowait()
                    wf_f1s.append(r2[1] if r2[0] == "ok" else 0.0)
                except:
                    wf_f1s.append(0.0)
        sc_tight[freq] = sum(sc_f1s) / len(sc_f1s)
        wf_tight[freq] = sum(wf_f1s) / len(wf_f1s)
        print(f"  freq={freq}: SC={sc_tight[freq]:.4f} WF={wf_tight[freq]:.4f}")

    sc_tight_macro = sum(sc_tight.values()) / len(sc_tight)
    wf_tight_macro = sum(wf_tight.values()) / len(wf_tight)
    combined_tight = (sc_tight_macro + wf_tight_macro) / 2
    print(f"\n  SC tight macro: {sc_tight_macro:.4f}")
    print(f"  WF tight macro: {wf_tight_macro:.4f}")
    print(f"  Combined tight: {combined_tight:.4f}")

    # ---- Comparison ----
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")

    # Get per-detector single generalist results for comparison
    print(f"\n{'Detector':<12} {'Gen F1':>8}")
    print(f"{'-'*12} {'-'*8}")
    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        gen_f1 = d.get("combined_f1", 0)
        print(f"{kind:<12} {gen_f1:>8.4f}")

    best_single = max(
        all_results.get(f"single_{k}", {}).get("combined_f1", 0)
        for k in DETECTORS
    )

    print(f"\n  Best single generalist:  {best_single:.4f}")
    print(f"  Ensemble (coord desc):   {combined:.4f}")
    print(f"  Ensemble tight:          {combined_tight:.4f}")

    if combined > best_single:
        print(f"\n  *** ENSEMBLE WINS by {combined - best_single:.4f} ***")
    else:
        print(f"\n  Ensemble is {best_single - combined:.4f} below best single")

    # Save final results
    output = {
        "slot_specs": [(k, p) for k, p in slot_specs],
        "best_criterion": best_criterion,
        "best_k": best_k,
        "best_dw": best_dw,
        "ensemble_f1": combined,
        "ensemble_f1_tight": combined_tight,
        "sc_detail": {str(f): v for f, v in sc_results.items()},
        "wf_detail": {str(f): v for f, v in wf_results.items()},
        "sc_detail_tight": {str(f): v for f, v in sc_tight.items()},
        "wf_detail_tight": {str(f): v for f, v in wf_tight.items()},
        "best_single_f1": best_single,
        "n_rounds": N_ROUNDS,
        "n_trials_per_det": N_TRIALS_PER_DET,
    }
    output_path = f"{OUTPUT_DIR}/coord_descent_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
