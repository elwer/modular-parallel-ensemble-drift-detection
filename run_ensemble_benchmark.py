#!/usr/bin/env python3
"""Ensemble benchmark: compare full 7-detector ensemble vs single generalists.

Goal: find a scenario where no single generalist detector is great, but the
ensemble of all 7 detectors (with majority voting) is significantly better.

Uses harder generator variants (SineClustersHard, WaveformDriftHard) with:
  - More noise, more concepts, more centroids (SineClusters)
  - Noise features, more h_functions (Waveform)
  - Wide frequency range [100, 200, 500, 1000, 2000]
  - Conservative suppression (50) to avoid overfitting
"""
import sys
import json
import time
import os
import multiprocessing as mp
import optuna
import warnings
import csv

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _suggest_detector_params, _instantiate,
    MAX_WINDOW_FRACTION, _cap, _f1_from_counts,
    CLASS_PATH, CANDIDATES,
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections
from optimization.synthetic_f1_multistream_optimize_optuna import GENERATORS as BASE_GENERATORS
from datasets.sineclusters_hard import SineClustersHard
from datasets.waveform_hard import WaveformDriftHard

# Register new generators
GENERATORS = dict(BASE_GENERATORS)
GENERATORS["SineClustersHard"] = SineClustersHard
GENERATORS["WaveformDriftHard"] = WaveformDriftHard


# --- Config ---
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
PER_STREAM_TIMEOUT = 30
N_SINGLE_TRIALS = 15
N_ENSEMBLE_TRIALS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Stream builder
# ---------------------------------------------------------------------------

def build_stream_local(generator_name, drift_frequency, stream_length, seed):
    cls = GENERATORS[generator_name]
    return cls(drift_frequency=drift_frequency,
               stream_length=stream_length,
               seed=seed)


# ---------------------------------------------------------------------------
# Single-detector worker
# ---------------------------------------------------------------------------

def _single_worker(queue, kind, params, freq, stream_seed, s_idx,
                   tolerance, suppression, generator):
    try:
        stream = build_stream_local(generator, freq, STREAM_LENGTH, stream_seed)
        known = list(stream.drifts)
        n_samples_key = "n_samples" if "n_samples" in params else "n_reference_samples"
        recent_size = params.get(n_samples_key, 200)
        det = _instantiate(kind, params, seed=SEED + 1000 * s_idx,
                           recent_samples_size=recent_size)
        _, raw = run_ensemble(stream, [det], [f"[000]{kind}"],
                              detector_criterion="any",
                              ensemble_criterion="any",
                              decision_window=1)
        dets = apply_suppression(raw, suppression)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        queue.put(("ok", f1, tp, fp, fn))
    except Exception as e:
        queue.put(("error", str(e)))


def eval_single(kind, params, generator, freqs, seeds,
                tolerance=TOLERANCE, suppression=SUPPRESSION):
    results = {}
    for freq in freqs:
        f1s = []
        for i, seed in enumerate(seeds):
            ctx = mp.get_context("fork")
            queue = ctx.Queue()
            proc = ctx.Process(target=_single_worker,
                               args=(queue, kind, params, freq, seed, i,
                                     tolerance, suppression, generator))
            proc.start()
            proc.join(PER_STREAM_TIMEOUT * 2)
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


# ---------------------------------------------------------------------------
# Ensemble worker (all 7 detectors)
# ---------------------------------------------------------------------------

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
            proc.join(PER_STREAM_TIMEOUT * 3)
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


# ---------------------------------------------------------------------------
# Single-detector generalist objective
# ---------------------------------------------------------------------------

def make_single_objective(kind):
    max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)

    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                for s_idx, seed in enumerate(TRAIN_SEEDS):
                    ctx = mp.get_context("fork")
                    queue = ctx.Queue()
                    proc = ctx.Process(target=_single_worker,
                                       args=(queue, kind, params, freq, seed, s_idx,
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
    return objective


def optimize_single_generalist(kind):
    print(f"\n  --- Single generalist ({kind}) ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_single_objective(kind),
                   n_trials=N_SINGLE_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print(f"  Train F1: {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)
    return best.params, best.value


# ---------------------------------------------------------------------------
# Full 7-detector ensemble objective
# ---------------------------------------------------------------------------

def make_ensemble_objective():
    max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)

    def objective(trial):
        slot_specs = []
        for i, kind in enumerate(DETECTORS):
            params = _suggest_detector_params(trial, f"d{i+1}_", kind, max_window=max_window)
            slot_specs.append((kind, params))

        criterion = trial.suggest_categorical("criterion", ["any", "majority", "all"])
        decision_window = trial.suggest_int("decision_window", 1, 5)

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
                    proc.join(PER_STREAM_TIMEOUT * 3)
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
    return objective


def optimize_ensemble():
    print(f"\n  --- Full 7-detector ensemble ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_ensemble_objective(),
                   n_trials=N_ENSEMBLE_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print(f"  Train F1: {best.value:.4f}")
    print(f"  Criterion: {best.params['criterion']}, DW: {best.params['decision_window']}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)

    slot_specs = []
    for i, kind in enumerate(DETECTORS):
        prefix = f"d{i+1}_"
        params = {k[len(prefix):]: v for k, v in best.params.items()
                  if k.startswith(prefix)}
        slot_specs.append((kind, params))

    return slot_specs, best.params["criterion"], best.params["decision_window"], best.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Ensemble Benchmark: Hard Generators + Full 7-DD Ensemble")
    print("=" * 70)
    print(f"Generators: {SCENARIO_GENERATORS}")
    print(f"Frequencies: {DRIFT_FREQS}")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Tolerance: {TOLERANCE} (standard), {TIGHT_TOLERANCE} (tight)")
    print(f"Suppression: {SUPPRESSION}")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors: {DETECTORS}")
    print(f"Single trials: {N_SINGLE_TRIALS}, Ensemble trials: {N_ENSEMBLE_TRIALS}")
    print(flush=True)

    all_results = {}
    start_time = time.time()

    # Resume support
    partial_path = f"{OUTPUT_DIR}/ensemble_benchmark_partial.json"
    if os.path.exists(partial_path):
        try:
            all_results = json.load(open(partial_path))
            print(f"Resumed from partial results: {list(all_results.keys())}", flush=True)
        except Exception:
            all_results = {}

    # ---- Phase 1: Single detector generalists ----
    print(f"\n{'='*70}")
    print(f"Phase 1: Single Detector Generalist Optimization")
    print(f"{'='*70}", flush=True)

    for kind in DETECTORS:
        if f"single_{kind}" in all_results:
            print(f"  {kind}: already done, skipping", flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"Detector: {kind}")
        print(f"{'='*70}", flush=True)

        try:
            params, train_f1 = optimize_single_generalist(kind)

            # Evaluate on held-out seeds
            print(f"  Evaluating on held-out seeds (tol={TOLERANCE})...", flush=True)
            sc_eval = eval_single(kind, params, SCENARIO_GENERATORS[0],
                                  DRIFT_FREQS, EVAL_SEEDS,
                                  tolerance=TOLERANCE, suppression=SUPPRESSION)
            wf_eval = eval_single(kind, params, SCENARIO_GENERATORS[1],
                                  DRIFT_FREQS, EVAL_SEEDS,
                                  tolerance=TOLERANCE, suppression=SUPPRESSION)
            sc_macro = macro_f1(sc_eval, DRIFT_FREQS)
            wf_macro = macro_f1(wf_eval, DRIFT_FREQS)
            combined = (sc_macro + wf_macro) / 2
            print(f"  SC macro F1: {sc_macro:.4f}, WF macro F1: {wf_macro:.4f}, Combined: {combined:.4f}")

            # Tight tolerance
            print(f"  Evaluating on held-out seeds (tol={TIGHT_TOLERANCE})...", flush=True)
            sc_eval_tight = eval_single(kind, params, SCENARIO_GENERATORS[0],
                                        DRIFT_FREQS, EVAL_SEEDS,
                                        tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            wf_eval_tight = eval_single(kind, params, SCENARIO_GENERATORS[1],
                                        DRIFT_FREQS, EVAL_SEEDS,
                                        tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            sc_tight = macro_f1(sc_eval_tight, DRIFT_FREQS)
            wf_tight = macro_f1(wf_eval_tight, DRIFT_FREQS)
            combined_tight = (sc_tight + wf_tight) / 2
            print(f"  SC tight F1: {sc_tight:.4f}, WF tight F1: {wf_tight:.4f}, Combined tight: {combined_tight:.4f}")

            all_results[f"single_{kind}"] = {
                "params": params,
                "train_f1": train_f1,
                "sc_f1": sc_macro,
                "wf_f1": wf_macro,
                "combined_f1": combined,
                "sc_f1_tight": sc_tight,
                "wf_f1_tight": wf_tight,
                "combined_f1_tight": combined_tight,
                "sc_detail": {str(f): sc_eval[f] for f in DRIFT_FREQS},
                "wf_detail": {str(f): wf_eval[f] for f in DRIFT_FREQS},
                "sc_detail_tight": {str(f): sc_eval_tight[f] for f in DRIFT_FREQS},
                "wf_detail_tight": {str(f): wf_eval_tight[f] for f in DRIFT_FREQS},
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"single_{kind}"] = {"error": str(e)}

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 2: Full 7-detector ensemble ----
    print(f"\n{'='*70}")
    print(f"Phase 2: Full 7-Detector Ensemble Optimization")
    print(f"{'='*70}", flush=True)

    if "ensemble" in all_results:
        print("  Ensemble: already done, skipping", flush=True)
    else:
        try:
            slot_specs, criterion, decision_window, train_f1 = optimize_ensemble()

            # Evaluate on held-out seeds
            print(f"\n  Evaluating ensemble on held-out seeds (tol={TOLERANCE})...", flush=True)
            ens_sc = eval_ensemble(slot_specs, criterion, decision_window,
                                   SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                   tolerance=TOLERANCE, suppression=SUPPRESSION)
            ens_wf = eval_ensemble(slot_specs, criterion, decision_window,
                                   SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                   tolerance=TOLERANCE, suppression=SUPPRESSION)
            ens_sc_macro = macro_f1(ens_sc, DRIFT_FREQS)
            ens_wf_macro = macro_f1(ens_wf, DRIFT_FREQS)
            ens_combined = (ens_sc_macro + ens_wf_macro) / 2
            print(f"  SC macro F1: {ens_sc_macro:.4f}, WF macro F1: {ens_wf_macro:.4f}, Combined: {ens_combined:.4f}")

            # Tight tolerance
            print(f"  Evaluating ensemble on held-out seeds (tol={TIGHT_TOLERANCE})...", flush=True)
            ens_sc_tight = eval_ensemble(slot_specs, criterion, decision_window,
                                         SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                         tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            ens_wf_tight = eval_ensemble(slot_specs, criterion, decision_window,
                                         SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                         tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            ens_sc_tight_macro = macro_f1(ens_sc_tight, DRIFT_FREQS)
            ens_wf_tight_macro = macro_f1(ens_wf_tight, DRIFT_FREQS)
            ens_combined_tight = (ens_sc_tight_macro + ens_wf_tight_macro) / 2
            print(f"  SC tight F1: {ens_sc_tight_macro:.4f}, WF tight F1: {ens_wf_tight_macro:.4f}, Combined tight: {ens_combined_tight:.4f}")

            all_results["ensemble"] = {
                "slot_specs": [(kind, params) for kind, params in slot_specs],
                "criterion": criterion,
                "decision_window": decision_window,
                "train_f1": train_f1,
                "sc_f1": ens_sc_macro,
                "wf_f1": ens_wf_macro,
                "combined_f1": ens_combined,
                "sc_f1_tight": ens_sc_tight_macro,
                "wf_f1_tight": ens_wf_tight_macro,
                "combined_f1_tight": ens_combined_tight,
                "sc_detail": {str(f): ens_sc[f] for f in DRIFT_FREQS},
                "wf_detail": {str(f): ens_wf[f] for f in DRIFT_FREQS},
                "sc_detail_tight": {str(f): ens_sc_tight[f] for f in DRIFT_FREQS},
                "wf_detail_tight": {str(f): ens_wf_tight[f] for f in DRIFT_FREQS},
            }
        except Exception as e:
            print(f"  ERROR in ensemble optimization: {e}")
            import traceback
            traceback.print_exc()
            all_results["ensemble"] = {"error": str(e)}

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 3: Also evaluate ensemble with each criterion on eval seeds ----
    print(f"\n{'='*70}")
    print(f"Phase 3: Ensemble criterion comparison")
    print(f"{'='*70}", flush=True)

    if "ensemble" in all_results and "error" not in all_results.get("ensemble", {}):
        ens_data = all_results["ensemble"]
        slot_specs = [(kind, params) for kind, params in ens_data["slot_specs"]]
        best_criterion = ens_data["criterion"]
        best_dw = ens_data["decision_window"]

        for crit in ["any", "majority", "all"]:
            key = f"ensemble_{crit}"
            if key in all_results:
                print(f"  {crit}: already done, skipping", flush=True)
                continue

            print(f"\n  Evaluating ensemble with criterion='{crit}', dw={best_dw}...", flush=True)
            sc_ev = eval_ensemble(slot_specs, crit, best_dw,
                                  SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                  tolerance=TOLERANCE, suppression=SUPPRESSION)
            wf_ev = eval_ensemble(slot_specs, crit, best_dw,
                                  SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                  tolerance=TOLERANCE, suppression=SUPPRESSION)
            sc_m = macro_f1(sc_ev, DRIFT_FREQS)
            wf_m = macro_f1(wf_ev, DRIFT_FREQS)
            comb = (sc_m + wf_m) / 2
            print(f"  SC: {sc_m:.4f}, WF: {wf_m:.4f}, Combined: {comb:.4f}", flush=True)

            all_results[key] = {
                "criterion": crit,
                "decision_window": best_dw,
                "sc_f1": sc_m,
                "wf_f1": wf_m,
                "combined_f1": comb,
                "sc_detail": {str(f): sc_ev[f] for f in DRIFT_FREQS},
                "wf_detail": {str(f): wf_ev[f] for f in DRIFT_FREQS},
            }
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ---- Final comparison table ----
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n--- Standard tolerance (tol={TOLERANCE}, supp={SUPPRESSION}) ---")
    print(f"{'Detector':<12} {'SC F1':>8} {'WF F1':>8} {'Combined':>10} {'Type':>12}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")

    best_single = -1
    best_single_name = ""
    best_ensemble = -1

    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        print(f"{kind:<12} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f} {'single':>12}")
        if comb > best_single:
            best_single = comb
            best_single_name = kind

    for crit in ["any", "majority", "all"]:
        d = all_results.get(f"ensemble_{crit}", {})
        if not d:
            continue
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        label = f"ens-{crit}"
        print(f"{label:<12} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f} {'ensemble':>12}")
        if comb > best_ensemble:
            best_ensemble = comb

    d = all_results.get("ensemble", {})
    if d and "error" not in d:
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        crit = d.get("criterion", "?")
        label = f"ens-opt({crit})"
        print(f"{label:<12} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f} {'ensemble':>12}")
        if comb > best_ensemble:
            best_ensemble = comb

    print(f"\n  Best single: {best_single_name} = {best_single:.4f}")
    print(f"  Best ensemble: {best_ensemble:.4f}")
    if best_ensemble > best_single:
        print(f"  *** ENSEMBLE WINS by {best_ensemble - best_single:.4f} ***")
    else:
        print(f"  Single detector wins by {best_single - best_ensemble:.4f}")

    # Tight tolerance table
    print(f"\n--- Tight tolerance (tol={TIGHT_TOLERANCE}, supp={SUPPRESSION}) ---")
    print(f"{'Detector':<12} {'SC F1':>8} {'WF F1':>8} {'Combined':>10}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10}")

    best_single_tight = -1
    best_ensemble_tight = -1

    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        sc = d.get("sc_f1_tight", 0)
        wf = d.get("wf_f1_tight", 0)
        comb = d.get("combined_f1_tight", 0)
        print(f"{kind:<12} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")
        if comb > best_single_tight:
            best_single_tight = comb

    d = all_results.get("ensemble", {})
    if d and "error" not in d:
        sc = d.get("sc_f1_tight", 0)
        wf = d.get("wf_f1_tight", 0)
        comb = d.get("combined_f1_tight", 0)
        crit = d.get("criterion", "?")
        print(f"ens-opt({crit}) {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")
        if comb > best_ensemble_tight:
            best_ensemble_tight = comb

    print(f"\n  Best single (tight): {best_single_tight:.4f}")
    print(f"  Best ensemble (tight): {best_ensemble_tight:.4f}")
    if best_ensemble_tight > best_single_tight:
        print(f"  *** ENSEMBLE WINS (tight) by {best_ensemble_tight - best_single_tight:.4f} ***")
    else:
        print(f"  Single detector wins (tight) by {best_single_tight - best_ensemble_tight:.4f}")

    # Per-frequency detail
    print(f"\n--- Per-frequency detail (tol={TOLERANCE}) ---")
    print(f"{'Detector':<12} {'SC f100':>8} {'SC f200':>8} {'SC f500':>8} {'SC f1k':>8} {'SC f2k':>8} | {'WF f100':>8} {'WF f200':>8} {'WF f500':>8} {'WF f1k':>8} {'WF f2k':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        sc_d = d.get("sc_detail", {})
        wf_d = d.get("wf_detail", {})
        sc_vals = [f"{sc_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS]
        wf_vals = [f"{wf_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS]
        print(f"{kind:<12} {sc_vals[0]:>8} {sc_vals[1]:>8} {sc_vals[2]:>8} {sc_vals[3]:>8} {sc_vals[4]:>8} | {wf_vals[0]:>8} {wf_vals[1]:>8} {wf_vals[2]:>8} {wf_vals[3]:>8} {wf_vals[4]:>8}")

    d = all_results.get("ensemble", {})
    if d and "error" not in d:
        sc_d = d.get("sc_detail", {})
        wf_d = d.get("wf_detail", {})
        sc_vals = [f"{sc_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS]
        wf_vals = [f"{wf_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS]
        crit = d.get("criterion", "?")
        print(f"ens({crit})  {sc_vals[0]:>8} {sc_vals[1]:>8} {sc_vals[2]:>8} {sc_vals[3]:>8} {sc_vals[4]:>8} | {wf_vals[0]:>8} {wf_vals[1]:>8} {wf_vals[2]:>8} {wf_vals[3]:>8} {wf_vals[4]:>8}")

    # Save CSV
    with open(f"{OUTPUT_DIR}/ensemble_benchmark_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "type", "criterion", "sc_f1_tol100", "wf_f1_tol100",
                     "combined_tol100", "sc_f1_tol20", "wf_f1_tol20", "combined_tol20"])
        for kind in DETECTORS:
            d = all_results.get(f"single_{kind}", {})
            if "error" in d or not d:
                continue
            w.writerow([kind, "single", "-",
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}",
                        f"{d.get('sc_f1_tight',0):.4f}", f"{d.get('wf_f1_tight',0):.4f}",
                        f"{d.get('combined_f1_tight',0):.4f}"])
        for crit in ["any", "majority", "all"]:
            d = all_results.get(f"ensemble_{crit}", {})
            if not d:
                continue
            w.writerow([f"ensemble_{crit}", "ensemble", crit,
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}", "0", "0", "0"])
        d = all_results.get("ensemble", {})
        if d and "error" not in d:
            w.writerow(["ensemble_opt", "ensemble", d.get("criterion", "?"),
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}",
                        f"{d.get('sc_f1_tight',0):.4f}", f"{d.get('wf_f1_tight',0):.4f}",
                        f"{d.get('combined_f1_tight',0):.4f}"])

    # Save full results
    with open(f"{OUTPUT_DIR}/ensemble_benchmark_final.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {OUTPUT_DIR}/ensemble_benchmark_*")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
