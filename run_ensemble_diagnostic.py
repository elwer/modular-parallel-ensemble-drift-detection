#!/usr/bin/env python3
"""Diagnostic: why does the ensemble underperform on hard generators?

Identified issues in run_expert_ensemble_hard.py:
  1. ARCHITECTURE: Cross-DD ensemble runs ALL 5 frequency-specific experts
     simultaneously on EVERY frequency. A freq=100 expert fires on freq=2000
     streams → false positives. "any" amplifies FPs, "all" never fires.
  2. FIXED TOL/SUPP: Original benchmark used freq-dependent tol=freq//10,
     supp=freq//2. Current run uses fixed tol=100, supp=50 for all freqs.
     For freq=2000, tol=100 is too strict (5% of gap); supp=50 too short.
  3. VOTING: Only tries any/majority/all with dw 1-5. Missing at_least_k
     which worked in coord_descent (at_least_4, dw=16 → F1=0.668).
  4. EXPERT OVERFITTING: Experts optimized on 1 seed, 10 trials only.

This script tests fixes:
  Phase 1: Generalist ensemble with freq-dependent tol/supp, all k×dw combos
  Phase 2: Generalist ensemble with fixed tol/supp (for comparison)
  Phase 3: Joint 2-detector ensemble optimization (like original winning approach)
  Phase 4: Coordinate descent with more trials and freq-dependent tol/supp
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
DRIFT_FREQS = [100, 200, 500, 1000, 2000]
K = len(DRIFT_FREQS)
STREAM_LENGTH = 5000
SEED = 1337
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 90
MAX_WINDOW = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)  # 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_stream_local(generator_name, drift_frequency, stream_length, seed):
    cls = GENERATORS[generator_name]
    return cls(drift_frequency=drift_frequency,
               stream_length=stream_length,
               seed=seed)


def freq_tolerance(freq):
    """Frequency-dependent tolerance (original benchmark style)."""
    return max(10, freq // 10)


def freq_suppression(freq):
    """Frequency-dependent suppression (original benchmark style)."""
    return max(25, freq // 2)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _ensemble_worker(queue, slot_specs, ensemble_criterion, decision_window,
                     freq, stream_seed, s_idx, tolerance, suppression, generator):
    try:
        stream = build_stream_local(generator, freq, STREAM_LENGTH, stream_seed)
        known = list(stream.drifts)
        detectors = []
        names = []
        for i, (kind, params) in enumerate(slot_specs):
            n_samples_key = ("n_samples" if "n_samples" in params
                             else "n_reference_samples")
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_ensemble_mp(slot_specs, criterion, dw, generator, freq, seeds,
                     tolerance, suppression):
    f1s = []
    for i, seed in enumerate(seeds):
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        proc = ctx.Process(target=_ensemble_worker,
                           args=(queue, slot_specs, criterion, dw, freq, seed, i,
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
                f1s.append(result[1] if result[0] == "ok" else 0.0)
            except Exception:
                f1s.append(0.0)
    return f1s


def _run_single_mp(kind, params, generator, freq, seeds, tolerance, suppression):
    f1s = []
    for i, seed in enumerate(seeds):
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        proc = ctx.Process(target=_single_worker,
                           args=(queue, kind, params, freq, seed, i,
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
                f1s.append(result[1] if result[0] == "ok" else 0.0)
            except Exception:
                f1s.append(0.0)
    return f1s


def eval_ensemble_freq_aware(slot_specs, criterion, dw, generators, freqs, seeds,
                             use_freq_dep=True):
    """Evaluate ensemble with per-frequency tolerance/suppression."""
    results = {}
    for freq in freqs:
        tol = freq_tolerance(freq) if use_freq_dep else 100
        supp = freq_suppression(freq) if use_freq_dep else 50
        f1s = []
        for generator in generators:
            f1s.extend(_run_ensemble_mp(slot_specs, criterion, dw, generator,
                                        freq, seeds, tol, supp))
        results[freq] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def eval_single_freq_aware(kind, params, generators, freqs, seeds,
                           use_freq_dep=True):
    """Evaluate single detector with per-frequency tolerance/suppression."""
    results = {}
    for freq in freqs:
        tol = freq_tolerance(freq) if use_freq_dep else 100
        supp = freq_suppression(freq) if use_freq_dep else 50
        f1s = []
        for generator in generators:
            f1s.extend(_run_single_mp(kind, params, generator, freq, seeds,
                                      tol, supp))
        results[freq] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def macro_f1(per_freq, freqs):
    return sum(per_freq[f] for f in freqs) / len(freqs)


def combined_f1(sc_results, wf_results, freqs):
    sc_macro = macro_f1(sc_results, freqs)
    wf_macro = macro_f1(wf_results, freqs)
    return (sc_macro + wf_macro) / 2


# ---------------------------------------------------------------------------
# Phase 1 & 2: Grid search over k × dw for generalist ensemble
# ---------------------------------------------------------------------------

def grid_search_voting(slot_specs, label, use_freq_dep, results_dict):
    """Try all at_least_k × dw combinations on eval seeds."""
    n = len(slot_specs)
    k_values = list(range(1, n + 1))
    dw_values = [1, 3, 5, 7, 10, 15, 20]

    tol_label = "freq_dep" if use_freq_dep else "fixed"
    print(f"\n{'='*70}")
    print(f"Grid Search: {label} (tol={tol_label})")
    print(f"  {n} detectors, k=[1..{n}], dw={dw_values}")
    print(f"  Total configs: {len(k_values) * len(dw_values)}")
    print(f"{'='*70}", flush=True)

    best_combined = -1
    best_config = None
    all_configs = []

    for k_val in k_values:
        criterion = f"at_least_{k_val}"
        for dw in dw_values:
            sc_res = eval_ensemble_freq_aware(
                slot_specs, criterion, dw,
                [SCENARIO_GENERATORS[0]], DRIFT_FREQS, EVAL_SEEDS,
                use_freq_dep=use_freq_dep)
            wf_res = eval_ensemble_freq_aware(
                slot_specs, criterion, dw,
                [SCENARIO_GENERATORS[1]], DRIFT_FREQS, EVAL_SEEDS,
                use_freq_dep=use_freq_dep)
            comb = combined_f1(sc_res, wf_res, DRIFT_FREQS)
            sc_macro = macro_f1(sc_res, DRIFT_FREQS)
            wf_macro = macro_f1(wf_res, DRIFT_FREQS)

            config_key = f"{label}_{tol_label}_k{k_val}_dw{dw}"
            all_configs.append({
                "label": label,
                "tol": tol_label,
                "criterion": criterion,
                "k": k_val,
                "dw": dw,
                "sc_f1": sc_macro,
                "wf_f1": wf_macro,
                "combined_f1": comb,
                "sc_detail": {str(f): sc_res[f] for f in DRIFT_FREQS},
                "wf_detail": {str(f): wf_res[f] for f in DRIFT_FREQS},
            })

            print(f"  k={k_val} dw={dw:2d}: SC={sc_macro:.4f} WF={wf_macro:.4f} "
                  f"Comb={comb:.4f}", flush=True)

            if comb > best_combined:
                best_combined = comb
                best_config = config_key

    print(f"\n  Best: {best_config} = {best_combined:.4f}")

    results_dict[f"grid_{label}_{tol_label}"] = {
        "best_config": best_config,
        "best_f1": best_combined,
        "all_configs": all_configs,
    }
    return best_combined, best_config


# ---------------------------------------------------------------------------
# Phase 3: Joint 2-detector ensemble optimization
# ---------------------------------------------------------------------------

def optimize_joint_pair(det_a, det_b, n_trials, use_freq_dep):
    """Jointly optimize both detectors' params + voting criterion + dw."""
    print(f"\n  --- Joint optimization: {det_a}+{det_b} ({n_trials} trials) ---",
          flush=True)
    t0 = time.time()

    def objective(trial):
        params_a = _suggest_detector_params(trial, "a_", det_a, max_window=MAX_WINDOW)
        params_b = _suggest_detector_params(trial, "b_", det_b, max_window=MAX_WINDOW)
        k = trial.suggest_int("k", 1, 2)
        dw = trial.suggest_int("decision_window", 1, 20)
        criterion = f"at_least_{k}"
        slot_specs = [(det_a, params_a), (det_b, params_b)]

        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                tol = freq_tolerance(freq) if use_freq_dep else 100
                supp = freq_suppression(freq) if use_freq_dep else 50
                f1s.extend(_run_ensemble_mp(slot_specs, criterion, dw,
                                            generator, freq, TRAIN_SEEDS,
                                            tol, supp))
        return sum(f1s) / len(f1s) if f1s else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_k = best.params["k"]
    best_dw = best.params["decision_window"]
    best_criterion = f"at_least_{best_k}"
    best_params_a = _suggest_detector_params(best, "a_", det_a, max_window=MAX_WINDOW)
    best_params_b = _suggest_detector_params(best, "b_", det_b, max_window=MAX_WINDOW)
    slot_specs = [(det_a, best_params_a), (det_b, best_params_b)]

    print(f"  Train F1: {best.value:.4f}, k={best_k}, dw={best_dw}")
    print(f"  {det_a}: {best_params_a}")
    print(f"  {det_b}: {best_params_b}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)

    # Evaluate on held-out seeds
    sc_res = eval_ensemble_freq_aware(
        slot_specs, best_criterion, best_dw,
        [SCENARIO_GENERATORS[0]], DRIFT_FREQS, EVAL_SEEDS,
        use_freq_dep=use_freq_dep)
    wf_res = eval_ensemble_freq_aware(
        slot_specs, best_criterion, best_dw,
        [SCENARIO_GENERATORS[1]], DRIFT_FREQS, EVAL_SEEDS,
        use_freq_dep=use_freq_dep)
    comb = combined_f1(sc_res, wf_res, DRIFT_FREQS)
    sc_macro = macro_f1(sc_res, DRIFT_FREQS)
    wf_macro = macro_f1(wf_res, DRIFT_FREQS)
    print(f"  Eval: SC={sc_macro:.4f} WF={wf_macro:.4f} Comb={comb:.4f}", flush=True)

    return {
        "det_a": det_a,
        "det_b": det_b,
        "params_a": best_params_a,
        "params_b": best_params_b,
        "criterion": best_criterion,
        "k": best_k,
        "dw": best_dw,
        "train_f1": best.value,
        "sc_f1": sc_macro,
        "wf_f1": wf_macro,
        "combined_f1": comb,
        "sc_detail": {str(f): sc_res[f] for f in DRIFT_FREQS},
        "wf_detail": {str(f): wf_res[f] for f in DRIFT_FREQS},
    }


# ---------------------------------------------------------------------------
# Phase 4: Coordinate descent with freq-dependent tol/supp
# ---------------------------------------------------------------------------

def coordinate_descent(initial_specs, n_rounds, n_trials_per_det, use_freq_dep):
    """Coordinate descent: optimize one detector at a time, then re-optimize k/dw."""
    slot_specs = list(initial_specs)
    n = len(slot_specs)
    tol_label = "freq_dep" if use_freq_dep else "fixed"

    # Initial k/dw optimization
    print(f"\n  Initial k/dw optimization...", flush=True)

    def kd_objective(trial):
        k = trial.suggest_int("k", 1, n)
        dw = trial.suggest_int("decision_window", 1, 20)
        criterion = f"at_least_{k}"
        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                tol = freq_tolerance(freq) if use_freq_dep else 100
                supp = freq_suppression(freq) if use_freq_dep else 50
                f1s.extend(_run_ensemble_mp(slot_specs, criterion, dw,
                                            generator, freq, TRAIN_SEEDS,
                                            tol, supp))
        return sum(f1s) / len(f1s) if f1s else 0.0

    kd_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    kd_study.optimize(kd_objective, n_trials=15, show_progress_bar=True)
    best_k = kd_study.best_trial.params["k"]
    best_dw = kd_study.best_trial.params["decision_window"]
    best_criterion = f"at_least_{best_k}"
    current_best_f1 = kd_study.best_value
    print(f"  Initial k={best_k}, dw={best_dw}, F1={current_best_f1:.4f}", flush=True)

    for round_idx in range(n_rounds):
        print(f"\n  Round {round_idx + 1}/{n_rounds} (k={best_k}, dw={best_dw})",
              flush=True)

        for det_idx, (kind, current_params) in enumerate(slot_specs):
            print(f"\n    [{det_idx}] Optimizing {kind}...", flush=True)
            t0 = time.time()
            fixed_specs = list(slot_specs)

            def objective(trial):
                new_params = _suggest_detector_params(
                    trial, f"r{round_idx}_d{det_idx}_", kind, max_window=MAX_WINDOW)
                trial_specs = list(fixed_specs)
                trial_specs[det_idx] = (kind, new_params)
                f1s = []
                for generator in SCENARIO_GENERATORS:
                    for freq in DRIFT_FREQS:
                        tol = freq_tolerance(freq) if use_freq_dep else 100
                        supp = freq_suppression(freq) if use_freq_dep else 50
                        f1s.extend(_run_ensemble_mp(trial_specs, best_criterion,
                                                    best_dw, generator, freq,
                                                    TRAIN_SEEDS, tol, supp))
                return sum(f1s) / len(f1s) if f1s else 0.0

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=SEED + det_idx + 100 * round_idx))
            study.optimize(objective, n_trials=n_trials_per_det,
                           show_progress_bar=True)

            best = study.best_trial
            best_params = _suggest_detector_params(
                best, f"r{round_idx}_d{det_idx}_", kind, max_window=MAX_WINDOW)

            if best.value > current_best_f1:
                print(f"    {kind}: F1={best.value:.4f} > {current_best_f1:.4f} ACCEPTED")
                slot_specs[det_idx] = (kind, best_params)
                current_best_f1 = best.value
            else:
                print(f"    {kind}: F1={best.value:.4f} <= {current_best_f1:.4f} REJECTED")

            print(f"    Time: {time.time() - t0:.1f}s", flush=True)

        # Re-optimize k/dw
        print(f"\n    Re-optimizing k/dw...", flush=True)

        def kd_objective2(trial):
            k = trial.suggest_int("k", 1, n)
            dw = trial.suggest_int("decision_window", 1, 20)
            criterion = f"at_least_{k}"
            f1s = []
            for generator in SCENARIO_GENERATORS:
                for freq in DRIFT_FREQS:
                    tol = freq_tolerance(freq) if use_freq_dep else 100
                    supp = freq_suppression(freq) if use_freq_dep else 50
                    f1s.extend(_run_ensemble_mp(slot_specs, criterion, dw,
                                                generator, freq, TRAIN_SEEDS,
                                                tol, supp))
            return sum(f1s) / len(f1s) if f1s else 0.0

        kd_study2 = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + 999 + round_idx))
        kd_study2.optimize(kd_objective2, n_trials=10, show_progress_bar=True)
        if kd_study2.best_value > current_best_f1:
            best_k = kd_study2.best_trial.params["k"]
            best_dw = kd_study2.best_trial.params["decision_window"]
            best_criterion = f"at_least_{best_k}"
            current_best_f1 = kd_study2.best_value
            print(f"    Updated k={best_k}, dw={best_dw}, F1={current_best_f1:.4f}",
                  flush=True)
        else:
            print(f"    k/dw unchanged", flush=True)

    # Final evaluation
    sc_res = eval_ensemble_freq_aware(
        slot_specs, best_criterion, best_dw,
        [SCENARIO_GENERATORS[0]], DRIFT_FREQS, EVAL_SEEDS,
        use_freq_dep=use_freq_dep)
    wf_res = eval_ensemble_freq_aware(
        slot_specs, best_criterion, best_dw,
        [SCENARIO_GENERATORS[1]], DRIFT_FREQS, EVAL_SEEDS,
        use_freq_dep=use_freq_dep)
    comb = combined_f1(sc_res, wf_res, DRIFT_FREQS)
    sc_macro = macro_f1(sc_res, DRIFT_FREQS)
    wf_macro = macro_f1(wf_res, DRIFT_FREQS)

    print(f"\n  Final ensemble: k={best_k}, dw={best_dw}")
    print(f"  Eval: SC={sc_macro:.4f} WF={wf_macro:.4f} Comb={comb:.4f}", flush=True)

    return {
        "slot_specs": [(k, p) for k, p in slot_specs],
        "criterion": best_criterion,
        "k": best_k,
        "dw": best_dw,
        "train_f1": current_best_f1,
        "sc_f1": sc_macro,
        "wf_f1": wf_macro,
        "combined_f1": comb,
        "sc_detail": {str(f): sc_res[f] for f in DRIFT_FREQS},
        "wf_detail": {str(f): wf_res[f] for f in DRIFT_FREQS},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Ensemble Diagnostic: Why does the ensemble underperform?")
    print("=" * 70)
    print(f"Generators: {SCENARIO_GENERATORS}")
    print(f"Frequencies: {DRIFT_FREQS}")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors: {DETECTORS}")
    print(f"Freq-dep tolerance: {[freq_tolerance(f) for f in DRIFT_FREQS]}")
    print(f"Freq-dep suppression: {[freq_suppression(f) for f in DRIFT_FREQS]}")
    print(f"Fixed tolerance: 100, Fixed suppression: 50")
    print(flush=True)

    # Load generalist params from completed run
    results_path = f"{OUTPUT_DIR}/expert_ensemble_hard_final.json"
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        return
    expert_results = json.load(open(results_path))

    # Build generalist slot specs
    gen_specs = []
    gen_single_f1s = {}
    for kind in DETECTORS:
        d = expert_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            print(f"  WARNING: {kind} generalist not found, skipping")
            continue
        params = d["params"]
        gen_specs.append((kind, params))
        gen_single_f1s[kind] = d.get("combined_f1", 0)
        print(f"  {kind}: params={params}, single F1={d.get('combined_f1', 0):.4f}")

    print(f"\nLoaded {len(gen_specs)} generalist configs")

    # Also load expert params for cross-DD comparison
    expert_specs_by_freq = {}
    for freq in DRIFT_FREQS:
        freq_key = str(freq)
        best_f1 = -1
        best_kind = None
        best_params = None
        for kind in DETECTORS:
            exp = expert_results.get("experts", {}).get(kind, {}).get(freq_key, {})
            if "error" in exp or not exp:
                continue
            if exp.get("train_f1", 0) > best_f1:
                best_f1 = exp["train_f1"]
                best_kind = kind
                best_params = exp["params"]
        if best_kind:
            expert_specs_by_freq[freq] = (best_kind, best_params)
            print(f"  Expert freq={freq}: {best_kind} (train F1={best_f1:.4f})")

    # Build cross-DD expert slot specs (one per frequency)
    cross_dd_specs = [(kind, params) for freq, (kind, params) in
                      sorted(expert_specs_by_freq.items())]

    all_results = {}
    start_time = time.time()

    # ---- Phase 1: Generalist ensemble grid search (freq-dep tol/supp) ----
    print(f"\n{'#'*70}")
    print(f"# Phase 1: Generalist Ensemble Grid Search (freq-dep tol/supp)")
    print(f"{'#'*70}", flush=True)

    best_gen_freqdep, best_gen_freqdep_config = grid_search_voting(
        gen_specs, "generalist", use_freq_dep=True, results_dict=all_results)

    # ---- Phase 1b: Cross-DD expert ensemble grid search (freq-dep tol/supp) ----
    print(f"\n{'#'*70}")
    print(f"# Phase 1b: Cross-DD Expert Ensemble Grid Search (freq-dep tol/supp)")
    print(f"{'#'*70}", flush=True)

    if len(cross_dd_specs) > 0:
        best_expert_freqdep, best_expert_freqdep_config = grid_search_voting(
            cross_dd_specs, "cross_dd_expert", use_freq_dep=True,
            results_dict=all_results)

    # ---- Phase 2: Generalist ensemble grid search (fixed tol/supp) ----
    print(f"\n{'#'*70}")
    print(f"# Phase 2: Generalist Ensemble Grid Search (fixed tol/supp)")
    print(f"{'#'*70}", flush=True)

    best_gen_fixed, best_gen_fixed_config = grid_search_voting(
        gen_specs, "generalist", use_freq_dep=False, results_dict=all_results)

    # ---- Phase 2b: Cross-DD expert ensemble grid search (fixed tol/supp) ----
    print(f"\n{'#'*70}")
    print(f"# Phase 2b: Cross-DD Expert Ensemble Grid Search (fixed tol/supp)")
    print(f"{'#'*70}", flush=True)

    if len(cross_dd_specs) > 0:
        best_expert_fixed, best_expert_fixed_config = grid_search_voting(
            cross_dd_specs, "cross_dd_expert", use_freq_dep=False,
            results_dict=all_results)

    # Save partial results
    partial_path = f"{OUTPUT_DIR}/ensemble_diagnostic_partial.json"
    with open(partial_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 3: Joint 2-detector ensemble optimization ----
    print(f"\n{'#'*70}")
    print(f"# Phase 3: Joint 2-Detector Ensemble Optimization (freq-dep tol/supp)")
    print(f"{'#'*70}", flush=True)

    # Select promising pairs based on generalist F1
    sorted_dets = sorted(gen_single_f1s.items(), key=lambda x: -x[1])
    top_dets = [d[0] for d in sorted_dets[:5]]
    print(f"  Top detectors by generalist F1: {top_dets}")
    print(f"  Pairs to optimize:")

    pairs = []
    for i in range(len(top_dets)):
        for j in range(i + 1, len(top_dets)):
            pairs.append((top_dets[i], top_dets[j]))
    print(f"  {pairs}")

    joint_results = {}
    for det_a, det_b in pairs:
        pair_key = f"{det_a}+{det_b}"
        print(f"\n  Optimizing pair: {pair_key}", flush=True)
        result = optimize_joint_pair(det_a, det_b, n_trials=50, use_freq_dep=True)
        joint_results[pair_key] = result

        # Save partial
        all_results["joint_pairs"] = joint_results
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 4: Coordinate descent (freq-dep tol/supp) ----
    print(f"\n{'#'*70}")
    print(f"# Phase 4: Coordinate Descent (freq-dep tol/supp)")
    print(f"{'#'*70}", flush=True)

    cd_result = coordinate_descent(
        gen_specs, n_rounds=2, n_trials_per_det=12, use_freq_dep=True)
    all_results["coord_descent_freqdep"] = cd_result

    with open(partial_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 5: Re-evaluate generalists with freq-dep tol/supp ----
    print(f"\n{'#'*70}")
    print(f"# Phase 5: Re-evaluate Generalists with freq-dep tol/supp")
    print(f"{'#'*70}", flush=True)

    gen_freqdep_results = {}
    for kind in DETECTORS:
        d = expert_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            continue
        params = d["params"]
        sc_res = eval_single_freq_aware(kind, params, [SCENARIO_GENERATORS[0]],
                                         DRIFT_FREQS, EVAL_SEEDS, use_freq_dep=True)
        wf_res = eval_single_freq_aware(kind, params, [SCENARIO_GENERATORS[1]],
                                         DRIFT_FREQS, EVAL_SEEDS, use_freq_dep=True)
        comb = combined_f1(sc_res, wf_res, DRIFT_FREQS)
        sc_macro = macro_f1(sc_res, DRIFT_FREQS)
        wf_macro = macro_f1(wf_res, DRIFT_FREQS)
        print(f"  Gen-{kind}: SC={sc_macro:.4f} WF={wf_macro:.4f} Comb={comb:.4f}")
        gen_freqdep_results[kind] = {
            "sc_f1": sc_macro,
            "wf_f1": wf_macro,
            "combined_f1": comb,
            "sc_detail": {str(f): sc_res[f] for f in DRIFT_FREQS},
            "wf_detail": {str(f): wf_res[f] for f in DRIFT_FREQS},
        }
    all_results["generalists_freqdep"] = gen_freqdep_results

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n--- Generalists (freq-dep tol/supp) ---")
    best_single_freqdep = -1
    best_single_name_freqdep = ""
    for kind, d in gen_freqdep_results.items():
        print(f"  Gen-{kind}: {d['combined_f1']:.4f}")
        if d["combined_f1"] > best_single_freqdep:
            best_single_freqdep = d["combined_f1"]
            best_single_name_freqdep = kind

    print(f"\n  Best single (freq-dep): {best_single_name_freqdep} = {best_single_freqdep:.4f}")

    print(f"\n--- Generalists (fixed tol/supp, from original run) ---")
    best_single_fixed = -1
    best_single_name_fixed = ""
    for kind in DETECTORS:
        d = expert_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            continue
        comb = d.get("combined_f1", 0)
        print(f"  Gen-{kind}: {comb:.4f}")
        if comb > best_single_fixed:
            best_single_fixed = comb
            best_single_name_fixed = kind
    print(f"  Best single (fixed): {best_single_name_fixed} = {best_single_fixed:.4f}")

    print(f"\n--- Ensemble Results ---")
    print(f"  Grid generalist (freq-dep): {best_gen_freqdep:.4f} ({best_gen_freqdep_config})")
    if len(cross_dd_specs) > 0:
        print(f"  Grid cross-DD expert (freq-dep): {best_expert_freqdep:.4f} ({best_expert_freqdep_config})")
    print(f"  Grid generalist (fixed): {best_gen_fixed:.4f} ({best_gen_fixed_config})")
    if len(cross_dd_specs) > 0:
        print(f"  Grid cross-DD expert (fixed): {best_expert_fixed:.4f} ({best_expert_fixed_config})")

    best_joint = -1
    best_joint_name = ""
    for pair_key, d in joint_results.items():
        print(f"  Joint {pair_key}: {d['combined_f1']:.4f} (k={d['k']}, dw={d['dw']})")
        if d["combined_f1"] > best_joint:
            best_joint = d["combined_f1"]
            best_joint_name = pair_key
    print(f"  Best joint: {best_joint_name} = {best_joint:.4f}")

    print(f"  Coord descent (freq-dep): {cd_result['combined_f1']:.4f} "
          f"(k={cd_result['k']}, dw={cd_result['dw']})")

    print(f"\n--- Comparison ---")
    best_ensemble = max(best_gen_freqdep, best_gen_fixed, best_joint,
                        cd_result["combined_f1"])
    if len(cross_dd_specs) > 0:
        best_ensemble = max(best_ensemble, best_expert_freqdep, best_expert_fixed)

    print(f"  Best single (freq-dep): {best_single_freqdep:.4f}")
    print(f"  Best ensemble:          {best_ensemble:.4f}")
    if best_ensemble > best_single_freqdep:
        print(f"  *** ENSEMBLE WINS by {best_ensemble - best_single_freqdep:.4f} ***")
    else:
        print(f"  Single wins by {best_single_freqdep - best_ensemble:.4f}")

    # Save final results
    final_path = f"{OUTPUT_DIR}/ensemble_diagnostic_final.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {OUTPUT_DIR}/ensemble_diagnostic_*")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
