#!/usr/bin/env python3
"""Overnight benchmark: all 7 detectors, generalist vs expert vs multi-DD ensemble.
Uses multiprocessing for hard per-stream timeouts.
Saves partial results after each phase.
"""
import sys
import json
import time
import csv
import os
import itertools
import multiprocessing as mp
import optuna
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _run_one_stream, _suggest_detector_params, _instantiate,
    build_stream, CANDIDATES, CLASS_PATH, MAX_WINDOW_FRACTION, _cap,
    _f1_from_counts
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections

# --- Config ---
GENERATOR = "SineClusters"
DRIFT_FREQS = [200, 500, 1000, 2000]
STREAM_LENGTH = 4000  # fixed: ensures freq=2000 has 1 drift
N_TRIALS = 25
SEED = 1337
TOLERANCES = [f // 10 for f in DRIFT_FREQS]  # 20, 50, 100, 200
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
# Order by expected speed: fastest first
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 15  # seconds per stream evaluation
EARLY_STOP_F1 = 1.0  # stop optimization when train F1 reaches this
# Limit ensemble search to pairs of top-N detectors (by expert macro F1)
ENSEMBLE_TOP_N = 5
ENSEMBLE_MAX_SIZE = 2  # only pairs, not triples

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-stream worker (multiprocessing for hard timeout)
# ---------------------------------------------------------------------------

def _single_stream_worker(queue, kind, params, freq, stream_seed, s_idx,
                          tolerance, suppression, stream_length, generator):
    try:
        n_samples_key = "n_samples" if "n_samples" in params else "n_reference_samples"
        recent_size = params.get(n_samples_key, 200)
        tp, fp, fn, delay, f1, prec, rec, n_known = _run_one_stream(
            generator_name=generator,
            drift_frequency=freq,
            stream_length=stream_length,
            stream_seed=stream_seed,
            tolerance=tolerance,
            slot_specs=[(kind, params)],
            detector_seed_base=SEED,
            s_idx=s_idx,
            detector_criterion="any",
            ensemble_criterion="any",
            decision_window=1,
            suppression_window=suppression,
            recent_samples_size=recent_size,
        )
        queue.put(("ok", tp, fp, fn, f1, prec, rec, n_known))
    except Exception as e:
        queue.put(("error", str(e)))


def run_single(kind, params, freq, stream_seed, s_idx,
               tolerance, suppression):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_single_stream_worker,
                       args=(queue, kind, params, freq, stream_seed, s_idx,
                             tolerance, suppression, STREAM_LENGTH, GENERATOR))
    proc.start()
    proc.join(PER_STREAM_TIMEOUT)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return 0.0
    try:
        result = queue.get_nowait()
        return result[4] if result[0] == "ok" else 0.0
    except:
        return 0.0


# ---------------------------------------------------------------------------
# Multi-detector ensemble stream worker
# ---------------------------------------------------------------------------

def _ensemble_stream_worker(queue, slot_specs, ensemble_criterion, freq,
                            stream_seed, s_idx, tolerance, suppression,
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
            decision_window=1,
        )
        dets = apply_suppression(raw_ensemble, suppression)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        queue.put(("ok", tp, fp, fn, f1, prec, rec, len(known)))
    except Exception as e:
        queue.put(("error", str(e)))


def run_ensemble_single(slot_specs, ensemble_criterion, freq, stream_seed,
                        s_idx, tolerance, suppression):
    """Run a multi-detector ensemble on one stream with hard timeout."""
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_ensemble_stream_worker,
                       args=(queue, slot_specs, ensemble_criterion, freq,
                             stream_seed, s_idx, tolerance, suppression,
                             STREAM_LENGTH, GENERATOR))
    proc.start()
    proc.join(PER_STREAM_TIMEOUT * 2)  # ensembles get double timeout
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return 0.0
    try:
        result = queue.get_nowait()
        return result[4] if result[0] == "ok" else 0.0
    except:
        return 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_params(kind, params, freqs, seeds):
    results_per_freq = {}
    for freq, tol in zip(freqs, [f // 10 for f in freqs]):
        supp = freq // 2
        f1s = []
        seed_results = []
        for i, seed in enumerate(seeds):
            ctx = mp.get_context("fork")
            queue = ctx.Queue()
            proc = ctx.Process(target=_single_stream_worker,
                               args=(queue, kind, params, freq, seed, 500 + i,
                                     tol, supp, STREAM_LENGTH, GENERATOR))
            proc.start()
            proc.join(PER_STREAM_TIMEOUT * 2)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                seed_results.append({"seed": seed, "f1": 0.0, "error": "timeout"})
                f1s.append(0.0)
            else:
                try:
                    result = queue.get_nowait()
                    if result[0] == "ok":
                        _, tp, fp, fn, f1, prec, rec, n_known = result
                        seed_results.append({
                            "seed": seed, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                            "precision": prec, "recall": rec, "n_drifts": n_known,
                        })
                        f1s.append(f1)
                    else:
                        seed_results.append({"seed": seed, "f1": 0.0, "error": result[1]})
                        f1s.append(0.0)
                except:
                    seed_results.append({"seed": seed, "f1": 0.0, "error": "no result"})
                    f1s.append(0.0)
        results_per_freq[freq] = {
            "seeds": seed_results,
            "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    return results_per_freq


def eval_ensemble(slot_specs, ensemble_criterion, freqs, seeds):
    """Evaluate a multi-DD ensemble on given freqs and seeds."""
    results_per_freq = {}
    for freq in freqs:
        tol = freq // 10
        supp = freq // 2
        f1s = []
        seed_results = []
        for i, seed in enumerate(seeds):
            f1 = run_ensemble_single(slot_specs, ensemble_criterion,
                                     freq, seed, 700 + i, tol, supp)
            seed_results.append({"seed": seed, "f1": f1})
            f1s.append(f1)
        results_per_freq[freq] = {
            "seeds": seed_results,
            "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    return results_per_freq


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def make_generalist_objective(kind):
    def objective(trial):
        max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
            supp = freq // 2
            for s_idx, seed in enumerate(TRAIN_SEEDS):
                f1 = run_single(kind, params, freq, seed, s_idx, tol, supp)
                f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def make_expert_objective(kind, freq, tol):
    supp = freq // 2
    max_window = int(freq * MAX_WINDOW_FRACTION)
    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for s_idx, seed in enumerate(TRAIN_SEEDS):
            f1 = run_single(kind, params, freq, seed, s_idx, tol, supp)
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def _early_stop_cb(study, trial):
    if study.best_value >= EARLY_STOP_F1:
        study.stop()


# ---------------------------------------------------------------------------
# Phase 1: Single-detector optimization
# ---------------------------------------------------------------------------

def optimize_detector(kind):
    print(f"\n{'='*70}")
    print(f"Detector: {kind}")
    print(f"{'='*70}", flush=True)

    det_results = {"generalist": {}, "experts": {}}

    # --- Generalist ---
    print(f"\n  --- Generalist ({kind}) ---", flush=True)
    t0 = time.time()
    gen_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    gen_study.optimize(make_generalist_objective(kind),
                       n_trials=N_TRIALS, show_progress_bar=True,
                       callbacks=[_early_stop_cb])
    gen_best = gen_study.best_trial
    print(f"  Generalist train F1: {gen_best.value:.4f}")
    print(f"  Generalist params: {gen_best.params}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)

    gen_eval = eval_params(kind, gen_best.params, DRIFT_FREQS, EVAL_SEEDS)
    gen_eval_f1s = [gen_eval[f]["mean_f1"] for f in DRIFT_FREQS]
    gen_macro = sum(gen_eval_f1s) / len(gen_eval_f1s)
    print(f"  Generalist eval macro F1: {gen_macro:.4f}")
    for f in DRIFT_FREQS:
        print(f"    freq={f}: F1={gen_eval[f]['mean_f1']:.4f}", flush=True)

    det_results["generalist"] = {
        "train_f1": gen_best.value,
        "params": gen_best.params,
        "eval": gen_eval,
        "eval_macro_f1": gen_macro,
    }

    # --- Experts ---
    for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
        print(f"\n  --- Expert {kind} for freq={freq} (tol={tol}) ---", flush=True)
        t0 = time.time()
        exp_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        exp_study.optimize(make_expert_objective(kind, freq, tol),
                           n_trials=N_TRIALS, show_progress_bar=True,
                           callbacks=[_early_stop_cb])
        exp_best = exp_study.best_trial
        print(f"  Expert train F1: {exp_best.value:.4f}")
        print(f"  Expert params: {exp_best.params}")
        print(f"  Time: {time.time() - t0:.1f}s", flush=True)

        exp_eval = eval_params(kind, exp_best.params, [freq], EVAL_SEEDS)
        exp_f1 = exp_eval[freq]["mean_f1"]
        print(f"  Expert eval F1 on freq={freq}: {exp_f1:.4f}", flush=True)

        det_results["experts"][freq] = {
            "train_f1": exp_best.value,
            "params": exp_best.params,
            "eval_f1": exp_f1,
            "eval_detail": exp_eval[freq],
        }

    exp_eval_f1s = [det_results["experts"][f]["eval_f1"] for f in DRIFT_FREQS]
    exp_macro = sum(exp_eval_f1s) / len(exp_eval_f1s)
    det_results["expert_macro_f1"] = exp_macro
    print(f"\n  >>> {kind}: Generalist={gen_macro:.4f}, Expert={exp_macro:.4f}, "
          f"Delta={exp_macro - gen_macro:+.4f}", flush=True)

    return det_results


# ---------------------------------------------------------------------------
# Phase 2: Multi-DD ensemble search
# ---------------------------------------------------------------------------

def search_multi_dd_ensembles(results):
    """Try all pairs and triples of detectors using their expert params.
    For each frequency, pick the best ensemble configuration."""
    print(f"\n{'='*70}")
    print(f"Phase 2: Multi-DD Ensemble Search")
    print(f"{'='*70}", flush=True)

    # Collect best expert params per detector per freq
    expert_params = {}  # {dd: {freq: params}}
    for dd in DETECTORS:
        if dd not in results["detectors"] or "error" in results["detectors"][dd]:
            continue
        det = results["detectors"][dd]
        expert_params[dd] = {}
        for freq in DRIFT_FREQS:
            exp = det.get("experts", {}).get(freq)
            if exp:
                expert_params[dd][freq] = exp["params"]

    available_dds = list(expert_params.keys())
    ensemble_criteria = ["any", "majority", "all"]

    # Rank detectors by expert macro F1 and keep top N
    dd_ranking = []
    for dd in available_dds:
        det = results["detectors"][dd]
        exp_macro = det.get("expert_macro_f1", 0)
        dd_ranking.append((dd, exp_macro))
    dd_ranking.sort(key=lambda x: x[1], reverse=True)
    top_dds = [dd for dd, _ in dd_ranking[:ENSEMBLE_TOP_N]]
    print(f"  Top {ENSEMBLE_TOP_N} detectors by expert macro F1: {[(dd, f'{f:.4f}') for dd, f in dd_ranking[:ENSEMBLE_TOP_N]]}", flush=True)

    best_ensemble_per_freq = {}
    best_ensemble_macro = -1
    best_ensemble_config = None

    # Try pairs (and optionally triples) of top detectors
    for size in range(2, ENSEMBLE_MAX_SIZE + 1):
        for combo in itertools.combinations(top_dds, size):
            for crit in ensemble_criteria:
                # For each freq, build slot_specs from expert params
                combo_f1s = []
                combo_detail = {}
                for freq in DRIFT_FREQS:
                    slot_specs = []
                    for dd in combo:
                        if freq in expert_params[dd]:
                            slot_specs.append((dd, expert_params[dd][freq]))
                    if not slot_specs:
                        combo_f1s.append(0.0)
                        continue

                    # Evaluate on eval seeds
                    f1s = []
                    for i, seed in enumerate(EVAL_SEEDS):
                        f1 = run_ensemble_single(
                            slot_specs, crit, freq, seed, 700 + i,
                            freq // 10, freq // 2)
                        f1s.append(f1)
                    mean_f1 = sum(f1s) / len(f1s)
                    combo_f1s.append(mean_f1)
                    combo_detail[freq] = mean_f1

                macro = sum(combo_f1s) / len(combo_f1s)

                combo_name = "+".join(combo)
                if macro > best_ensemble_macro:
                    best_ensemble_macro = macro
                    best_ensemble_config = {
                        "detectors": list(combo),
                        "criterion": crit,
                        "macro_f1": macro,
                        "per_freq": combo_detail,
                    }
                    print(f"  NEW BEST: {combo_name} ({crit}) macro={macro:.4f}", flush=True)

                # Track per-frequency best
                for freq in DRIFT_FREQS:
                    prev = best_ensemble_per_freq.get(freq, {})
                    prev_macro = prev.get("macro_f1", -1)
                    if combo_detail.get(freq, 0) > prev_macro:
                        best_ensemble_per_freq[freq] = {
                            "detectors": list(combo),
                            "criterion": crit,
                            "macro_f1": combo_detail[freq],
                        }

    # Also try: per-frequency best ensemble (different combo per freq)
    print(f"\n  --- Per-frequency best ensemble ---", flush=True)
    pf_f1s = []
    pf_detail = {}
    for freq in DRIFT_FREQS:
        if freq in best_ensemble_per_freq:
            cfg = best_ensemble_per_freq[freq]
            combo = cfg["detectors"]
            crit = cfg["criterion"]
            slot_specs = [(dd, expert_params[dd][freq]) for dd in combo
                          if freq in expert_params[dd]]
            f1s = []
            for i, seed in enumerate(EVAL_SEEDS):
                f1 = run_ensemble_single(
                    slot_specs, crit, freq, seed, 800 + i,
                    freq // 10, freq // 2)
                f1s.append(f1)
            mean_f1 = sum(f1s) / len(f1s)
            pf_detail[freq] = {
                "detectors": combo, "criterion": crit, "f1": mean_f1,
            }
            pf_f1s.append(mean_f1)
            print(f"    freq={freq}: {'+'.join(combo)} ({crit}) F1={mean_f1:.4f}", flush=True)
        else:
            pf_f1s.append(0.0)
    pf_macro = sum(pf_f1s) / len(pf_f1s)
    print(f"  Per-freq ensemble macro F1: {pf_macro:.4f}", flush=True)

    # Also try: cross-DD ensemble (best single expert per freq)
    print(f"\n  --- Cross-DD single-expert ensemble ---", flush=True)
    cross_f1s = []
    cross_detail = {}
    for freq in DRIFT_FREQS:
        best_dd = None
        best_train_f1 = -1
        for dd in available_dds:
            exp = results["detectors"][dd].get("experts", {}).get(freq)
            if exp and exp["train_f1"] > best_train_f1:
                best_train_f1 = exp["train_f1"]
                best_dd = dd
        if best_dd:
            params = expert_params[best_dd][freq]
            f1s = []
            for i, seed in enumerate(EVAL_SEEDS):
                f1 = run_single(best_dd, params, freq, seed, 900 + i,
                                freq // 10, freq // 2)
                f1s.append(f1)
            mean_f1 = sum(f1s) / len(f1s)
            cross_detail[freq] = {"detector": best_dd, "f1": mean_f1}
            cross_f1s.append(mean_f1)
            print(f"    freq={freq}: {best_dd} F1={mean_f1:.4f}", flush=True)
        else:
            cross_f1s.append(0.0)
    cross_macro = sum(cross_f1s) / len(cross_f1s)
    print(f"  Cross-DD single-expert macro F1: {cross_macro:.4f}", flush=True)

    return {
        "best_fixed_ensemble": best_ensemble_config,
        "per_freq_ensemble": {"macro_f1": pf_macro, "detail": pf_detail},
        "cross_dd_single_expert": {"macro_f1": cross_macro, "detail": cross_detail},
        "best_ensemble_per_freq": best_ensemble_per_freq,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Overnight Benchmark v2: All 7 Detectors + Multi-DD Ensemble Search")
    print(f"Frequencies: {DRIFT_FREQS}")
    print(f"Tolerances: {TOLERANCES}")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Trials per optimization: {N_TRIALS}")
    print(f"Per-stream timeout: {PER_STREAM_TIMEOUT}s")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors (ordered by speed): {DETECTORS}")
    print(f"Output: {OUTPUT_DIR}/")
    print(flush=True)

    results = {
        "config": {
            "generator": GENERATOR,
            "drift_freqs": DRIFT_FREQS,
            "stream_length": STREAM_LENGTH,
            "n_trials": N_TRIALS,
            "tolerances": TOLERANCES,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "detectors": DETECTORS,
            "per_stream_timeout": PER_STREAM_TIMEOUT,
        },
        "detectors": {},
    }

    start_time = time.time()

    # Phase 1: Single-detector optimization
    for kind in DETECTORS:
        try:
            det_results = optimize_detector(kind)
            results["detectors"][kind] = det_results
            with open(f"{OUTPUT_DIR}/results_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"\n  ERROR with {kind}: {e}")
            import traceback
            traceback.print_exc()
            results["detectors"][kind] = {"error": str(e)}
            with open(f"{OUTPUT_DIR}/results_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Phase 2: Multi-DD ensemble search
    try:
        ensemble_results = search_multi_dd_ensembles(results)
        results["ensemble_search"] = ensemble_results
        with open(f"{OUTPUT_DIR}/results_partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"\nERROR in ensemble search: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nSingle detectors:")
    print(f"{'DD':<10} {'Gen F1':>8} {'Exp F1':>8} {'Delta':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    best_single_expert = -1
    best_single_expert_dd = None
    best_single_gen = -1
    best_single_gen_dd = None
    for dd in DETECTORS:
        if dd not in results["detectors"] or "error" in results["detectors"][dd]:
            print(f"{dd:<10} {'ERROR':>8}")
            continue
        det = results["detectors"][dd]
        gen = det.get("generalist", {}).get("eval_macro_f1", 0)
        exp = det.get("expert_macro_f1", 0)
        print(f"{dd:<10} {gen:>8.4f} {exp:>8.4f} {exp - gen:>+8.4f}")
        if exp > best_single_expert:
            best_single_expert = exp
            best_single_expert_dd = dd
        if gen > best_single_gen:
            best_single_gen = gen
            best_single_gen_dd = dd

    print(f"\nBest single generalist: {best_single_gen_dd} = {best_single_gen:.4f}")
    print(f"Best single expert: {best_single_expert_dd} = {best_single_expert:.4f}")

    if "ensemble_search" in results:
        es = results["ensemble_search"]
        if es.get("best_fixed_ensemble"):
            be = es["best_fixed_ensemble"]
            print(f"\nBest fixed ensemble: {'+'.join(be['detectors'])} ({be['criterion']}) = {be['macro_f1']:.4f}")
        if es.get("per_freq_ensemble"):
            print(f"Per-freq ensemble: {es['per_freq_ensemble']['macro_f1']:.4f}")
        if es.get("cross_dd_single_expert"):
            print(f"Cross-DD single-expert: {es['cross_dd_single_expert']['macro_f1']:.4f}")

        # Determine overall winner
        candidates = {
            "best_single_gen": best_single_gen,
            "best_single_expert": best_single_expert,
        }
        if es.get("best_fixed_ensemble"):
            candidates["best_fixed_ensemble"] = es["best_fixed_ensemble"]["macro_f1"]
        if es.get("per_freq_ensemble"):
            candidates["per_freq_ensemble"] = es["per_freq_ensemble"]["macro_f1"]
        if es.get("cross_dd_single_expert"):
            candidates["cross_dd_single_expert"] = es["cross_dd_single_expert"]["macro_f1"]

        winner = max(candidates, key=candidates.get)
        print(f"\n*** OVERALL WINNER: {winner} = {candidates[winner]:.4f} ***")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    with open(f"{OUTPUT_DIR}/results_final.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(f"{OUTPUT_DIR}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "gen_train_f1", "gen_eval_macro_f1",
                     "exp_eval_macro_f1", "delta"])
        for dd in DETECTORS:
            if dd not in results["detectors"] or "error" in results["detectors"][dd]:
                w.writerow([dd, "ERROR", "", "", ""])
                continue
            det = results["detectors"][dd]
            gen_train = det.get("generalist", {}).get("train_f1", 0)
            gen_eval = det.get("generalist", {}).get("eval_macro_f1", 0)
            exp_eval = det.get("expert_macro_f1", 0)
            w.writerow([dd, f"{gen_train:.4f}", f"{gen_eval:.4f}",
                        f"{exp_eval:.4f}", f"{exp_eval - gen_eval:+.4f}"])
        if "ensemble_search" in results:
            es = results["ensemble_search"]
            w.writerow([])
            w.writerow(["ensemble_type", "detectors", "criterion", "macro_f1"])
            if es.get("best_fixed_ensemble"):
                be = es["best_fixed_ensemble"]
                w.writerow(["fixed", "+".join(be["detectors"]), be["criterion"],
                            f"{be['macro_f1']:.4f}"])
            if es.get("per_freq_ensemble"):
                w.writerow(["per_freq", "varies", "varies",
                            f"{es['per_freq_ensemble']['macro_f1']:.4f}"])
            if es.get("cross_dd_single_expert"):
                w.writerow(["cross_dd_single", "varies", "n/a",
                            f"{es['cross_dd_single_expert']['macro_f1']:.4f}"])

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
