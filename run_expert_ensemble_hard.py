#!/usr/bin/env python3
"""Expert Ensemble Optimization with Hard Generators.

Re-establishes the original expert ensemble approach with equal optimization
budget for experts and generalists.

Approach:
1. Train K×7 experts (per frequency, per detector) - m trials each
   - Each expert is optimized on 2 streams (both generators at one frequency)
   - K=5 frequencies, 7 detectors → 35 expert studies
2. Train 7 generalists - K×m trials each (equal total budget per detector)
   - Each generalist is optimized on all 10 streams (2 generators × 5 frequencies)
3. Cross-DD ensemble: select best detector per frequency by train F1
   - Build a 5-expert ensemble (one per frequency, potentially different DD types)
4. Optimize voting (criterion + decision_window) for the cross-DD ensemble
5. Evaluate all configurations on held-out seeds
6. Compare ensemble vs generalists vs cross-DD single-expert

Expected: ensemble F1 > 0.9, generalist F1s 0.6-0.8
"""
import sys
import json
import time
import os
import csv
import multiprocessing as mp
import optuna
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _suggest_detector_params, _instantiate,
    MAX_WINDOW_FRACTION, _cap, _f1_from_counts,
    CLASS_PATH,
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections
from optimization.synthetic_f1_multistream_optimize_optuna import GENERATORS as BASE_GENERATORS
from datasets.sineclusters_hard import SineClustersHard
from datasets.waveform_hard import WaveformDriftHard

GENERATORS = dict(BASE_GENERATORS)
GENERATORS["SineClustersHard"] = SineClustersHard
GENERATORS["WaveformDriftHard"] = WaveformDriftHard

# --- Config ---
SCENARIO_GENERATORS = ["SineClustersHard", "WaveformDriftHard"]
DRIFT_FREQS = [100, 200, 500, 1000, 2000]
K = len(DRIFT_FREQS)  # 5
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
N_EXPERT_TRIALS = 10  # m
N_GENERALIST_TRIALS = K * N_EXPERT_TRIALS  # 50 (equal budget: K*m)
N_VOTING_TRIALS = 15

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
# Workers (multiprocessing for hard timeout)
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_single_mp(kind, params, generator, freq, seeds,
                   tolerance, suppression):
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
            except Exception:
                f1s.append(0.0)
    return f1s


def eval_single(kind, params, generator, freqs, seeds,
                tolerance=TOLERANCE, suppression=SUPPRESSION):
    results = {}
    for freq in freqs:
        f1s = _run_single_mp(kind, params, generator, freq, seeds,
                             tolerance, suppression)
        results[freq] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def eval_ensemble(slot_specs, criterion, dw, generator, freqs, seeds,
                  tolerance=TOLERANCE, suppression=SUPPRESSION):
    results = {}
    for freq in freqs:
        f1s = _run_ensemble_mp(slot_specs, criterion, dw, generator, freq, seeds,
                               tolerance, suppression)
        results[freq] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def macro_f1(per_freq, freqs):
    return sum(per_freq[f] for f in freqs) / len(freqs)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def make_expert_objective(kind, freq):
    max_window = int(freq * MAX_WINDOW_FRACTION)

    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for generator in SCENARIO_GENERATORS:
            f1s.extend(_run_single_mp(kind, params, generator, freq,
                                      TRAIN_SEEDS, TOLERANCE, SUPPRESSION))
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def make_generalist_objective(kind):
    max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)

    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                f1s.extend(_run_single_mp(kind, params, generator, freq,
                                          TRAIN_SEEDS, TOLERANCE, SUPPRESSION))
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def make_voting_objective(slot_specs):
    def objective(trial):
        criterion = trial.suggest_categorical("criterion", ["any", "majority", "all"])
        dw = trial.suggest_int("decision_window", 1, 5)
        f1s = []
        for generator in SCENARIO_GENERATORS:
            for freq in DRIFT_FREQS:
                f1s.extend(_run_ensemble_mp(slot_specs, criterion, dw, generator,
                                            freq, TRAIN_SEEDS, TOLERANCE, SUPPRESSION))
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


# ---------------------------------------------------------------------------
# Optimization functions
# ---------------------------------------------------------------------------

def optimize_expert(kind, freq):
    print(f"\n  --- Expert {kind} for freq={freq} ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_expert_objective(kind, freq),
                   n_trials=N_EXPERT_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print(f"  Train F1: {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)
    return best.params, best.value


def optimize_generalist(kind):
    print(f"\n  --- Generalist {kind} ({N_GENERALIST_TRIALS} trials) ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_generalist_objective(kind),
                   n_trials=N_GENERALIST_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print(f"  Train F1: {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)
    return best.params, best.value


def optimize_voting(slot_specs):
    print(f"\n  --- Voting optimization ({N_VOTING_TRIALS} trials) ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_voting_objective(slot_specs),
                   n_trials=N_VOTING_TRIALS, show_progress_bar=True)
    best = study.best_trial
    criterion = best.params["criterion"]
    dw = best.params["decision_window"]
    print(f"  Train F1: {best.value:.4f}")
    print(f"  Criterion: {criterion}, DW: {dw}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)
    return criterion, dw, best.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Expert Ensemble Optimization (Hard Generators)")
    print("=" * 70)
    print(f"Generators: {SCENARIO_GENERATORS}")
    print(f"Frequencies: {DRIFT_FREQS} (K={K})")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Tolerance: {TOLERANCE} (standard), {TIGHT_TOLERANCE} (tight)")
    print(f"Suppression: {SUPPRESSION}")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors: {DETECTORS}")
    print(f"Expert trials: {N_EXPERT_TRIALS} (m)")
    print(f"Generalist trials: {N_GENERALIST_TRIALS} (K*m={K}*{N_EXPERT_TRIALS})")
    print(f"Voting trials: {N_VOTING_TRIALS}")
    print(f"Budget per detector: experts={K}*{N_EXPERT_TRIALS}={K*N_EXPERT_TRIALS}, "
          f"generalist={N_GENERALIST_TRIALS} → EQUAL")
    print(flush=True)

    all_results = {}
    start_time = time.time()

    # Resume support
    partial_path = f"{OUTPUT_DIR}/expert_ensemble_hard_partial.json"
    if os.path.exists(partial_path):
        try:
            all_results = json.load(open(partial_path))
            print(f"Resumed from partial results: {list(all_results.keys())}", flush=True)
        except Exception:
            all_results = {}

    # ---- Phase 1: Expert optimization (per frequency, per detector) ----
    print(f"\n{'='*70}")
    print(f"Phase 1: Expert Optimization ({K}×{len(DETECTORS)}={K*len(DETECTORS)} experts)")
    print(f"{'='*70}", flush=True)

    if "experts" not in all_results:
        all_results["experts"] = {}

    for kind in DETECTORS:
        if kind not in all_results["experts"]:
            all_results["experts"][kind] = {}

        for freq in DRIFT_FREQS:
            freq_key = str(freq)
            if freq_key in all_results["experts"][kind]:
                print(f"  {kind} freq={freq}: already done, skipping", flush=True)
                continue

            print(f"\n{'='*70}")
            print(f"Expert: {kind} @ freq={freq}")
            print(f"{'='*70}", flush=True)

            try:
                params, train_f1 = optimize_expert(kind, freq)

                # Evaluate expert on eval seeds at this frequency
                print(f"  Evaluating on held-out seeds...", flush=True)
                sc_eval = eval_single(kind, params, SCENARIO_GENERATORS[0],
                                      [freq], EVAL_SEEDS,
                                      tolerance=TOLERANCE, suppression=SUPPRESSION)
                wf_eval = eval_single(kind, params, SCENARIO_GENERATORS[1],
                                      [freq], EVAL_SEEDS,
                                      tolerance=TOLERANCE, suppression=SUPPRESSION)
                eval_f1 = (sc_eval[freq] + wf_eval[freq]) / 2
                print(f"  Eval F1: SC={sc_eval[freq]:.4f}, WF={wf_eval[freq]:.4f}, "
                      f"Combined={eval_f1:.4f}", flush=True)

                all_results["experts"][kind][freq_key] = {
                    "params": params,
                    "train_f1": train_f1,
                    "eval_sc_f1": sc_eval[freq],
                    "eval_wf_f1": wf_eval[freq],
                    "eval_combined_f1": eval_f1,
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results["experts"][kind][freq_key] = {"error": str(e)}

            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 2: Generalist optimization ----
    print(f"\n{'='*70}")
    print(f"Phase 2: Generalist Optimization ({len(DETECTORS)} detectors, "
          f"{N_GENERALIST_TRIALS} trials each)")
    print(f"{'='*70}", flush=True)

    if "generalists" not in all_results:
        all_results["generalists"] = {}

    for kind in DETECTORS:
        if kind in all_results["generalists"]:
            print(f"  {kind}: already done, skipping", flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"Generalist: {kind}")
        print(f"{'='*70}", flush=True)

        try:
            params, train_f1 = optimize_generalist(kind)

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
            print(f"  SC macro F1: {sc_macro:.4f}, WF macro F1: {wf_macro:.4f}, "
                  f"Combined: {combined:.4f}")

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
            print(f"  SC tight F1: {sc_tight:.4f}, WF tight F1: {wf_tight:.4f}, "
                  f"Combined tight: {combined_tight:.4f}")

            all_results["generalists"][kind] = {
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
            all_results["generalists"][kind] = {"error": str(e)}

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 3: Cross-DD ensemble selection + voting optimization ----
    print(f"\n{'='*70}")
    print(f"Phase 3: Cross-DD Ensemble Selection + Voting Optimization")
    print(f"{'='*70}", flush=True)

    if "cross_dd_ensemble" not in all_results:
        # Select best detector per frequency by train F1
        cross_dd_slots = []
        cross_dd_selection = {}
        for freq in DRIFT_FREQS:
            freq_key = str(freq)
            best_f1 = -1
            best_kind = None
            best_params = None
            for kind in DETECTORS:
                exp = all_results["experts"].get(kind, {}).get(freq_key, {})
                if "error" in exp or not exp:
                    continue
                train_f1 = exp["train_f1"]
                if train_f1 > best_f1:
                    best_f1 = train_f1
                    best_kind = kind
                    best_params = exp["params"]

            if best_kind:
                cross_dd_slots.append((best_kind, best_params))
                cross_dd_selection[freq_key] = {
                    "detector": best_kind,
                    "train_f1": best_f1,
                    "params": best_params,
                }
                print(f"  freq={freq}: {best_kind} (train F1={best_f1:.4f})")
            else:
                print(f"  freq={freq}: NO VALID EXPERT")

        if len(cross_dd_slots) > 0:
            print(f"\n  Cross-DD ensemble: {len(cross_dd_slots)} experts")
            for i, (kind, params) in enumerate(cross_dd_slots):
                print(f"    Slot {i}: {kind} → {params}")

            # Optimize voting
            criterion, dw, vote_train_f1 = optimize_voting(cross_dd_slots)

            # Evaluate on held-out seeds
            print(f"\n  Evaluating cross-DD ensemble on held-out seeds "
                  f"(tol={TOLERANCE})...", flush=True)
            ens_sc = eval_ensemble(cross_dd_slots, criterion, dw,
                                   SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                   tolerance=TOLERANCE, suppression=SUPPRESSION)
            ens_wf = eval_ensemble(cross_dd_slots, criterion, dw,
                                   SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                   tolerance=TOLERANCE, suppression=SUPPRESSION)
            ens_sc_macro = macro_f1(ens_sc, DRIFT_FREQS)
            ens_wf_macro = macro_f1(ens_wf, DRIFT_FREQS)
            ens_combined = (ens_sc_macro + ens_wf_macro) / 2
            print(f"  SC macro F1: {ens_sc_macro:.4f}, WF macro F1: {ens_wf_macro:.4f}, "
                  f"Combined: {ens_combined:.4f}")

            # Tight tolerance
            print(f"  Evaluating cross-DD ensemble (tol={TIGHT_TOLERANCE})...", flush=True)
            ens_sc_tight = eval_ensemble(cross_dd_slots, criterion, dw,
                                         SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                         tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            ens_wf_tight = eval_ensemble(cross_dd_slots, criterion, dw,
                                         SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                         tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
            ens_sc_tight_macro = macro_f1(ens_sc_tight, DRIFT_FREQS)
            ens_wf_tight_macro = macro_f1(ens_wf_tight, DRIFT_FREQS)
            ens_combined_tight = (ens_sc_tight_macro + ens_wf_tight_macro) / 2
            print(f"  SC tight F1: {ens_sc_tight_macro:.4f}, "
                  f"WF tight F1: {ens_wf_tight_macro:.4f}, "
                  f"Combined tight: {ens_combined_tight:.4f}")

            all_results["cross_dd_ensemble"] = {
                "slot_specs": [(kind, params) for kind, params in cross_dd_slots],
                "selection": cross_dd_selection,
                "criterion": criterion,
                "decision_window": dw,
                "train_f1": vote_train_f1,
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

            # Also evaluate with each criterion
            for crit in ["any", "majority", "all"]:
                key = f"cross_dd_ensemble_{crit}"
                if key in all_results:
                    continue
                print(f"\n  Evaluating cross-DD ensemble with criterion='{crit}'...",
                      flush=True)
                sc_ev = eval_ensemble(cross_dd_slots, crit, dw,
                                      SCENARIO_GENERATORS[0], DRIFT_FREQS, EVAL_SEEDS,
                                      tolerance=TOLERANCE, suppression=SUPPRESSION)
                wf_ev = eval_ensemble(cross_dd_slots, crit, dw,
                                      SCENARIO_GENERATORS[1], DRIFT_FREQS, EVAL_SEEDS,
                                      tolerance=TOLERANCE, suppression=SUPPRESSION)
                sc_m = macro_f1(sc_ev, DRIFT_FREQS)
                wf_m = macro_f1(wf_ev, DRIFT_FREQS)
                comb = (sc_m + wf_m) / 2
                print(f"  SC: {sc_m:.4f}, WF: {wf_m:.4f}, Combined: {comb:.4f}",
                      flush=True)
                all_results[key] = {
                    "criterion": crit,
                    "decision_window": dw,
                    "sc_f1": sc_m,
                    "wf_f1": wf_m,
                    "combined_f1": comb,
                    "sc_detail": {str(f): sc_ev[f] for f in DRIFT_FREQS},
                    "wf_detail": {str(f): wf_ev[f] for f in DRIFT_FREQS},
                }
        else:
            print("  No valid experts for cross-DD ensemble!", flush=True)
            all_results["cross_dd_ensemble"] = {"error": "no valid experts"}

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 4: Cross-DD single-expert (per-frequency oracle) ----
    print(f"\n{'='*70}")
    print(f"Phase 4: Cross-DD Single-Expert (per-frequency selection)")
    print(f"{'='*70}", flush=True)

    if "cross_dd_single_by_train" not in all_results:
        # By train F1
        by_train = {}
        train_f1s = []
        for freq in DRIFT_FREQS:
            freq_key = str(freq)
            best_f1 = -1
            best_kind = None
            best_eval_f1 = 0.0
            for kind in DETECTORS:
                exp = all_results["experts"].get(kind, {}).get(freq_key, {})
                if "error" in exp or not exp:
                    continue
                if exp["train_f1"] > best_f1:
                    best_f1 = exp["train_f1"]
                    best_kind = kind
                    best_eval_f1 = exp["eval_combined_f1"]
            by_train[freq_key] = {
                "detector": best_kind,
                "train_f1": best_f1,
                "eval_f1": best_eval_f1,
            }
            train_f1s.append(best_eval_f1)
            print(f"  freq={freq} (by train): {best_kind} "
                  f"(train={best_f1:.4f}, eval={best_eval_f1:.4f})")

        by_train_macro = sum(train_f1s) / len(train_f1s) if train_f1s else 0.0
        print(f"  Cross-DD single by train: macro F1={by_train_macro:.4f}")
        all_results["cross_dd_single_by_train"] = {
            "per_freq": by_train,
            "macro_f1": by_train_macro,
        }

    if "cross_dd_single_by_eval" not in all_results:
        # By eval F1 (oracle)
        by_eval = {}
        eval_f1s = []
        for freq in DRIFT_FREQS:
            freq_key = str(freq)
            best_f1 = -1
            best_kind = None
            best_train_f1 = 0.0
            for kind in DETECTORS:
                exp = all_results["experts"].get(kind, {}).get(freq_key, {})
                if "error" in exp or not exp:
                    continue
                if exp["eval_combined_f1"] > best_f1:
                    best_f1 = exp["eval_combined_f1"]
                    best_kind = kind
                    best_train_f1 = exp["train_f1"]
            by_eval[freq_key] = {
                "detector": best_kind,
                "eval_f1": best_f1,
                "train_f1": best_train_f1,
            }
            eval_f1s.append(best_f1)
            print(f"  freq={freq} (by eval): {best_kind} "
                  f"(eval={best_f1:.4f}, train={best_train_f1:.4f})")

        by_eval_macro = sum(eval_f1s) / len(eval_f1s) if eval_f1s else 0.0
        print(f"  Cross-DD single by eval (oracle): macro F1={by_eval_macro:.4f}")
        all_results["cross_dd_single_by_eval"] = {
            "per_freq": by_eval,
            "macro_f1": by_eval_macro,
        }

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Phase 5: Final comparison table ----
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n--- Standard tolerance (tol={TOLERANCE}, supp={SUPPRESSION}) ---")
    print(f"{'Approach':<25} {'SC F1':>8} {'WF F1':>8} {'Combined':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")

    best_single = -1
    best_single_name = ""
    best_ensemble = -1

    # Generalists
    for kind in DETECTORS:
        d = all_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            continue
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        label = f"Gen-{kind}"
        print(f"{label:<25} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")
        if comb > best_single:
            best_single = comb
            best_single_name = kind

    # Cross-DD ensemble (optimized criterion)
    d = all_results.get("cross_dd_ensemble", {})
    if d and "error" not in d:
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        crit = d.get("criterion", "?")
        dw = d.get("decision_window", "?")
        label = f"CrossDD-ens({crit},dw{dw})"
        print(f"{label:<25} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")
        if comb > best_ensemble:
            best_ensemble = comb

    # Cross-DD ensemble with each criterion
    for crit in ["any", "majority", "all"]:
        d = all_results.get(f"cross_dd_ensemble_{crit}", {})
        if not d:
            continue
        sc = d.get("sc_f1", 0)
        wf = d.get("wf_f1", 0)
        comb = d.get("combined_f1", 0)
        label = f"CrossDD-ens-{crit}"
        print(f"{label:<25} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")
        if comb > best_ensemble:
            best_ensemble = comb

    # Cross-DD single-expert
    d = all_results.get("cross_dd_single_by_train", {})
    if d:
        macro = d.get("macro_f1", 0)
        print(f"{'CrossDD-single(train)':<25} {'':>8} {'':>8} {macro:>10.4f}")
        if macro > best_ensemble:
            best_ensemble = macro

    d = all_results.get("cross_dd_single_by_eval", {})
    if d:
        macro = d.get("macro_f1", 0)
        print(f"{'CrossDD-single(eval*)':<25} {'':>8} {'':>8} {macro:>10.4f}")
        if macro > best_ensemble:
            best_ensemble = macro

    print(f"\n  Best single generalist: {best_single_name} = {best_single:.4f}")
    print(f"  Best ensemble: {best_ensemble:.4f}")
    if best_ensemble > best_single:
        print(f"  *** ENSEMBLE WINS by {best_ensemble - best_single:.4f} ***")
    else:
        print(f"  Single detector wins by {best_single - best_ensemble:.4f}")

    # Tight tolerance table
    print(f"\n--- Tight tolerance (tol={TIGHT_TOLERANCE}, supp={SUPPRESSION}) ---")
    print(f"{'Approach':<25} {'SC F1':>8} {'WF F1':>8} {'Combined':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")

    for kind in DETECTORS:
        d = all_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            continue
        sc = d.get("sc_f1_tight", 0)
        wf = d.get("wf_f1_tight", 0)
        comb = d.get("combined_f1_tight", 0)
        label = f"Gen-{kind}"
        print(f"{label:<25} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")

    d = all_results.get("cross_dd_ensemble", {})
    if d and "error" not in d:
        sc = d.get("sc_f1_tight", 0)
        wf = d.get("wf_f1_tight", 0)
        comb = d.get("combined_f1_tight", 0)
        crit = d.get("criterion", "?")
        print(f"{'CrossDD-ens('+crit+')':<25} {sc:>8.4f} {wf:>8.4f} {comb:>10.4f}")

    # Per-frequency detail for generalists
    print(f"\n--- Per-frequency detail (tol={TOLERANCE}) ---")
    freq_header = "  ".join(f"f{f:>4}" for f in DRIFT_FREQS)
    print(f"{'Approach':<20} {'SC':>3} {freq_header}  | {'WF':>3} {freq_header}")
    print(f"{'-'*20} {'-'*3} {'-'*35}  | {'-'*3} {'-'*35}")

    for kind in DETECTORS:
        d = all_results.get("generalists", {}).get(kind, {})
        if "error" in d or not d:
            continue
        sc_d = d.get("sc_detail", {})
        wf_d = d.get("wf_detail", {})
        sc_vals = "  ".join(f"{sc_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS)
        wf_vals = "  ".join(f"{wf_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS)
        print(f"Gen-{kind:<15} {'SC':>3} {sc_vals}  | {'WF':>3} {wf_vals}")

    d = all_results.get("cross_dd_ensemble", {})
    if d and "error" not in d:
        sc_d = d.get("sc_detail", {})
        wf_d = d.get("wf_detail", {})
        sc_vals = "  ".join(f"{sc_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS)
        wf_vals = "  ".join(f"{wf_d.get(str(f), 0):.4f}" for f in DRIFT_FREQS)
        crit = d.get("criterion", "?")
        print(f"CrossDD-ens({crit:<7}) {'SC':>3} {sc_vals}  | {'WF':>3} {wf_vals}")

    # Cross-DD selection detail
    print(f"\n--- Cross-DD selection (best detector per frequency) ---")
    sel = all_results.get("cross_dd_ensemble", {}).get("selection", {})
    for freq in DRIFT_FREQS:
        freq_key = str(freq)
        s = sel.get(freq_key, {})
        det = s.get("detector", "?")
        train_f1 = s.get("train_f1", 0)
        print(f"  freq={freq}: {det} (train F1={train_f1:.4f})")

    # Save CSV summary
    csv_path = f"{OUTPUT_DIR}/expert_ensemble_hard_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "type", "criterion", "dw",
                     "sc_f1_tol100", "wf_f1_tol100", "combined_tol100",
                     "sc_f1_tol20", "wf_f1_tol20", "combined_tol20"])
        for kind in DETECTORS:
            d = all_results.get("generalists", {}).get(kind, {})
            if "error" in d or not d:
                continue
            w.writerow([f"gen_{kind}", "generalist", "-", "-",
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}",
                        f"{d.get('sc_f1_tight',0):.4f}", f"{d.get('wf_f1_tight',0):.4f}",
                        f"{d.get('combined_f1_tight',0):.4f}"])
        d = all_results.get("cross_dd_ensemble", {})
        if d and "error" not in d:
            w.writerow(["cross_dd_ensemble", "ensemble",
                        d.get("criterion", "?"), d.get("decision_window", "?"),
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}",
                        f"{d.get('sc_f1_tight',0):.4f}", f"{d.get('wf_f1_tight',0):.4f}",
                        f"{d.get('combined_f1_tight',0):.4f}"])
        for crit in ["any", "majority", "all"]:
            d = all_results.get(f"cross_dd_ensemble_{crit}", {})
            if not d:
                continue
            w.writerow([f"cross_dd_ens_{crit}", "ensemble", crit,
                        d.get("decision_window", "?"),
                        f"{d.get('sc_f1',0):.4f}", f"{d.get('wf_f1',0):.4f}",
                        f"{d.get('combined_f1',0):.4f}", "0", "0", "0"])
        d = all_results.get("cross_dd_single_by_train", {})
        if d:
            w.writerow(["cross_dd_single_train", "oracle", "-", "-",
                        "0", "0", f"{d.get('macro_f1',0):.4f}", "0", "0", "0"])
        d = all_results.get("cross_dd_single_by_eval", {})
        if d:
            w.writerow(["cross_dd_single_eval", "oracle", "-", "-",
                        "0", "0", f"{d.get('macro_f1',0):.4f}", "0", "0", "0"])

    # Save final results
    final_path = f"{OUTPUT_DIR}/expert_ensemble_hard_final.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {OUTPUT_DIR}/expert_ensemble_hard_*")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
