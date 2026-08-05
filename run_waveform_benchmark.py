#!/usr/bin/env python3
"""Overnight benchmark: WaveformDrift2 + cross-generator transfer + mixed ensembles.

Phases:
  1. Single-detector optimization on WaveformDrift2 (generalist + expert per freq)
     - Fair settings: fixed tolerance=100, fixed suppression=100
  2. Cross-generator transfer evaluation:
     - SineClusters-optimized params evaluated on WaveformDrift2
     - WaveformDrift2-optimized params evaluated on SineClusters
  3. WaveformDrift2 ensemble search (pairs of top 5)
     - Also: mixed ensemble (best SineClusters detector + best WaveformDrift2 detector)
  4. Joint 2-DD ensemble optimization on WaveformDrift2 for top pairs

Outputs to overnight_results/waveform_*
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
import math

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _run_one_stream, _suggest_detector_params, _instantiate,
    build_stream, CANDIDATES, CLASS_PATH, MAX_WINDOW_FRACTION, _cap,
    _f1_from_counts
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections

# --- Config ---
GENERATOR = "WaveformDrift2"
DRIFT_FREQS = [200, 500, 1000, 2000]
STREAM_LENGTH = 4000
N_TRIALS = 25
SEED = 1337
FIXED_TOLERANCE = 100
FIXED_SUPPRESSION = 100
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 30
EARLY_STOP_F1 = 1.0
ENSEMBLE_TOP_N = 5
ENSEMBLE_MAX_SIZE = 2
N_ENSEMBLE_TRIALS = 30
ENSEMBLE_PER_STREAM_TIMEOUT = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Workers
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
               tolerance, suppression, generator=GENERATOR):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_single_stream_worker,
                       args=(queue, kind, params, freq, stream_seed, s_idx,
                             tolerance, suppression, STREAM_LENGTH, generator))
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


def _ensemble_stream_worker(queue, slot_specs, detector_criterion, ensemble_criterion,
                            decision_window, freq, stream_seed, s_idx,
                            tolerance, suppression, stream_length, generator):
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
            detector_criterion=detector_criterion,
            ensemble_criterion=ensemble_criterion,
            decision_window=decision_window,
        )
        dets = apply_suppression(raw_ensemble, suppression)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        queue.put(("ok", tp, fp, fn, f1, prec, rec, len(known)))
    except Exception as e:
        queue.put(("error", str(e)))


def run_ensemble_eval(slot_specs, detector_criterion, ensemble_criterion,
                      decision_window, freq, stream_seed, s_idx,
                      tolerance=FIXED_TOLERANCE,
                      suppression=FIXED_SUPPRESSION,
                      generator=GENERATOR):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_ensemble_stream_worker,
                       args=(queue, slot_specs, detector_criterion,
                             ensemble_criterion, decision_window,
                             freq, stream_seed, s_idx, tolerance, suppression,
                             STREAM_LENGTH, generator))
    proc.start()
    proc.join(ENSEMBLE_PER_STREAM_TIMEOUT)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return None
    try:
        result = queue.get_nowait()
        if result[0] == "ok":
            _, tp, fp, fn, f1, prec, rec, n_known = result
            return {"tp": tp, "fp": fp, "fn": fn, "f1": f1,
                    "precision": prec, "recall": rec, "n_drifts": n_known}
    except:
        pass
    return None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_params(kind, params, freqs, seeds, generator=GENERATOR):
    results_per_freq = {}
    for freq in freqs:
        f1s = []
        seed_results = []
        for i, seed in enumerate(seeds):
            ctx = mp.get_context("fork")
            queue = ctx.Queue()
            proc = ctx.Process(target=_single_stream_worker,
                               args=(queue, kind, params, freq, seed, 500 + i,
                                     FIXED_TOLERANCE, FIXED_SUPPRESSION,
                                     STREAM_LENGTH, generator))
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


def eval_ensemble(slot_specs, detector_criterion, ensemble_criterion,
                  decision_window, freqs, seeds, generator=GENERATOR):
    results_per_freq = {}
    for freq in freqs:
        f1s = []
        seed_results = []
        for i, seed in enumerate(seeds):
            result = run_ensemble_eval(
                slot_specs, detector_criterion, ensemble_criterion,
                decision_window, freq, seed, 700 + i,
                FIXED_TOLERANCE, FIXED_SUPPRESSION, generator)
            if result:
                f1s.append(result["f1"])
                seed_results.append({"seed": seed, "f1": result["f1"],
                                     "tp": result["tp"], "fp": result["fp"],
                                     "fn": result["fn"]})
            else:
                f1s.append(0.0)
                seed_results.append({"seed": seed, "f1": 0.0, "error": "timeout"})
        results_per_freq[freq] = {
            "seeds": seed_results,
            "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    return results_per_freq


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def make_generalist_objective(kind, generator=GENERATOR):
    def objective(trial):
        max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for freq in DRIFT_FREQS:
            for s_idx, seed in enumerate(TRAIN_SEEDS):
                f1 = run_single(kind, params, freq, seed, s_idx,
                                FIXED_TOLERANCE, FIXED_SUPPRESSION, generator)
                f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def make_expert_objective(kind, freq, generator=GENERATOR):
    max_window = int(freq * MAX_WINDOW_FRACTION)
    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for s_idx, seed in enumerate(TRAIN_SEEDS):
            f1 = run_single(kind, params, freq, seed, s_idx,
                            FIXED_TOLERANCE, FIXED_SUPPRESSION, generator)
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def _early_stop_cb(study, trial):
    if study.best_value >= EARLY_STOP_F1:
        study.stop()


# ---------------------------------------------------------------------------
# Phase 1: Single-detector optimization on WaveformDrift2
# ---------------------------------------------------------------------------

def optimize_detector(kind):
    print(f"\n{'='*70}")
    print(f"Phase 1 — Detector: {kind} on {GENERATOR}")
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
    for freq in DRIFT_FREQS:
        print(f"\n  --- Expert {kind} for freq={freq} ---", flush=True)
        t0 = time.time()
        exp_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        exp_study.optimize(make_expert_objective(kind, freq),
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
# Phase 2: Cross-generator transfer evaluation
# ---------------------------------------------------------------------------

def cross_generator_transfer(waveform_results):
    """Evaluate SineClusters-optimized params on WaveformDrift2 and vice versa."""
    print(f"\n{'='*70}")
    print(f"Phase 2: Cross-Generator Transfer Evaluation")
    print(f"{'='*70}", flush=True)

    # Load SineClusters results
    sc_path = f"{OUTPUT_DIR}/results_final.json"
    if not os.path.exists(sc_path):
        print("  SineClusters results not found, skipping transfer.")
        return {}

    sc_results = json.load(open(sc_path))
    transfer = {"sc_to_wf": {}, "wf_to_sc": {}, "mixed_ensemble": {}}

    # --- SC → WF: SineClusters-optimized params evaluated on WaveformDrift2 ---
    print(f"\n  --- SineClusters params → WaveformDrift2 ---", flush=True)
    for dd in DETECTORS:
        if dd not in sc_results.get("detectors", {}):
            continue
        det = sc_results["detectors"][dd]
        if "error" in det:
            continue

        # Generalist transfer
        gen_params = det.get("generalist", {}).get("params", {})
        if gen_params:
            gen_eval = eval_params(dd, gen_params, DRIFT_FREQS, EVAL_SEEDS,
                                   generator="WaveformDrift2")
            gen_f1s = [gen_eval[f]["mean_f1"] for f in DRIFT_FREQS]
            gen_macro = sum(gen_f1s) / len(gen_f1s)
            sc_gen_macro = det.get("generalist", {}).get("eval_macro_f1", 0)
            print(f"    {dd} gen: SC={sc_gen_macro:.4f} → WF={gen_macro:.4f} "
                  f"(Δ={gen_macro - sc_gen_macro:+.4f})", flush=True)
            transfer["sc_to_wf"][dd] = {
                "generalist": {"sc_f1": sc_gen_macro, "wf_f1": gen_macro,
                               "delta": gen_macro - sc_gen_macro,
                               "params": gen_params, "eval": gen_eval},
            }

        # Expert transfer
        for freq in DRIFT_FREQS:
            exp = det.get("experts", {}).get(str(freq))
            if not exp:
                continue
            exp_params = exp.get("params", {})
            if exp_params:
                exp_eval = eval_params(dd, exp_params, [freq], EVAL_SEEDS,
                                       generator="WaveformDrift2")
                exp_f1 = exp_eval[freq]["mean_f1"]
                sc_exp_f1 = exp.get("eval_f1", 0)
                if dd not in transfer["sc_to_wf"]:
                    transfer["sc_to_wf"][dd] = {"experts": {}}
                elif "experts" not in transfer["sc_to_wf"][dd]:
                    transfer["sc_to_wf"][dd]["experts"] = {}
                transfer["sc_to_wf"][dd]["experts"][freq] = {
                    "sc_f1": sc_exp_f1, "wf_f1": exp_f1,
                    "delta": exp_f1 - sc_exp_f1,
                }

    # --- WF → SC: WaveformDrift2-optimized params evaluated on SineClusters ---
    print(f"\n  --- WaveformDrift2 params → SineClusters ---", flush=True)
    for dd in DETECTORS:
        if dd not in waveform_results.get("detectors", {}):
            continue
        det = waveform_results["detectors"][dd]
        if "error" in det:
            continue

        gen_params = det.get("generalist", {}).get("params", {})
        if gen_params:
            gen_eval = eval_params(dd, gen_params, DRIFT_FREQS, EVAL_SEEDS,
                                   generator="SineClusters")
            gen_f1s = [gen_eval[f]["mean_f1"] for f in DRIFT_FREQS]
            gen_macro = sum(gen_f1s) / len(gen_f1s)
            wf_gen_macro = det.get("generalist", {}).get("eval_macro_f1", 0)
            print(f"    {dd} gen: WF={wf_gen_macro:.4f} → SC={gen_macro:.4f} "
                  f"(Δ={gen_macro - wf_gen_macro:+.4f})", flush=True)
            transfer["wf_to_sc"][dd] = {
                "generalist": {"wf_f1": wf_gen_macro, "sc_f1": gen_macro,
                               "delta": gen_macro - wf_gen_macro,
                               "params": gen_params, "eval": gen_eval},
            }

    # --- Mixed ensemble: best SC detector + best WF detector ---
    print(f"\n  --- Mixed generator ensemble ---", flush=True)
    # Find best SC generalist and best WF generalist
    sc_best_dd = None
    sc_best_f1 = -1
    for dd, det in sc_results.get("detectors", {}).items():
        if "error" in det:
            continue
        f1 = det.get("generalist", {}).get("eval_macro_f1", 0)
        if f1 > sc_best_f1:
            sc_best_f1 = f1
            sc_best_dd = dd

    wf_best_dd = None
    wf_best_f1 = -1
    for dd, det in waveform_results.get("detectors", {}).items():
        if "error" in det:
            continue
        f1 = det.get("generalist", {}).get("eval_macro_f1", 0)
        if f1 > wf_best_f1:
            wf_best_f1 = f1
            wf_best_dd = dd

    if sc_best_dd and wf_best_dd and sc_best_dd != wf_best_dd:
        sc_params = sc_results["detectors"][sc_best_dd]["generalist"]["params"]
        wf_params = waveform_results["detectors"][wf_best_dd]["generalist"]["params"]
        slot_specs = [(sc_best_dd, sc_params), (wf_best_dd, wf_params)]

        print(f"  Mixed ensemble: {sc_best_dd}(SC) + {wf_best_dd}(WF)", flush=True)

        # Evaluate on both generators with different criteria
        for ens_crit in ["any", "majority", "all"]:
            for gen_name in ["SineClusters", "WaveformDrift2"]:
                eval_res = eval_ensemble(
                    slot_specs, "any", ens_crit, 1,
                    DRIFT_FREQS, EVAL_SEEDS, generator=gen_name)
                f1s = [eval_res[f]["mean_f1"] for f in DRIFT_FREQS]
                macro = sum(f1s) / len(f1s)
                key = f"{sc_best_dd}+{wf_best_dd}_{ens_crit}_{gen_name}"
                transfer["mixed_ensemble"][key] = {
                    "detectors": [sc_best_dd, wf_best_dd],
                    "ens_crit": ens_crit,
                    "generator": gen_name,
                    "macro_f1": macro,
                    "per_freq": {f: eval_res[f]["mean_f1"] for f in DRIFT_FREQS},
                }
                print(f"    {ens_crit} on {gen_name}: macro F1={macro:.4f}", flush=True)

        # Also try with decision_window=5
        for ens_crit in ["all"]:
            for gen_name in ["SineClusters", "WaveformDrift2"]:
                eval_res = eval_ensemble(
                    slot_specs, "any", ens_crit, 5,
                    DRIFT_FREQS, EVAL_SEEDS, generator=gen_name)
                f1s = [eval_res[f]["mean_f1"] for f in DRIFT_FREQS]
                macro = sum(f1s) / len(f1s)
                key = f"{sc_best_dd}+{wf_best_dd}_{ens_crit}_dw5_{gen_name}"
                transfer["mixed_ensemble"][key] = {
                    "detectors": [sc_best_dd, wf_best_dd],
                    "ens_crit": ens_crit,
                    "decision_window": 5,
                    "generator": gen_name,
                    "macro_f1": macro,
                    "per_freq": {f: eval_res[f]["mean_f1"] for f in DRIFT_FREQS},
                }
                print(f"    {ens_crit} dw=5 on {gen_name}: macro F1={macro:.4f}", flush=True)
    else:
        print(f"  Best SC={sc_best_dd}({sc_best_f1:.4f}), Best WF={wf_best_dd}({wf_best_f1:.4f})"
              f" — same detector, skipping mixed ensemble.", flush=True)

    return transfer


# ---------------------------------------------------------------------------
# Phase 3: Multi-DD ensemble search on WaveformDrift2
# ---------------------------------------------------------------------------

def search_multi_dd_ensembles(results):
    print(f"\n{'='*70}")
    print(f"Phase 3: Multi-DD Ensemble Search on {GENERATOR}")
    print(f"{'='*70}", flush=True)

    expert_params = {}
    for dd in DETECTORS:
        if dd not in results["detectors"] or "error" in results["detectors"][dd]:
            continue
        det = results["detectors"][dd]
        expert_params[dd] = {}
        for freq in DRIFT_FREQS:
            exp = det.get("experts", {}).get(str(freq) if isinstance(list(det.get("experts", {}).keys())[0] if det.get("experts") else "0", str) else freq)
            if not exp:
                exp = det.get("experts", {}).get(freq)
            if exp:
                expert_params[dd][freq] = exp["params"]

    available_dds = list(expert_params.keys())
    ensemble_criteria = ["any", "majority", "all"]

    dd_ranking = []
    for dd in available_dds:
        det = results["detectors"][dd]
        exp_macro = det.get("expert_macro_f1", 0)
        dd_ranking.append((dd, exp_macro))
    dd_ranking.sort(key=lambda x: x[1], reverse=True)
    top_dds = [dd for dd, _ in dd_ranking[:ENSEMBLE_TOP_N]]
    print(f"  Top {ENSEMBLE_TOP_N} detectors by expert macro F1: "
          f"{[(dd, f'{f:.4f}') for dd, f in dd_ranking[:ENSEMBLE_TOP_N]]}", flush=True)

    best_ensemble_per_freq = {}
    best_ensemble_macro = -1
    best_ensemble_config = None

    for size in range(2, ENSEMBLE_MAX_SIZE + 1):
        for combo in itertools.combinations(top_dds, size):
            for crit in ensemble_criteria:
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

                    f1s = []
                    for i, seed in enumerate(EVAL_SEEDS):
                        result = run_ensemble_eval(
                            slot_specs, "any", crit, 1,
                            freq, seed, 700 + i,
                            FIXED_TOLERANCE, FIXED_SUPPRESSION)
                        f1s.append(result["f1"] if result else 0.0)
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

                for freq in DRIFT_FREQS:
                    prev = best_ensemble_per_freq.get(freq, {})
                    prev_macro = prev.get("macro_f1", -1)
                    if combo_detail.get(freq, 0) > prev_macro:
                        best_ensemble_per_freq[freq] = {
                            "detectors": list(combo),
                            "criterion": crit,
                            "macro_f1": combo_detail[freq],
                        }

    # Cross-DD best-by-eval ensemble
    print(f"\n  --- Cross-DD best-by-eval ensemble ---", flush=True)
    cross_f1s = []
    cross_detail = {}
    for freq in DRIFT_FREQS:
        best_dd = None
        best_eval_f1 = -1
        for dd in available_dds:
            det = results["detectors"][dd]
            exp = det.get("experts", {}).get(str(freq)) or det.get("experts", {}).get(freq)
            if exp and exp.get("eval_f1", 0) > best_eval_f1:
                best_eval_f1 = exp["eval_f1"]
                best_dd = dd
        if best_dd:
            params = expert_params[best_dd][freq]
            f1s = []
            for i, seed in enumerate(EVAL_SEEDS):
                f1 = run_single(best_dd, params, freq, seed, 900 + i,
                                FIXED_TOLERANCE, FIXED_SUPPRESSION)
                f1s.append(f1)
            mean_f1 = sum(f1s) / len(f1s)
            cross_detail[freq] = {"detector": best_dd, "f1": mean_f1}
            cross_f1s.append(mean_f1)
            print(f"    freq={freq}: {best_dd} F1={mean_f1:.4f}", flush=True)
        else:
            cross_f1s.append(0.0)
    cross_macro = sum(cross_f1s) / len(cross_f1s)
    print(f"  Cross-DD best-by-eval macro F1: {cross_macro:.4f}", flush=True)

    return {
        "best_fixed_ensemble": best_ensemble_config,
        "cross_dd_best_by_eval": {"macro_f1": cross_macro, "detail": cross_detail},
        "best_ensemble_per_freq": best_ensemble_per_freq,
        "dd_ranking": [(dd, f) for dd, f in dd_ranking],
    }


# ---------------------------------------------------------------------------
# Phase 4: Joint 2-DD ensemble optimization on WaveformDrift2
# ---------------------------------------------------------------------------

def make_joint_ensemble_objective(dd1, dd2, freqs, seeds):
    max_window = int(min(freqs) * MAX_WINDOW_FRACTION)

    def objective(trial):
        params1 = _suggest_detector_params(trial, "d1_", dd1, max_window=max_window)
        params2 = _suggest_detector_params(trial, "d2_", dd2, max_window=max_window)
        ens_crit = trial.suggest_categorical("ensemble_criterion",
                                              ["any", "majority", "all"])
        det_crit = trial.suggest_categorical("detector_criterion",
                                              ["any", "majority", "all"])
        dec_window = trial.suggest_int("decision_window", 1, 10)

        f1s = []
        for freq in freqs:
            for i, seed in enumerate(seeds):
                slot_specs = [(dd1, params1), (dd2, params2)]
                result = run_ensemble_eval(
                    slot_specs, det_crit, ens_crit, dec_window,
                    freq, seed, 200 + i,
                    FIXED_TOLERANCE, FIXED_SUPPRESSION)
                f1s.append(result["f1"] if result else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def optimize_joint_ensembles(results):
    print(f"\n{'='*70}")
    print(f"Phase 4: Joint 2-DD Ensemble Optimization on {GENERATOR}")
    print(f"{'='*70}", flush=True)

    dd_ranking = results.get("ensemble_search", {}).get("dd_ranking", [])
    if not dd_ranking:
        dd_ranking = []
        for dd in DETECTORS:
            if dd in results["detectors"] and "error" not in results["detectors"][dd]:
                exp_macro = results["detectors"][dd].get("expert_macro_f1", 0)
                dd_ranking.append((dd, exp_macro))
        dd_ranking.sort(key=lambda x: x[1], reverse=True)

    top_dds = [dd for dd, _ in dd_ranking[:3]]
    print(f"  Top 3 detectors for joint optimization: {top_dds}", flush=True)

    joint_results = {}
    for dd1, dd2 in itertools.combinations(top_dds, 2):
        pair_name = f"{dd1}+{dd2}"
        print(f"\n  --- Joint optimization: {pair_name} ---", flush=True)
        t0 = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(
            make_joint_ensemble_objective(dd1, dd2, DRIFT_FREQS, TRAIN_SEEDS),
            n_trials=N_ENSEMBLE_TRIALS,
            show_progress_bar=True,
            callbacks=[_early_stop_cb])

        best = study.best_trial
        print(f"  Best train F1: {best.value:.4f}")
        print(f"  Best params: {best.params}")
        print(f"  Time: {time.time() - t0:.1f}s", flush=True)

        params1 = {k[3:]: v for k, v in best.params.items() if k.startswith("d1_")}
        params2 = {k[3:]: v for k, v in best.params.items() if k.startswith("d2_")}
        ens_crit = best.params.get("ensemble_criterion", "all")
        det_crit = best.params.get("detector_criterion", "any")
        dec_window = best.params.get("decision_window", 1)

        slot_specs = [(dd1, params1), (dd2, params2)]
        eval_results = eval_ensemble(
            slot_specs, det_crit, ens_crit, dec_window,
            DRIFT_FREQS, EVAL_SEEDS)
        eval_f1s = [eval_results[f]["mean_f1"] for f in DRIFT_FREQS]
        eval_macro = sum(eval_f1s) / len(eval_f1s)
        print(f"  Eval macro F1: {eval_macro:.4f}")
        for f in DRIFT_FREQS:
            print(f"    freq={f}: F1={eval_results[f]['mean_f1']:.4f}", flush=True)

        joint_results[pair_name] = {
            "train_f1": best.value,
            "params_d1": params1,
            "params_d2": params2,
            "ensemble_criterion": ens_crit,
            "detector_criterion": det_crit,
            "decision_window": dec_window,
            "eval_macro_f1": eval_macro,
            "eval_per_freq": {f: eval_results[f]["mean_f1"] for f in DRIFT_FREQS},
        }

    return joint_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Overnight Benchmark: WaveformDrift2 + Cross-Generator Transfer")
    print(f"Generator: {GENERATOR}")
    print(f"Frequencies: {DRIFT_FREQS}")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Tolerance: {FIXED_TOLERANCE} (fixed)")
    print(f"Suppression: {FIXED_SUPPRESSION} (fixed)")
    print(f"Trials per optimization: {N_TRIALS}")
    print(f"Per-stream timeout: {PER_STREAM_TIMEOUT}s")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors: {DETECTORS}")
    print(f"Output: {OUTPUT_DIR}/waveform_*")
    print(flush=True)

    results = {
        "config": {
            "generator": GENERATOR,
            "drift_freqs": DRIFT_FREQS,
            "stream_length": STREAM_LENGTH,
            "n_trials": N_TRIALS,
            "tolerance": FIXED_TOLERANCE,
            "suppression": FIXED_SUPPRESSION,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "detectors": DETECTORS,
            "per_stream_timeout": PER_STREAM_TIMEOUT,
        },
        "detectors": {},
    }

    start_time = time.time()

    # Phase 1: Single-detector optimization
    print(f"\n{'#'*70}")
    print(f"# PHASE 1: Single-detector optimization on {GENERATOR}")
    print(f"{'#'*70}", flush=True)

    for kind in DETECTORS:
        try:
            det_results = optimize_detector(kind)
            results["detectors"][kind] = det_results
            with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"\n  ERROR with {kind}: {e}")
            import traceback
            traceback.print_exc()
            results["detectors"][kind] = {"error": str(e)}
            with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Phase 1b: Tight-tolerance evaluation (tol=20) to expose delay differences
    print(f"\n{'#'*70}")
    print(f"# PHASE 1b: Tight-tolerance evaluation (tol=20) on both generators")
    print(f"{'#'*70}", flush=True)

    TIGHT_TOL = 20
    tight_results = {"waveform": {}, "sineclusters": {}}
    for dd in DETECTORS:
        if dd not in results["detectors"] or "error" in results["detectors"][dd]:
            continue
        det = results["detectors"][dd]
        gen_params = det.get("generalist", {}).get("params", {})
        if not gen_params:
            continue
        # Evaluate on WaveformDrift2 with tight tolerance
        wf_eval = eval_params(dd, gen_params, DRIFT_FREQS, EVAL_SEEDS,
                              generator="WaveformDrift2")
        # Override tolerance in the worker calls
        wf_f1s = []
        for freq in DRIFT_FREQS:
            f1s = []
            for i, seed in enumerate(EVAL_SEEDS):
                f1 = run_single(dd, gen_params, freq, seed, 500 + i,
                                TIGHT_TOL, FIXED_SUPPRESSION, generator="WaveformDrift2")
                f1s.append(f1)
            wf_f1s.append(sum(f1s) / len(f1s))
        wf_macro = sum(wf_f1s) / len(wf_f1s)

        # Evaluate on SineClusters with tight tolerance
        sc_f1s = []
        for freq in DRIFT_FREQS:
            f1s = []
            for i, seed in enumerate(EVAL_SEEDS):
                f1 = run_single(dd, gen_params, freq, seed, 500 + i,
                                TIGHT_TOL, FIXED_SUPPRESSION, generator="SineClusters")
                f1s.append(f1)
            sc_f1s.append(sum(f1s) / len(f1s))
        sc_macro = sum(sc_f1s) / len(sc_f1s)

        tight_results["waveform"][dd] = {"macro_f1": wf_macro,
                                          "per_freq": dict(zip(DRIFT_FREQS, wf_f1s))}
        tight_results["sineclusters"][dd] = {"macro_f1": sc_macro,
                                               "per_freq": dict(zip(DRIFT_FREQS, sc_f1s))}
        print(f"  {dd}: WF={wf_macro:.4f}, SC={sc_macro:.4f} (tol=20)", flush=True)

    results["tight_tolerance"] = tight_results
    with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Phase 2: Cross-generator transfer
    print(f"\n{'#'*70}")
    print(f"# PHASE 2: Cross-generator transfer evaluation")
    print(f"{'#'*70}", flush=True)

    try:
        transfer = cross_generator_transfer(results)
        results["cross_transfer"] = transfer
        with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"\nERROR in transfer: {e}")
        import traceback
        traceback.print_exc()

    # Phase 3: Multi-DD ensemble search
    print(f"\n{'#'*70}")
    print(f"# PHASE 3: Multi-DD ensemble search on {GENERATOR}")
    print(f"{'#'*70}", flush=True)

    try:
        ensemble_results = search_multi_dd_ensembles(results)
        results["ensemble_search"] = ensemble_results
        with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"\nERROR in ensemble search: {e}")
        import traceback
        traceback.print_exc()

    # Phase 4: Joint ensemble optimization
    print(f"\n{'#'*70}")
    print(f"# PHASE 4: Joint 2-DD ensemble optimization on {GENERATOR}")
    print(f"{'#'*70}", flush=True)

    try:
        joint_results = optimize_joint_ensembles(results)
        results["joint_ensembles"] = joint_results
        with open(f"{OUTPUT_DIR}/waveform_results_partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"\nERROR in joint optimization: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — {GENERATOR}")
    print(f"{'='*70}")
    print(f"\nSingle detectors (fair: tol={FIXED_TOLERANCE}, supp={FIXED_SUPPRESSION}):")
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
        if es.get("cross_dd_best_by_eval"):
            print(f"Cross-DD best-by-eval: {es['cross_dd_best_by_eval']['macro_f1']:.4f}")

    if "joint_ensembles" in results:
        print(f"\nJoint ensembles:")
        for pair, jr in results["joint_ensembles"].items():
            print(f"  {pair}: train={jr['train_f1']:.4f} eval={jr['eval_macro_f1']:.4f} "
                  f"(crit={jr['ensemble_criterion']}, dw={jr['decision_window']})")

    if "cross_transfer" in results:
        ct = results["cross_transfer"]
        print(f"\nCross-generator transfer:")
        if ct.get("sc_to_wf"):
            print(f"  SineClusters → WaveformDrift2 (generalist):")
            for dd, t in ct["sc_to_wf"].items():
                if "generalist" in t:
                    print(f"    {dd}: {t['generalist']['sc_f1']:.4f} → {t['generalist']['wf_f1']:.4f} "
                          f"(Δ={t['generalist']['delta']:+.4f})")
        if ct.get("wf_to_sc"):
            print(f"  WaveformDrift2 → SineClusters (generalist):")
            for dd, t in ct["wf_to_sc"].items():
                if "generalist" in t:
                    print(f"    {dd}: {t['generalist']['wf_f1']:.4f} → {t['generalist']['sc_f1']:.4f} "
                          f"(Δ={t['generalist']['delta']:+.4f})")
        if ct.get("mixed_ensemble"):
            print(f"  Mixed ensembles:")
            for key, me in ct["mixed_ensemble"].items():
                print(f"    {key}: {me['macro_f1']:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Save final results
    with open(f"{OUTPUT_DIR}/waveform_results_final.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary CSV
    with open(f"{OUTPUT_DIR}/waveform_summary.csv", "w", newline="") as f:
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
            if es.get("cross_dd_best_by_eval"):
                w.writerow(["cross_dd_best", "varies", "n/a",
                            f"{es['cross_dd_best_by_eval']['macro_f1']:.4f}"])
        if "joint_ensembles" in results:
            w.writerow([])
            for pair, jr in results["joint_ensembles"].items():
                w.writerow(["joint", pair, jr["ensemble_criterion"],
                            f"{jr['eval_macro_f1']:.4f}"])

    print(f"\nResults saved to {OUTPUT_DIR}/waveform_*")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
