#!/usr/bin/env python3
"""Ensemble benchmark with custom threshold criteria.

Uses fixed optimized single-detector params and searches over:
  - threshold k (at_least_1 through at_least_7)
  - decision_window (1-20)

Also evaluates on high-frequency-only scenario (1000, 2000) where
all single detectors are weak, to find where ensemble truly shines.
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
HIGH_FREQS = [1000, 2000]
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
N_TRIALS = 20

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


def run_eval_combo(slot_specs, criterion, dw, freqs, label=""):
    """Evaluate one (criterion, dw) combo on all generators and both tolerances."""
    sc_ev = eval_ensemble(slot_specs, criterion, dw,
                          SCENARIO_GENERATORS[0], freqs, EVAL_SEEDS,
                          tolerance=TOLERANCE, suppression=SUPPRESSION)
    wf_ev = eval_ensemble(slot_specs, criterion, dw,
                          SCENARIO_GENERATORS[1], freqs, EVAL_SEEDS,
                          tolerance=TOLERANCE, suppression=SUPPRESSION)
    sc_m = macro_f1(sc_ev, freqs)
    wf_m = macro_f1(wf_ev, freqs)
    comb = (sc_m + wf_m) / 2

    sc_tight = eval_ensemble(slot_specs, criterion, dw,
                             SCENARIO_GENERATORS[0], freqs, EVAL_SEEDS,
                             tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
    wf_tight = eval_ensemble(slot_specs, criterion, dw,
                             SCENARIO_GENERATORS[1], freqs, EVAL_SEEDS,
                             tolerance=TIGHT_TOLERANCE, suppression=SUPPRESSION)
    sc_t = macro_f1(sc_tight, freqs)
    wf_t = macro_f1(wf_tight, freqs)
    comb_t = (sc_t + wf_t) / 2

    print(f"  {label} crit={criterion} dw={dw}: "
          f"SC={sc_m:.4f} WF={wf_m:.4f} Comb={comb:.4f} | "
          f"tight: SC={sc_t:.4f} WF={wf_t:.4f} Comb={comb_t:.4f}", flush=True)

    return {
        "criterion": criterion, "decision_window": dw,
        "sc_f1": sc_m, "wf_f1": wf_m, "combined_f1": comb,
        "sc_f1_tight": sc_t, "wf_f1_tight": wf_t, "combined_f1_tight": comb_t,
        "sc_detail": {str(f): sc_ev[f] for f in freqs},
        "wf_detail": {str(f): wf_ev[f] for f in freqs},
        "sc_detail_tight": {str(f): sc_tight[f] for f in freqs},
        "wf_detail_tight": {str(f): wf_tight[f] for f in freqs},
    }


def main():
    print("=" * 70)
    print("Ensemble Threshold Search Benchmark")
    print("=" * 70)

    # Load optimized single-detector params
    partial_path = f"{OUTPUT_DIR}/ensemble_benchmark_partial.json"
    if not os.path.exists(partial_path):
        print(f"ERROR: {partial_path} not found.")
        return

    all_results = json.load(open(partial_path))

    slot_specs = []
    for kind in DETECTORS:
        key = f"single_{kind}"
        if key not in all_results or "error" in all_results.get(key, {}):
            print(f"WARNING: {key} not found or has error")
            continue
        params = all_results[key]["params"]
        slot_specs.append((kind, params))
        print(f"  {kind}: loaded params")

    print(f"\nLoaded {len(slot_specs)} detector configs")

    # ---- Phase 1: Optimize on ALL frequencies ----
    print(f"\n{'='*70}")
    print(f"Phase 1: Optimize threshold+dw on ALL freqs {ALL_FREQS}")
    print(f"{'='*70}", flush=True)

    def make_objective(freqs):
        def objective(trial):
            k = trial.suggest_int("k", 1, len(slot_specs))
            criterion = f"at_least_{k}"
            dw = trial.suggest_int("decision_window", 1, 20)

            f1s = []
            for generator in SCENARIO_GENERATORS:
                for freq in freqs:
                    for s_idx, seed in enumerate(TRAIN_SEEDS):
                        ctx = mp.get_context("fork")
                        queue = ctx.Queue()
                        proc = ctx.Process(target=_ensemble_worker,
                                           args=(queue, slot_specs, criterion,
                                                 dw, freq, seed, s_idx,
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

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(ALL_FREQS), n_trials=N_TRIALS, show_progress_bar=True)
    best_k = study.best_trial.params["k"]
    best_dw = study.best_trial.params["decision_window"]
    best_criterion = f"at_least_{best_k}"
    print(f"\nBest (all freqs): k={best_k}, dw={best_dw}, train F1={study.best_value:.4f}")
    print(f"Time: {time.time() - t0:.1f}s")

    # ---- Phase 2: Evaluate all thresholds with best dw on eval seeds ----
    print(f"\n{'='*70}")
    print(f"Phase 2: Evaluate all thresholds (dw={best_dw}) on eval seeds")
    print(f"{'='*70}", flush=True)

    results_all = {}
    for k in range(1, len(slot_specs) + 1):
        crit = f"at_least_{k}"
        results_all[f"all_freqs_k{k}"] = run_eval_combo(
            slot_specs, crit, best_dw, ALL_FREQS,
            label=f"[ALL freqs]")

    # Also try a few different dws with best k
    for dw in [1, 3, 5, 7, 10, 15, 20]:
        if dw == best_dw:
            continue
        results_all[f"all_freqs_k{best_k}_dw{dw}"] = run_eval_combo(
            slot_specs, best_criterion, dw, ALL_FREQS,
            label=f"[ALL freqs, dw={dw}]")

    # ---- Phase 3: Optimize on HIGH frequencies only ----
    print(f"\n{'='*70}")
    print(f"Phase 3: Optimize threshold+dw on HIGH freqs {HIGH_FREQS}")
    print(f"{'='*70}", flush=True)

    study2 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    study2.optimize(make_objective(HIGH_FREQS), n_trials=N_TRIALS, show_progress_bar=True)
    best_k_hi = study2.best_trial.params["k"]
    best_dw_hi = study2.best_trial.params["decision_window"]
    best_criterion_hi = f"at_least_{best_k_hi}"
    print(f"\nBest (high freqs): k={best_k_hi}, dw={best_dw_hi}, train F1={study2.best_value:.4f}")
    print(f"Time: {time.time() - t0:.1f}s")

    # ---- Phase 4: Evaluate all thresholds on high freqs ----
    print(f"\n{'='*70}")
    print(f"Phase 4: Evaluate all thresholds on HIGH freqs (dw={best_dw_hi})")
    print(f"{'='*70}", flush=True)

    results_high = {}
    for k in range(1, len(slot_specs) + 1):
        crit = f"at_least_{k}"
        results_high[f"high_freqs_k{k}"] = run_eval_combo(
            slot_specs, crit, best_dw_hi, HIGH_FREQS,
            label=f"[HIGH freqs]")

    for dw in [1, 3, 5, 7, 10, 15, 20]:
        if dw == best_dw_hi:
            continue
        results_high[f"high_freqs_k{best_k_hi}_dw{dw}"] = run_eval_combo(
            slot_specs, best_criterion_hi, dw, HIGH_FREQS,
            label=f"[HIGH freqs, dw={dw}]")

    # ---- Phase 5: Also evaluate single detectors on high freqs ----
    print(f"\n{'='*70}")
    print(f"Phase 5: Single detector comparison on HIGH freqs")
    print(f"{'='*70}", flush=True)

    print(f"\n{'Detector':<12} {'SC f1k':>8} {'SC f2k':>8} {'SC macro':>10} {'WF f1k':>8} {'WF f2k':>8} {'WF macro':>10} {'Combined':>10}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    single_high = {}
    for kind in DETECTORS:
        d = all_results.get(f"single_{kind}", {})
        if "error" in d or not d:
            continue
        sc_d = d.get("sc_detail", {})
        wf_d = d.get("wf_detail", {})
        sc_vals = [sc_d.get(str(f), 0) for f in HIGH_FREQS]
        wf_vals = [wf_d.get(str(f), 0) for f in HIGH_FREQS]
        sc_m = sum(sc_vals) / len(sc_vals)
        wf_m = sum(wf_vals) / len(wf_vals)
        comb = (sc_m + wf_m) / 2
        print(f"{kind:<12} {sc_vals[0]:>8.4f} {sc_vals[1]:>8.4f} {sc_m:>10.4f} {wf_vals[0]:>8.4f} {wf_vals[1]:>8.4f} {wf_m:>10.4f} {comb:>10.4f}")
        single_high[kind] = {"sc_macro": sc_m, "wf_macro": wf_m, "combined": comb}

    # ---- Final comparison ----
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")

    best_single_all = max(
        (all_results.get(f"single_{k}", {}).get("combined_f1", 0) for k in DETECTORS),
        default=0)
    best_single_high = max(
        (v["combined"] for v in single_high.values()),
        default=0)

    best_ens_all = max(
        (v["combined_f1"] for v in results_all.values()),
        default=0)
    best_ens_high = max(
        (v["combined_f1"] for v in results_high.values()),
        default=0)

    print(f"\n  ALL freqs:  best single={best_single_all:.4f}, best ensemble={best_ens_all:.4f}")
    print(f"  HIGH freqs: best single={best_single_high:.4f}, best ensemble={best_ens_high:.4f}")

    if best_ens_high > best_single_high:
        print(f"  *** ENSEMBLE WINS on HIGH freqs by {best_ens_high - best_single_high:.4f} ***")
    if best_ens_all > best_single_all:
        print(f"  *** ENSEMBLE WINS on ALL freqs by {best_ens_all - best_single_all:.4f} ***")

    # Save
    output = {
        "results_all_freqs": results_all,
        "results_high_freqs": results_high,
        "single_high_freqs": single_high,
        "best_k_all": best_k, "best_dw_all": best_dw,
        "best_k_high": best_k_hi, "best_dw_high": best_dw_hi,
    }
    output_path = f"{OUTPUT_DIR}/ensemble_threshold_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
