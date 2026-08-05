#!/usr/bin/env python3
"""Combined benchmark: evaluate all detectors on BOTH SineClusters and WaveformDrift2.

Produces a single side-by-side comparison table showing each detector's F1
on each generator, with three param sources:
  1. SC-optimized generalist params (from SineClusters benchmark)
  2. WF-optimized generalist params (from WaveformDrift2 benchmark)
  3. Joint generalist params (optimized on both generators simultaneously)

Also evaluates with tight tolerance (tol=20) to expose detection delay.
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
    _run_one_stream, _suggest_detector_params, _instantiate,
    build_stream, CANDIDATES, CLASS_PATH, MAX_WINDOW_FRACTION, _cap,
    _f1_from_counts
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections

# --- Config ---
GENERATORS = ["SineClusters", "WaveformDrift2"]
DRIFT_FREQS = [200, 500, 1000, 2000]
STREAM_LENGTH = 4000
SEED = 1337
FIXED_TOLERANCE = 100
FIXED_SUPPRESSION = 100
TIGHT_TOLERANCE = 20
TRAIN_SEEDS = [42]
EVAL_SEEDS = [45, 46]
DETECTORS = ["OCDD", "IBDD", "UDetect", "SPLL", "D3", "CSDDM", "BNDM"]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 30
N_JOINT_TRIALS = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-stream evaluation
# ---------------------------------------------------------------------------

def _single_worker(queue, kind, params, freq, stream_seed, s_idx,
                   tolerance, suppression, generator):
    try:
        n_samples_key = "n_samples" if "n_samples" in params else "n_reference_samples"
        recent_size = params.get(n_samples_key, 200)
        tp, fp, fn, delay, f1, prec, rec, n_known = _run_one_stream(
            generator_name=generator,
            drift_frequency=freq,
            stream_length=STREAM_LENGTH,
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
        queue.put(("ok", tp, fp, fn, f1, prec, rec, n_known, delay))
    except Exception as e:
        queue.put(("error", str(e)))


def eval_on_generator(kind, params, generator, freqs, seeds,
                      tolerance=FIXED_TOLERANCE,
                      suppression=FIXED_SUPPRESSION):
    """Evaluate a single detector on one generator, return per-freq results."""
    results = {}
    for freq in freqs:
        seed_results = []
        f1s = []
        for i, seed in enumerate(seeds):
            ctx = mp.get_context("fork")
            queue = ctx.Queue()
            proc = ctx.Process(target=_single_worker,
                               args=(queue, kind, params, freq, seed, 500 + i,
                                     tolerance, suppression, generator))
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
                        _, tp, fp, fn, f1, prec, rec, n_known, delay = result
                        seed_results.append({
                            "seed": seed, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                            "precision": prec, "recall": rec, "n_drifts": n_known,
                            "delay": delay,
                        })
                        f1s.append(f1)
                    else:
                        seed_results.append({"seed": seed, "f1": 0.0, "error": result[1]})
                        f1s.append(0.0)
                except:
                    seed_results.append({"seed": seed, "f1": 0.0, "error": "no result"})
                    f1s.append(0.0)
        results[freq] = {
            "seeds": seed_results,
            "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    return results


def macro_f1(eval_results, freqs):
    return sum(eval_results[f]["mean_f1"] for f in freqs) / len(freqs)


# ---------------------------------------------------------------------------
# Joint generalist optimization (both generators)
# ---------------------------------------------------------------------------

def make_joint_generalist_objective(kind):
    max_window = int(min(DRIFT_FREQS) * MAX_WINDOW_FRACTION)
    def objective(trial):
        params = _suggest_detector_params(trial, "", kind, max_window=max_window)
        f1s = []
        for generator in GENERATORS:
            for freq in DRIFT_FREQS:
                for s_idx, seed in enumerate(TRAIN_SEEDS):
                    ctx = mp.get_context("fork")
                    queue = ctx.Queue()
                    proc = ctx.Process(target=_single_worker,
                                       args=(queue, kind, params, freq, seed, s_idx,
                                             FIXED_TOLERANCE, FIXED_SUPPRESSION,
                                             generator))
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
                            f1s.append(result[4] if result[0] == "ok" else 0.0)
                        except:
                            f1s.append(0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0
    return objective


def optimize_joint_generalist(kind):
    print(f"\n  --- Joint generalist ({kind}) ---", flush=True)
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_joint_generalist_objective(kind),
                   n_trials=N_JOINT_TRIALS, show_progress_bar=True)
    best = study.best_trial
    print(f"  Joint train F1: {best.value:.4f}")
    print(f"  Joint params: {best.params}")
    print(f"  Time: {time.time() - t0:.1f}s", flush=True)
    return best.params, best.value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Combined Benchmark: SineClusters + WaveformDrift2")
    print("=" * 70)
    print(f"Generators: {GENERATORS}")
    print(f"Frequencies: {DRIFT_FREQS}")
    print(f"Stream length: {STREAM_LENGTH}")
    print(f"Tolerance: {FIXED_TOLERANCE} (standard), {TIGHT_TOLERANCE} (tight)")
    print(f"Suppression: {FIXED_SUPPRESSION}")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Detectors: {DETECTORS}")
    print(flush=True)

    # Load existing optimized params
    sc_results = json.load(open(f"{OUTPUT_DIR}/results_final.json"))
    wf_results = json.load(open(f"{OUTPUT_DIR}/waveform_results_final.json"))

    all_results = {}

    # Resume from partial results if available
    partial_path = f"{OUTPUT_DIR}/combined_results_partial.json"
    if os.path.exists(partial_path):
        try:
            all_results = json.load(open(partial_path))
            print(f"Resumed from partial results: {list(all_results.keys())}", flush=True)
        except Exception:
            all_results = {}

    start_time = time.time()

    for kind in DETECTORS:
        print(f"\n{'='*70}")
        print(f"Detector: {kind}")
        print(f"{'='*70}", flush=True)

        if kind in all_results:
            print(f"  Already completed, skipping.", flush=True)
            continue

        det_result = {}

        # Get existing params
        sc_gen_params = sc_results.get("detectors", {}).get(kind, {}).get("generalist", {}).get("params", {})
        wf_gen_params = wf_results.get("detectors", {}).get(kind, {}).get("generalist", {}).get("params", {})

        # --- Evaluate SC-optimized params on both generators ---
        if sc_gen_params:
            print(f"\n  SC-optimized params on SineClusters (tol=100):", flush=True)
            sc_on_sc = eval_on_generator(kind, sc_gen_params, "SineClusters",
                                         DRIFT_FREQS, EVAL_SEEDS)
            sc_sc_macro = macro_f1(sc_on_sc, DRIFT_FREQS)
            print(f"    macro F1 = {sc_sc_macro:.4f}")

            print(f"  SC-optimized params on WaveformDrift2 (tol=100):", flush=True)
            sc_on_wf = eval_on_generator(kind, sc_gen_params, "WaveformDrift2",
                                         DRIFT_FREQS, EVAL_SEEDS)
            sc_wf_macro = macro_f1(sc_on_wf, DRIFT_FREQS)
            print(f"    macro F1 = {sc_wf_macro:.4f}")

            # Tight tolerance
            print(f"  SC-optimized params on SineClusters (tol=20):", flush=True)
            sc_on_sc_tight = eval_on_generator(kind, sc_gen_params, "SineClusters",
                                                DRIFT_FREQS, EVAL_SEEDS,
                                                tolerance=TIGHT_TOLERANCE)
            sc_sc_tight = macro_f1(sc_on_sc_tight, DRIFT_FREQS)
            print(f"    macro F1 = {sc_sc_tight:.4f}")

            print(f"  SC-optimized params on WaveformDrift2 (tol=20):", flush=True)
            sc_on_wf_tight = eval_on_generator(kind, sc_gen_params, "WaveformDrift2",
                                                DRIFT_FREQS, EVAL_SEEDS,
                                                tolerance=TIGHT_TOLERANCE)
            sc_wf_tight = macro_f1(sc_on_wf_tight, DRIFT_FREQS)
            print(f"    macro F1 = {sc_wf_tight:.4f}")

            det_result["sc_params"] = {
                "params": sc_gen_params,
                "sc_f1": sc_sc_macro,
                "wf_f1": sc_wf_macro,
                "sc_f1_tight": sc_sc_tight,
                "wf_f1_tight": sc_wf_tight,
                "sc_detail": {f: sc_on_sc[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail": {f: sc_on_wf[f]["mean_f1"] for f in DRIFT_FREQS},
                "sc_detail_tight": {f: sc_on_sc_tight[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail_tight": {f: sc_on_wf_tight[f]["mean_f1"] for f in DRIFT_FREQS},
            }

        # --- Evaluate WF-optimized params on both generators ---
        if wf_gen_params:
            print(f"\n  WF-optimized params on SineClusters (tol=100):", flush=True)
            wf_on_sc = eval_on_generator(kind, wf_gen_params, "SineClusters",
                                         DRIFT_FREQS, EVAL_SEEDS)
            wf_sc_macro = macro_f1(wf_on_sc, DRIFT_FREQS)
            print(f"    macro F1 = {wf_sc_macro:.4f}")

            print(f"  WF-optimized params on WaveformDrift2 (tol=100):", flush=True)
            wf_on_wf = eval_on_generator(kind, wf_gen_params, "WaveformDrift2",
                                         DRIFT_FREQS, EVAL_SEEDS)
            wf_wf_macro = macro_f1(wf_on_wf, DRIFT_FREQS)
            print(f"    macro F1 = {wf_wf_macro:.4f}")

            # Tight tolerance
            print(f"  WF-optimized params on SineClusters (tol=20):", flush=True)
            wf_on_sc_tight = eval_on_generator(kind, wf_gen_params, "SineClusters",
                                                DRIFT_FREQS, EVAL_SEEDS,
                                                tolerance=TIGHT_TOLERANCE)
            wf_sc_tight = macro_f1(wf_on_sc_tight, DRIFT_FREQS)
            print(f"    macro F1 = {wf_sc_tight:.4f}")

            print(f"  WF-optimized params on WaveformDrift2 (tol=20):", flush=True)
            wf_on_wf_tight = eval_on_generator(kind, wf_gen_params, "WaveformDrift2",
                                                DRIFT_FREQS, EVAL_SEEDS,
                                                tolerance=TIGHT_TOLERANCE)
            wf_wf_tight = macro_f1(wf_on_wf_tight, DRIFT_FREQS)
            print(f"    macro F1 = {wf_wf_tight:.4f}")

            det_result["wf_params"] = {
                "params": wf_gen_params,
                "sc_f1": wf_sc_macro,
                "wf_f1": wf_wf_macro,
                "sc_f1_tight": wf_sc_tight,
                "wf_f1_tight": wf_wf_tight,
                "sc_detail": {f: wf_on_sc[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail": {f: wf_on_wf[f]["mean_f1"] for f in DRIFT_FREQS},
                "sc_detail_tight": {f: wf_on_sc_tight[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail_tight": {f: wf_on_wf_tight[f]["mean_f1"] for f in DRIFT_FREQS},
            }

        # --- Joint generalist optimization ---
        try:
            joint_params, joint_train_f1 = optimize_joint_generalist(kind)

            print(f"\n  Joint-optimized params on SineClusters (tol=100):", flush=True)
            j_on_sc = eval_on_generator(kind, joint_params, "SineClusters",
                                        DRIFT_FREQS, EVAL_SEEDS)
            j_sc_macro = macro_f1(j_on_sc, DRIFT_FREQS)
            print(f"    macro F1 = {j_sc_macro:.4f}")

            print(f"  Joint-optimized params on WaveformDrift2 (tol=100):", flush=True)
            j_on_wf = eval_on_generator(kind, joint_params, "WaveformDrift2",
                                        DRIFT_FREQS, EVAL_SEEDS)
            j_wf_macro = macro_f1(j_on_wf, DRIFT_FREQS)
            print(f"    macro F1 = {j_wf_macro:.4f}")

            # Tight tolerance
            print(f"  Joint-optimized params on SineClusters (tol=20):", flush=True)
            j_on_sc_tight = eval_on_generator(kind, joint_params, "SineClusters",
                                               DRIFT_FREQS, EVAL_SEEDS,
                                               tolerance=TIGHT_TOLERANCE)
            j_sc_tight = macro_f1(j_on_sc_tight, DRIFT_FREQS)
            print(f"    macro F1 = {j_sc_tight:.4f}")

            print(f"  Joint-optimized params on WaveformDrift2 (tol=20):", flush=True)
            j_on_wf_tight = eval_on_generator(kind, joint_params, "WaveformDrift2",
                                               DRIFT_FREQS, EVAL_SEEDS,
                                               tolerance=TIGHT_TOLERANCE)
            j_wf_tight = macro_f1(j_on_wf_tight, DRIFT_FREQS)
            print(f"    macro F1 = {j_wf_tight:.4f}")

            det_result["joint_params"] = {
                "params": joint_params,
                "train_f1": joint_train_f1,
                "sc_f1": j_sc_macro,
                "wf_f1": j_wf_macro,
                "sc_f1_tight": j_sc_tight,
                "wf_f1_tight": j_wf_tight,
                "sc_detail": {f: j_on_sc[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail": {f: j_on_wf[f]["mean_f1"] for f in DRIFT_FREQS},
                "sc_detail_tight": {f: j_on_sc_tight[f]["mean_f1"] for f in DRIFT_FREQS},
                "wf_detail_tight": {f: j_on_wf_tight[f]["mean_f1"] for f in DRIFT_FREQS},
            }
        except Exception as e:
            print(f"  ERROR in joint optimization: {e}")
            import traceback
            traceback.print_exc()

        all_results[kind] = det_result

        # Save partial
        with open(f"{OUTPUT_DIR}/combined_results_partial.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- Print final comparison table ---
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    # Table 1: Standard tolerance (tol=100)
    print(f"\n--- Standard tolerance (tol=100, supp=100) ---")
    print(f"{'Detector':<10} {'Param src':<12} {'SC F1':>8} {'WF F1':>8} {'Combined':>8} {'Gap':>8}")
    print(f"{'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for kind in DETECTORS:
        dr = all_results.get(kind, {})
        for src_label, src_key in [("SC-opt", "sc_params"), ("WF-opt", "wf_params"), ("Joint", "joint_params")]:
            d = dr.get(src_key)
            if not d:
                continue
            sc_f1 = d.get("sc_f1", 0)
            wf_f1 = d.get("wf_f1", 0)
            combined = (sc_f1 + wf_f1) / 2
            gap = abs(sc_f1 - wf_f1)
            print(f"{kind:<10} {src_label:<12} {sc_f1:>8.4f} {wf_f1:>8.4f} {combined:>8.4f} {gap:>8.4f}")
        print()

    # Table 2: Tight tolerance (tol=20)
    print(f"\n--- Tight tolerance (tol=20, supp=100) ---")
    print(f"{'Detector':<10} {'Param src':<12} {'SC F1':>8} {'WF F1':>8} {'Combined':>8} {'Gap':>8}")
    print(f"{'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for kind in DETECTORS:
        dr = all_results.get(kind, {})
        for src_label, src_key in [("SC-opt", "sc_params"), ("WF-opt", "wf_params"), ("Joint", "joint_params")]:
            d = dr.get(src_key)
            if not d:
                continue
            sc_f1 = d.get("sc_f1_tight", 0)
            wf_f1 = d.get("wf_f1_tight", 0)
            combined = (sc_f1 + wf_f1) / 2
            gap = abs(sc_f1 - wf_f1)
            print(f"{kind:<10} {src_label:<12} {sc_f1:>8.4f} {wf_f1:>8.4f} {combined:>8.4f} {gap:>8.4f}")
        print()

    # Table 3: Per-frequency detail for joint params (tol=100)
    print(f"\n--- Per-frequency detail (joint params, tol=100) ---")
    print(f"{'Detector':<10} {'SC f200':>8} {'SC f500':>8} {'SC f1000':>8} {'SC f2000':>8} | {'WF f200':>8} {'WF f500':>8} {'WF f1000':>8} {'WF f2000':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for kind in DETECTORS:
        dr = all_results.get(kind, {}).get("joint_params")
        if not dr:
            continue
        sc_d = dr.get("sc_detail", {})
        wf_d = dr.get("wf_detail", {})
        sc_vals = [f"{sc_d.get(f, 0):.4f}" for f in DRIFT_FREQS]
        wf_vals = [f"{wf_d.get(f, 0):.4f}" for f in DRIFT_FREQS]
        print(f"{kind:<10} {sc_vals[0]:>8} {sc_vals[1]:>8} {sc_vals[2]:>8} {sc_vals[3]:>8} | {wf_vals[0]:>8} {wf_vals[1]:>8} {wf_vals[2]:>8} {wf_vals[3]:>8}")

    # Save CSV
    with open(f"{OUTPUT_DIR}/combined_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "param_source", "sc_f1_tol100", "wf_f1_tol100",
                     "combined_tol100", "sc_f1_tol20", "wf_f1_tol20", "combined_tol20"])
        for kind in DETECTORS:
            dr = all_results.get(kind, {})
            for src_label, src_key in [("SC-opt", "sc_params"), ("WF-opt", "wf_params"), ("Joint", "joint_params")]:
                d = dr.get(src_key)
                if not d:
                    continue
                sc100 = d.get("sc_f1", 0)
                wf100 = d.get("wf_f1", 0)
                sc20 = d.get("sc_f1_tight", 0)
                wf20 = d.get("wf_f1_tight", 0)
                w.writerow([kind, src_label, f"{sc100:.4f}", f"{wf100:.4f}",
                            f"{(sc100+wf100)/2:.4f}", f"{sc20:.4f}", f"{wf20:.4f}",
                            f"{(sc20+wf20)/2:.4f}"])

    # Save full results
    with open(f"{OUTPUT_DIR}/combined_results_final.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to {OUTPUT_DIR}/combined_*")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
