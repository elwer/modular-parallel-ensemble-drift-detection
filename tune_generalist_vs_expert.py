#!/usr/bin/env python3
"""Test whether a single generalist OCDD can handle multiple drift frequencies,
or if per-frequency experts are needed.

Generalist: one param set optimized across all frequencies.
Expert: separate param set per frequency, then combined.
"""
import sys
import time
import optuna
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import _run_one_stream

# --- Config ---
GENERATOR = "SineClusters"
DRIFT_FREQS = [200, 400, 800]
STREAM_LENGTH = 5000
N_TRIALS = 150
SEED = 1337
# Use freq//10 tolerance (strict, no artificial inflation)
TOLERANCES = [f // 10 for f in DRIFT_FREQS]  # 20, 40, 80
SUPPRESSION = None  # will use freq // 2 per stream

TRAIN_SEEDS = [42, 43, 44]
EVAL_SEEDS = [45, 46]


def run_ocdd(params, drift_freq, stream_seed, s_idx, tolerance, suppression):
    """Run single OCDD detector on one stream."""
    tp, fp, fn, delay, f1, prec, rec, n_known = _run_one_stream(
        generator_name=GENERATOR,
        drift_frequency=drift_freq,
        stream_length=STREAM_LENGTH,
        stream_seed=stream_seed,
        tolerance=tolerance,
        slot_specs=[("OCDD", params)],
        detector_seed_base=SEED,
        s_idx=s_idx,
        detector_criterion="any",
        ensemble_criterion="any",
        decision_window=1,
        suppression_window=suppression,
        recent_samples_size=params["n_samples"],
    )
    return tp, fp, fn, f1, prec, rec, n_known


def generalist_objective(trial):
    """One param set for all frequencies."""
    params = {
        "n_samples": trial.suggest_int("n_samples", 50, 500),
        "threshold": trial.suggest_float("threshold", 0.1, 0.9),
    }
    f1s = []
    for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
        supp = freq // 2
        for s_idx, seed in enumerate(TRAIN_SEEDS):
            tp, fp, fn, f1, _, _, _ = run_ocdd(params, freq, seed, s_idx, tol, supp)
            f1s.append(f1)
    return sum(f1s) / len(f1s)


def make_expert_objective(freq, tol):
    """Optimize params for a single drift frequency."""
    supp = freq // 2
    def objective(trial):
        params = {
            "n_samples": trial.suggest_int("n_samples", 50, 500),
            "threshold": trial.suggest_float("threshold", 0.1, 0.9),
        }
        f1s = []
        for s_idx, seed in enumerate(TRAIN_SEEDS):
            tp, fp, fn, f1, _, _, _ = run_ocdd(params, freq, seed, s_idx, tol, supp)
            f1s.append(f1)
        return sum(f1s) / len(f1s)
    return objective


def main():
    print(f"Generalist vs Expert: OCDD on {GENERATOR}")
    print(f"Drift frequencies: {DRIFT_FREQS}")
    print(f"Tolerances: {TOLERANCES}")
    print(f"Suppression: freq//2 per stream")
    print(f"Stream length: {STREAM_LENGTH}, Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Trials: {N_TRIALS}")
    print()

    # --- Generalist: one param set for all ---
    print("=== Generalist (one param set for all freqs) ===")
    t0 = time.time()
    gen_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    gen_study.optimize(generalist_objective, n_trials=N_TRIALS, show_progress_bar=True)
    gen_best = gen_study.best_trial
    print(f"  Best train macro F1: {gen_best.value:.4f}")
    print(f"  Best params: {gen_best.params}")
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n  --- Generalist eval ---")
    gen_eval_f1s = []
    for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
        supp = freq // 2
        for seed in EVAL_SEEDS:
            tp, fp, fn, f1, prec, rec, n_drifts = run_ocdd(
                gen_best.params, freq, seed, 100, tol, supp)
            gen_eval_f1s.append(f1)
            print(f"  freq={freq} seed={seed}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} "
                  f"TP={tp} FP={fp} FN={fn} drifts={n_drifts}")
    print(f"  Generalist eval macro F1: {sum(gen_eval_f1s)/len(gen_eval_f1s):.4f}")

    # --- Experts: one param set per frequency ---
    print("\n=== Experts (separate params per freq) ===")
    expert_params = {}
    for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
        t0 = time.time()
        print(f"\n  --- Expert for freq={freq} (tol={tol}, supp={freq//2}) ---")
        exp_study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=SEED))
        exp_study.optimize(make_expert_objective(freq, tol),
                           n_trials=N_TRIALS, show_progress_bar=True)
        exp_best = exp_study.best_trial
        expert_params[freq] = exp_best.params
        print(f"  Best train F1: {exp_best.value:.4f}")
        print(f"  Best params: {exp_best.params}")
        print(f"  Time: {time.time() - t0:.1f}s")

    print("\n  --- Expert eval (each expert on its own freq) ---")
    exp_eval_f1s = []
    for freq, tol in zip(DRIFT_FREQS, TOLERANCES):
        supp = freq // 2
        params = expert_params[freq]
        for seed in EVAL_SEEDS:
            tp, fp, fn, f1, prec, rec, n_drifts = run_ocdd(
                params, freq, seed, 200, tol, supp)
            exp_eval_f1s.append(f1)
            print(f"  freq={freq} seed={seed}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} "
                  f"TP={tp} FP={fp} FN={fn} drifts={n_drifts}")
    print(f"  Expert eval macro F1: {sum(exp_eval_f1s)/len(exp_eval_f1s):.4f}")

    # --- Summary ---
    gen_eval = sum(gen_eval_f1s) / len(gen_eval_f1s)
    exp_eval = sum(exp_eval_f1s) / len(exp_eval_f1s)
    print(f"\n=== SUMMARY ===")
    print(f"Generalist eval F1: {gen_eval:.4f}")
    print(f"Expert eval F1:     {exp_eval:.4f}")
    print(f"Improvement:        {exp_eval - gen_eval:+.4f}")


if __name__ == "__main__":
    main()
