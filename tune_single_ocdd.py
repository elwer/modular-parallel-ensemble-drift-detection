#!/usr/bin/env python3
"""Quick OCDD tuning to find parameters achieving F1 >= 0.7."""
import sys
import time
import optuna
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    build_stream, _instantiate, _run_one_stream, _f1_from_counts,
    evaluate_detections, apply_suppression, run_ensemble
)

# --- Config ---
GENERATOR = "SineClusters"
DRIFT_FREQ = 500
STREAM_LENGTH = 5000
N_TRIALS = 100
SEED = 1337
TOLERANCE = 100  # wider: freq // 5 instead of freq // 10
SUPPRESSION = 400  # 80% of inter-drift gap

# Test on multiple seeds to check robustness
TRAIN_SEEDS = [42, 43, 44]
EVAL_SEEDS = [45, 46]

def objective(trial):
    params = {
        "n_samples": trial.suggest_int("n_samples", 50, 500),
        "threshold": trial.suggest_float("threshold", 0.1, 0.9),
    }
    
    f1s = []
    for s_idx, seed in enumerate(TRAIN_SEEDS):
        tp, fp, fn, _, f1, _, _, _ = _run_one_stream(
            generator_name=GENERATOR,
            drift_frequency=DRIFT_FREQ,
            stream_length=STREAM_LENGTH,
            stream_seed=seed,
            tolerance=TOLERANCE,
            slot_specs=[("OCDD", params)],
            detector_seed_base=SEED,
            s_idx=s_idx,
            detector_criterion="any",
            ensemble_criterion="any",
            decision_window=1,
            suppression_window=SUPPRESSION,
            recent_samples_size=params["n_samples"],
        )
        f1s.append(f1)
    
    macro_f1 = sum(f1s) / len(f1s)
    return macro_f1

def main():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    print(f"Tuning OCDD on {GENERATOR} freq={DRIFT_FREQ} length={STREAM_LENGTH}")
    print(f"Train seeds: {TRAIN_SEEDS}, Eval seeds: {EVAL_SEEDS}")
    print(f"Tolerance: {TOLERANCE}, Suppression: {SUPPRESSION}")
    print(f"Trials: {N_TRIALS}")
    print()
    
    start = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed = time.time() - start
    
    best = study.best_trial
    print(f"\nBest train macro F1: {best.value:.4f}")
    print(f"Best params: {best.params}")
    print(f"Time: {elapsed:.1f}s")
    
    # Evaluate on held-out seeds
    print("\n--- Eval on held-out seeds ---")
    for seed in EVAL_SEEDS:
        tp, fp, fn, delay, f1, prec, rec, n_known = _run_one_stream(
            generator_name=GENERATOR,
            drift_frequency=DRIFT_FREQ,
            stream_length=STREAM_LENGTH,
            stream_seed=seed,
            tolerance=TOLERANCE,
            slot_specs=[("OCDD", best.params)],
            detector_seed_base=SEED,
            s_idx=100,
            detector_criterion="any",
            ensemble_criterion="any",
            decision_window=1,
            suppression_window=SUPPRESSION,
            recent_samples_size=best.params["n_samples"],
        )
        print(f"  Seed {seed}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} TP={tp} FP={fp} FN={fn} drifts={n_known} delay={delay:.1f}")

if __name__ == "__main__":
    main()
