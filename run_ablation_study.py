#!/usr/bin/env python3
"""Ablation study: sweep ensemble parameters and record F1/TP/FP/FN/delay.

Sweeps:
  - detector_criterion (level 1): any, all, majority
  - ensemble_criterion (level 2): any, all, majority
  - decision_window: 1, 2, 3, 5, 10
  - suppression_window: 0, freq//8, freq//4, freq//2, freq

For each combination, evaluates on all 4 drift frequencies and 2 eval seeds.
Uses the best ensemble configs from the benchmark:
  1. UDetect+SPLL (joint-optimized params)
  2. UDetect alone (generalist params)
  3. UDetect+SPLL (expert params per freq)

Outputs:
  - overnight_results/ablation_fair_results.csv
  - overnight_results/ablation_fair_parallel_plot.png
  - overnight_results/ablation_fair_report.md
"""
import sys
import json
import time
import csv
import math
import itertools
import multiprocessing as mp
import os

sys.path.insert(0, ".")
from optimization.synthetic_f1_multistream_optimize_optuna import (
    _instantiate, build_stream, MAX_WINDOW_FRACTION, _f1_from_counts
)
from main_synthetic import run_ensemble, apply_suppression, evaluate_detections

# --- Config (must match benchmark) ---
GENERATOR = "SineClusters"
DRIFT_FREQS = [200, 500, 1000, 2000]
STREAM_LENGTH = 4000
SEED = 1337
EVAL_SEEDS = [45, 46]
OUTPUT_DIR = "overnight_results"
PER_STREAM_TIMEOUT = 30  # generous for ensembles

# Ablation grid
DETECTOR_CRITERIA = ["any", "all", "majority"]
ENSEMBLE_CRITERIA = ["any", "all", "majority"]
DECISION_WINDOWS = [1, 3, 5, 10]
# Fixed suppression windows (same across all frequencies — no oracle knowledge)
SUPPRESSION_WINDOWS = [0, 50, 100, 200, 500, 1000]
# Fixed tolerance (same across all frequencies — no oracle knowledge)
FIXED_TOLERANCE = 100


def _ablation_worker(queue, slot_specs, detector_criterion, ensemble_criterion,
                     decision_window, suppression_window, freq, stream_seed,
                     s_idx, tolerance, stream_length, generator):
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
        dets = apply_suppression(raw_ensemble, suppression_window)
        tp, fp, fn, mean_delay = evaluate_detections(dets, known, tolerance)
        f1 = _f1_from_counts(tp, fp, fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        delay = mean_delay if not math.isnan(mean_delay) else -1.0
        queue.put(("ok", tp, fp, fn, f1, prec, rec, delay, len(known)))
    except Exception as e:
        queue.put(("error", str(e)))


def run_ablation_point(slot_specs, detector_criterion, ensemble_criterion,
                       decision_window, suppression_window, freq, stream_seed,
                       s_idx, tolerance):
    """Run one ablation point with hard timeout, return metrics dict."""
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_ablation_worker,
                       args=(queue, slot_specs, detector_criterion,
                             ensemble_criterion, decision_window,
                             suppression_window, freq, stream_seed, s_idx,
                             tolerance, STREAM_LENGTH, GENERATOR))
    proc.start()
    proc.join(PER_STREAM_TIMEOUT)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return None
    try:
        result = queue.get_nowait()
        if result[0] == "ok":
            _, tp, fp, fn, f1, prec, rec, delay, n_known = result
            return {"tp": tp, "fp": fp, "fn": fn, "f1": f1,
                    "precision": prec, "recall": rec, "delay": delay,
                    "n_drifts": n_known}
    except:
        pass
    return None


def load_params():
    """Load best params from benchmark results."""
    results = json.load(open(f"{OUTPUT_DIR}/results_final.json"))
    ensemble = json.load(open(f"{OUTPUT_DIR}/ensemble_results.json"))

    configs = {}

    # 1. UDetect generalist
    configs["UDetect_gen"] = {
        "slots": [("UDetect", results["detectors"]["UDetect"]["generalist"]["params"])],
        "description": "UDetect generalist (single DD, generalist params)",
    }

    # 2. UDetect+SPLL joint ensemble
    joint = ensemble["joint_ensembles"]["UDetect+SPLL"]
    configs["UDetect_SPLL_joint"] = {
        "slots": [("UDetect", joint["params_d1"]), ("SPLL", joint["params_d2"])],
        "description": f"UDetect+SPLL joint (criterion={joint['criterion']}, dw={joint['decision_window']})",
    }

    # 3. UDetect+SPLL expert params per freq (cross-DD best-by-eval)
    expert_params = {}
    for dd in ["UDetect", "SPLL"]:
        expert_params[dd] = {}
        for freq in DRIFT_FREQS:
            exp = results["detectors"][dd]["experts"][str(freq)]
            expert_params[dd][freq] = exp["params"]

    configs["UDetect_SPLL_expert"] = {
        "slots_per_freq": {
            200: [("UDetect", expert_params["UDetect"][200])],
            500: [("UDetect", expert_params["UDetect"][500])],
            1000: [("SPLL", expert_params["SPLL"][1000])],
            2000: [("UDetect", expert_params["UDetect"][2000])],
        },
        "description": "Cross-DD best-by-eval (UDetect for 200/500/2000, SPLL for 1000)",
    }

    # 4. UDetect expert params per freq
    configs["UDetect_expert"] = {
        "slots_per_freq": {
            200: [("UDetect", expert_params["UDetect"][200])],
            500: [("UDetect", expert_params["UDetect"][500])],
            1000: [("UDetect", expert_params["UDetect"][1000])],
            2000: [("UDetect", expert_params["UDetect"][2000])],
        },
        "description": "UDetect expert (per-freq expert params)",
    }

    # 5. SPLL expert params per freq
    configs["SPLL_expert"] = {
        "slots_per_freq": {
            200: [("SPLL", expert_params["SPLL"][200])],
            500: [("SPLL", expert_params["SPLL"][500])],
            1000: [("SPLL", expert_params["SPLL"][1000])],
            2000: [("SPLL", expert_params["SPLL"][2000])],
        },
        "description": "SPLL expert (per-freq expert params)",
    }

    return configs


def run_ablation():
    configs = load_params()
    all_rows = []

    # Estimate total points
    total_points = 0
    for config_name, config in configs.items():
        is_multi = ("SPLL" in config_name and "UDetect" in config_name)
        if is_multi:
            n_combos = len(DETECTOR_CRITERIA) * len(ENSEMBLE_CRITERIA) * len(DECISION_WINDOWS) * len(SUPPRESSION_WINDOWS)
        else:
            # Single detector: detector_criterion and ensemble_criterion have no effect
            # Only sweep "any" for both, vary decision_window and suppression
            n_combos = 1 * 1 * len(DECISION_WINDOWS) * len(SUPPRESSION_WINDOWS)
        total_points += n_combos * len(DRIFT_FREQS) * len(EVAL_SEEDS)

    print(f"Total ablation points: {total_points}")
    print(f"  Estimated time: ~{total_points * 3 / 3600:.1f}h (assuming 3s per point)")
    print()

    t0 = time.time()
    done = 0

    for config_name, config in configs.items():
        print(f"\n=== Config: {config_name} ===")
        print(f"  {config['description']}")

        is_multi = ("SPLL" in config_name and "UDetect" in config_name)
        # For single-detector configs, only sweep decision_window and suppression
        det_crits = DETECTOR_CRITERIA if is_multi else ["any"]
        ens_crits = ENSEMBLE_CRITERIA if is_multi else ["any"]

        for det_crit in det_crits:
            for ens_crit in ens_crits:
                for dw in DECISION_WINDOWS:
                    for supp in SUPPRESSION_WINDOWS:
                        for freq in DRIFT_FREQS:
                            tol = FIXED_TOLERANCE

                            # Get slot specs (fixed or per-freq)
                            if "slots" in config:
                                slots = config["slots"]
                            else:
                                slots = config["slots_per_freq"][freq]

                            f1s = []
                            tps = []
                            fps = []
                            fns = []
                            delays = []
                            precs = []
                            recs = []

                            for i, seed in enumerate(EVAL_SEEDS):
                                result = run_ablation_point(
                                    slots, det_crit, ens_crit, dw, supp,
                                    freq, seed, i, tol)
                                if result:
                                    f1s.append(result["f1"])
                                    tps.append(result["tp"])
                                    fps.append(result["fp"])
                                    fns.append(result["fn"])
                                    delays.append(result["delay"])
                                    precs.append(result["precision"])
                                    recs.append(result["recall"])
                                else:
                                    f1s.append(0.0)
                                    tps.append(0)
                                    fps.append(0)
                                    fns.append(0)
                                    delays.append(-1.0)
                                    precs.append(0.0)
                                    recs.append(0.0)
                                done += 1

                            mean_f1 = sum(f1s) / len(f1s)
                            mean_tp = sum(tps) / len(tps)
                            mean_fp = sum(fps) / len(fps)
                            mean_fn = sum(fns) / len(fns)
                            valid_delays = [d for d in delays if d >= 0]
                            mean_delay = sum(valid_delays) / len(valid_delays) if valid_delays else -1.0
                            mean_prec = sum(precs) / len(precs)
                            mean_rec = sum(recs) / len(recs)

                            row = {
                                "config": config_name,
                                "n_detectors": len(slots),
                                "detector_criterion": det_crit,
                                "ensemble_criterion": ens_crit,
                                "decision_window": dw,
                                "suppression_window": supp,
                                "freq": freq,
                                "tolerance": tol,
                                "f1": mean_f1,
                                "tp": mean_tp,
                                "fp": mean_fp,
                                "fn": mean_fn,
                                "delay": mean_delay,
                                "precision": mean_prec,
                                "recall": mean_rec,
                            }
                            all_rows.append(row)

                        elapsed = time.time() - t0
                        if done % 50 == 0:
                            print(f"  [{done}/{total_points}] {elapsed:.0f}s elapsed, "
                                  f"est {elapsed/done*total_points:.0f}s total", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {done} points in {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/ablation_fair_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved to {csv_path}")

    return all_rows


def generate_parallel_plot(rows):
    """Generate parallel coordinates plot using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as mpatches
    import numpy as np

    # Aggregate: macro F1 across freqs per param combo
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        key = (r["config"], r["detector_criterion"], r["ensemble_criterion"],
               r["decision_window"], r["suppression_window"])
        agg[key].append(r)

    # Compute macro F1 per combo
    plot_data = []
    for key, rs in agg.items():
        config, det_crit, ens_crit, dw, supp = key
        # Group by freq and take mean f1, then macro across freqs
        freq_f1s = {}
        freq_tp = {}
        freq_fp = {}
        freq_fn = {}
        freq_delay = {}
        for r in rs:
            if r["freq"] not in freq_f1s:
                freq_f1s[r["freq"]] = []
                freq_tp[r["freq"]] = []
                freq_fp[r["freq"]] = []
                freq_fn[r["freq"]] = []
                freq_delay[r["freq"]] = []
            freq_f1s[r["freq"]].append(r["f1"])
            freq_tp[r["freq"]].append(r["tp"])
            freq_fp[r["freq"]].append(r["fp"])
            freq_fn[r["freq"]].append(r["fn"])
            freq_delay[r["freq"]].append(r["delay"])

        macro_f1 = sum(sum(v)/len(v) for v in freq_f1s.values()) / len(freq_f1s)
        macro_tp = sum(sum(v)/len(v) for v in freq_tp.values()) / len(freq_tp)
        macro_fp = sum(sum(v)/len(v) for v in freq_fp.values()) / len(freq_fp)
        macro_fn = sum(sum(v)/len(v) for v in freq_fn.values()) / len(freq_fn)
        all_delays = [d for v in freq_delay.values() for d in v if d >= 0]
        macro_delay = sum(all_delays)/len(all_delays) if all_delays else -1.0

        plot_data.append({
            "config": config,
            "det_crit": det_crit,
            "ens_crit": ens_crit,
            "decision_window": dw,
            "suppression_window": supp,
            "macro_f1": macro_f1,
            "tp": macro_tp,
            "fp": macro_fp,
            "fn": macro_fn,
            "delay": macro_delay,
        })

    # Map categorical to int
    crit_order = ["any", "all", "majority"]
    crit_to_int = {c: i for i, c in enumerate(crit_order)}
    config_names = sorted(set(d["config"] for d in plot_data))
    config_to_int = {c: i for i, c in enumerate(config_names)}

    # Dimensions for parallel plot
    dims = [
        ("Config", [config_to_int[d["config"]] for d in plot_data],
         list(range(len(config_names))), config_names),
        ("Det Crit (L1)", [crit_to_int[d["det_crit"]] for d in plot_data],
         list(range(len(crit_order))), crit_order),
        ("Ens Crit (L2)", [crit_to_int[d["ens_crit"]] for d in plot_data],
         list(range(len(crit_order))), crit_order),
        ("Decision W", [d["decision_window"] for d in plot_data], None, None),
        ("Suppression W", [d["suppression_window"] for d in plot_data], None, None),
        ("Macro F1", [d["macro_f1"] for d in plot_data], None, None),
        ("TP", [d["tp"] for d in plot_data], None, None),
        ("FP", [d["fp"] for d in plot_data], None, None),
        ("FN", [d["fn"] for d in plot_data], None, None),
    ]

    n_dims = len(dims)
    fig, ax = plt.subplots(figsize=(16, 8))

    # Normalize each dimension to [0, 1]
    norm_vals = []
    tick_positions = []
    tick_labels = []

    for i, (label, vals, tickvals, ticktext) in enumerate(dims):
        vmin = min(vals)
        vmax = max(vals)
        if vmax == vmin:
            norm = [0.5] * len(vals)
        else:
            norm = [(v - vmin) / (vmax - vmin) for v in vals]
        norm_vals.append(norm)

        if tickvals is not None:
            ticks = [(t - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                     for t in tickvals]
            tick_positions.append(ticks)
            tick_labels.append(ticktext)
        else:
            # Show min, mid, max
            n_ticks = min(5, len(set(vals)))
            unique_vals = sorted(set(vals))
            if n_ticks >= 2:
                step = max(1, len(unique_vals) // 4)
                selected = unique_vals[::step][:5]
            else:
                selected = unique_vals
            ticks = [(t - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                     for t in selected]
            tick_positions.append(ticks)
            tick_labels.append([str(t) for t in selected])

    # Draw lines colored by macro F1
    f1_vals = [d["macro_f1"] for d in plot_data]
    f1_min = min(f1_vals)
    f1_max = max(f1_vals)

    for j in range(len(plot_data)):
        color = plt.cm.RdYlGn((f1_vals[j] - f1_min) / (f1_max - f1_min)
                              if f1_max > f1_min else 0.5)
        xs = list(range(n_dims))
        ys = [norm_vals[i][j] for i in range(n_dims)]
        ax.plot(xs, ys, color=color, alpha=0.3, linewidth=0.8)

    # Set axis labels and ticks
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([d[0] for d in dims], fontsize=9, rotation=15, ha="right")
    ax.set_xlim(-0.5, n_dims - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])

    for i in range(n_dims):
        for tp, tl in zip(tick_positions[i], tick_labels[i]):
            ax.annotate(tl, xy=(i, tp), fontsize=7, ha="center",
                        xytext=(-5, 0), textcoords="offset points",
                        color="gray")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=f1_min, vmax=f1_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Macro F1", fontsize=10)

    ax.set_title("Ablation: Ensemble Parameters vs Macro F1\n"
                 "(each line = one parameter combination, color = macro F1)",
                 fontsize=12)
    ax.set_xlabel("Parameter / Metric", fontsize=10)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/ablation_fair_parallel_plot.png", dpi=150, bbox_inches="tight")
    print(f"Saved parallel plot to {OUTPUT_DIR}/ablation_fair_parallel_plot.png")
    plt.close(fig)

    # Also generate per-config plots
    for config_name in config_names:
        config_data = [d for d in plot_data if d["config"] == config_name]
        if len(config_data) < 2:
            continue

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        dims2 = [
            ("Det Crit (L1)", [crit_to_int[d["det_crit"]] for d in config_data],
             list(range(len(crit_order))), crit_order),
            ("Ens Crit (L2)", [crit_to_int[d["ens_crit"]] for d in config_data],
             list(range(len(crit_order))), crit_order),
            ("Decision W", [d["decision_window"] for d in config_data], None, None),
            ("Suppression W", [d["suppression_window"] for d in config_data], None, None),
            ("Macro F1", [d["macro_f1"] for d in config_data], None, None),
            ("TP", [d["tp"] for d in config_data], None, None),
            ("FP", [d["fp"] for d in config_data], None, None),
            ("FN", [d["fn"] for d in config_data], None, None),
        ]

        n_dims2 = len(dims2)
        norm_vals2 = []
        tick_positions2 = []
        tick_labels2 = []

        for i, (label, vals, tickvals, ticktext) in enumerate(dims2):
            vmin = min(vals)
            vmax = max(vals)
            if vmax == vmin:
                norm = [0.5] * len(vals)
            else:
                norm = [(v - vmin) / (vmax - vmin) for v in vals]
            norm_vals2.append(norm)
            if tickvals is not None:
                ticks = [(t - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                         for t in tickvals]
                tick_positions2.append(ticks)
                tick_labels2.append(ticktext)
            else:
                unique_vals = sorted(set(vals))
                step = max(1, len(unique_vals) // 4)
                selected = unique_vals[::step][:5]
                ticks = [(t - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                         for t in selected]
                tick_positions2.append(ticks)
                tick_labels2.append([str(t) for t in selected])

        f1_vals2 = [d["macro_f1"] for d in config_data]
        f1_min2 = min(f1_vals2)
        f1_max2 = max(f1_vals2)

        for j in range(len(config_data)):
            color = plt.cm.RdYlGn((f1_vals2[j] - f1_min2) / (f1_max2 - f1_min2)
                                  if f1_max2 > f1_min2 else 0.5)
            xs = list(range(n_dims2))
            ys = [norm_vals2[i][j] for i in range(n_dims2)]
            ax2.plot(xs, ys, color=color, alpha=0.4, linewidth=1.0)

        ax2.set_xticks(range(n_dims2))
        ax2.set_xticklabels([d[0] for d in dims2], fontsize=9, rotation=15, ha="right")
        ax2.set_xlim(-0.5, n_dims2 - 0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])

        for i in range(n_dims2):
            for tp, tl in zip(tick_positions2[i], tick_labels2[i]):
                ax2.annotate(tl, xy=(i, tp), fontsize=7, ha="center",
                             xytext=(-5, 0), textcoords="offset points",
                             color="gray")

        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                    norm=plt.Normalize(vmin=f1_min2, vmax=f1_max2))
        sm2.set_array([])
        cbar2 = fig2.colorbar(sm2, ax=ax2, pad=0.01)
        cbar2.set_label("Macro F1", fontsize=10)

        ax2.set_title(f"Ablation: {config_name}\n"
                      f"(each line = one param combo, color = macro F1)", fontsize=11)

        plt.tight_layout()
        safe_name = config_name.replace("/", "_").replace(" ", "_")
        fig2.savefig(f"{OUTPUT_DIR}/ablation_fair_plot_{safe_name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved per-config plot to {OUTPUT_DIR}/ablation_fair_plot_{safe_name}.png")
        plt.close(fig2)


def generate_marginal_analysis(rows):
    """Compute marginal effects of each parameter on F1."""
    from collections import defaultdict

    # Aggregate macro F1 per (config, param_combo) first
    combo_f1 = defaultdict(list)
    for r in rows:
        key = (r["config"], r["detector_criterion"], r["ensemble_criterion"],
               r["decision_window"], r["suppression_window"])
        combo_f1[key].append(r["f1"])

    # Now compute macro F1 per combo (mean across freqs)
    combo_macro = {}
    for key, f1s in combo_f1.items():
        # f1s has one per freq (already averaged across seeds)
        combo_macro[key] = sum(f1s) / len(f1s)

    # Marginal effect of each parameter
    params = ["detector_criterion", "ensemble_criterion",
              "decision_window", "suppression_window"]
    idx_map = {"detector_criterion": 1, "ensemble_criterion": 2,
               "decision_window": 3, "suppression_window": 4}

    results = {}
    for param in params:
        idx = idx_map[param]
        marginal = defaultdict(list)
        for key, macro_f1 in combo_macro.items():
            marginal[key[idx]].append(macro_f1)
        results[param] = {val: sum(v)/len(v) for val, v in marginal.items()}

    return results


def write_ablation_report(rows, marginal):
    """Write ablation report as markdown."""
    from collections import defaultdict

    # Aggregate macro F1 per combo
    combo_f1 = defaultdict(list)
    combo_details = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["config"], r["detector_criterion"], r["ensemble_criterion"],
               r["decision_window"], r["suppression_window"])
        combo_f1[key].append(r["f1"])
        combo_details[key][r["freq"]].append(r)

    combo_macro = {}
    for key, f1s in combo_f1.items():
        combo_macro[key] = sum(f1s) / len(f1s)

    # Best and worst per config
    by_config = defaultdict(list)
    for key, macro_f1 in combo_macro.items():
        config = key[0]
        by_config[config].append((macro_f1, key))

    lines = []
    lines.append("# Ablation Study: Ensemble Parameters vs F1\n")
    lines.append("## Overview\n")
    lines.append("This ablation sweeps four ensemble parameters to understand "
                 "their individual and combined effects on drift detection F1:\n")
    lines.append("- **Detector criterion (Level 1)**: How raw detector outputs are "
                 "smoothed over the decision window\n")
    lines.append("- **Ensemble criterion (Level 2)**: How per-detector decisions are "
                 "combined into the ensemble decision\n")
    lines.append("- **Decision window**: Sliding window size for Level 1 smoothing\n")
    lines.append("- **Suppression window**: Post-detection suppression period that "
                 "collapses repeated detections\n")
    lines.append("\n## Parameter Grid\n")
    lines.append(f"| Parameter | Values |")
    lines.append(f"|---|---|")
    lines.append(f"| Detector criterion (L1) | {', '.join(DETECTOR_CRITERIA)} |")
    lines.append(f"| Ensemble criterion (L2) | {', '.join(ENSEMBLE_CRITERIA)} |")
    lines.append(f"| Decision window | {', '.join(str(d) for d in DECISION_WINDOWS)} |")
    lines.append(f"| Suppression window | {', '.join(str(s) for s in SUPPRESSION_WINDOWS)} (fixed, no oracle) |")
    lines.append(f"| Tolerance | {FIXED_TOLERANCE} (fixed, no oracle) |")
    lines.append(f"| Eval seeds | {EVAL_SEEDS} |")
    lines.append(f"| Frequencies | {DRIFT_FREQS} |")
    lines.append(f"| Stream length | {STREAM_LENGTH} |")
    lines.append(f"| Total ablation points | {len(rows)} |")

    lines.append("\n## Configurations Ablated\n")
    configs = load_params()
    for name, cfg in configs.items():
        lines.append(f"- **{name}**: {cfg['description']}")

    lines.append("\n## Marginal Effects\n")
    lines.append("Mean macro F1 when varying each parameter (averaged over all others):\n")

    for param in ["detector_criterion", "ensemble_criterion",
                  "decision_window", "suppression_window"]:
        lines.append(f"\n### {param}\n")
        lines.append(f"| Value | Mean Macro F1 |")
        lines.append(f"|---|---|")
        for val in sorted(marginal[param].keys(), key=lambda x: (str(x))):
            lines.append(f"| {val} | {marginal[param][val]:.4f} |")

    lines.append("\n## Best and Worst Configurations\n")
    for config_name in sorted(by_config.keys()):
        combos = sorted(by_config[config_name], reverse=True)
        best = combos[0]
        worst = combos[-1]
        lines.append(f"\n### {config_name}\n")
        lines.append(f"| Rank | Det Crit | Ens Crit | Dec W | Supp W | Macro F1 |")
        lines.append(f"|---|---|---|---|---|---|")
        for i, (macro_f1, key) in enumerate(combos[:5]):
            _, dc, ec, dw, sw = key
            lines.append(f"| {i+1} | {dc} | {ec} | {dw} | {sw} | {macro_f1:.4f} |")
        lines.append(f"| ... | | | | | |")
        for i, (macro_f1, key) in enumerate(combos[-3:]):
            _, dc, ec, dw, sw = key
            rank = len(combos) - 3 + i + 1
            lines.append(f"| {rank} | {dc} | {ec} | {dw} | {sw} | {macro_f1:.4f} |")

    # Key observations
    lines.append("\n## Key Observations\n")

    # Effect of ensemble criterion for multi-DD configs
    multi_configs = [c for c in by_config if "SPLL" in c and "UDetect" in c]
    if multi_configs:
        lines.append("\n### Ensemble Criterion Effect (multi-DD configs)\n")
        lines.append("| Config | Det Crit | Ens Crit | Dec W | Supp W | Macro F1 |")
        lines.append("|---|---|---|---|---|---|")
        for config_name in multi_configs:
            for (macro_f1, key) in sorted(by_config[config_name], reverse=True)[:10]:
                _, dc, ec, dw, sw = key
                lines.append(f"| {config_name} | {dc} | {ec} | {dw} | {sw} | {macro_f1:.4f} |")

    # Suppression window effect
    lines.append("\n### Suppression Window Effect\n")
    lines.append("Mean macro F1 by fixed suppression window (across all configs):\n")
    supp_marginal = defaultdict(list)
    for key, macro_f1 in combo_macro.items():
        sw = key[4]
        supp_marginal[sw].append(macro_f1)
    lines.append("| Suppression Window | Mean Macro F1 |")
    lines.append("|---|---|")
    for sw in sorted(supp_marginal.keys()):
        vals = supp_marginal[sw]
        lines.append(f"| {sw} | {sum(vals)/len(vals):.4f} |")

    # Decision window effect
    lines.append("\n### Decision Window Effect\n")
    lines.append("Mean macro F1 by decision window (across all configs):\n")
    dw_marginal = defaultdict(list)
    for key, macro_f1 in combo_macro.items():
        dw_marginal[key[3]].append(macro_f1)
    lines.append("| Decision Window | Mean Macro F1 |")
    lines.append("|---|---|")
    for dw in sorted(dw_marginal.keys()):
        vals = dw_marginal[dw]
        lines.append(f"| {dw} | {sum(vals)/len(vals):.4f} |")

    report = "\n".join(lines)
    report_path = f"{OUTPUT_DIR}/ablation_fair_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved ablation report to {report_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run ablation
    rows = run_ablation()

    # Generate parallel plot
    generate_parallel_plot(rows)

    # Marginal analysis
    marginal = generate_marginal_analysis(rows)

    # Write report
    write_ablation_report(rows, marginal)

    print("\n=== Ablation study complete ===")
    print(f"  CSV: {OUTPUT_DIR}/ablation_fair_results.csv")
    print(f"  Parallel plot: {OUTPUT_DIR}/ablation_fair_parallel_plot.png")
    print(f"  Per-config plots: {OUTPUT_DIR}/ablation_fair_plot_*.png")
    print(f"  Report: {OUTPUT_DIR}/ablation_fair_report.md")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
