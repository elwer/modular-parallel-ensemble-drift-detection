"""
Greedy forward-selection of a MOPEDDS ensemble from a pool of well-tuned
SINGLE-detector configurations.

Motivation
----------
Jointly optimizing ``size * (1 categorical + ~5 numerical)`` hyperparameters
with Optuna in a single study works for small ``size`` but collapses for
large ``size`` because the search space explodes faster than the trial
budget. As a result, a naive joint study tends to find a config that
clones one good single-detector across all slots, which majority/all/any
voting cannot improve on -- so the F1-vs-N curve flatlines or decreases.

This script avoids the search-space explosion by:

1. Reading a pool of *individually well-tuned* single-detector configs from
   a previous ``synthetic_f1_multistream_optimize_optuna.py`` N=1 study
   (per-trial CSVs and / or the Optuna sqlite DB).
2. Growing the ensemble greedily: at each step it tries every pool member
   as the next slot and (optionally) re-tunes the ensemble voting rule
   ``ensemble_decision_criteria in {any, majority, all}``, keeping the
   addition that maximizes train macro F1.
3. Evaluating each accepted ensemble on the held-out eval streams so the
   train-vs-eval generalization curve is comparable to the joint-search
   results.

Stream set + train/eval split are identical to the multistream optimizer
so the two methods are directly comparable.

Usage
-----
    python optimization/greedy_ensemble_from_pool.py \\
        --generator SineClusters --n-streams 10 \\
        --drift-frequencies 200,400,500,750,1000,1250,1500,2000,2500,3000 \\
        --stream-length 10000 --eval-stream-indices 1,4,8 \\
        --pool-csv path/to/synthF1ms_SineClusters_N1_S10.csv \\
        --max-n 32 --top-k-overall 30 --top-k-per-type 5 \\
        --output-csv greedy_SineClusters.csv
"""

from __future__ import annotations

import os
import sys
import csv
import json
import glob
import math
import logging
import multiprocessing as mp
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Make repository importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.synthetic_f1_multistream_optimize_optuna import (  # noqa: E402
    CANDIDATES,
    _default_tolerances,
    _f1_from_counts,
    _parse_pin_globals,
    _resolve_generators,
    _resolve_list,
    _resolve_stream_seeds,
    _resolve_study_tag,
    _run_one_stream,
    GENERATORS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool entry
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """A single well-tuned detector config taken from an N=1 study."""
    source: str                  # provenance, e.g. "csv:foo.csv#trial=123"
    kind: str                    # detector class short name (CANDIDATES)
    params: Dict[str, object]    # detector-specific hyperparameters
    macro_f1: float              # macro F1 reported for this trial (N=1)
    # global params from the trial (kept as defaults if --globals=inherit-best)
    detector_decision_criteria: Optional[str] = None
    decision_window: Optional[int] = None
    suppression_window: Optional[int] = None
    recent_samples_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Pool readers
# ---------------------------------------------------------------------------

_GLOBAL_KEYS = {
    "detector_decision_criteria",
    "ensemble_decision_criteria",
    "decision_window",
    "suppression_window",
    "recent_samples_size",
}


def _safe_float(v) -> Optional[float]:
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    f = _safe_float(v)
    return None if f is None else int(f)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Robust ragged-CSV reader (Optuna writer may append columns)."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        rows_raw = list(reader)
    if not rows_raw:
        return []
    max_w = max((len(r) for r in rows_raw), default=len(header))
    if max_w > len(header):
        header = list(header) + [f"extra_{i}" for i in range(max_w - len(header))]
    out = []
    for r in rows_raw:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        out.append(dict(zip(header, r)))
    return out


def _row_to_pool_entry(row: Dict[str, str], source: str) -> Optional[PoolEntry]:
    """Convert one CSV row (assumed slot0_*-style N=1 trial) to a PoolEntry."""
    if (row.get("error") or "").strip():
        return None
    kind = (row.get("slot0_type") or "").strip()
    if kind not in CANDIDATES:
        return None
    macro = _safe_float(row.get("macro_f1"))
    if macro is None:
        return None
    prefix = f"slot0_{kind}_"
    params: Dict[str, object] = {}
    for k, v in row.items():
        if not k.startswith(prefix):
            continue
        if v in (None, ""):
            continue
        suffix = k[len(prefix):]
        # Detect bools, ints, floats heuristically.
        sv = str(v).strip()
        lv = sv.lower()
        if lv in ("true", "false"):
            params[suffix] = (lv == "true")
        else:
            try:
                fv = float(sv)
                params[suffix] = int(fv) if fv.is_integer() and "." not in sv else fv
            except ValueError:
                params[suffix] = sv
    return PoolEntry(
        source=source,
        kind=kind,
        params=params,
        macro_f1=float(macro),
        detector_decision_criteria=(row.get("detector_decision_criteria") or None),
        decision_window=_safe_int(row.get("decision_window")),
        suppression_window=_safe_int(row.get("suppression_window")),
        recent_samples_size=_safe_int(row.get("recent_samples_size")),
    )


def load_pool_from_csvs(paths: List[str]) -> List[PoolEntry]:
    pool: List[PoolEntry] = []
    for p in paths:
        rows = _read_csv_rows(p)
        for r in rows:
            entry = _row_to_pool_entry(r, source=f"csv:{os.path.basename(p)}#"
                                                f"trial={r.get('trial_id','?')}")
            if entry is not None:
                pool.append(entry)
    return pool


def load_pool_from_optuna(storage_url: str, study_name: str) -> List[PoolEntry]:
    """Fallback path: read trials directly from an Optuna sqlite DB."""
    import optuna  # local import: keep CLI --help cheap
    from optuna.trial import TrialState
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    pool: List[PoolEntry] = []
    for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if t.user_attrs.get("error"):
            continue
        params = dict(t.params)
        kind = params.get("slot0_type")
        if kind not in CANDIDATES:
            continue
        prefix = f"slot0_{kind}_"
        det_params = {k[len(prefix):]: v for k, v in params.items()
                      if k.startswith(prefix)}
        macro = t.user_attrs.get("macro_f1", t.value)
        if macro is None:
            continue
        pool.append(PoolEntry(
            source=f"optuna:{study_name}#trial={t.number}",
            kind=kind,
            params=det_params,
            macro_f1=float(macro),
            detector_decision_criteria=params.get("detector_decision_criteria"),
            decision_window=_safe_int(params.get("decision_window")),
            suppression_window=_safe_int(params.get("suppression_window")),
            recent_samples_size=_safe_int(params.get("recent_samples_size")),
        ))
    return pool


def filter_pool(pool: List[PoolEntry],
                top_k_overall: int,
                top_k_per_type: int) -> List[PoolEntry]:
    """Keep the top-K configs overall and the top-K per detector type to
    guarantee that every type is representable in the greedy candidate set."""
    by_score = sorted(pool, key=lambda e: e.macro_f1, reverse=True)
    keep: List[PoolEntry] = []
    seen_ids: set = set()

    def _id(e: PoolEntry) -> int:
        return id(e)

    if top_k_overall > 0:
        for e in by_score[:top_k_overall]:
            if _id(e) not in seen_ids:
                keep.append(e)
                seen_ids.add(_id(e))

    if top_k_per_type > 0:
        per_type: Dict[str, List[PoolEntry]] = {k: [] for k in CANDIDATES}
        for e in by_score:
            if len(per_type[e.kind]) < top_k_per_type:
                per_type[e.kind].append(e)
        for kind in CANDIDATES:
            for e in per_type[kind]:
                if _id(e) not in seen_ids:
                    keep.append(e)
                    seen_ids.add(_id(e))

    return keep


# ---------------------------------------------------------------------------
# Subset evaluation (uses _run_one_stream from the multistream module)
# ---------------------------------------------------------------------------

@dataclass
class GlobalConfig:
    detector_decision_criteria: str
    ensemble_decision_criteria: str
    decision_window: int
    suppression_window: int
    recent_samples_size: int


def evaluate_ensemble(*, generators: List[str],
                      drift_frequencies: List[int],
                      stream_length: int,
                      stream_seeds: List[int],
                      tolerances: List[int],
                      indices: List[int],
                      slot_specs: List[Tuple[str, Dict[str, object]]],
                      detector_seed: int,
                      g: GlobalConfig) -> Dict[str, object]:
    per_f1: List[float] = []
    tp_total = fp_total = fn_total = 0
    for s_idx in indices:
        tp, fp, fn, _md, f1, _p, _r, _n = _run_one_stream(
            generator_name=generators[s_idx],
            drift_frequency=drift_frequencies[s_idx],
            stream_length=stream_length,
            stream_seed=stream_seeds[s_idx],
            tolerance=tolerances[s_idx],
            slot_specs=slot_specs,
            detector_seed_base=detector_seed,
            s_idx=s_idx,
            detector_criterion=g.detector_decision_criteria,
            ensemble_criterion=g.ensemble_decision_criteria,
            decision_window=g.decision_window,
            suppression_window=g.suppression_window,
            recent_samples_size=g.recent_samples_size,
        )
        per_f1.append(float(f1))
        tp_total += int(tp); fp_total += int(fp); fn_total += int(fn)
    macro = sum(per_f1) / len(per_f1) if per_f1 else 0.0
    micro = _f1_from_counts(tp_total, fp_total, fn_total)
    return {"macro_f1": macro, "micro_f1": micro,
            "per_stream_f1": per_f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total}


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

ENS_CRITS = ("any", "majority", "all")


def _evaluate_candidate_task(args_tuple: Tuple) -> Tuple[float, List[float], Dict, str, Dict[str, object]]:
    """Worker function for parallel candidate evaluation.
    
    Args are primitive types only (no complex objects) for reliable pickling.
    Returns: (macro_f1, per_stream_f1, candidate_dict, ens_crit, metrics)
    """
    try:
        (cand_dict, ensemble_dicts, ec, base_global_dict, generators, drift_frequencies,
         stream_length, stream_seeds, tolerances, train_indices, detector_seed) = args_tuple
        
        trial_specs = [(e["kind"], e["params"]) for e in ensemble_dicts] + [(cand_dict["kind"], cand_dict["params"])]
        g = GlobalConfig(
            detector_decision_criteria=base_global_dict["detector_decision_criteria"],
            ensemble_decision_criteria=ec,
            decision_window=base_global_dict["decision_window"],
            suppression_window=base_global_dict["suppression_window"],
            recent_samples_size=base_global_dict["recent_samples_size"],
        )
        m = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            indices=train_indices,
            slot_specs=trial_specs,
            detector_seed=detector_seed,
            g=g,
        )
        return (m["macro_f1"], m["per_stream_f1"], cand_dict, ec, m)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def _compute_complementarity_score(
    candidate_per_stream_f1: List[float],
    current_per_stream_f1: List[float],
) -> float:
    """Compute complementarity score based on weak-stream boost.
    
    For each stream, weight the improvement by how weak the current ensemble is.
    Streams where current ensemble is weak get higher weight.
    """
    score = 0.0
    for cand_f1, curr_f1 in zip(candidate_per_stream_f1, current_per_stream_f1):
        # Improvement on this stream
        improvement = cand_f1 - curr_f1
        # Weight by inverse of current performance (weaker streams get higher weight)
        # Add epsilon to avoid division by zero
        weight = 1.0 / (curr_f1 + 1e-6)
        score += improvement * weight
    return score


def greedy_select(*, pool: List[PoolEntry],
                  generators: List[str],
                  drift_frequencies: List[int],
                  stream_length: int,
                  stream_seeds: List[int],
                  tolerances: List[int],
                  train_indices: List[int],
                  eval_indices: List[int],
                  base_global: GlobalConfig,
                  inner_search_ens_crit: bool,
                  detector_seed: int,
                  max_n: int,
                  stop_on_no_improve: bool,
                  n_workers: int = 1,
                  selection_strategy: str = "complementarity"):
    history: List[Dict[str, object]] = []
    ensemble: List[PoolEntry] = []
    current_train = 0.0
    current_eval = 0.0
    current_train_per_stream = [0.0] * len(train_indices)

    ens_crits = ENS_CRITS if inner_search_ens_crit else (base_global.ensemble_decision_criteria,)
    use_mp = n_workers > 1

    for step in range(1, max_n + 1):
        best_candidate: Optional[PoolEntry] = None
        best_ens_crit: Optional[str] = base_global.ensemble_decision_criteria
        best_train_metrics: Optional[Dict[str, object]] = None
        best_train_macro = -math.inf

        # Build task list for this step (convert to primitive types for pickling)
        base_global_dict = {
            "detector_decision_criteria": base_global.detector_decision_criteria,
            "ensemble_decision_criteria": base_global.ensemble_decision_criteria,
            "decision_window": base_global.decision_window,
            "suppression_window": base_global.suppression_window,
            "recent_samples_size": base_global.recent_samples_size,
        }
        ensemble_dicts = [{"kind": e.kind, "params": e.params} for e in ensemble]
        
        tasks = []
        for cand in pool:
            if any(cand is e for e in ensemble):
                continue
            cand_dict = {"kind": cand.kind, "params": cand.params, "source": cand.source, "macro_f1": cand.macro_f1}
            for ec in ens_crits:
                tasks.append((
                    cand_dict, ensemble_dicts, ec, base_global_dict,
                    generators, drift_frequencies, stream_length,
                    stream_seeds, tolerances, train_indices, detector_seed
                ))

        logger.info("Step %d: use_mp=%s, n_workers=%d, tasks=%d", step, use_mp, n_workers, len(tasks))
        
        if use_mp and tasks:
            # Parallel evaluation
            logger.info("Step %d: evaluating %d candidates with %d workers", step, len(tasks), n_workers)
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=min(n_workers, len(tasks))) as pool_mp:
                    results = pool_mp.map(_evaluate_candidate_task, tasks, chunksize=1)
                for macro_f1, per_stream_f1, cand_dict, ec, m in results:
                    # Reconstruct PoolEntry from dict
                    cand = PoolEntry(
                        kind=cand_dict["kind"],
                        params=cand_dict["params"],
                        source=cand_dict["source"],
                        macro_f1=cand_dict["macro_f1"],
                    )
                    if selection_strategy == "complementarity":
                        score = _compute_complementarity_score(per_stream_f1, current_train_per_stream)
                    else:  # macro_f1
                        score = macro_f1
                    if score > best_train_macro:
                        best_train_macro = score
                        best_candidate = cand
                        best_ens_crit = ec
                        best_train_metrics = m
                logger.info("Step %d: best train score = %.4f", step, best_train_macro)
            except Exception as e:
                logger.warning("Parallel evaluation failed: %s. Falling back to sequential.", e)
                logger.info("Step %d: falling back to sequential evaluation", step)
                use_mp = False
        else:
            # Sequential evaluation (original behavior)
            for cand in pool:
                if any(cand is e for e in ensemble):
                    continue
                trial_specs = [(e.kind, e.params) for e in ensemble] + [(cand.kind, cand.params)]
                for ec in ens_crits:
                    g = GlobalConfig(
                        detector_decision_criteria=base_global.detector_decision_criteria,
                        ensemble_decision_criteria=ec,
                        decision_window=base_global.decision_window,
                        suppression_window=base_global.suppression_window,
                        recent_samples_size=base_global.recent_samples_size,
                    )
                    m = evaluate_ensemble(
                        generators=generators,
                        drift_frequencies=drift_frequencies,
                        stream_length=stream_length,
                        stream_seeds=stream_seeds,
                        tolerances=tolerances,
                        indices=train_indices,
                        slot_specs=trial_specs,
                        detector_seed=detector_seed,
                        g=g,
                    )
                    if selection_strategy == "complementarity":
                        score = _compute_complementarity_score(m["per_stream_f1"], current_train_per_stream)
                    else:  # macro_f1
                        score = m["macro_f1"]
                    if score > best_train_macro:
                        best_train_macro = score
                        best_candidate = cand
                        best_ens_crit = ec
                        best_train_metrics = m

        if best_candidate is None:
            logger.warning("No candidate available; stopping at N=%d", len(ensemble))
            break

        improved = best_train_macro > current_train + 1e-9
        if stop_on_no_improve and step > 1 and not improved:
            logger.info("No train-F1 improvement at step %d "
                        "(best train=%.4f, prev=%.4f); stopping.",
                        step, best_train_macro, current_train)
            break

        ensemble.append(best_candidate)
        # Evaluate the accepted ensemble on the held-out eval indices.
        g_final = GlobalConfig(
            detector_decision_criteria=base_global.detector_decision_criteria,
            ensemble_decision_criteria=best_ens_crit,
            decision_window=base_global.decision_window,
            suppression_window=base_global.suppression_window,
            recent_samples_size=base_global.recent_samples_size,
        )
        eval_metrics = evaluate_ensemble(
            generators=generators,
            drift_frequencies=drift_frequencies,
            stream_length=stream_length,
            stream_seeds=stream_seeds,
            tolerances=tolerances,
            indices=eval_indices,
            slot_specs=[(e.kind, e.params) for e in ensemble],
            detector_seed=detector_seed,
            g=g_final,
        )
        current_train = best_train_metrics["macro_f1"]
        current_train_per_stream = best_train_metrics["per_stream_f1"]
        current_eval = eval_metrics["macro_f1"]
        train_delta = current_train - (history[-1]["train_macro_f1"] if history else 0.0)
        eval_delta = current_eval - (history[-1]["eval_macro_f1"] if history else 0.0)
        
        record = {
            "step": step,
            "n": len(ensemble),
            "added_kind": best_candidate.kind,
            "added_source": best_candidate.source,
            "added_params": best_candidate.params,
            "ens_crit": best_ens_crit,
            "det_crit": base_global.detector_decision_criteria,
            "decision_window": base_global.decision_window,
            "suppression_window": base_global.suppression_window,
            "recent_samples_size": base_global.recent_samples_size,
            "train_macro_f1": best_train_macro,
            "train_delta": train_delta,
            "train_per_stream_f1": best_train_metrics["per_stream_f1"],
            "eval_macro_f1": eval_metrics["macro_f1"],
            "eval_delta": eval_delta,
            "eval_per_stream_f1": eval_metrics["per_stream_f1"],
            "members": [{"kind": e.kind, "source": e.source} for e in ensemble],
        }
        history.append(record)
        logger.info(
            "step=%d N=%d +%s  train_macroF1=%.4f (%+.4f)  eval_macroF1=%.4f (%+.4f)  ens=%s",
            step, len(ensemble), best_candidate.kind,
            best_train_macro, train_delta, eval_metrics["macro_f1"], eval_delta, best_ens_crit,
        )

    return history


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_history_csv(path: str, history: List[Dict[str, object]]) -> None:
    if not history:
        logger.warning("No history rows to write.")
        return
    flat_rows: List[Dict[str, object]] = []
    fieldnames = ["step", "n", "added_kind", "ens_crit", "det_crit",
                  "decision_window", "suppression_window",
                  "recent_samples_size",
                  "train_macro_f1", "eval_macro_f1",
                  "train_per_stream_f1", "eval_per_stream_f1",
                  "members_kinds", "members_sources",
                  "added_params_json", "added_source"]
    for h in history:
        flat_rows.append({
            "step": h["step"],
            "n": h["n"],
            "added_kind": h["added_kind"],
            "ens_crit": h["ens_crit"],
            "det_crit": h["det_crit"],
            "decision_window": h["decision_window"],
            "suppression_window": h["suppression_window"],
            "recent_samples_size": h["recent_samples_size"],
            "train_macro_f1": f"{h['train_macro_f1']:.6f}",
            "eval_macro_f1": f"{h['eval_macro_f1']:.6f}",
            "train_per_stream_f1": json.dumps(h["train_per_stream_f1"]),
            "eval_per_stream_f1": json.dumps(h["eval_per_stream_f1"]),
            "members_kinds": json.dumps([m["kind"] for m in h["members"]]),
            "members_sources": json.dumps([m["source"] for m in h["members"]]),
            "added_params_json": json.dumps(h["added_params"]),
            "added_source": h["added_source"],
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)
    logger.info("Wrote %d rows to %s", len(flat_rows), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument("--generator", default=None, choices=list(GENERATORS.keys()),
                    help="DEPRECATED: single stream generator class. Use "
                         "--generators for mixed-generator runs.")
    ap.add_argument("--generators", default=None,
                    help="Comma-separated per-stream generator names. Must "
                         "have length 1 (broadcast) or --n-streams. Overrides "
                         "--generator. Allowed values: "
                         + ", ".join(sorted(GENERATORS.keys())) + ".")
    ap.add_argument("--study-tag", default=None,
                    help="Identifier used for logging only (output path is set "
                         "explicitly via --output-csv). Defaults to the single "
                         "generator name when homogeneous, otherwise 'Mix_' + "
                         "'+'-joined sorted unique generators.")
    ap.add_argument("--pin-globals", default=None,
                    help="Comma-separated 'key=value' overrides applied AFTER "
                         "--globals resolution. Use to force specific values "
                         "of detector_decision_criteria, ensemble_decision_"
                         "criteria, decision_window, suppression_window, "
                         "recent_samples_size. If ensemble_decision_criteria "
                         "is pinned, --inner-search-ens-crit is implicitly "
                         "disabled.")
    ap.add_argument("--n-streams", type=int, default=10)
    ap.add_argument("--base-stream-seed", type=int, default=42)
    ap.add_argument("--stream-seeds", default=None)
    ap.add_argument("--drift-frequencies",
                    default="200,400,500,750,1000,1250,1500,2000,2500,3000")
    ap.add_argument("--stream-length", type=int, default=10000)
    ap.add_argument("--tolerances", default=None)
    ap.add_argument("--eval-stream-indices", default="1,4,8")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Base detector seed (matches multistream optimizer).")
    # Pool source: at least one of --pool-csv / --pool-glob / --optuna-storage
    ap.add_argument("--pool-csv", action="append", default=[],
                    help="Path to a per-trial CSV from a previous N=1 study. "
                         "May be repeated.")
    ap.add_argument("--pool-glob", default=None,
                    help="Optional glob expanding to per-trial CSVs.")
    ap.add_argument("--optuna-storage", default=None,
                    help="Optional Optuna storage URL to load N=1 trials from.")
    ap.add_argument("--optuna-study", default=None,
                    help="Optuna study name (required if --optuna-storage given).")
    ap.add_argument("--top-k-overall", type=int, default=30)
    ap.add_argument("--top-k-per-type", type=int, default=5)
    ap.add_argument("--max-n", type=int, default=32)
    ap.add_argument("--inner-search-ens-crit", action="store_true",
                    help="At each greedy step also pick the best "
                         "ensemble_decision_criteria in {any, majority, all}.")
    ap.add_argument("--stop-on-no-improve", action="store_true")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Number of parallel workers for candidate evaluation. "
                         "Default 1 (sequential). Use >1 to speed up greedy selection.")
    ap.add_argument("--selection-strategy", default="complementarity",
                    choices=["complementarity", "macro_f1"],
                    help="Strategy for selecting candidates: 'complementarity' "
                         "(boost weak streams, default) or 'macro_f1' (maximize average).")
    # Anchors for global params (det_crit, decision_window, ...). 'best' means
    # inherit from the top-1 pool entry; otherwise override explicitly.
    ap.add_argument("--globals", default="best",
                    help="'best' (inherit from top-1 pool entry) or "
                         "'manual' (use --det-crit/--ens-crit/--decision-window/"
                         "--suppression-window/--recent-samples-size).")
    ap.add_argument("--det-crit", default="majority", choices=["any", "majority", "all"])
    ap.add_argument("--ens-crit", default="any", choices=["any", "majority", "all"])
    ap.add_argument("--decision-window", type=int, default=10)
    ap.add_argument("--suppression-window", type=int, default=None)
    ap.add_argument("--recent-samples-size", type=int, default=500)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    # Stream set / split (must mirror the multistream optimizer exactly).
    stream_seeds = _resolve_stream_seeds(args.stream_seeds, args.n_streams,
                                         args.base_stream_seed)
    drift_frequencies = _resolve_list(args.drift_frequencies, args.n_streams,
                                      name="--drift-frequencies")
    generators_list = _resolve_generators(args.generators, args.generator,
                                          args.n_streams)
    study_tag = _resolve_study_tag(args.study_tag, generators_list)
    tolerances = (_resolve_list(args.tolerances, args.n_streams, "--tolerances")
                  if args.tolerances else _default_tolerances(drift_frequencies))
    eval_indices = sorted({int(s.strip())
                           for s in args.eval_stream_indices.split(",")
                           if s.strip() != ""})
    for idx in eval_indices:
        if idx < 0 or idx >= args.n_streams:
            raise ValueError(f"--eval-stream-indices contains out-of-range "
                             f"index {idx} (must be in [0,{args.n_streams - 1}]).")
    train_indices = [i for i in range(args.n_streams) if i not in set(eval_indices)]
    if not train_indices:
        raise ValueError("All stream indices were assigned to evaluation.")

    # Build the pool.
    csv_paths: List[str] = list(args.pool_csv)
    if args.pool_glob:
        csv_paths += sorted(glob.glob(args.pool_glob))
    pool: List[PoolEntry] = []
    if csv_paths:
        pool += load_pool_from_csvs(csv_paths)
    if args.optuna_storage:
        if not args.optuna_study:
            raise ValueError("--optuna-storage requires --optuna-study.")
        pool += load_pool_from_optuna(args.optuna_storage, args.optuna_study)
    if not pool:
        raise ValueError("Empty pool: pass --pool-csv / --pool-glob / "
                         "--optuna-storage so the script has candidates.")

    pool = filter_pool(pool, args.top_k_overall, args.top_k_per_type)
    pool.sort(key=lambda e: e.macro_f1, reverse=True)
    logger.info("Pool size after filtering: %d", len(pool))
    type_counts: Dict[str, int] = {}
    for e in pool:
        type_counts[e.kind] = type_counts.get(e.kind, 0) + 1
    logger.info("Pool type counts: %s", type_counts)
    logger.info("Best pool entry: kind=%s macroF1=%.4f source=%s",
                pool[0].kind, pool[0].macro_f1, pool[0].source)

    # Resolve global config anchor.
    if args.globals == "best":
        top = pool[0]
        det_crit = top.detector_decision_criteria or args.det_crit
        decision_window = top.decision_window or args.decision_window
        suppression_window = (top.suppression_window
                              if top.suppression_window is not None
                              else (args.suppression_window
                                    if args.suppression_window is not None
                                    else max(0, min(tolerances))))
        recent_samples_size = top.recent_samples_size or args.recent_samples_size
        ens_crit = args.ens_crit  # starting ens_crit; --inner-search-ens-crit may override per step
    elif args.globals == "manual":
        det_crit = args.det_crit
        ens_crit = args.ens_crit
        decision_window = args.decision_window
        suppression_window = (args.suppression_window
                              if args.suppression_window is not None
                              else max(0, min(tolerances)))
        recent_samples_size = args.recent_samples_size
    else:
        raise ValueError(f"Unknown --globals: {args.globals}")

    # Apply --pin-globals overrides on top of the resolved base config.
    pinned_globals = _parse_pin_globals(args.pin_globals)
    if pinned_globals:
        logger.info("Pinning globals: %s", pinned_globals)
        det_crit = pinned_globals.get("detector_decision_criteria", det_crit)
        ens_crit = pinned_globals.get("ensemble_decision_criteria", ens_crit)
        decision_window = pinned_globals.get("decision_window", decision_window)
        suppression_window = pinned_globals.get(
            "suppression_window", suppression_window)
        recent_samples_size = pinned_globals.get(
            "recent_samples_size", recent_samples_size)

    # If ens_crit is pinned, force-disable inner search over it so the
    # ablation actually fixes that dimension.
    inner_search_ens_crit = args.inner_search_ens_crit
    if "ensemble_decision_criteria" in pinned_globals and inner_search_ens_crit:
        logger.info("ensemble_decision_criteria is pinned; "
                    "disabling --inner-search-ens-crit.")
        inner_search_ens_crit = False

    base_global = GlobalConfig(
        detector_decision_criteria=det_crit,
        ensemble_decision_criteria=ens_crit,
        decision_window=int(decision_window),
        suppression_window=int(suppression_window),
        recent_samples_size=int(recent_samples_size),
    )

    print("=" * 80)
    print("Greedy ensemble from pool")
    print("=" * 80)
    print(f"  Study tag           : {study_tag}")
    print(f"  Generators (all)    : {generators_list}")
    print(f"  Train indices       : {train_indices}")
    print(f"    generators        : {[generators_list[i] for i in train_indices]}")
    print(f"  Eval indices        : {eval_indices}")
    print(f"    generators        : {[generators_list[i] for i in eval_indices]}")
    print(f"  Drift frequencies   : {drift_frequencies}")
    print(f"  Tolerances          : {tolerances}")
    print(f"  Pool size           : {len(pool)}")
    print(f"  Max N               : {args.max_n}")
    print(f"  det_crit            : {base_global.detector_decision_criteria}")
    print(f"  ens_crit start      : {base_global.ensemble_decision_criteria} "
          f"(inner-search={args.inner_search_ens_crit})")
    print(f"  decision_window     : {base_global.decision_window}")
    print(f"  suppression_window  : {base_global.suppression_window}")
    print(f"  recent_samples_size : {base_global.recent_samples_size}")
    print(f"  selection_strategy  : {args.selection_strategy}")
    print("=" * 80, flush=True)

    history = greedy_select(
        pool=pool,
        generators=generators_list,
        drift_frequencies=drift_frequencies,
        stream_length=args.stream_length,
        stream_seeds=stream_seeds,
        tolerances=tolerances,
        train_indices=train_indices,
        eval_indices=eval_indices,
        base_global=base_global,
        inner_search_ens_crit=inner_search_ens_crit,
        detector_seed=args.seed,
        max_n=args.max_n,
        stop_on_no_improve=args.stop_on_no_improve,
        n_workers=args.n_workers,
        selection_strategy=args.selection_strategy,
    )

    write_history_csv(args.output_csv, history)


if __name__ == "__main__":
    main()
