# MoPEDDS — Modular Parallel Ensemble Drift Detection

Evaluation framework for parallel ensemble-based unsupervised concept drift
detection. The repository contains three core components:

1. **Synthetic ablation study** — Compares per-DD ensembles, a cross-DD
   ensemble, and generalist detectors on synthetic streams with budget-matched
   Optuna optimization.
2. **Real-world Optuna study** — Evaluates ML model-based pipelines equipped
   with MoPEDDS against single-DD pipelines on real-world datasets.
3. **Trace data** — OTF2 traces from an instrumented DD run for parallelism
   analysis with Vampir.

## Supported Detectors

- **CSDDM** — Clustered Statistical Test Drift Detection Method
- **D3** — Discriminative Drift Detector
- **IBDD** — Image-Based Drift Detector
- **OCDD** — One-Class Drift Detector
- **SPLL** — Semi-Parametric Log-Likelihood
- **UDetect** — Unsupervised Change Detection for Activity Recognition
- **MoPEDDS** — Modular Parallel Ensemble of the above detectors

## Requirements

Python 3.13 with the dependencies listed in `requirements.txt`:

```
numpy, pandas, scikit-learn, scipy, river, optuna, torch, matplotlib,
pyitlib, memory-profiler
```

On HPC, the environment is set up via `setup.sh`, which loads the required
modules and activates the virtual environment.

## Core Components

### 1. Synthetic Ablation Study

**Script:** `run_ensemble_vs_generalist.py`

Compares per-DD ensembles, a cross-DD ensemble, and generalist detectors on
synthetic streams (SineClusters, WaveformDrift2) with a budget-matched Optuna
optimization. The experiment trains K=3 experts per stream, optimizes
deployment parameters (voting scheme, decision window, suppression), and
evaluates on held-out test streams.

Results are stored in JSON format in the specified output directory.

### 2. Real-World Optuna Study

Per-dataset hyperparameter optimization of both single DDs and the
MoPEDDS ensemble on real-world datasets, using Optuna with a TPE sampler.

**Script:** `run_cross_dataset_evaluation.py`

Wraps `optimization/single_dd_optimize_optuna.py` and
`optimization/mopedds_optimize_optuna.py`, running per-dataset optimization
across all supported datasets. By default, all datasets are optimized;
a subset can be selected via `--datasets`.

**Note:** Only the Electricity dataset is available by default. Additional
datasets (GasSensor, PokerHand, RialtoBridgeTimelapse, etc.) must be
downloaded manually from the
[USP DS repository](https://github.com/EliasWang49/usp_ds) and placed in
`datasets/files/`. The corresponding dataset classes must also be
uncommented in `datasets/__init__.py`.

### 3. Trace Data

**Directory:** `trace_data/trace_dscal/`

Contains OTF2 trace data from an instrumented DD run, collected with Score-P.
The traces can be visualized with Vampir to analyze the runtime behaviour and
parallelism of the ensemble pipeline.

## Repository Structure

```
.
├── run_ensemble_vs_generalist.py    # Synthetic ablation study
├── run_cross_dataset_evaluation.py  # Real-world Optuna study (wrapper)
├── ensemble_vs_generalist.sbatch    # SLURM script for ablation study
├── submit_ensemble_vs_generalist.sh # Submit script for ablation study
├── optimization/                    # Optuna optimization scripts
│   ├── single_dd_optimize_optuna.py   # Single-DD optimization per dataset
│   └── mopedds_optimize_optuna.py     # MoPEDDS ensemble optimization per dataset
├── main.py                          # ML pipeline runner
├── trace_data/                      # OTF2 traces for Vampir
├── detectors/                       # Detector implementations
├── model/                           # ML model pipeline components
├── metrics/                         # Evaluation metrics
├── datasets/                        # Dataset loaders
├── split_pipeline/                  # Pipeline splitting utilities
├── test/                            # Unit tests
├── main_synthetic.py                # Synthetic stream utilities
├── setup.sh                         # HPC environment setup
├── requirements.txt                 # Python dependencies
├── references.bib                   # Bibliography
├── report_ensemble_vs_generalist.tex # LaTeX report
└── further_studies/                 # Additional experiments and scripts
```

## Further Studies

Will be added in future
