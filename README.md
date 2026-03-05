# Unsupervised Concept Drift Detection

## Detectors

The following unsupervised drift detectors are supported:

- **BNDM** — Bayesian Nonparametric Drift Detection
- **CSDDM** — Clustered Statistical Test Drift Detection Method
- **D3** — Discriminative Drift Detector
- **IBDD** — Image-Based Drift Detector
- **OCDD** — One-Class Drift Detector
- **SPLL** — Semi-Parametric Log-Likelihood
- **UDetect** — Unsupervised Change Detection for Activity Recognition

## Hyperparameter Optimization

Optimization uses [Optuna](https://optuna.org/) with a TPE sampler.

### Submit all optimization jobs (SLURM)

The `submit_dds_optimization.sh` script submits one SLURM job per detector:

```bash
bash submit_dds_optimization.sh
```

An optional argument overrides the default number of trials (1000):

```bash
bash submit_dds_optimization.sh 500
```

### Submit individual jobs manually

```bash
sbatch --job-name="DD_CSDDM" \
       --export=ALL,DETECTOR="CSDDM",N_TRIALS="1000" \
       optimize_single_dd.sbatch
```

### Run optimization locally (without SLURM)

```bash
python optimization/single_dd_optimize_optuna.py --n_trials 100 --detectors CSDDM D3
```
