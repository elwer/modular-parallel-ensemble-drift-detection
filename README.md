# Parallel Unsupervised Concept Drift Detection

## Detectors

The following unsupervised drift detectors are supported:

- **BNDM** — Bayesian Nonparametric Drift Detection
- **CSDDM** — Clustered Statistical Test Drift Detection Method
- **D3** — Discriminative Drift Detector
- **IBDD** — Image-Based Drift Detector
- **OCDD** — One-Class Drift Detector
- **SPLL** — Semi-Parametric Log-Likelihood
- **UDetect** — Unsupervised Change Detection for Activity Recognition
- **MOPEDDS** — Ensemble Window-based Drift Detection (ensemble of the above)

## Hyperparameter Optimization

Optimization uses [Optuna](https://optuna.org/) with a TPE sampler.

### Submit all optimization jobs (SLURM)

The `submit_dds_optimization.sh` script submits one SLURM job per single detector plus one for the MOPEDDS ensemble (8 jobs total):

```bash
bash submit_dds_optimization.sh
```

An optional argument overrides the default number of trials (1000):

```bash
bash submit_dds_optimization.sh 500
```

### Submit individual jobs manually

Single detector:

```bash
sbatch --job-name="DD_CSDDM" \
       --export=ALL,DETECTOR="CSDDM",N_TRIALS="1000" \
       optimize_single_dd.sbatch
```

MOPEDDS ensemble:

```bash
sbatch --export=ALL,N_TRIALS="1000" optimize_mopedds.sbatch
```

### Run optimization locally (without SLURM)

Single detector (one or more):

```bash
python optimization/single_dd_optimize_optuna.py --n_trials 100 --detectors CSDDM D3
```

MOPEDDS ensemble:

```bash
python optimization/mopedds_optimize_optuna.py --n_trials 100
```

## Testing Configurations

After optimization, use `run_config_detectors.py` to evaluate a configuration on a dataset.

### Usage

```bash
python run_config_detectors.py <Dataset> <ConfigPath> <RecentSamplesSize> <TrainSamples> <Accuracy> <Runtime> <ReqLabels>
```

### Arguments

| Argument            | Description                                              |
|---------------------|----------------------------------------------------------|
| `Dataset`           | Dataset name, e.g. `Electricity` or `ForestCovertype`    |
| `ConfigPath`        | Path to the YAML config file with detector definitions   |
| `RecentSamplesSize` | Number of recent samples for drift detection (int)       |
| `TrainSamples`      | Number of training samples (int)                         |
| `Accuracy`          | Output accuracy metric (`True`/`False`)                  |
| `Runtime`           | Output runtime metric (`True`/`False`)                   |
| `ReqLabels`         | Output requested labels metric (`True`/`False`)          |

### Example

```bash
python run_config_detectors.py Electricity detectors/mopedds/configs/mopedds.config 500 1600 True True False
```

### Submit as a SLURM job

```bash
sbatch run_single_dds_with_mopedds_config.sbatch
```
