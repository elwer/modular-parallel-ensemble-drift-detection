"""
MOPEDDS - Optimized Ensemble Wrapper Drift Detector.

Redesigned for minimal runtime overhead.
"""

import os
import yaml
import logging
from typing import Optional, List
from pathlib import Path
from collections import deque
from itertools import islice
import scorep

try:
    from ..base import UnsupervisedDriftDetector
    from .threads_deployment import ThreadsDeployment
    from metrics.computational_metrics import computational_metrics
except ImportError:
    from detectors.base import UnsupervisedDriftDetector
    from detectors.mopedds.threads_deployment import ThreadsDeployment
    from metrics.computational_metrics import computational_metrics

logger = logging.getLogger(__name__)


class MOPEDDS(UnsupervisedDriftDetector):
    """
    Optimized Ensemble Wrapper Drift Detector.
    """
    
    DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"
    
    def __init__(
        self,
        config_path: str = None,
        seed: Optional[int] = None,
        recent_samples_size: int = 500,
        **kwargs
    ):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)
        
        self.detectors = []
        self.config_path = config_path
        self.deployment = None
        self.detector_decision_criteria = "majority"  # Level 1: per-detector decision
        self.ensemble_decision_criteria = "any"       # Level 2: ensemble decision
        self.verbose = False
        self.decision_window = 10
        self.suppression_window = None  # If None, uses decision_window
        
        # Decision state
        self.sample_counter = 0
        self.drift_reported_at_sample = -1
        self.in_suppression = False
        
        if config_path:
            self._load_config()

    @scorep.user.region("MOPEDDS._load_config")
    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Level 1: How each detector decides drift across the window (default: majority)
        self.detector_decision_criteria = config.get('detector_decision_criteria', 'majority').lower()
        # Level 2: How ensemble combines detector decisions (default: any)
        self.ensemble_decision_criteria = config.get('ensemble_decision_criteria', 'any').lower()
        # Backward compatibility: if old 'decision_criteria' exists, use it for ensemble
        if 'decision_criteria' in config and 'ensemble_decision_criteria' not in config:
            self.ensemble_decision_criteria = config.get('decision_criteria', 'any').lower()
        self.verbose = config.get('verbose', False)
        self.decision_window = config.get('decision_window', 10)
        self.suppression_window = config.get('suppression_window', None)  # Separate suppression window
        
        for detector_config in config.get('detectors', []):
            detector_class = detector_config['class']
            params = detector_config.get('params', {})
            self._init_detector(detector_class, **params)
        
        logger.info(f"MOPEDDS loaded {len(self.detectors)} detectors")

    @scorep.user.region("MOPEDDS._init_detector")
    def _init_detector(self, detector_class: str, **kwargs):
        try:
            module_path, class_name = detector_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            detector = cls(
                seed=self.seed,
                recent_samples_size=self.recent_samples_size,
                **kwargs
            )
            self.detectors.append(detector)
        except Exception as e:
            logger.error(f"Failed to init {detector_class}: {e}")

    @scorep.user.region("MOPEDDS.deploy")
    def deploy(self):
        # Use suppression_window if set, otherwise fall back to decision_window
        effective_suppression = self.suppression_window if self.suppression_window is not None else self.decision_window
        
        print(f"\n=== MOPEDDS Deployment ===")
        print(f"Ensemble decision criteria: {self.ensemble_decision_criteria}")
        print(f"Detector decision criteria: {self.detector_decision_criteria}")
        print(f"Decision window: {self.decision_window}")
        print(f"Suppression window: {effective_suppression}")
        print(f"Number of detectors: {len(self.detectors)}")
        print(f"\nDeployed detectors:")
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__
            detector_params = {k: v for k, v in vars(detector).items() 
                              if not k.startswith('_') and k not in ('seed', 'recent_samples_size', 'recent_samples')}
            print(f"  [{i+1}] {detector_name}: {detector_params}")
        print(f"========================\n")
        
        self.deployment = ThreadsDeployment(
            self.detectors,
            self.verbose,
            self,  # Pass reference to MOPEDDS for suppression flag
            detector_decision_criteria=self.detector_decision_criteria,
            decision_window=self.decision_window
        )
        self.deployment.initialize()
        return self.deployment

    @scorep.user.region("MOPEDDS.shutdown")
    def shutdown(self):
        if self.deployment:
            self.deployment.shutdown()
            self.deployment = None

    @scorep.user.region("MOPEDDS.update")
    def update(self, data: dict) -> bool:
        if not self.deployment:
            self.deploy()
        
        self.sample_counter += 1
        sample_id = self.sample_counter
        
        # Check suppression state (use suppression_window if set, otherwise decision_window)
        effective_suppression = self.suppression_window if self.suppression_window is not None else self.decision_window
        if self.drift_reported_at_sample >= 0:
            samples_since = sample_id - self.drift_reported_at_sample
            if samples_since <= effective_suppression:
                scorep.user.parameter_int("in_suppression", samples_since)
                self.in_suppression = True
                if self.verbose:
                    print(f"  [MOPEDDS] Sample {sample_id}: IN SUPPRESSION (samples_since={samples_since}, window={effective_suppression})")
            else:
                scorep.user.parameter_int("in_suppression", samples_since)
                self.in_suppression = False
                self.drift_reported_at_sample = -1
        
        # Get results from deployment
        results = self.deployment.update_all_detectors(data)
        
        # If in suppression, return immediately
        if self.in_suppression:
            # Debug: show when suppression blocks drift reporting (results may be None during suppression)
            if self.verbose:
                print(f"  [MOPEDDS] Sample {sample_id}: SUPPRESSED (drift_reported_at={self.drift_reported_at_sample}, results={results})")
            return False
        
        # Apply level 2 decision criteria (results already contain level 1 decisions from workers)
        drift_decision = self._apply_level2_decision(results)
        
        # Debug: print when any DD reports drift
        if self.verbose and results and any(results):
            print(f"  [MOPEDDS] Sample {sample_id}: DD results={results}, ensemble decision={drift_decision}, in_suppression={self.in_suppression}")
        
        # Mark suppression if drift detected (don't clear histories - DDs should continue normally)
        if drift_decision:
            self.drift_reported_at_sample = sample_id
            scorep.user.parameter_int("in_suppression", 0)
            self.in_suppression = True
        
        return drift_decision
    
    @scorep.user.region("MOPEDDS._apply_level2_decision")
    def _apply_level2_decision(self, detector_decisions: list) -> bool:
        """Level 2 decision criteria.
        
        Combines individual detector decisions (already computed by workers using level 1)
        into the final ensemble decision.
        
        Args:
            detector_decisions: List of bools, one per detector, representing their
                               level 1 decisions computed over the decision window.
        """
        if not detector_decisions:
            return False
        
        num_detectors = len(self.detectors)
        detectors_reporting_drift = sum(detector_decisions)
        
        if self.ensemble_decision_criteria == "any":
            # Ensemble reports drift if ANY detector reports drift
            return detectors_reporting_drift > 0
        elif self.ensemble_decision_criteria == "all":
            # Ensemble reports drift only if ALL detectors report drift
            return detectors_reporting_drift == num_detectors
        else:  # "majority" (default)
            # Ensemble reports drift if majority of detectors report drift
            return detectors_reporting_drift >= (num_detectors + 1) // 2

    @computational_metrics
    @scorep.user.region("MOPEDDS.process_main_stream")
    def process_main_stream(self, stream, n_training_samples: int, classifier):
        """Override to only train on NEW samples since last retraining, avoiding duplicate training."""
        # Track the last sample index we trained on
        last_trained_idx = n_training_samples - 1
        
        for i, (x, y) in enumerate(islice(stream, n_training_samples, None),
                                   start=n_training_samples):

            if self.single_variate:
                if not self.feature_key:
                    self.retrieve_key_from_idx(list(x.keys()))
                x = {self.feature_key: x[self.feature_key]}

            self.recent_samples.pop(0)
            self.recent_samples.append((x, y))

            self.predictions.append(classifier.predict(x))
            self.labels.append(y)

            if self.update(x):  # Use MOPEDDS's update method
                with scorep.user.region("MOPEDDS.update_classifier"):
                    self.drifts.append(i)
                    
                    # Only train on samples AFTER last_trained_idx (avoid duplicates)
                    # Calculate how many new samples we have since last training
                    new_samples_count = i - last_trained_idx
                    
                    if new_samples_count > 0:
                        # Get only the most recent new_samples_count samples
                        samples_to_train = list(self.recent_samples)[-new_samples_count:]
                        for new_x, new_y in samples_to_train:
                            classifier.fit(new_x, new_y)
                        
                        # Update used_labels_set for only the new samples
                        self.used_labels_set.update(range(last_trained_idx + 1, i + 1))
                    
                    # Update last trained index
                    last_trained_idx = i

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    @scorep.user.region("MOPEDDS.run_stream")
    def run_stream(self, stream, n_training_samples: int, classifier_path):
        try:
            self.deploy()
            return super().run_stream(stream, n_training_samples, classifier_path)
        finally:
            self.shutdown()
    
    def __del__(self):
        self.shutdown()
