#!/usr/bin/env python3
"""
Test to verify that detector state (recent_samples) is properly maintained
during parallel processing in both ThreadsDeployment and MultiprocessingDeployment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location("base", "detectors/base.py")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
UnsupervisedDriftDetector = base_module.UnsupervisedDriftDetector

spec = importlib.util.spec_from_file_location("mopedds", "detectors/mopedds.py")
mopedds_module = importlib.util.module_from_spec(spec)
sys.modules['detectors.base'] = base_module  # Make base available for mopedds
spec.loader.exec_module(mopedds_module)
ThreadsDeployment = mopedds_module.ThreadsDeployment
MultiprocessingDeployment = mopedds_module.MultiprocessingDeployment


class TestDetector(UnsupervisedDriftDetector):
    """Test detector that tracks recent_samples."""
    
    def __init__(self, detector_id=0, **kwargs):
        super().__init__(**kwargs)
        self.detector_id = detector_id
        self.update_count = 0
    
    def update(self, data: dict) -> bool:
        """Update detector and add to recent_samples."""
        self.update_count += 1
        
        # Add data to recent_samples (simulating what real detectors do)
        if hasattr(data, 'get'):
            value = data.get('feature', 0.0)
        else:
            value = float(data)
        
        self.recent_samples.append(value)
        
        # Keep only recent_samples_size items
        if len(self.recent_samples) > self.recent_samples_size:
            self.recent_samples = self.recent_samples[-self.recent_samples_size:]
        
        # Never detect drift in this test
        return False


def test_threads_deployment_state():
    """Test that ThreadsDeployment maintains detector state."""
    print("\n" + "="*70)
    print("TEST 1: ThreadsDeployment State Preservation")
    print("="*70)
    
    # Create detectors
    detectors = [
        TestDetector(detector_id=0, seed=42, recent_samples_size=100),
        TestDetector(detector_id=1, seed=42, recent_samples_size=100),
        TestDetector(detector_id=2, seed=42, recent_samples_size=100),
    ]
    
    # Create deployment
    deployment = ThreadsDeployment(detectors)
    deployment.initialize()
    
    # Verify initial state
    for i, detector in enumerate(deployment.detectors):
        assert len(detector.recent_samples) == 0, f"Detector {i} should start with empty recent_samples"
        print(f"  Detector {i}: recent_samples size = {len(detector.recent_samples)} (initial)")
    
    # Update 10 times
    for update_num in range(10):
        data = {"feature": update_num * 0.1}
        deployment.update_all_detectors(data)
    
    # Verify state grew
    print("\n  After 10 updates:")
    for i, detector in enumerate(deployment.detectors):
        samples_count = len(detector.recent_samples)
        print(f"  Detector {i}: recent_samples size = {samples_count}")
        assert samples_count == 10, f"Detector {i} should have 10 samples, got {samples_count}"
        assert detector.update_count == 10, f"Detector {i} should have 10 updates, got {detector.update_count}"
    
    # Update 100 more times
    for update_num in range(10, 110):
        data = {"feature": update_num * 0.1}
        deployment.update_all_detectors(data)
    
    # Verify state is capped at recent_samples_size
    print("\n  After 110 updates:")
    for i, detector in enumerate(deployment.detectors):
        samples_count = len(detector.recent_samples)
        print(f"  Detector {i}: recent_samples size = {samples_count} (capped at recent_samples_size)")
        assert samples_count == 100, f"Detector {i} should have 100 samples (capped), got {samples_count}"
        assert detector.update_count == 110, f"Detector {i} should have 110 updates, got {detector.update_count}"
    
    deployment.shutdown()
    print("\n✓ ThreadsDeployment state preservation test PASSED!")


def test_multiprocessing_deployment_state():
    """Test that MultiprocessingDeployment maintains detector state."""
    print("\n" + "="*70)
    print("TEST 2: MultiprocessingDeployment State Preservation")
    print("="*70)
    
    # Create detectors
    detectors = [
        TestDetector(detector_id=0, seed=42, recent_samples_size=100),
        TestDetector(detector_id=1, seed=42, recent_samples_size=100),
        TestDetector(detector_id=2, seed=42, recent_samples_size=100),
    ]
    
    # Create deployment
    deployment = MultiprocessingDeployment(detectors)
    deployment.initialize()
    
    # Verify initial state
    for i, detector in enumerate(deployment.detectors):
        assert len(detector.recent_samples) == 0, f"Detector {i} should start with empty recent_samples"
        print(f"  Detector {i}: recent_samples size = {len(detector.recent_samples)} (initial)")
    
    # Update 10 times
    for update_num in range(10):
        data = {"feature": update_num * 0.1}
        deployment.update_all_detectors(data)
    
    # Verify state grew
    print("\n  After 10 updates:")
    for i, detector in enumerate(deployment.detectors):
        samples_count = len(detector.recent_samples)
        print(f"  Detector {i}: recent_samples size = {samples_count}")
        assert samples_count == 10, f"Detector {i} should have 10 samples, got {samples_count}"
    
    # Update 100 more times
    for update_num in range(10, 110):
        data = {"feature": update_num * 0.1}
        deployment.update_all_detectors(data)
    
    # Verify state is capped at recent_samples_size
    print("\n  After 110 updates:")
    for i, detector in enumerate(deployment.detectors):
        samples_count = len(detector.recent_samples)
        print(f"  Detector {i}: recent_samples size = {samples_count} (capped at recent_samples_size)")
        assert samples_count == 100, f"Detector {i} should have 100 samples (capped), got {samples_count}"
    
    deployment.shutdown()
    print("\n✓ MultiprocessingDeployment state preservation test PASSED!")


if __name__ == "__main__":
    try:
        test_threads_deployment_state()
        test_multiprocessing_deployment_state()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nDetector state (recent_samples) is properly maintained in both")
        print("ThreadsDeployment and MultiprocessingDeployment.")
        print("="*70 + "\n")
        
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
