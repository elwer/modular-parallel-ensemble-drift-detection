#!/usr/bin/env python3
"""
Quick test to verify the deployment architecture works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need to avoid dependency issues
from detectors.mopedds import ThreadsDeployment, MultiprocessingDeployment
from detectors.base import UnsupervisedDriftDetector


class MockDetector(UnsupervisedDriftDetector):
    """Mock detector for testing."""
    
    def __init__(self, detector_id=0, **kwargs):
        super().__init__(**kwargs)
        self.detector_id = detector_id
        self.update_count = 0
    
    def update(self, data: dict) -> bool:
        """Mock update that returns False (no drift)."""
        self.update_count += 1
        return False


def test_threads_deployment():
    """Test ThreadsDeployment with persistent thread pool."""
    print("Testing ThreadsDeployment...")
    
    # Create mock detectors
    detectors = [MockDetector(detector_id=i, seed=42) for i in range(3)]
    
    # Create deployment
    deployment = ThreadsDeployment(detectors)
    
    # Initialize (creates thread pool)
    deployment.initialize()
    assert deployment._initialized, "Deployment should be initialized"
    assert deployment.executor is not None, "Thread pool should be created"
    
    # Update multiple times (should reuse thread pool)
    for i in range(5):
        data = {"feature": i * 0.1}
        drift = deployment.update_all_detectors(data)
        assert not drift, "Mock detectors should not detect drift"
    
    # Verify all detectors were updated
    for detector in detectors:
        assert detector.update_count == 5, f"Detector should have been updated 5 times, got {detector.update_count}"
    
    # Shutdown
    deployment.shutdown()
    assert not deployment._initialized, "Deployment should be shutdown"
    assert deployment.executor is None, "Thread pool should be cleaned up"
    
    print("✓ ThreadsDeployment test passed!")


def test_multiprocessing_deployment():
    """Test MultiprocessingDeployment with persistent process pool."""
    print("Testing MultiprocessingDeployment...")
    
    # Create mock detectors
    detectors = [MockDetector(detector_id=i, seed=42) for i in range(3)]
    
    # Create deployment
    deployment = MultiprocessingDeployment(detectors)
    
    # Initialize (creates process pool)
    deployment.initialize()
    assert deployment._initialized, "Deployment should be initialized"
    assert deployment.pool is not None, "Process pool should be created"
    
    # Update multiple times (should reuse process pool)
    for i in range(3):
        data = {"feature": i * 0.1}
        drift = deployment.update_all_detectors(data)
        assert not drift, "Mock detectors should not detect drift"
    
    # Shutdown
    deployment.shutdown()
    assert not deployment._initialized, "Deployment should be shutdown"
    assert deployment.pool is None, "Process pool should be cleaned up"
    
    print("✓ MultiprocessingDeployment test passed!")


def test_context_manager():
    """Test context manager support."""
    print("Testing context manager...")
    
    detectors = [MockDetector(detector_id=i, seed=42) for i in range(2)]
    
    with ThreadsDeployment(detectors) as deployment:
        assert deployment._initialized, "Should be initialized in context"
        data = {"feature": 0.5}
        deployment.update_all_detectors(data)
    
    # Should be cleaned up after context
    assert not deployment._initialized, "Should be shutdown after context"
    
    print("✓ Context manager test passed!")


if __name__ == "__main__":
    try:
        test_threads_deployment()
        test_multiprocessing_deployment()
        test_context_manager()
        print("\n✓ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
