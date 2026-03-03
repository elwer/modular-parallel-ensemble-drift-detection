#!/usr/bin/env python3
"""
Simple test to demonstrate the state preservation fix for MultiprocessingDeployment.

This test shows that the fix properly returns and updates detector objects
so that their internal state (like recent_samples) is preserved across updates.
"""

from multiprocessing import Pool


class SimpleDetector:
    """Simplified detector for testing state preservation."""
    
    def __init__(self, detector_id):
        self.detector_id = detector_id
        self.samples = []  # Simulates recent_samples
        self.update_count = 0
    
    def update(self, value):
        """Update the detector with a new value."""
        self.update_count += 1
        self.samples.append(value)
        return False  # Never detect drift in this test


def update_detector_worker_OLD(detector, value):
    """OLD implementation - only returns drift status, loses state."""
    drift = detector.update(value)
    return drift  # State changes are lost!


def update_detector_worker_NEW(detector, value):
    """NEW implementation - returns both drift status and updated detector."""
    drift = detector.update(value)
    return (drift, detector)  # State is preserved!


def test_old_implementation():
    """Test OLD implementation - state is NOT preserved."""
    print("\n" + "="*70)
    print("TEST 1: OLD Implementation (State NOT Preserved)")
    print("="*70)
    
    detectors = [SimpleDetector(i) for i in range(3)]
    
    print(f"\nInitial state:")
    for i, det in enumerate(detectors):
        print(f"  Detector {i}: samples={len(det.samples)}, updates={det.update_count}")
    
    # Simulate 5 updates using multiprocessing
    with Pool(processes=3) as pool:
        for update_num in range(5):
            args_list = [(det, update_num * 0.1) for det in detectors]
            results = pool.starmap(update_detector_worker_OLD, args_list)
            # Only get drift status back, detector state is lost!
    
    print(f"\nAfter 5 updates (OLD implementation):")
    for i, det in enumerate(detectors):
        print(f"  Detector {i}: samples={len(det.samples)}, updates={det.update_count}")
    
    # Check if state was preserved
    all_preserved = all(len(det.samples) == 5 for det in detectors)
    if all_preserved:
        print("\n✓ State WAS preserved (unexpected!)")
    else:
        print("\n✗ State was NOT preserved (expected with OLD implementation)")
        print("   This is the BUG - detector state is lost in multiprocessing!")


def test_new_implementation():
    """Test NEW implementation - state IS preserved."""
    print("\n" + "="*70)
    print("TEST 2: NEW Implementation (State IS Preserved)")
    print("="*70)
    
    detectors = [SimpleDetector(i) for i in range(3)]
    
    print(f"\nInitial state:")
    for i, det in enumerate(detectors):
        print(f"  Detector {i}: samples={len(det.samples)}, updates={det.update_count}")
    
    # Simulate 5 updates using multiprocessing with NEW implementation
    with Pool(processes=3) as pool:
        for update_num in range(5):
            args_list = [(det, update_num * 0.1) for det in detectors]
            results = pool.starmap(update_detector_worker_NEW, args_list)
            
            # Update detectors with the returned updated versions
            for i, (drift, updated_detector) in enumerate(results):
                detectors[i] = updated_detector  # KEY FIX: Replace with updated detector
    
    print(f"\nAfter 5 updates (NEW implementation):")
    for i, det in enumerate(detectors):
        print(f"  Detector {i}: samples={len(det.samples)}, updates={det.update_count}")
    
    # Check if state was preserved
    all_preserved = all(len(det.samples) == 5 and det.update_count == 5 for det in detectors)
    if all_preserved:
        print("\n✓ State WAS preserved (expected with NEW implementation)")
        print("   The FIX works - detector state is properly maintained!")
        return True
    else:
        print("\n✗ State was NOT preserved (unexpected!)")
        return False


def main():
    print("\n" + "#"*70)
    print("# State Preservation Test for MultiprocessingDeployment Fix")
    print("#"*70)
    print("\nThis demonstrates why we need to return the updated detector object")
    print("from worker processes, not just the drift status.")
    
    test_old_implementation()
    success = test_new_implementation()
    
    print("\n" + "="*70)
    if success:
        print("✓ FIX VERIFIED: Detector state is properly preserved!")
    else:
        print("✗ FIX FAILED: Detector state is still being lost!")
    print("="*70)
    
    print("\nSummary:")
    print("  - OLD: Worker returns only 'bool' -> state lost")
    print("  - NEW: Worker returns '(bool, detector)' -> state preserved")
    print("  - Applied to: MultiprocessingDeployment.update_all_detectors()")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
