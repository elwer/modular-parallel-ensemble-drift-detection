"""
Optimized thread-based deployment for MOPEDDS.

Design goals:
- Minimal main thread overhead
- Lock-free data passing where possible
- Efficient synchronization using barriers/events
- Workers run continuously without idle time
"""

import os
import threading
import logging
import time
from collections import deque
from typing import List, Optional, Set

import scorep.user

try:
    from ..base import UnsupervisedDriftDetector
except ImportError:
    from detectors.base import UnsupervisedDriftDetector

logger = logging.getLogger(__name__)


class WorkerSlot:
    """
    Minimal shared memory slot for worker communication.
    Uses threading.Event for low-overhead synchronization.
    """
    __slots__ = ['sample_id', 'data', 'result', 'result_ready',
                 'decision_criteria', 'decision_window', 'clear_history',
                 'data_ready_event', 'result_ready_event']

    def __init__(self, decision_criteria: str = "majority", decision_window: int = 10):
        self.sample_id: int = 0
        self.data: Optional[dict] = None
        self.result: bool = False  # Now represents level 1 decision, not raw detector output
        self.result_ready: bool = False
        self.decision_criteria: str = decision_criteria
        self.decision_window: int = decision_window
        self.clear_history: bool = False  # Signal to clear decision history
        self.data_ready_event = threading.Event()
        self.result_ready_event = threading.Event()


class DetectorWorker(threading.Thread):
    """
    Optimized worker thread. Waits on data_ready_event for new samples.
    Maintains local decision history and computes level 1 decisions.
    """
    __slots__ = ['detector_idx', 'detector', 'slot', 'running', 'last_processed_id',
                 'deployment', 'name', 'decision_history', 'drift_count',
                 'track_stats', 'work_time', 'total_time', 'run_start',
                 'cpu_affinity']
    
    def __init__(self, detector_idx: int, detector: UnsupervisedDriftDetector,
                 slot: WorkerSlot, deployment, track_stats: bool = False,
                 cpu_affinity: Optional[int] = None):
        super().__init__(daemon=True)
        self.detector_idx = detector_idx
        self.detector = detector
        self.slot = slot
        self.deployment = deployment
        self.running = True
        self.last_processed_id = 0
        self.name = f"Worker-{detector_idx}-{detector.__class__.__name__}"
        self.decision_history: deque = deque(maxlen=slot.decision_window)
        self.drift_count: int = 0  # Running count of True values in history
        self.track_stats = track_stats
        self.work_time: float = 0.0
        self.total_time: float = 0.0
        self.run_start: float = 0.0
        self.cpu_affinity = cpu_affinity
    
    def _add_to_history(self, result: bool) -> None:
        """Add result to history and update drift_count incrementally."""
        # If deque is full, the oldest element will be evicted
        if len(self.decision_history) == self.decision_history.maxlen:
            evicted = self.decision_history[0]
            if evicted:
                self.drift_count -= 1
        
        # Add new result
        self.decision_history.append(result)
        if result:
            self.drift_count += 1
    
    def _clear_history(self) -> None:
        """Clear history and reset drift_count."""
        self.decision_history.clear()
        self.drift_count = 0
    
    def _apply_level1_decision(self) -> bool:
        """Apply level 1 decision criteria. O(1) using maintained drift_count."""
        if not self.decision_history:
            return False
        
        num_samples = len(self.decision_history)
        criteria = self.slot.decision_criteria
        
        if criteria == "any":
            return self.drift_count > 0
        elif criteria == "all":
            return self.drift_count == num_samples
        else:  # "majority" (default)
            return self.drift_count >= (num_samples + 1) // 2
    
    def run(self):
        with scorep.user.region(f"{self.name}.run"):
            # Pin this thread to a specific CPU core if requested
            if self.cpu_affinity is not None:
                try:
                    os.sched_setaffinity(0, {self.cpu_affinity})
                except (OSError, AttributeError):
                    pass  # Not supported on this platform

            slot = self.slot  # Local reference for speed
            detector = self.detector
            deployment = self.deployment
            last_id = 0
            self.run_start = time.perf_counter() if self.track_stats else 0.0

            while self.running:
                # Wait for new data (blocks without burning CPU)
                slot.data_ready_event.wait()
                slot.data_ready_event.clear()

                # Check if we need to clear history (after drift detection)
                if slot.clear_history:
                    self._clear_history()
                    slot.clear_history = False

                current_id = slot.sample_id
                if current_id <= last_id:
                    continue  # Spurious wakeup or already processed

                # Process the sample
                data = slot.data
                if self.track_stats:
                    _t0 = time.perf_counter()
                try:
                    raw_result = detector.update(data)
                except Exception:
                    raw_result = False
                if self.track_stats:
                    self.work_time += time.perf_counter() - _t0

                # Debug: print raw DD result when drift detected
                if raw_result and self.deployment.verbose:
                    print(f"    [Worker-{self.detector_idx}] Raw DD result: {raw_result} at sample_id={current_id}")

                # Always add raw result to local history (DDs should always process)
                self._add_to_history(raw_result)

                # Compute level 1 decision (O(1) using drift_count)
                level1_decision = self._apply_level1_decision()
                slot.result = level1_decision
                slot.result_ready = True
                slot.result_ready_event.set()

                last_id = current_id

            if self.track_stats:
                self.total_time = time.perf_counter() - self.run_start
    
    def stop(self):
        self.running = False
        self.slot.data_ready_event.set()  # Wake up worker if blocked on wait()

    def get_stats(self) -> dict:
        if self.track_stats and self.running and self.run_start > 0:
            total = time.perf_counter() - self.run_start
        else:
            total = self.total_time
        busy_wait = total - self.work_time
        return {
            "worker_idx": self.detector_idx,
            "worker_name": self.name,
            "work_time": self.work_time,
            "total_time": total,
            "busy_wait_time": busy_wait,
            "work_ratio": self.work_time / total if total > 0 else 0.0,
        }


class ThreadsDeployment:
    """
    Optimized parallel deployment with minimal main thread overhead.
    
    Architecture:
    - One WorkerSlot per detector (contains data + result)
    - Main thread writes data and signals workers via data_ready_event
    - Workers wait on data_ready_event, process, signal result via result_ready_event
    - Main thread waits on result_ready_event for all workers
    - Workers compute level 1 decisions locally (distributed computation)
    """
    
    def __init__(self, detectors: List[UnsupervisedDriftDetector],
                 verbose: bool = False,
                 mopedds=None,
                 detector_decision_criteria: str = "majority",
                 decision_window: int = 10,
                 track_stats: bool = False,
                 pin_cpus: bool = False):
        self.detectors = detectors
        self.verbose = verbose
        self.mopedds = mopedds  # Reference to MOPEDDS for suppression flag
        self.detector_decision_criteria = detector_decision_criteria
        self.decision_window = decision_window
        self.track_stats = track_stats
        self.pin_cpus = pin_cpus
        
        # Worker slots and threads
        self.slots: List[WorkerSlot] = []
        self.workers: List[DetectorWorker] = []
        
        # State tracking
        self.sample_counter = 0
        
        self._initialized = False

    @scorep.user.region("ThreadsDeployment.initialize")
    def initialize(self):
        if self._initialized:
            return
        
        num_detectors = len(self.detectors)
        
        # Create slots and workers with decision criteria config
        self.slots = [
            WorkerSlot(
                decision_criteria=self.detector_decision_criteria,
                decision_window=self.decision_window
            ) 
            for _ in range(num_detectors)
        ]
        self.workers = []
        
        # Determine CPU affinity for workers
        cpu_affinity = None
        if self.pin_cpus:
            avail_cpus = sorted(os.sched_getaffinity(0))
            # Reserve CPU 0 for the main thread, pin workers to CPUs 1..N
            worker_cpus = avail_cpus[1:num_detectors + 1] if len(avail_cpus) > num_detectors else avail_cpus

        for idx, detector in enumerate(self.detectors):
            cpu_id = None
            if self.pin_cpus and idx < len(worker_cpus):
                cpu_id = worker_cpus[idx]
            worker = DetectorWorker(idx, detector, self.slots[idx], self,
                                    track_stats=self.track_stats,
                                    cpu_affinity=cpu_id)
            worker.start()
            self.workers.append(worker)
        
        self._initialized = True
        logger.debug(f"MOPEDDS initialized with {num_detectors} workers (level1={self.detector_decision_criteria}, window={self.decision_window})")

    def clear_all_histories(self):
        """Signal all workers to clear their decision histories."""
        for slot in self.slots:
            slot.clear_history = True

    @scorep.user.region("ThreadsDeployment.shutdown")
    def shutdown(self):
        if not self._initialized:
            return
        
        for worker in self.workers:
            worker.stop()
        
        # Give workers a chance to exit
        for worker in self.workers:
            worker.join(timeout=0.1)
        
        self.workers = []
        self.slots = []
        self._initialized = False

    @scorep.user.region("ThreadsDeployment.update_all_detectors")
    def update_all_detectors(self, data: dict) -> List[bool]:
        """Update all detectors and return raw results list.
        
        Returns:
            List of bools indicating drift detection per detector,
            or None if in suppression mode.
        """
        if not self._initialized:
            self.initialize()
        
        # Use MOPEDDS's sample counter for consistency
        sample_id = self.mopedds.sample_counter if self.mopedds else self.sample_counter + 1
        if not self.mopedds:
            self.sample_counter += 1
        
        # Write data to all slots and signal workers
        for slot in self.slots:
            slot.data = data
            slot.result_ready = False
            slot.result_ready_event.clear()
            slot.sample_id = sample_id
            slot.data_ready_event.set()  # Signal worker to start
        
        # If in suppression, return immediately (workers still process but don't write results)
        if self.mopedds and self.mopedds.in_suppression:
            return None
        
        # Wait for all results (blocks without burning CPU)
        num_workers = len(self.slots)
        results = [False] * num_workers
        
        for idx in range(num_workers):
            self.slots[idx].result_ready_event.wait()
            results[idx] = self.slots[idx].result
        
        return results

    def get_worker_stats(self) -> List[dict]:
        if not self.track_stats:
            return []
        return [w.get_stats() for w in self.workers]

