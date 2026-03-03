import os
import time
from memory_profiler import memory_usage


def _is_scorep_active():
    """Check if Score-P profiling/tracing is enabled."""
    return os.environ.get('SCOREP_ENABLE_PROFILING') is not None or \
           os.environ.get('SCOREP_ENABLE_TRACING') is not None


def computational_metrics(func):
    def wrapper(*args, **kwargs):
        # Skip memory profiling when Score-P is active
        if _is_scorep_active():
            result = func(*args, **kwargs)
            return result + (0.0,) + (0.0,) + (0.0,)
        
        start = time.time()
        result = memory_usage((func, args, kwargs), interval=10, max_iterations=1, retval=True)
        duration = time.time() - start
        if not isinstance(result, tuple):
            raise ValueError(
                """Unexpected result, should be tuple with: [drifts], [labels],
                [predictions], n_req_labels"""
            )
        peak_mem = max(result[0])
        mean_mem = sum(result[0]) / len(result[0])
        return result[1] + (duration,) + (peak_mem,) + (mean_mem,)

    return wrapper
