"""
Backward compatibility module for EWDD.

The EWDD implementation has been moved to a package structure at detectors/ewdd/.
This module provides backward compatibility by re-exporting the main classes.

For new code, prefer importing directly from the package:
    from detectors.ewdd import EWDD, ThreadsDeployment, MultiprocessingDeployment
"""

# Re-export main classes for backward compatibility
from .ewdd import (
    EWDD,
    DriftDetectorDeployment,
    ThreadsDeployment,
    MultiprocessingDeployment,
    OpenMPDeployment,
    OpenMPCythonDeployment,
    DaskDeployment,
    DaskBagDeployment,
    MPIDeployment,
    MPIAsyncDeployment,
)

__all__ = [
    'EWDD',
    'DriftDetectorDeployment',
    'ThreadsDeployment',
    'MultiprocessingDeployment',
    'OpenMPDeployment',
    'OpenMPCythonDeployment',
    'DaskDeployment',
    'DaskBagDeployment',
    'MPIDeployment',
    'MPIAsyncDeployment',
]
