"""
Backward compatibility module for MOPEDDS.

The MOPEDDS implementation has been moved to a package structure at detectors/mopedds/.
This module provides backward compatibility by re-exporting the main classes.

For new code, prefer importing directly from the package:
    from detectors.mopedds import MOPEDDS, ThreadsDeployment, MultiprocessingDeployment
"""

# Re-export main classes for backward compatibility
from .mopedds import (
    MOPEDDS,
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
    'MOPEDDS',
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
