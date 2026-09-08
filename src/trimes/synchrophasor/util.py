"""Small numeric helpers shared across the synchrophasor layer.

Kept in one place so :mod:`trimes.synchrophasor.signal`, ``.generators``,
``.metrics`` and ``.plot`` agree on the phasor and angle conventions.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TWO_PI", "ROCOF_STEP", "wrap_pi", "rms_phasor"]

TWO_PI = 2.0 * np.pi

#: Half-step of the central difference used for the ROCOF truth, in seconds.
ROCOF_STEP = 1e-4


def wrap_pi(x) -> np.ndarray:
    """Wrap an angle (radians) into ``[-pi, pi)``."""
    return (np.asarray(x, dtype=float) + np.pi) % TWO_PI - np.pi


def rms_phasor(amplitude, phase) -> np.ndarray:
    """RMS complex phasor from a peak amplitude and a phase in radians.

    ``(amplitude / sqrt(2)) * exp(1j * phase)`` -- the convention an IEC/IEEE
    60255-118-1 estimator reports.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase, dtype=float)
    return (amplitude / np.sqrt(2.0)) * np.exp(1j * phase)
