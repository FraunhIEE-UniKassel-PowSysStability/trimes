"""Backward-compatible alias for the phasor/angle helpers.

``wrap_pi``, ``rms_phasor``, ``TWO_PI`` and ``ROCOF_STEP`` moved to the core
module :mod:`trimes.analytic`. Import from there in new code.
"""

from __future__ import annotations

from trimes.analytic import ROCOF_STEP, TWO_PI, rms_phasor, wrap_pi

__all__ = ["TWO_PI", "ROCOF_STEP", "wrap_pi", "rms_phasor"]
