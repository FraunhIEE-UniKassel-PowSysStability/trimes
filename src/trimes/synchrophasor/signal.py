"""Backward-compatible alias of :mod:`trimes.analytic`.

:class:`AnalyticSignal` and friends moved to the core module
:mod:`trimes.analytic` so that :mod:`trimes.synchrophasor` sits cleanly on top
of them (and ``import trimes`` no longer drags in the whole benchmark layer).
Import from :mod:`trimes.analytic` in new code.
"""

from __future__ import annotations

from trimes.analytic import (
    ROCOF_STEP,
    TWO_PI,
    AnalyticSignal,
    build_analytic_signal,
    rms_phasor,
    wrap_pi,
    _segment_truth_fns,
)

__all__ = ["AnalyticSignal", "build_analytic_signal"]
