"""Synchrophasor test signals, analytic ground truth, compliance metrics.

The IEC/IEEE 60255-118-1 benchmark layer:

* :class:`AnalyticSignal` -- a sampled waveform plus exact
  ``amplitude`` / ``frequency`` / ``total_phase`` callables, from which the
  ``phase`` / ``phasor`` / ``rocof`` / ``ground_truth`` an estimator is scored
  against are derived at arbitrary instants.
* :func:`build_analytic_signal` -- factory from closed-form callables.
* basic generators (:func:`steady_state`, :func:`modulation`,
  :func:`frequency_ramp`, :func:`amplitude_step`, :func:`phase_step`).
* :func:`concat_segments` -- phase-continuous concatenation (Phase 2).
* :func:`three_phase` -- abc expansion with symmetrical-component content (Phase 3).
* :mod:`~trimes.synchrophasor.conditions` -- the 60255-118-1 test matrix (Phase 4).
* :mod:`~trimes.synchrophasor.metrics` -- TVE / FE / RFE, ``summary``,
  ``compliance`` (Phase 5).
* :mod:`~trimes.synchrophasor.limits` -- ``P_CLASS_LIMITS`` / ``M_CLASS_LIMITS``.
* :mod:`~trimes.synchrophasor.plot` -- overlays and sweep heatmaps (Phase 6,
  needs matplotlib; not imported here).

numpy + pandas only (``plot`` aside).
"""

from trimes.synchrophasor import conditions, limits, metrics
from trimes.synchrophasor.concat import concat_segments
from trimes.synchrophasor.generators import (
    amplitude_step,
    frequency_ramp,
    modulation,
    phase_step,
    steady_state,
)
from trimes.synchrophasor.limits import M_CLASS_LIMITS, P_CLASS_LIMITS
from trimes.synchrophasor.metrics import (
    compliance,
    errors,
    fe,
    rfe,
    summary,
    tve,
)
from trimes.synchrophasor.signal import AnalyticSignal, build_analytic_signal
from trimes.synchrophasor.three_phase import three_phase
from trimes.synchrophasor.util import ROCOF_STEP, TWO_PI, rms_phasor, wrap_pi

__all__ = [
    "AnalyticSignal",
    "build_analytic_signal",
    "steady_state",
    "amplitude_step",
    "phase_step",
    "frequency_ramp",
    "modulation",
    "concat_segments",
    "three_phase",
    "conditions",
    "metrics",
    "limits",
    "P_CLASS_LIMITS",
    "M_CLASS_LIMITS",
    "errors",
    "summary",
    "compliance",
    "tve",
    "fe",
    "rfe",
    "wrap_pi",
    "rms_phasor",
    "TWO_PI",
    "ROCOF_STEP",
]
