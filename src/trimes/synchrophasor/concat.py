"""Phase 2: phase-continuous concatenation of analytic segments.

:func:`concat_segments` builds one :class:`~trimes.synchrophasor.signal.AnalyticSignal`
from a list of ``(duration, spec)`` segments, carrying the fundamental phase
across every boundary so the concatenated truth is C0 unless a jump is asked
for. It replaces hand-rolled steady -> event -> recovery builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trimes.synchrophasor.signal import AnalyticSignal
from trimes.synchrophasor.util import TWO_PI

__all__ = ["concat_segments"]


def _spec_local_callables(spec: dict, duration: float, nominal: float):
    """Resolve a dict spec to (amplitude_fn, frequency_fn, total_phase_fn).

    All three are functions of *local* time in ``[0, duration]`` and
    ``total_phase_fn(0) == 0`` so the caller can anchor the phase freely.

    Recognised keys (all optional): ``amplitude`` (scalar or ``(start, end)``
    ramp, default 1.0), ``frequency`` (scalar or ramp, default ``nominal``),
    ``rocof`` (Hz/s -- frequency ramps from ``frequency`` by this slope),
    ``am_depth`` / ``pm_depth`` / ``modulation_frequency`` (AM+PM modulation).
    """
    d = dict(spec)
    amp = d.get("amplitude", 1.0)

    if {"am_depth", "pm_depth", "modulation_frequency"} & d.keys():
        base = float(amp) if np.ndim(amp) == 0 else float(amp[0])
        kx = float(d.get("am_depth", 0.0))  # amplitude-modulation depth
        ka = float(d.get("pm_depth", 0.0))  # phase-modulation depth
        fm = float(d.get("modulation_frequency", 1.0))
        f0 = float(d.get("frequency", nominal))
        wm = TWO_PI * fm
        # 60255-118-1 Eq 25 sign convention; the -(cos - 1) keeps total_phase(0) = 0
        return (
            lambda t: base * (1.0 + kx * np.cos(wm * np.asarray(t, float))),
            lambda t: f0 + ka * fm * np.sin(wm * np.asarray(t, float)),
            lambda t: TWO_PI * f0 * np.asarray(t, float)
            - ka * (np.cos(wm * np.asarray(t, float)) - 1.0),
        )

    if np.ndim(amp) == 0:
        amp_fn = lambda t, a=float(amp): np.full(np.shape(np.asarray(t, float)), a)
    else:
        a0, a1 = float(amp[0]), float(amp[1])
        amp_fn = lambda t: a0 + (a1 - a0) / duration * np.asarray(t, float)

    if "rocof" in d:
        f0 = float(d.get("frequency", nominal))
        freq = (f0, f0 + float(d["rocof"]) * duration)
    else:
        freq = d.get("frequency", nominal)

    if np.ndim(freq) == 0:
        f0 = float(freq)
        freq_fn = lambda t, f=f0: np.full(np.shape(np.asarray(t, float)), f)
        phase_fn = lambda t, f=f0: TWO_PI * f * np.asarray(t, float)
    else:
        f0, f1 = float(freq[0]), float(freq[1])
        slope = (f1 - f0) / duration
        freq_fn = lambda t, f0=f0, s=slope: f0 + s * np.asarray(t, float)
        phase_fn = lambda t, f0=f0, s=slope: TWO_PI * (
            f0 * np.asarray(t, float) + 0.5 * s * np.asarray(t, float) ** 2
        )

    return amp_fn, freq_fn, phase_fn


def _segment_local_callables(spec, duration: float, nominal: float):
    """Local-time (amp, freq, total_phase) callables for one segment spec."""
    if isinstance(spec, AnalyticSignal):
        t0 = float(spec.time[0])
        p0 = float(np.atleast_1d(spec.total_phase_fn(np.array([t0])))[0])
        return (
            lambda t: np.asarray(spec.amplitude_fn(np.asarray(t, float) + t0), float),
            lambda t: np.asarray(spec.frequency_fn(np.asarray(t, float) + t0), float),
            lambda t: np.asarray(spec.total_phase_fn(np.asarray(t, float) + t0), float)
            - p0,
        )
    # PeriodicSignal (imported lazily to avoid an import cycle)
    from trimes.signal_generation import PeriodicSignal

    if isinstance(spec, PeriodicSignal):
        from trimes.synchrophasor.signal import _segment_truth_fns

        amp, freq, phase, ts, _ = _segment_truth_fns(spec)
        p0 = float(np.atleast_1d(phase(np.array([ts])))[0])
        return (
            lambda t: np.asarray(amp(np.asarray(t, float) + ts), float),
            lambda t: np.asarray(freq(np.asarray(t, float) + ts), float),
            lambda t: np.asarray(phase(np.asarray(t, float) + ts), float) - p0,
        )
    if isinstance(spec, dict):
        return _spec_local_callables(spec, duration, nominal)
    if callable(spec):
        result = spec(duration)
        if not (isinstance(result, tuple) and len(result) == 3):
            raise TypeError(
                "a callable segment spec must return "
                "(amplitude_fn, frequency_fn, total_phase_fn)"
            )
        return result
    raise TypeError(
        f"segment spec must be a dict, PeriodicSignal, AnalyticSignal or callable, "
        f"got {type(spec).__name__}"
    )


def concat_segments(
    segments,
    *,
    fs: float,
    nominal: float = 50.0,
    amplitude: float = 1.0,
    phi0: float = 0.0,
    jumps: dict | None = None,
    name: str = "concat",
) -> AnalyticSignal:
    """Concatenate ``(duration, spec)`` segments into one continuous signal.

    Parameters
    ----------
    segments : list of (float, spec)
        ``spec`` is a dict (see :func:`_spec_local_callables`), a
        :class:`~trimes.signal_generation.PeriodicSignal`, an
        :class:`~trimes.synchrophasor.signal.AnalyticSignal`, or a callable
        ``duration -> (amplitude_fn, frequency_fn, total_phase_fn)``.
    fs : float
        Sampling rate in Hz.
    nominal : float, default 50.0
        Phase reference in Hz and default segment frequency.
    amplitude : float, default 1.0
        Overall amplitude scale (segment specs multiply on top of this).
    phi0 : float, default 0.0
        Initial total phase in radians.
    jumps : dict, optional
        ``{boundary: {"phase_deg": float, "magnitude": float}}`` -- a deliberate
        discontinuity at an interior boundary. ``boundary`` is either the
        segment index it precedes (1 .. n-1) or a time matched to the nearest
        boundary. ``phase_deg`` adds to the fundamental phase; ``magnitude``
        multiplies the amplitude, and both persist for the rest of the record.
    name : str, default "concat"
        Name of the wrapped Series.

    Returns
    -------
    AnalyticSignal
        ``metadata`` carries ``transition_times`` (every interior boundary) and
        ``boundaries`` (all segment edges, including 0 and the end).
    """
    segs = [(float(d), spec) for d, spec in segments]
    if not segs:
        raise ValueError("at least one segment is required")
    durations = np.array([d for d, _ in segs], dtype=float)
    if np.any(durations <= 0.0):
        raise ValueError("every segment duration must be positive")
    boundaries = np.concatenate([[0.0], np.cumsum(durations)])
    total = float(boundaries[-1])

    resolved: dict[int, dict] = {}
    for key, value in (jumps or {}).items():
        idx = int(key) if isinstance(key, (int, np.integer)) else int(
            np.argmin(np.abs(boundaries - float(key)))
        )
        if not 1 <= idx <= len(segs) - 1:
            raise ValueError(f"jump key {key!r} is not an interior boundary")
        resolved[idx] = dict(value)

    global_fns = []  # (t_start, amplitude_fn, frequency_fn, total_phase_fn)
    phi_carry = float(phi0)
    mag_carry = float(amplitude)
    transition_times: list[float] = []

    for i, ((dur, spec), t_start) in enumerate(zip(segs, boundaries[:-1])):
        a_loc, f_loc, p_loc = _segment_local_callables(spec, dur, nominal)
        jump = resolved.get(i, {})
        phi_carry += np.radians(float(jump.get("phase_deg", 0.0)))
        mag_carry *= float(jump.get("magnitude", 1.0))

        ts = float(t_start)
        phi_start = phi_carry
        mag = mag_carry

        global_fns.append(
            (
                ts,
                lambda t, a=a_loc, ts=ts, m=mag: m
                * np.asarray(a(np.asarray(t, float) - ts), float),
                lambda t, f=f_loc, ts=ts: np.asarray(
                    f(np.asarray(t, float) - ts), float
                ),
                lambda t, p=p_loc, ts=ts, phi=phi_start: phi
                + np.asarray(p(np.asarray(t, float) - ts), float),
            )
        )
        phi_carry = phi_start + float(np.atleast_1d(p_loc(np.array([dur])))[0])
        if i + 1 < len(segs):
            transition_times.append(float(boundaries[i + 1]))

    next_starts = [g[0] for g in global_fns[1:]]
    n = len(global_fns)

    def _piecewise(component: int):
        def evaluate(t):
            t = np.asarray(t, float)
            flat = np.atleast_1d(t)
            out = np.empty(flat.shape, float)
            for k, g in enumerate(global_fns):
                lo = -np.inf if k == 0 else next_starts[k - 1]
                hi = np.inf if k == n - 1 else next_starts[k]
                mask = (flat >= lo) & (flat < hi)
                if np.any(mask):
                    out[mask] = np.asarray(g[component](flat[mask]), float)
            return out.reshape(t.shape) if t.ndim else out[0]

        return evaluate

    amplitude_fn = _piecewise(1)
    frequency_fn = _piecewise(2)
    total_phase_fn = _piecewise(3)

    t = np.arange(0.0, total, 1.0 / float(fs))
    series = pd.Series(
        amplitude_fn(t) * np.cos(total_phase_fn(t)),
        index=pd.Index(t, name="time"),
        name=name,
    )
    return AnalyticSignal(
        signal=series,
        amplitude_fn=amplitude_fn,
        frequency_fn=frequency_fn,
        total_phase_fn=total_phase_fn,
        nominal=float(nominal),
        metadata={
            "transition_times": sorted(set(transition_times)),
            "boundaries": [float(b) for b in boundaries],
        },
    )
