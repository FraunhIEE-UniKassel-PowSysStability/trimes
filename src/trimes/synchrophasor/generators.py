"""Analytic test-signal generators.

Each function returns an :class:`~trimes.synchrophasor.signal.AnalyticSignal`
whose truth callables are exact closed forms. The signatures mirror
``pmubench.signals`` so the benchmark can migrate onto trimes with thin
re-exports.

Common parameters
-----------------
``fs`` : float
    Sampling rate in Hz.
``nominal`` : float, default 50.0
    Nominal system frequency in Hz (the phase reference).
``amplitude`` : float, default 1.0
    Peak amplitude of the fundamental.
``duration`` : float
    Record length in seconds.
``harmonics`` : dict, optional
    ``{order: relative_amplitude}`` added to the fundamental, each scaled by
    ``amplitude``.
``noise`` : float, default 0.0
    White-noise standard deviation as a fraction of the peak amplitude.
``seed`` : int, default 0
    Noise RNG seed.
"""

from __future__ import annotations

import numpy as np

from trimes.synchrophasor.signal import AnalyticSignal, build_analytic_signal
from trimes.synchrophasor.util import TWO_PI

__all__ = [
    "steady_state",
    "amplitude_step",
    "phase_step",
    "frequency_ramp",
    "modulation",
]


def _const(value):
    return lambda t: np.full(np.shape(np.asarray(t, dtype=float)), float(value))


def steady_state(
    fs,
    nominal=50.0,
    frequency=None,
    amplitude=1.0,
    phase=0.0,
    duration=1.0,
    harmonics=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Constant amplitude and frequency, optionally off-nominal.

    Parameters
    ----------
    frequency : float, optional
        Tone frequency in Hz; defaults to ``nominal``.
    phase : float, default 0.0
        Constant phase offset in radians.

    Returns
    -------
    AnalyticSignal
        ``metadata = {"frequency": frequency, "transition_times": []}``.
    """
    f = float(nominal if frequency is None else frequency)
    return build_analytic_signal(
        _const(amplitude),
        _const(f),
        lambda t: TWO_PI * f * np.asarray(t, dtype=float) + float(phase),
        fs=fs,
        duration=duration,
        nominal=nominal,
        harmonics=harmonics,
        noise=noise,
        seed=seed,
        name=f"steady_{f:g}Hz",
        metadata={"frequency": f, "transition_times": []},
    )


def amplitude_step(
    fs,
    nominal=50.0,
    amplitude=1.0,
    step=0.1,
    step_time=0.5,
    duration=1.0,
    frequency=None,
    harmonics=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """A relative amplitude step at ``step_time``.

    Parameters
    ----------
    step : float, default 0.1
        Fractional amplitude change applied for ``t >= step_time``
        (``0.1`` = +10 %).
    step_time : float, default 0.5
        Step instant in seconds.

    Returns
    -------
    AnalyticSignal
        ``metadata`` carries ``step_time``, ``step`` and
        ``transition_times = [step_time]``.
    """
    f = float(nominal if frequency is None else frequency)
    st = float(step_time)
    return build_analytic_signal(
        lambda t: float(amplitude)
        * (1.0 + float(step) * (np.asarray(t, dtype=float) >= st)),
        _const(f),
        lambda t: TWO_PI * f * np.asarray(t, dtype=float),
        fs=fs,
        duration=duration,
        nominal=nominal,
        harmonics=harmonics,
        noise=noise,
        seed=seed,
        name=f"amplitude_step_{step:+.0%}",
        metadata={"step_time": st, "step": float(step), "transition_times": [st]},
    )


def phase_step(
    fs,
    nominal=50.0,
    amplitude=1.0,
    step_deg=10.0,
    step_time=0.5,
    duration=1.0,
    frequency=None,
    harmonics=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """A phase step at ``step_time``; true ROCOF is 0 away from the step.

    Parameters
    ----------
    step_deg : float, default 10.0
        Phase change in degrees applied for ``t >= step_time``.
    step_time : float, default 0.5
        Step instant in seconds.

    Returns
    -------
    AnalyticSignal
        ``metadata`` carries ``step_time``, ``step_deg`` and
        ``transition_times = [step_time]``.
    """
    f = float(nominal if frequency is None else frequency)
    st = float(step_time)
    delta = np.radians(float(step_deg))
    return build_analytic_signal(
        _const(amplitude),
        _const(f),
        lambda t: TWO_PI * f * np.asarray(t, dtype=float)
        + delta * (np.asarray(t, dtype=float) >= st),
        fs=fs,
        duration=duration,
        nominal=nominal,
        harmonics=harmonics,
        noise=noise,
        seed=seed,
        name=f"phase_step_{step_deg:+g}deg",
        metadata={
            "step_time": st,
            "step_deg": float(step_deg),
            "transition_times": [st],
        },
    )


def frequency_ramp(
    fs,
    nominal=50.0,
    amplitude=1.0,
    rocof=1.0,
    ramp_time=0.5,
    duration=1.5,
    harmonics=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """A linear frequency ramp from ``t = 0`` for ``ramp_time``, then constant.

    Parameters
    ----------
    rocof : float, default 1.0
        Ramp slope in Hz/s.
    ramp_time : float, default 0.5
        Ramp duration in seconds; frequency holds at
        ``nominal + rocof * ramp_time`` afterwards.

    Returns
    -------
    AnalyticSignal
        ``metadata`` carries ``rocof``, ``ramp_time`` and
        ``transition_times = [0.0, ramp_time]`` (the frequency has a corner at
        each).
    """
    f0, r, rt = float(nominal), float(rocof), float(ramp_time)

    def frequency_fn(t):
        t = np.asarray(t, dtype=float)
        return f0 + r * np.clip(t, 0.0, rt)

    def total_phase_fn(t):
        t = np.asarray(t, dtype=float)
        tau = np.clip(t, 0.0, rt)
        return TWO_PI * (f0 * t + r * (0.5 * tau**2 + rt * (t - tau)))

    return build_analytic_signal(
        _const(amplitude),
        frequency_fn,
        total_phase_fn,
        fs=fs,
        duration=duration,
        nominal=nominal,
        harmonics=harmonics,
        noise=noise,
        seed=seed,
        name=f"frequency_ramp_{rocof:+g}Hz_s",
        metadata={"rocof": r, "ramp_time": rt, "transition_times": [0.0, rt]},
    )


def modulation(
    fs,
    nominal=50.0,
    amplitude=1.0,
    am_depth=0.1,
    pm_depth=0.1,
    modulation_frequency=2.0,
    duration=2.0,
    harmonics=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Simultaneous amplitude and phase modulation (IEC/IEEE 60255-118-1 Eq 25).

    The reference signal is
    ``A[1 + kx cos(wm t)] * cos[2*pi*f0*t + ka cos(wm t - pi)]``, i.e. the
    phase-modulation term is ``-ka cos(wm t)`` and the true frequency is
    ``f0 + ka*fm*sin(wm t)``.

    Parameters
    ----------
    am_depth : float, default 0.1
        Amplitude-modulation depth ``kx`` (fraction of ``amplitude``).
    pm_depth : float, default 0.1
        Phase-modulation depth ``ka`` in radians.
    modulation_frequency : float, default 2.0
        Modulation frequency ``fm`` in Hz for both AM and PM.

    Returns
    -------
    AnalyticSignal
        ``metadata`` carries ``am_depth``, ``pm_depth``,
        ``modulation_frequency`` and ``transition_times = []``.
    """
    f0, fm = float(nominal), float(modulation_frequency)
    kx, ka = float(am_depth), float(pm_depth)
    wm = TWO_PI * fm

    return build_analytic_signal(
        lambda t: float(amplitude) * (1.0 + kx * np.cos(wm * np.asarray(t, dtype=float))),
        lambda t: f0 + ka * fm * np.sin(wm * np.asarray(t, dtype=float)),
        lambda t: TWO_PI * f0 * np.asarray(t, dtype=float)
        - ka * np.cos(wm * np.asarray(t, dtype=float)),
        fs=fs,
        duration=duration,
        nominal=nominal,
        harmonics=harmonics,
        noise=noise,
        seed=seed,
        name=f"modulation_{fm:g}Hz",
        metadata={
            "am_depth": ka,
            "pm_depth": kx,
            "modulation_frequency": fm,
            "transition_times": [],
        },
    )
