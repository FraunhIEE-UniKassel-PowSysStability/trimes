"""Phase 4: IEC/IEEE 60255-118-1:2018 test-condition generators.

Each function returns an :class:`~trimes.analytic.AnalyticSignal`
whose ``metadata`` carries ``test``, ``class`` (``"P"`` / ``"M"``), ``limits``
(the ``tve`` / ``fe`` / ``rfe`` values that apply -- reporting-rate-dependent
where the standard makes them so) and the condition's parameter values.

The ``*_sweep`` functions return a list of signals covering the standard's
parameter range for that class.
"""

from __future__ import annotations

import numpy as np

from trimes.synchrophasor import generators as _gen
from trimes.synchrophasor import limits as _lim
from trimes.synchrophasor.concat import concat_segments
from trimes.analytic import AnalyticSignal, build_analytic_signal, TWO_PI

__all__ = [
    "frequency_range",
    "frequency_range_sweep",
    "magnitude",
    "magnitude_sweep",
    "single_harmonic",
    "harmonic_sweep",
    "out_of_band",
    "modulation",
    "modulation_sweep",
    "frequency_ramp",
    "magnitude_step",
    "phase_step",
]

DEFAULT_FS = _lim.DEFAULT_REPORTING_RATE_HZ


def _stamp(
    signal: AnalyticSignal,
    test: str,
    class_: str,
    *,
    reporting_rate: float | None = None,
    limits: dict | None = None,
    **params,
) -> AnalyticSignal:
    signal.metadata.update(
        test=test,
        **{"class": class_.upper()},
        limits=_lim.error_limits(test, class_, reporting_rate)
        if limits is None
        else limits,
        **({"reporting_rate": float(reporting_rate)} if reporting_rate else {}),
        **params,
    )
    return signal


# --------------------------------------------------------------------------- #
# Steady state
# --------------------------------------------------------------------------- #
def frequency_range(
    fs,
    class_="P",
    *,
    fdev=None,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Off-nominal steady tone at ``nominal + fdev`` Hz (Eq 12).

    ``fdev`` defaults to the class's full deviation: ``+-2`` Hz (P), or for M
    ``+-2`` / ``+-Fs/5`` / ``+-5`` Hz depending on the reporting rate.
    """
    rng = _lim.frequency_range_hz(class_, reporting_rate)
    fdev = float(rng if fdev is None else fdev)
    sig = _gen.steady_state(
        fs,
        nominal=nominal,
        frequency=nominal + fdev,
        amplitude=amplitude,
        duration=duration,
        noise=noise,
        seed=seed,
    )
    return _stamp(
        sig, "frequency_range", class_, reporting_rate=reporting_rate, fdev=fdev,
        range_hz=rng,
    )


def frequency_range_sweep(
    fs, class_="P", *, step=0.5, reporting_rate=DEFAULT_FS, **kwargs
) -> list[AnalyticSignal]:
    """A :func:`frequency_range` signal at each ``fdev`` across the class range."""
    rng = _lim.frequency_range_hz(class_, reporting_rate)
    fdevs = np.round(np.arange(-rng, rng + step / 2.0, step), 6)
    return [
        frequency_range(fs, class_, fdev=float(fd), reporting_rate=reporting_rate, **kwargs)
        for fd in fdevs
    ]


def magnitude(
    fs,
    class_="P",
    *,
    level=1.0,
    kind="V",
    nominal=50.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Steady nominal-frequency tone at ``level`` per-unit amplitude (Eq 12)."""
    sig = _gen.steady_state(
        fs,
        nominal=nominal,
        amplitude=float(level),
        duration=duration,
        noise=noise,
        seed=seed,
    )
    return _stamp(sig, "magnitude", class_, level=float(level), kind=kind.upper())


def magnitude_sweep(
    fs, class_="P", *, kind="V", step=0.1, **kwargs
) -> list[AnalyticSignal]:
    """A :func:`magnitude` signal at each level across the class range.

    Voltage: P 80-120 %, M 10-120 %. Current: 10-200 % both.
    """
    key = "range_v" if kind.upper() == "V" else "range_i"
    lo, hi = _lim.limits_for("magnitude", class_)[key]
    levels = np.round(np.arange(lo, hi + step / 2.0, step), 6)
    return [magnitude(fs, class_, level=float(x), kind=kind, **kwargs) for x in levels]


def single_harmonic(
    fs,
    class_="P",
    *,
    order=2,
    level=None,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Steady tone with one added harmonic (Eq 15).

    ``level`` defaults to the class level: 1 % (P) / 10 % (M).
    """
    level = float(
        _lim.limits_for("harmonic", class_)["level"] if level is None else level
    )
    sig = _gen.steady_state(
        fs,
        nominal=nominal,
        amplitude=amplitude,
        duration=duration,
        harmonics={int(order): level},
        noise=noise,
        seed=seed,
    )
    return _stamp(
        sig, "harmonic", class_, reporting_rate=reporting_rate,
        order=int(order), level=level,
    )


def harmonic_sweep(fs, class_="P", **kwargs) -> list[AnalyticSignal]:
    """A :func:`single_harmonic` signal for every order 2..50."""
    lo, hi = _lim.limits_for("harmonic", class_)["orders"]
    return [single_harmonic(fs, class_, order=h, **kwargs) for h in range(lo, hi + 1)]


def out_of_band(
    fs,
    class_="M",
    *,
    interharmonic_hz=None,
    level=0.10,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Steady tone plus an interharmonic outside the passband (Eq 18).

    The passband is ``|f - f0| < Fs/2`` (``Fs`` = ``reporting_rate``); the
    interfering tone must satisfy ``|f - f0| >= Fs/2`` and lies between 10 Hz
    and ``2*f0``. ``interharmonic_hz`` defaults to ``f0 + Fs/2`` (the passband
    edge). P class is exempt: the signal then carries ``metadata["exempt"]``
    and empty ``limits``.
    """
    half_band = reporting_rate / 2.0
    if interharmonic_hz is None:
        interharmonic_hz = nominal + half_band
    in_band = abs(interharmonic_hz - nominal) < half_band

    sig = build_analytic_signal(
        lambda t: np.full(np.shape(np.asarray(t, float)), float(amplitude)),
        lambda t: np.full(np.shape(np.asarray(t, float)), float(nominal)),
        lambda t: TWO_PI * float(nominal) * np.asarray(t, float),
        fs=fs,
        duration=duration,
        nominal=nominal,
        extra_tones={float(interharmonic_hz): float(level)},
        noise=noise,
        seed=seed,
        name=f"out_of_band_{interharmonic_hz:g}Hz",
    )
    exempt = _lim.limits_for("out_of_band", class_) is None
    _stamp(
        sig,
        "out_of_band",
        class_,
        reporting_rate=reporting_rate,
        interharmonic_hz=float(interharmonic_hz),
        level=float(level),
        in_band=bool(in_band),
        limits={} if exempt else None,
    )
    if exempt:
        sig.metadata["exempt"] = True
    return sig


# --------------------------------------------------------------------------- #
# Dynamic
# --------------------------------------------------------------------------- #
def modulation(
    fs,
    class_="P",
    *,
    fm=None,
    reporting_rate=DEFAULT_FS,
    am_depth=None,
    pm_depth=None,
    nominal=50.0,
    amplitude=1.0,
    duration=None,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """AM+PM modulation test (Eq 25). ``fm`` defaults to the class maximum ``Fr``.

    The ``limits`` in ``metadata`` are computed for this reporting rate:
    ``fe = 0.03*Fr`` (P) / ``0.06*Fr`` (M), ``rfe = 0.18*pi*Fr**2``.
    """
    lim = _lim.limits_for("modulation", class_)
    fr = _lim.modulation_fr_hz(class_, reporting_rate)
    fm = float(fr if fm is None else fm)
    am_depth = float(lim["am_depth"] if am_depth is None else am_depth)
    pm_depth = float(lim["pm_depth"] if pm_depth is None else pm_depth)
    if duration is None:
        duration = max(5.0, 2.0 / fm)  # >= 5 s or two modulation cycles
    sig = _gen.modulation(
        fs,
        nominal=nominal,
        amplitude=amplitude,
        am_depth=am_depth,
        pm_depth=pm_depth,
        modulation_frequency=fm,
        duration=duration,
        noise=noise,
        seed=seed,
    )
    return _stamp(
        sig, "modulation", class_, reporting_rate=reporting_rate,
        fm=fm, fr_max=fr, am_depth=am_depth, pm_depth=pm_depth,
    )


def modulation_sweep(
    fs, class_="P", *, reporting_rate=DEFAULT_FS, step=0.2, **kwargs
) -> list[AnalyticSignal]:
    """A :func:`modulation` signal at each ``fm`` from 0.1 Hz to ``Fr``.

    The standard sweeps ``fm`` in steps of 0.2 Hz or smaller.
    """
    fr = _lim.modulation_fr_hz(class_, reporting_rate)
    fms = np.round(np.arange(0.1, fr + step / 2.0, step), 6)
    fms = fms[fms <= fr + 1e-9]
    return [
        modulation(fs, class_, fm=float(fm), reporting_rate=reporting_rate, **kwargs)
        for fm in fms
    ]


def frequency_ramp(
    fs,
    class_="P",
    *,
    direction=1,
    rocof=None,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    settle_s=0.2,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Linear frequency ramp across the full class range at ``+-1`` Hz/s (Eq 33).

    Built as steady ``f0 - range`` -> ramp -> steady ``f0 + range`` (or the
    reverse for ``direction=-1``). ``metadata["exclusion"]`` gives the
    ``(start, stop)`` windows around the two ramp corners that the standard
    excludes: half-width ``max(n/Fs, n/f0)`` with ``n`` = 2 (P) / 7 (M).

    ``rocof`` overrides the standard ``+-1`` Hz/s magnitude (e.g. to shorten a
    wide M-class ramp for a quick run); the ``metadata`` records what was used.
    """
    magnitude_hz_s = (
        float(_lim.limits_for("frequency_ramp", class_)["rocof_hz_s"])
        if rocof is None
        else abs(float(rocof))
    )
    rocof = magnitude_hz_s * (np.sign(rocof) if rocof else np.sign(direction or 1))
    rng = _lim.frequency_range_hz(class_, reporting_rate)
    ramp_time = 2.0 * rng / abs(rocof)
    lo_f, hi_f = nominal - rng, nominal + rng
    start_f, end_f = (lo_f, hi_f) if rocof > 0 else (hi_f, lo_f)

    sig = concat_segments(
        [
            (settle_s, {"frequency": start_f}),
            (ramp_time, {"frequency": (start_f, end_f)}),
            (settle_s, {"frequency": end_f}),
        ],
        fs=fs,
        nominal=nominal,
        amplitude=amplitude,
        name=f"frequency_ramp_{rocof:+g}Hz_s",
    )
    if noise:
        rng_gen = np.random.default_rng(seed)
        peak = float(np.max(np.abs(sig.amplitude(sig.time))))
        sig.signal.iloc[:] = sig.signal.to_numpy() + rng_gen.normal(
            0.0, noise * peak, sig.signal.size
        )

    half = _lim.ramp_exclusion_s(class_, reporting_rate, nominal)
    corners = [settle_s, settle_s + ramp_time]
    sig.metadata["exclusion"] = [(c - half, c + half) for c in corners]
    return _stamp(
        sig, "frequency_ramp", class_, reporting_rate=reporting_rate,
        rocof=rocof, ramp_time=ramp_time, range_hz=rng,
    )


def _step(
    kind: str,
    fs,
    class_,
    *,
    step,
    step_deg,
    step_time,
    reporting_rate,
    nominal,
    amplitude,
    duration,
    noise,
    seed,
) -> AnalyticSignal:
    if kind == "magnitude":
        step = float(_lim.limits_for("magnitude_step", class_)["step"] if step is None else step)
        sig = _gen.amplitude_step(
            fs, nominal=nominal, amplitude=amplitude, step=step,
            step_time=step_time, duration=duration, noise=noise, seed=seed,
        )
        params = {"step": step}
        test = "magnitude_step"
    else:
        step_deg = float(
            _lim.limits_for("phase_step", class_)["step_deg"] if step_deg is None else step_deg
        )
        sig = _gen.phase_step(
            fs, nominal=nominal, amplitude=amplitude, step_deg=step_deg,
            step_time=step_time, duration=duration, noise=noise, seed=seed,
        )
        params = {"step_deg": step_deg}
        test = "phase_step"

    overshoot = _lim.limits_for(test, class_)["overshoot"]
    return _stamp(
        sig, test, class_, reporting_rate=reporting_rate,
        step_time=float(step_time),
        response_time_s=_lim.step_response_time_s(class_, reporting_rate, nominal),
        delay_time_s=_lim.step_delay_time_s(reporting_rate),
        overshoot_limit=overshoot,
        **params,
    )


def magnitude_step(
    fs,
    class_="P",
    *,
    step=None,
    step_time=0.5,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Relative amplitude step (default ``+-10 %``) at ``step_time`` (Eq 41)."""
    return _step(
        "magnitude", fs, class_, step=step, step_deg=None, step_time=step_time,
        reporting_rate=reporting_rate, nominal=nominal, amplitude=amplitude,
        duration=duration, noise=noise, seed=seed,
    )


def phase_step(
    fs,
    class_="P",
    *,
    step_deg=None,
    step_time=0.5,
    reporting_rate=DEFAULT_FS,
    nominal=50.0,
    amplitude=1.0,
    duration=1.0,
    noise=0.0,
    seed=0,
) -> AnalyticSignal:
    """Phase step (default ``+-10 degrees`` = ``+-pi/18`` rad) at ``step_time``."""
    return _step(
        "phase", fs, class_, step=None, step_deg=step_deg, step_time=step_time,
        reporting_rate=reporting_rate, nominal=nominal, amplitude=amplitude,
        duration=duration, noise=noise, seed=seed,
    )
