"""Sampled signals that carry their own analytical ground truth.

An :class:`AnalyticSignal` wraps the pandas object a generator produces (it
does **not** replace it) and adds three callables of a time array:

``amplitude_fn(t)``
    true peak amplitude of the fundamental,
``frequency_fn(t)``
    true instantaneous frequency in Hz,
``total_phase_fn(t)``
    true *total* phase of the carrier in radians (the argument of the cosine,
    not yet referenced to the nominal frequency).

From those three, the phasor / frequency / ROCOF quantities an estimator
reports are derived and can be evaluated at *arbitrary* instants -- not only
the sample grid -- because a block estimator timestamps its result at a window
centre and a recursive one at the current sample.

This is a core module: numpy + pandas only, importable without scipy,
matplotlib or any optional extra. :mod:`trimes.synchrophasor` builds its
IEC/IEEE 60255-118-1 benchmark layer on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "AnalyticSignal",
    "build_analytic_signal",
    "wrap_pi",
    "rms_phasor",
    "TWO_PI",
    "ROCOF_STEP",
]

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


@dataclass
class AnalyticSignal:
    """A sampled waveform plus the analytic truth behind it.

    Parameters
    ----------
    signal : pandas.Series or pandas.DataFrame
        The sampled waveform, exactly as produced by the generator it came
        from. Its index is time in seconds. This object is *wrapped*, never
        rebuilt.
    amplitude_fn, frequency_fn, total_phase_fn : callable
        Functions of a time array (seconds) returning, respectively, the true
        peak amplitude, the true frequency in Hz and the true total phase in
        radians. Each must accept a scalar or an ndarray and return a matching
        shape (a scalar constant is broadcast for you).
    nominal : float, default 50.0
        Nominal system frequency in Hz -- the phase reference for
        :meth:`phase`, :meth:`phasor` and :meth:`ground_truth`.
    metadata : dict, optional
        Scenario extras. ``metadata["transition_times"]`` (always present, a
        possibly empty list) holds the instants where the analytic model is
        not smooth -- injected phase / magnitude steps, ramp corners and the
        boundaries between concatenated segments of different frequency.

    Attributes
    ----------
    time, t : numpy.ndarray
        Sample instants in seconds (``t`` is an alias).
    channel : numpy.ndarray
        The wrapped waveform as a 1-D float ndarray, ready to feed an estimator.
    fs : float
        Sampling rate in Hz, inferred from the time index.
    name
        The wrapped Series' ``name``.

    Notes
    -----
    The truth callables are exact closed forms. When the wrapped ``signal``
    was sampled by a generator that integrates frequency numerically (a
    :class:`~trimes.signal_generation.PeriodicSignal` frequency ramp, for
    instance), the samples approximate that closed form to
    :math:`O(\\Delta t^2)` per step; the callables remain exact.
    :func:`build_analytic_signal` samples straight from ``total_phase_fn`` and
    has no such gap.
    """

    signal: "pd.Series | pd.DataFrame"
    amplitude_fn: Callable[[np.ndarray], np.ndarray]
    frequency_fn: Callable[[np.ndarray], np.ndarray]
    total_phase_fn: Callable[[np.ndarray], np.ndarray]
    nominal: float = 50.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nominal = float(self.nominal)
        self.metadata.setdefault("transition_times", [])

    # -- grid ---------------------------------------------------------------
    @property
    def time(self) -> np.ndarray:
        """The wrapped signal's sample instants as a float ndarray."""
        return self.signal.index.to_numpy(dtype=float)

    #: Short alias of :attr:`time`, for callers that think in ``t``.
    @property
    def t(self) -> np.ndarray:
        """Alias of :attr:`time` -- the sample instants as a float ndarray."""
        return self.time

    @property
    def channel(self) -> np.ndarray:
        """The wrapped waveform as a 1-D float ndarray, ready to feed an estimator.

        For a single-column :class:`~pandas.DataFrame` the sole column is
        returned; a multi-column frame raises (pick a column yourself).
        """
        data = self.signal
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(
                    "channel is only defined for a Series or a single-column "
                    f"DataFrame; this signal has {data.shape[1]} columns"
                )
            data = data.iloc[:, 0]
        return np.asarray(data.to_numpy(), dtype=float)

    @property
    def fs(self) -> float:
        """Sampling rate in Hz, inferred from the (assumed uniform) time index.

        ``1 / median(diff(time))``. Raises for a signal with fewer than two
        samples.
        """
        t = self.time
        if t.size < 2:
            raise ValueError("fs needs at least two samples to be defined")
        return float(1.0 / np.median(np.diff(t)))

    @property
    def name(self):
        """The wrapped Series' ``name`` (``None`` for a DataFrame)."""
        return getattr(self.signal, "name", None)

    @property
    def duration(self) -> float:
        """``time[-1] - time[0]`` in seconds (0.0 for an empty signal)."""
        t = self.time
        return float(t[-1] - t[0]) if t.size else 0.0

    @property
    def transition_times(self) -> list:
        """Instants where the analytic model is not smooth (may be empty)."""
        return list(self.metadata.get("transition_times", []))

    # -- primitive truth --------------------------------------------------
    @staticmethod
    def _t(times) -> np.ndarray:
        return np.asarray(times, dtype=float)

    def amplitude(self, t) -> np.ndarray:
        """True peak amplitude at ``t``."""
        t = self._t(t)
        return np.broadcast_to(
            np.asarray(self.amplitude_fn(t), dtype=float), t.shape
        ).copy()

    def frequency(self, t) -> np.ndarray:
        """True instantaneous frequency in Hz at ``t``."""
        t = self._t(t)
        return np.broadcast_to(
            np.asarray(self.frequency_fn(t), dtype=float), t.shape
        ).copy()

    def total_phase(self, t) -> np.ndarray:
        """True total carrier phase in radians at ``t`` (not wrapped)."""
        t = self._t(t)
        return np.broadcast_to(
            np.asarray(self.total_phase_fn(t), dtype=float), t.shape
        ).copy()

    # -- derived synchrophasor truth ------------------------------------
    def phase(self, t, nominal: float | None = None) -> np.ndarray:
        """Nominal-referenced phase, wrapped to ``[-pi, pi)``.

        ``wrap_pi(total_phase(t) - 2*pi*nominal*t)``.
        """
        nominal = self.nominal if nominal is None else float(nominal)
        t = self._t(t)
        return wrap_pi(self.total_phase(t) - TWO_PI * nominal * t)

    def phasor(self, t, nominal: float | None = None) -> np.ndarray:
        """RMS complex phasor: ``(amplitude/sqrt2) * exp(1j*phase)``."""
        return rms_phasor(self.amplitude(t), self.phase(t, nominal))

    def rocof(self, t) -> np.ndarray:
        """Rate of change of frequency in Hz/s.

        A central difference of ``frequency_fn`` with half-step
        :data:`~trimes.analytic.ROCOF_STEP`. For a piecewise-linear
        frequency this returns the mean of the one-sided slopes at a corner,
        which is the honest answer there; at a genuine frequency *jump* it is
        not meaningful (a phase step has no finite ROCOF).
        """
        t = self._t(t)
        forward = np.asarray(self.frequency_fn(t + ROCOF_STEP), dtype=float)
        backward = np.asarray(self.frequency_fn(t - ROCOF_STEP), dtype=float)
        return np.broadcast_to(
            (forward - backward) / (2.0 * ROCOF_STEP), t.shape
        ).copy()

    def ground_truth(self, t=None, nominal: float | None = None) -> pd.DataFrame:
        """Evaluate the full truth at arbitrary instants.

        Parameters
        ----------
        t : array_like, optional
            Instants in seconds. Defaults to the wrapped signal's own sample
            grid.
        nominal : float, optional
            Phase reference in Hz; defaults to :attr:`nominal`.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``t`` (index name ``"time"``) with columns
            ``frequency`` (Hz), ``amplitude``, ``phase`` (rad, wrapped),
            ``rocof`` (Hz/s) and ``phasor`` (RMS complex) -- the column set of
            an estimator trace.
        """
        t = self.time if t is None else self._t(t)
        t = np.atleast_1d(t)
        amplitude = self.amplitude(t)
        phase = self.phase(t, nominal)
        return pd.DataFrame(
            {
                "frequency": self.frequency(t),
                "amplitude": amplitude,
                "phase": phase,
                "rocof": self.rocof(t),
                "phasor": rms_phasor(amplitude, phase),
            },
            index=pd.Index(t, name="time"),
        )


def build_analytic_signal(
    amplitude_fn: Callable[[np.ndarray], np.ndarray],
    frequency_fn: Callable[[np.ndarray], np.ndarray],
    total_phase_fn: Callable[[np.ndarray], np.ndarray],
    *,
    fs: float,
    duration: float,
    t0: float = 0.0,
    nominal: float = 50.0,
    harmonics: dict | None = None,
    extra_tones: dict | None = None,
    noise: float = 0.0,
    seed: int = 0,
    name: str | None = None,
    metadata: dict | None = None,
) -> AnalyticSignal:
    """Assemble an :class:`AnalyticSignal` from its defining callables.

    The waveform is sampled straight from the callables --
    ``amplitude_fn(t) * cos(total_phase_fn(t))`` plus harmonics and noise -- so
    the samples are exactly consistent with the truth (no integration drift).

    Parameters
    ----------
    amplitude_fn, frequency_fn, total_phase_fn : callable
        Closed-form truth, functions of a time array.
    fs : float
        Sampling rate in Hz.
    duration : float
        Record length in seconds; samples land on
        ``arange(t0, t0 + duration, 1/fs)``.
    t0 : float, default 0.0
        First sample instant.
    nominal : float, default 50.0
        Phase reference in Hz.
    harmonics : dict, optional
        ``{order: relative_amplitude}`` harmonics of the total phase, each
        scaled by the (time-varying) fundamental amplitude.
    extra_tones : dict, optional
        ``{frequency_hz: relative_amplitude}`` tones at arbitrary (non-integer,
        possibly out-of-band) frequencies, scaled by the fundamental amplitude
        and phased as a cosine at ``t0``. Used for interharmonic and
        out-of-band interference tests.
    noise : float, default 0.0
        White-noise standard deviation as a fraction of the peak amplitude.
    seed : int, default 0
        Noise RNG seed.
    name : str, optional
        Name of the wrapped Series.
    metadata : dict, optional
        Attached verbatim (``transition_times`` is added if missing).

    Returns
    -------
    AnalyticSignal
    """
    t = np.arange(float(t0), float(t0) + float(duration), 1.0 / float(fs))
    psi = np.asarray(total_phase_fn(t), dtype=float)
    amplitude = np.broadcast_to(
        np.asarray(amplitude_fn(t), dtype=float), t.shape
    ).astype(float)

    x = amplitude * np.cos(psi)
    for order, relative in (harmonics or {}).items():
        x = x + relative * amplitude * np.cos(order * psi)
    for frequency_hz, relative in (extra_tones or {}).items():
        x = x + relative * amplitude * np.cos(2.0 * np.pi * float(frequency_hz) * (t - t[0]))
    if noise:
        rng = np.random.default_rng(seed)
        peak = float(np.max(np.abs(amplitude))) if amplitude.size else 1.0
        x = x + rng.normal(0.0, float(noise) * peak, t.size)

    series = pd.Series(x, index=pd.Index(t, name="time"), name=name or "signal")
    return AnalyticSignal(
        signal=series,
        amplitude_fn=amplitude_fn,
        frequency_fn=frequency_fn,
        total_phase_fn=total_phase_fn,
        nominal=float(nominal),
        metadata=dict(metadata or {}),
    )


def _segment_truth_fns(periodic_signal, *, phi_override: float | None = None):
    """Closed-form truth callables for one :class:`PeriodicSignal`-like object.

    Reads ``.t``, ``.func``, ``.phi`` and the *original* ``f`` / ``mag`` /
    ``offset`` specs (``._f_spec`` etc., falling back to the resolved
    attributes). Handles the analytically tractable cases:

    * constant frequency, or a ``(f_start, f_end)`` linear ramp,
    * constant amplitude, or a ``(m_start, m_end)`` linear ramp,
    * zero offset and the default ``func=np.cos`` carrier.

    Raises ``ValueError`` for anything else (an explicit frequency / amplitude
    array has no closed form).

    Returns
    -------
    (amplitude_fn, frequency_fn, total_phase_fn, t_start, t_end)
    """
    t = np.asarray(periodic_signal.t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("PeriodicSignal.t must be a 1-D array with at least two samples")
    t_start, t_end = float(t[0]), float(t[-1])
    span = t_end - t_start
    if span <= 0.0:
        raise ValueError("PeriodicSignal.t must be strictly increasing")

    func = getattr(periodic_signal, "func", np.cos)
    if func is not np.cos:
        name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"analytic truth is only defined for the default carrier func=np.cos, got {name!r}"
        )

    offset_spec = getattr(periodic_signal, "_offset_spec", periodic_signal.offset)
    if np.any(np.asarray(offset_spec, dtype=float) != 0.0):
        raise ValueError(
            "analytic truth requires offset == 0 (a DC term breaks the phasor model)"
        )

    phi = float(periodic_signal.phi if phi_override is None else phi_override)

    f_spec = getattr(periodic_signal, "_f_spec", periodic_signal.f)
    if isinstance(f_spec, tuple):
        f_start, f_end = float(f_spec[0]), float(f_spec[1])
        f_slope = (f_end - f_start) / span
    elif np.ndim(f_spec) == 0:
        f_start = float(f_spec)
        f_slope = 0.0
    else:
        raise ValueError(
            "analytic truth needs a scalar frequency or a (f_start, f_end) tuple; "
            "an explicit frequency array has no closed form"
        )

    mag_spec = getattr(periodic_signal, "_mag_spec", periodic_signal.mag)
    if isinstance(mag_spec, tuple):
        m_start, m_end = float(mag_spec[0]), float(mag_spec[1])
        m_slope = (m_end - m_start) / span
    elif np.ndim(mag_spec) == 0:
        m_start = float(mag_spec)
        m_slope = 0.0
    else:
        raise ValueError(
            "analytic truth needs a scalar magnitude or a (m_start, m_end) tuple"
        )

    def amplitude_fn(times):
        times = np.asarray(times, dtype=float)
        return m_start + m_slope * (times - t_start)

    def frequency_fn(times):
        times = np.asarray(times, dtype=float)
        return f_start + f_slope * (times - t_start)

    def total_phase_fn(times):
        times = np.asarray(times, dtype=float)
        d = times - t_start
        return phi + TWO_PI * (f_start * d + 0.5 * f_slope * d * d)

    return amplitude_fn, frequency_fn, total_phase_fn, t_start, t_end
