"""Synthesise time-domain signals: periodic waveforms (harmonics, ramps,
multi-phase sets), phase-continuous concatenation of segments, piecewise-linear
signals, and additive noise.

For a periodic signal that also carries its exact analytic ground truth, see
:meth:`PeriodicSignal.analytic_signal` and
:func:`superimpose_and_concat_analytic_signal`, which wrap the waveform in a
:class:`trimes.analytic.AnalyticSignal`.
"""

from collections.abc import Callable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from trimes.analytic import AnalyticSignal, _segment_truth_fns
from trimes.base import superpose_series, get_index

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "PeriodicSignal",
    "superimpose_and_concat_periodic_signals",
    "superimpose_and_concat_analytic_signal",
    "get_attributes_of_superimposed_and_concatenated_signals_over_time",
    "linear_time_series",
    "make_monotonically_increasing",
    "get_interpolated",
    "mirror_y",
    "discretize",
    "create_ts_from_dict_with_varying_length",
    "add_normal_noise",
    "add_uniform_noise",
]


def _pyplot():
    """Import :mod:`matplotlib.pyplot` lazily (matplotlib is the ``viz`` extra)."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "plotting needs matplotlib; install it with: pip install 'trimes[viz]'"
        ) from exc
    return plt


class PeriodicSignal:
    """A periodic waveform ``mag * func(angle(t)) + offset`` sampled on ``t``.

    Frequency, magnitude and offset may each be a constant, an explicit array
    over ``t``, or a ``(start, end)`` tuple for a linear ramp. The phase angle
    is the running integral of ``2*pi*f``.

    Parameters
    ----------
    t : array_like
        Sample instants in seconds.
    func : callable, default numpy.cos
        Carrier waveform, applied to the phase angle.
    f : array_like or tuple, default 1.0
        Frequency in Hz: a scalar, an array over ``t``, or ``(f_start, f_end)``
        for a linear ramp.
    mag : array_like or tuple, default 1.0
        Amplitude, in the same three forms as ``f``.
    offset : float or array_like or tuple, default 0.0
        Additive offset, in the same three forms as ``f``.
    phi : float, default 0.0
        Initial phase angle in radians.
    """

    def __init__(
        self,
        t: ArrayLike,
        func: Callable = np.cos,
        f: ArrayLike | tuple = 1.0,
        mag: ArrayLike | tuple = 1.0,
        offset: float = 0.0,
        phi: float = 0.0,
    ) -> None:
        self.t: ArrayLike = t
        "time"
        self.func: Callable = func
        "function to create periodic signals (e.g. np.sin)"
        self._f_spec = f
        self._mag_spec = mag
        self._offset_spec = offset
        "original f/mag/offset arguments, kept for the analytic-truth model"
        if isinstance(f, tuple):
            f = np.linspace(f[0], f[1], len(self.t))
        self.f: ArrayLike = f
        "frequency"
        if isinstance(mag, tuple):
            mag = np.linspace(mag[0], mag[1], len(self.t))
        self.mag: ArrayLike = mag
        "magnitude"
        if isinstance(offset, tuple):
            offset = np.linspace(offset[0], offset[1], len(self.t))
        self.offset: ArrayLike = offset
        "y offset"
        self.phi: float = phi
        "initial angle"

    def analytic_signal(self, nominal: float = 50.0) -> AnalyticSignal:
        """Wrap this signal in a :class:`~trimes.analytic.AnalyticSignal`.

        Builds exact closed-form ``amplitude`` / ``frequency`` / ``total_phase``
        callables from the constructor arguments. Supported: a scalar frequency
        or a ``(f_start, f_end)`` linear ramp, a scalar magnitude or a
        ``(m_start, m_end)`` ramp, zero offset and ``func=np.cos``. Anything
        else (an explicit ``f``/``mag`` array) raises ``ValueError`` -- it has
        no closed form.

        Parameters
        ----------
        nominal : float, default 50.0
            Nominal frequency in Hz, the phase reference of the returned
            object.

        Returns
        -------
        AnalyticSignal
            Wrapping :meth:`get_signal_series`. ``metadata["transition_times"]``
            is empty (a single ``PeriodicSignal`` is continuous).

        Notes
        -----
        For a scalar frequency the wrapped samples equal the truth to floating
        point. For a frequency ramp the wrapped samples (built by the existing
        cumulative-sum integrator) approximate the exact closed form to
        ``O(dt**2)`` per step; the callables stay exact.
        """
        amplitude_fn, frequency_fn, total_phase_fn, _, _ = _segment_truth_fns(self)
        return AnalyticSignal(
            signal=self.get_signal_series(),
            amplitude_fn=amplitude_fn,
            frequency_fn=frequency_fn,
            total_phase_fn=total_phase_fn,
            nominal=float(nominal),
            metadata={"transition_times": []},
        )

    def get_signal(self) -> ArrayLike:
        return self.mag * self.func(self.get_angle_over_time()) + self.offset

    def get_signal_from_angle(self, angle: ArrayLike) -> ArrayLike:
        return self.mag * self.func(angle) + self.offset

    def get_signal_series(self) -> pd.Series:
        series = pd.Series(self.get_signal(), index=self.t)
        series.index.name = "time"
        return series

    def get_signal_series_from_angle(self, angle: ArrayLike) -> pd.Series:
        series = pd.Series(self.get_signal_from_angle(angle), index=self.t)
        series.index.name = "time"
        return series

    def get_signal_n_phases(self, n: int) -> pd.DataFrame:
        return superimpose_and_concat_periodic_signals([[self]], num_phases=n)

    def get_angle_over_time(self) -> ArrayLike:
        """Phase angle over ``t`` in radians.

        For a scalar frequency this is ``2*pi*f*(t - t[0]) + phi``; for a
        varying frequency it is the cumulative integral of ``2*pi*f``.
        """
        if isinstance(self.f, Number):
            return 2 * np.pi * self.f * (self.t - self.t[0]) + self.phi
        else:
            frequency_integrated = (
                np.cumsum(2 * np.pi * self.f[1:] * np.diff(self.t)) + self.phi
            )
            return np.array([self.phi, *frequency_integrated])

    def get_signal_attributes_over_time(
        self, attributes: list[str] | None = None
    ) -> pd.DataFrame:
        if attributes is None:
            attributes = ["f", "mag", "offset"]
        attrs_over_time = pd.DataFrame(columns=attributes, index=self.t, dtype=float)
        for attr in attributes:
            attrs_over_time[attr] = self.__getattribute__(attr)
        return attrs_over_time

    def plot(self, **kwargs):
        series = self.get_signal_series()
        series.plot(**kwargs)

    def plot_with_attributes(
        self, attributes: list[str] | None = None
    ) -> "list[Axes]":
        """Plot the waveform above one subplot per attribute over time.

        Parameters
        ----------
        attributes : list of str, optional
            Which of ``"f"``, ``"mag"``, ``"offset"`` to show. Defaults to all.

        Returns
        -------
        list of matplotlib.axes.Axes
            The waveform axis followed by one axis per attribute.
        """
        plt = _pyplot()
        attr = self.get_signal_attributes_over_time(attributes)
        fig, axes = plt.subplots(nrows=attr.shape[1] + 1, ncols=1)
        fig.tight_layout()
        plt.sca(axes[0])
        self.plot()
        plt.title("signal")
        plt.xlabel("")
        for col, ax in enumerate(axes[1:]):
            ax.plot(attr.index, attr.iloc[:, col])
            ax.grid()
            ax.set_title(attr.columns[col])
        return axes


def superimpose_and_concat_periodic_signals(
    signals: list[list[PeriodicSignal]],
    num_phases: int = 1,
    angle_continuation: bool = True,
    angular_jump: None | ArrayLike = None,
) -> pd.Series | pd.DataFrame:
    """Superimpose and concatenate a grid of :class:`PeriodicSignal` objects.

    Each inner list ("row") is summed, and the resulting segments are
    concatenated in time. Optionally expands to several phases and keeps the
    fundamental phase continuous across segment boundaries.

    Parameters
    ----------
    signals : list of list of PeriodicSignal
        Rows are superimposed; the resulting segments are concatenated.
    num_phases : int, default 1
        Number of phases, each shifted by ``2*pi/num_phases``.
    angle_continuation : bool, default True
        If True, adjust each segment's initial phase ``phi`` so the fundamental
        angle is continuous with the previous segment.
    angular_jump : array_like, optional
        Deliberate fundamental phase jump (radians) at each interior boundary;
        length ``len(signals) - 1``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A Series for ``num_phases == 1``, otherwise a DataFrame with one column
        per phase.

    Examples
    --------
    >>> step_size = 1e-3
    >>> signals = [
    ...     [
    ...         PeriodicSignal(np.arange(0, 0.1, step_size), f=(50, 52)),
    ...         PeriodicSignal(np.arange(0, 0.1, step_size), f=5 * 50, mag=(0.03, 0.06)),
    ...     ],
    ...     [
    ...         PeriodicSignal(np.arange(0.1, 0.3, step_size), f=50),
    ...         PeriodicSignal(np.arange(0.1, 0.3, step_size), f=7 * 50, mag=0.06),
    ...     ],
    ... ]
    >>> res = superimpose_and_concat_periodic_signals(signals, num_phases=3)
    """
    try:
        signals = np.array(signals)
    except ValueError:
        raise ValueError(
            "Items in 'signals' must be of same length (same number of Periodic signals in each row)."
        )
    if angular_jump is not None:
        # build the continuous angle first, then add the deliberate jump
        angle_continuation = True
        angular_jump = np.array(angular_jump)
        angular_jump = np.insert(angular_jump, 0, 0, axis=0)  # first jump is zero
    elif angle_continuation:
        angular_jump = [0] * signals.shape[0]
    phase_signals = [None] * num_phases
    for phase in range(num_phases):
        phase_angle = phase * (-2 * np.pi / num_phases)
        series_to_concat = [None] * signals.shape[0]
        if angle_continuation:
            phi_initial_next_row = [
                signal.phi for signal in signals[0]
            ]  # is updated for each row
        for n, signals_to_superpose in enumerate(signals):
            signals_superposed = [None] * signals.shape[1]
            for m, signal in enumerate(signals_to_superpose):
                phi_initial = signal.phi  # store to later reset original value
                if angle_continuation:
                    signal.phi = (
                        phi_initial_next_row[m] + angular_jump[n]
                    )  # set from last row
                signal.phi += phase_angle
                angle = signal.get_angle_over_time()
                signals_superposed[m] = signal.get_signal_series_from_angle(angle)
                if angle_continuation:
                    # Initial angle for this row in the *next* segment. The next
                    # segment starts one sample step after this segment's last
                    # sample, so extrapolate the angle by one step
                    # (angle[-1] - angle[-2]); then undo the per-phase shift so
                    # it can be re-applied for each phase. Exact when segments
                    # are contiguous with a uniform step (the usual case).
                    phi_initial_next_row[m] = (
                        angle[-1] + (angle[-1] - angle[-2]) - phase_angle
                    )
                signal.phi = phi_initial
            series_to_concat[n] = superpose_series(signals_superposed)

        phase_signals[phase] = pd.concat(series_to_concat, axis=0)
    if num_phases == 1:
        return phase_signals[0]
    else:
        return pd.concat(phase_signals, axis=1)


def superimpose_and_concat_analytic_signal(
    signals: list[list[PeriodicSignal]],
    *,
    num_phases: int = 1,
    angle_continuation: bool = True,
    angular_jump: None | ArrayLike = None,
    nominal: float = 50.0,
) -> AnalyticSignal:
    """Concatenate an array of periodic signals and keep the analytic truth.

    The waveform is exactly :func:`superimpose_and_concat_periodic_signals`
    (same arguments, same result). The analytic truth is built from the
    **fundamental of each row** -- ``signals[n][0]`` -- with fundamental phase
    carried across every boundary (plus any ``angular_jump``), so ``phasor`` /
    ``frequency`` / ``rocof`` truth is available at arbitrary instants.

    Every fundamental must be analytically tractable (see
    :meth:`PeriodicSignal.analytic_signal`). The remaining signals in a row
    (harmonics, interharmonics, noise-like tones) shape the waveform but not
    the truth.

    Parameters
    ----------
    signals : list[list[PeriodicSignal]]
        Rows are superimposed; the resulting segments are concatenated.
    num_phases : int, default 1
        Number of phases; the truth is always the positive-sequence (phase a)
        fundamental.
    angle_continuation : bool, default True
        Forwarded to :func:`superimpose_and_concat_periodic_signals`.
    angular_jump : array_like, optional
        Deliberate fundamental phase jump (radians) at each interior boundary;
        length ``len(signals) - 1``. Forwarded to the waveform builder and
        applied to the truth.
    nominal : float, default 50.0
        Phase reference in Hz.

    Returns
    -------
    AnalyticSignal
        Wrapping the concatenated Series (``num_phases == 1``) or DataFrame.
        ``metadata["transition_times"]`` lists the boundaries that carry a jump
        or a change of fundamental frequency.
    """
    rows = [list(row) for row in signals]
    if not rows or any(not row for row in rows):
        raise ValueError("'signals' must be a non-empty list of non-empty rows")

    waveform = superimpose_and_concat_periodic_signals(
        rows,
        num_phases=num_phases,
        angle_continuation=angle_continuation,
        angular_jump=angular_jump,
    )

    fundamentals = [row[0] for row in rows]
    n_seg = len(fundamentals)
    if angular_jump is None:
        jumps = np.zeros(max(n_seg - 1, 0))
    else:
        jumps = np.asarray(angular_jump, dtype=float).reshape(-1)
        if jumps.shape != (n_seg - 1,):
            raise ValueError(
                f"angular_jump must have length {n_seg - 1} (segments - 1)"
            )

    # (t_start, t_end, amplitude_fn, frequency_fn, total_phase_fn) per segment
    segments: list[tuple] = []
    next_starts: list[float] = []
    transition_times: list[float] = []
    phi_carry = float(fundamentals[0].phi)
    for i, fund in enumerate(fundamentals):
        amp_fn, freq_fn, phase_fn, t_start, t_end = _segment_truth_fns(
            fund, phi_override=phi_carry
        )
        segments.append((t_start, t_end, amp_fn, freq_fn, phase_fn))
        if i + 1 < n_seg:
            nxt = fundamentals[i + 1]
            next_start = float(np.asarray(nxt.t, dtype=float)[0])
            next_starts.append(next_start)
            phi_end = float(np.atleast_1d(phase_fn(np.array([next_start])))[0])
            jump = float(jumps[i])
            phi_carry = phi_end + jump
            f_left = float(np.atleast_1d(freq_fn(np.array([next_start])))[0])
            _, next_freq_fn, _, _, _ = _segment_truth_fns(nxt)
            f_right = float(np.atleast_1d(next_freq_fn(np.array([next_start])))[0])
            if jump != 0.0 or not np.isclose(f_left, f_right):
                transition_times.append(next_start)

    def _piecewise(component: int) -> Callable:
        def evaluate(times):
            times = np.asarray(times, dtype=float)
            flat = np.atleast_1d(times)
            out = np.empty(flat.shape, dtype=float)
            for idx, segment in enumerate(segments):
                lo = -np.inf if idx == 0 else next_starts[idx - 1]
                hi = np.inf if idx == n_seg - 1 else next_starts[idx]
                mask = (flat >= lo) & (flat < hi)
                if np.any(mask):
                    out[mask] = np.asarray(
                        segment[component](flat[mask]), dtype=float
                    )
            return out.reshape(times.shape) if times.ndim else out[0]

        return evaluate

    return AnalyticSignal(
        signal=waveform,
        amplitude_fn=_piecewise(2),
        frequency_fn=_piecewise(3),
        total_phase_fn=_piecewise(4),
        nominal=float(nominal),
        metadata={"transition_times": sorted(set(transition_times))},
    )


def get_attributes_of_superimposed_and_concatenated_signals_over_time(
    signals: list[list[PeriodicSignal]],
    columns: list[int] | None = None,
    attributes: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Per-column attribute traces for a grid of :class:`PeriodicSignal` objects.

    Mirrors the layout of :func:`superimpose_and_concat_periodic_signals`: for
    each requested column, the attribute frames of every row are concatenated
    in time.

    Parameters
    ----------
    signals : list of list of PeriodicSignal
        Rows are superimposed; columns are concatenated.
    columns : list of int, optional
        Which columns to return. Defaults to all.
    attributes : list of str, optional
        Which of ``"f"``, ``"mag"``, ``"offset"`` to include. Defaults to all.

    Returns
    -------
    list of pandas.DataFrame
        One frame per requested column.
    """
    if columns is None:
        columns = range(0, len(signals[0]))
    attrs_of_columns = [None] * len(columns)
    for n, col in enumerate(columns):
        frames = [pd.DataFrame()] * len(signals)
        for row in range(len(signals)):
            frames[row] = signals[row][col].get_signal_attributes_over_time(attributes)
        attrs_of_columns[n] = pd.concat(frames)
    return attrs_of_columns


def linear_time_series(
    t: ArrayLike, y: ArrayLike, sample_time: float
) -> pd.DataFrame | pd.Series:
    """Piecewise-linear signal through the points ``(t, y)``, sampled at ``sample_time``.

    Parameters
    ----------
    t : array_like
        Break-point times. Non-increasing steps are nudged forward by
        ``sample_time`` (see :func:`make_monotonically_increasing`) so a
        vertical step becomes a steep ramp.
    y : array_like
        Values at the break points.
    sample_time : float
        Output sample step.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The interpolated signal.
    """
    t = make_monotonically_increasing(t, sample_time)
    ts = get_interpolated(t, y, sample_time)
    return ts


def make_monotonically_increasing(t: ArrayLike, sample_time: float) -> np.ndarray:
    """Force ``t`` to strictly increase.

    Where a step is non-positive (``t[k] <= t[k-1]``), that entry is moved
    forward by ``sample_time``.

    Parameters
    ----------
    t : array_like
        Times.
    sample_time : float
        Amount to nudge non-increasing entries by.

    Returns
    -------
    numpy.ndarray
        The adjusted times.
    """
    t = np.array(t, dtype=float)
    diff_t_is_zero = np.concatenate([[False], np.diff(t) <= 0])
    t[diff_t_is_zero] = t[diff_t_is_zero] + sample_time
    return t


#: Deprecated misspelled alias of :func:`make_monotonically_increasing`.
make_monotonously_increasing = make_monotonically_increasing


def get_interpolated(
    t: ArrayLike, y: ArrayLike | list[ArrayLike], sample_time: float
) -> pd.DataFrame | pd.Series:
    """Linearly interpolate values ``y`` given at times ``t`` onto a uniform grid.

    Parameters
    ----------
    t : array_like
        Times at which ``y`` is given.
    y : array_like or list of array_like
        Values at ``t``; pass a list to interpolate several signals at once
        (the result is then a DataFrame).
    sample_time : float
        Output sample step; the grid runs from ``t[0]`` to ``t[-1]``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The interpolated signal(s).
    """
    index = np.arange(t[0], t[-1] + sample_time, sample_time)
    if not isinstance(y, list):
        return_series = True
        y = [y]
    else:
        return_series = False
    ts = pd.DataFrame(np.nan, index=index, columns=range(len(y)))
    ts.iloc[get_index(ts, t), :] = np.transpose(y)
    ts.interpolate(inplace=True)
    ts.index.name = "time"
    if return_series:
        ts = ts.squeeze()
    return ts


def mirror_y(
    ts: pd.Series, y: float, inplace: bool = False
) -> pd.DataFrame | pd.Series:
    """Reflect ``ts`` about the horizontal line ``y`` (``2*y - ts``).

    Parameters
    ----------
    ts : pandas.Series
        Time series.
    y : float
        Value of the mirror line.
    inplace : bool, default False
        If True, return a two-column DataFrame with the original and the
        mirrored signal; otherwise return only the mirrored Series.

    Returns
    -------
    pandas.Series or pandas.DataFrame
    """
    mirrored_y = 2 * y - ts
    if inplace:
        ts = ts.to_frame()
        ts[1] = mirrored_y
        return ts
    else:
        return mirrored_y


def discretize(
    ts: pd.DataFrame | pd.Series, delta_t: float = 1e-6
) -> pd.DataFrame | pd.Series:
    """Turn a sampled signal into a staircase by adding a sample just before each step.

    For example, ``ts`` with index ``[0, 2, 3]`` and values ``[20, 50, 60]``
    becomes index ``[0, 1.999, 2, 2.999, 3]`` with values
    ``[20, 20, 50, 50, 60]`` (for ``delta_t = 1e-3``).

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    delta_t : float, default 1e-6
        Gap placed before each step.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The staircase signal.
    """
    is_frame = isinstance(ts, pd.DataFrame)
    ts2 = ts.copy()
    ts2.index = ts.index - delta_t
    ts = pd.concat([ts, ts2])
    ts.sort_index(inplace=True)
    if is_frame:
        return pd.DataFrame(
            ts.iloc[0:-1, :].to_numpy(), index=ts.index[1:], columns=ts.columns
        )
    else:
        return pd.Series(ts.iloc[0:-1].to_numpy(), index=ts.index[1:], name=ts.name)


def create_ts_from_dict_with_varying_length(
    d: dict,
    time_label: str = "time",
) -> pd.DataFrame:
    """Build a DataFrame from a dict whose value arrays may be shorter than time.

    Any value array shorter than ``d[time_label]`` is forward-filled with its
    last element.

    Parameters
    ----------
    d : dict
        Maps column names to value arrays, plus one key holding the time
        vector.
    time_label : str, default "time"
        The key in ``d`` that holds the time vector.

    Returns
    -------
    pandas.DataFrame
        Indexed by the time vector.
    """
    df = pd.DataFrame()
    time = d[time_label]
    df.index = time
    df.index.name = time_label
    for variable, values in d.items():
        if variable != time_label:
            if len(values) < len(time):
                diff_len = len(time) - len(values)
                values = np.concatenate([values, np.repeat(values[-1], diff_len)])
            df[variable] = values
    return df


def add_normal_noise(
    df: pd.DataFrame | pd.Series,
    scale: float,
    mean: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame | pd.Series:
    """Add Gaussian noise to a time series.

    Parameters
    ----------
    df : pandas.Series or pandas.DataFrame
        Time series.
    scale : float
        Standard deviation of the noise (non-negative).
    mean : float, default 0.0
        Mean of the noise.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`; ``None`` draws fresh entropy.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        ``df`` plus noise.
    """
    rng = np.random.default_rng(seed=seed)
    return df + rng.normal(mean, scale, df.shape)


def add_uniform_noise(
    df: pd.DataFrame | pd.Series,
    low: float = 0.0,
    high: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame | pd.Series:
    """Add uniform noise on ``[low, high)`` to a time series.

    Parameters
    ----------
    df : pandas.Series or pandas.DataFrame
        Time series.
    low, high : float, default 0.0 and 1.0
        Bounds of the noise interval (``high - low`` non-negative).
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`; ``None`` draws fresh entropy.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        ``df`` plus noise.
    """
    rng = np.random.default_rng(seed=seed)
    return df + rng.uniform(low, high, df.shape)
