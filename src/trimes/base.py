"""Access, interpolation and resampling helpers for numerically indexed
time series -- ``pandas`` objects whose index is time as plain numbers
(seconds), not a ``DatetimeIndex``.
"""

from collections.abc import Callable, Iterable
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

FrameOrSeries = pd.Series | pd.DataFrame

__all__ = [
    "FrameOrSeries",
    "get_sample",
    "get_sample_shifted",
    "get_samples_around",
    "get_sample_const",
    "interp_df",
    "interp_np_matrix",
    "interp_series",
    "get_between",
    "get_between_and_around",
    "get_between_interp",
    "get_index",
    "resample",
    "get_delta",
    "get_delta_shift",
    "get_delta_around_event",
    "get_delta_interp_df",
    "get_delta_interp_series",
    "get_duration",
    "create_pandas_series_or_frame_with_same_columns_and_index",
    "superpose_series",
    "apply_to_columns",
    "time_to_seconds",
    "is_uniformly_sampled",
    "get_sampling_rate",
    "resample_uniform",
]


def get_sample(ts: FrameOrSeries, time: float | ArrayLike) -> FrameOrSeries:
    """First sample at or after ``time``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series with a numeric, increasing index.
    time : float or array_like
        Instant(s) in time.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Row(s) at the first index position ``>= time`` (binary search).
    """
    return ts.iloc[np.searchsorted(ts.index.values, time)]  # binary search


def get_sample_shifted(
    ts: FrameOrSeries, time: float | ArrayLike, shift: int
) -> FrameOrSeries:
    """Sample at or after ``time``, then moved by ``shift`` samples.

    Example: ``shift = -1`` locates the first sample at or after ``time`` and
    returns the one before it, i.e. the last sample strictly before ``time``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time : float or array_like
        Instant(s) in time.
    shift : int
        Offset in samples applied to the located position.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The shifted sample(s).
    """
    return ts.iloc[np.searchsorted(ts.index.values, time) + shift]


def get_samples_around(
    ts: FrameOrSeries,
    time: float,
    num_samples_before: int = -1,
    num_samples_after: int = 0,
) -> FrameOrSeries:
    """Samples in a window around ``time``.

    ``num_samples_before`` and ``num_samples_after`` are offsets in samples
    relative to the first sample at or after ``time``. The defaults return the
    sample just before and the sample just after ``time``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time : float
        Instant in time.
    num_samples_before : int, default -1
        Start of the window, as a sample offset (usually negative).
    num_samples_after : int, default 0
        End of the window (exclusive), as a sample offset.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The samples in the window.
    """
    index = np.searchsorted(ts.index.values, time)
    return ts.iloc[index + num_samples_before : index + num_samples_after]


def get_sample_const(df: pd.DataFrame, time: ArrayLike) -> FrameOrSeries:
    """Sample(s) at ``time`` assuming a constant sampling interval.

    Faster than :func:`get_sample` (arithmetic on the first two index values
    instead of a search), but only correct for a uniformly sampled ``df``.
    """
    return df.iloc[
        np.rint(
            (time - df.index.values[0]) / (df.index.values[1] - df.index.values[0])
        ).astype(int)
    ]


def interp_df(df: pd.DataFrame, time: ArrayLike) -> pd.DataFrame:
    """Linearly interpolate every column of ``df`` at ``time``.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series.
    time : array_like
        Instant(s) in time.

    Returns
    -------
    pandas.DataFrame
        Interpolated values, indexed by ``time``.
    """
    return pd.DataFrame(
        interp_np_matrix(time, df.index.to_numpy(), df.to_numpy()),
        columns=df.columns,
        index=time,
    )


def interp_np_matrix(new_x: ArrayLike, old_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linearly interpolate each column of a 2-D array at ``new_x``.

    Parameters
    ----------
    new_x : array_like
        New x values.
    old_x : numpy.ndarray
        Existing x values (increasing).
    y : numpy.ndarray
        Existing values, shape ``(len(old_x), n_columns)``.

    Returns
    -------
    numpy.ndarray
        Interpolated values, shape ``(len(new_x), n_columns)``.
    """
    interpolated = np.empty((len(new_x), y.shape[1]), dtype=np.float64)

    for y_col in range(y.shape[1]):
        interpolated[:, y_col] = np.interp(new_x, old_x, y[:, y_col])
    return interpolated


def interp_series(ts: pd.Series, time: ArrayLike) -> pd.Series:
    """Linearly interpolate ``ts`` at ``time``.

    Parameters
    ----------
    ts : pandas.Series
        Time series.
    time : array_like
        Instant(s) in time.

    Returns
    -------
    pandas.Series
        Interpolated values, indexed by ``time``.
    """
    return pd.Series(np.interp(time, ts.index.values, ts.values), index=time)


def get_between(ts: FrameOrSeries, tstart: float, tend: float) -> FrameOrSeries:
    """Samples with ``tstart <= t < tend``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    tstart, tend : float
        Start (inclusive) and end (exclusive) times.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Samples in the range.
    """
    indices = np.searchsorted(ts.index.to_numpy(), [tstart, tend])
    return ts.iloc[indices[0] : indices[1]]


def get_between_and_around(
    ts: FrameOrSeries,
    tstart: float,
    tend: float,
    num_samples_start: int = -1,
    num_samples_end: int = 1,
) -> FrameOrSeries:
    """Like :func:`get_between`, but padded by a few samples on each side.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    tstart, tend : float
        Start and end times.
    num_samples_start : int, default -1
        Sample offset applied to the start index; a negative value includes
        samples before ``tstart``.
    num_samples_end : int, default 1
        Sample offset applied to the end index; a positive value includes
        samples after ``tend``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Samples in the range plus the padding.
    """
    indices = np.searchsorted(ts.index.to_numpy(), [tstart, tend])
    return ts.iloc[indices[0] + num_samples_start : indices[1] + num_samples_end]


def get_between_interp(
    ts: FrameOrSeries, tstart: float, tend: float
) -> FrameOrSeries:
    """Samples between ``tstart`` and ``tend``, with the endpoints interpolated.

    Unlike :func:`get_between`, the first and last rows sit exactly at
    ``tstart`` and ``tend`` (linearly interpolated) rather than on the
    surrounding sample grid.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    tstart, tend : float
        Start and end times.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Samples in the range, bounded exactly by ``tstart`` and ``tend``.
    """
    indices = np.searchsorted(ts.index.to_numpy(), [tstart, tend])
    ts_between = ts.iloc[indices[0] - 1 : indices[1] + 1].copy()
    index = ts.index.values[
        indices[0] - 1 : indices[1] + 1
    ].copy()  # copying the index separately is important, otherwise the index is not copied and the original index of 'ts' may be altered.
    if isinstance(ts, pd.DataFrame):
        if not ts_between.index[0] == tstart:
            ts_between.iloc[0] = interp_df(ts_between, [tstart]).to_numpy()
            index[0] = tstart
        if not index[-1] == tend:
            ts_between.iloc[-1] = interp_df(ts_between, [tend]).to_numpy()
            index[-1] = tend
    else:
        if not index[0] == tstart:
            ts_between.iloc[0] = interp_series(ts_between, [tstart]).to_numpy()
            index[0] = tstart
        if not index[-1] == tend:
            ts_between.iloc[-1] = interp_series(ts_between, [tend]).to_numpy()
            index[-1] = tend
    ts_between.index = index
    return ts_between


def get_index(ts: FrameOrSeries, time: float | ArrayLike) -> int | np.ndarray:
    """Index position of the first sample at or after ``time``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time : float or array_like
        Instant(s) in time.

    Returns
    -------
    int or numpy.ndarray
        The integer position(s), as returned by :func:`numpy.searchsorted`.
    """
    return np.searchsorted(ts.index.values, time)


def resample(
    ts: FrameOrSeries, new_time: np.ndarray | Iterable | float
) -> FrameOrSeries:
    """Resample ``ts`` onto ``new_time`` by linear interpolation.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    new_time : numpy.ndarray or iterable or float
        Either the new sample instants, or a scalar step -- in which case the
        grid ``arange(t[0], t[-1], new_time)`` is used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Resampled values.
    """
    if not isinstance(new_time, np.ndarray):
        new_time = np.arange(ts.index.values[0], ts.index.values[-1], new_time)
    if isinstance(ts, pd.DataFrame):
        return interp_df(ts, new_time)
    else:
        return interp_series(ts, new_time)


def get_delta(ts: FrameOrSeries, tstart: float, tend: float) -> FrameOrSeries:
    """Change in value between ``tstart`` and ``tend`` (``value(tend) - value(tstart)``).

    Uses the nearest sample at or after each instant (see :func:`get_sample`).
    """
    return get_sample(ts, tend) - get_sample(ts, tstart)


def get_delta_shift(
    ts: FrameOrSeries,
    time: float,
    num_samples_before: int,
    num_samples_after: int,
) -> FrameOrSeries:
    """Change in value across ``time``, measured between two shifted samples.

    Both offsets are relative to the first sample at or after ``time``
    (offset ``0``). The result is
    ``ts[idx + num_samples_after] - ts[idx + num_samples_before]``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time : float
        Instant in time.
    num_samples_before : int
        Sample offset of the "before" point (usually negative).
    num_samples_after : int
        Sample offset of the "after" point (usually positive).

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The difference.
    """
    index = np.searchsorted(ts.index.values, time)
    return ts.iloc[index + num_samples_after] - ts.iloc[index + num_samples_before]


def get_delta_around_event(
    ts: FrameOrSeries,
    event_time: float,
    num_samples_before: int = -2,
    num_samples_after: int = 1,
) -> FrameOrSeries:
    """Change in value across an event -- :func:`get_delta_shift` with event-friendly defaults.

    Exists alongside :func:`get_delta_shift` for its self-documenting name at
    call sites in time-domain simulation code. The defaults (``-2`` and ``1``)
    keep a one-step margin from the event, because numerical solvers sometimes
    place erratic samples immediately either side of it.
    """
    return get_delta_shift(ts, event_time, num_samples_before, num_samples_after)


def get_delta_interp_df(df: pd.DataFrame, tstart: float, tend: float) -> pd.Series:
    """Change in each column between ``tstart`` and ``tend``, both linearly interpolated."""
    values = interp_df(df, (tstart, tend))
    return values.iloc[1, :] - values.iloc[0, :]


def get_delta_interp_series(s: pd.Series, tstart: float, tend: float) -> float:
    """Change in value between ``tstart`` and ``tend``, both linearly interpolated."""
    values = interp_series(s, (tstart, tend))
    return values.iloc[1] - values.iloc[0]


def get_duration(ts: FrameOrSeries) -> float:
    """Time span of ``ts`` in seconds (``index[-1] - index[0]``)."""
    return ts.index.values[-1] - ts.index.values[0]


def create_pandas_series_or_frame_with_same_columns_and_index(
    data: np.ndarray, original: FrameOrSeries
) -> FrameOrSeries:
    """Wrap ``data`` in a pandas object matching ``original``'s type and labels.

    A :class:`~pandas.DataFrame` gets ``original``'s columns and index; a
    :class:`~pandas.Series` gets ``original``'s name and index.
    """
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(data, columns=original.columns, index=original.index)
    else:
        return pd.Series(data, name=original.name, index=original.index)


def superpose_series(series: list[pd.Series]) -> pd.Series:
    """Sum a list of Series, aligning on the index (missing points count as 0)."""
    return pd.concat(series, axis=1).sum(axis=1)


def apply_to_columns(
    ts: FrameOrSeries, func: Callable, **kwargs
) -> FrameOrSeries:
    """Apply ``func`` to each column of ``ts``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series. A Series is treated as a single column.
    func : callable
        Applied column-wise (``axis=0``).
    **kwargs
        Bound to ``func`` via :func:`functools.partial`; they may fill either
        keyword or positional parameters of ``func``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Same type as ``ts``.
    """
    func_partial = partial(func, **kwargs)
    if isinstance(ts, pd.Series):
        ts = ts.to_frame().apply(func_partial, axis=0)
        return ts.squeeze()
    else:
        return ts.apply(func_partial, axis=0)


# --------------------------------------------------------------------------- #
# Index / sampling helpers
# --------------------------------------------------------------------------- #
def time_to_seconds(index: pd.Index) -> np.ndarray:
    """Convert a time index to float seconds relative to its first element.

    Parameters
    ----------
    index : pandas.Index
        Time index. Supported types:

        * :class:`pandas.DatetimeIndex` — differences to the first sample are
          taken in the index's native integer resolution and then converted
          to seconds, so any unit (``datetime64[s|ms|us|ns]``) is handled.
        * :class:`pandas.TimedeltaIndex` — converted via
          :meth:`~pandas.TimedeltaIndex.total_seconds`.
        * any numeric index — interpreted as seconds already.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float`` array of length ``len(index)``. The first
        element is always ``0.0``.

    Notes
    -----
    Anchoring at zero avoids loss of precision for ``DatetimeIndex`` objects,
    whose absolute values since the epoch are far too large to be represented
    accurately as ``float64`` seconds. The subtraction is performed on the
    integer datetimes *before* the conversion to float, so full resolution is
    retained regardless of the record's absolute date.

    Examples
    --------
    >>> idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
    >>> time_to_seconds(idx)
    array([0., 1.])
    """
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return (index - index[0]).total_seconds().to_numpy(dtype=float)
    values = np.asarray(index, dtype=float)
    return values - values[0]


def is_uniformly_sampled(data: FrameOrSeries, rtol: float = 1e-6) -> bool:
    """Check whether the time steps of a time series are equidistant.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Time series whose index is inspected. The values are ignored.
    rtol : float, default 1e-6
        Relative tolerance passed to :func:`numpy.allclose`. Each sampling
        interval must deviate from the mean interval by no more than
        ``rtol * mean_interval``.

    Returns
    -------
    bool
        ``True`` if all sampling intervals agree within ``rtol``, or if the
        series has fewer than three samples (in which case uniformity is
        trivially satisfied). ``False`` otherwise.

    Notes
    -----
    The default tolerance is loose enough to accept indices generated by
    :func:`numpy.arange` or :func:`numpy.linspace`, whose steps differ in the
    last bits, but tight enough to detect dropped or duplicated samples.
    """
    t = time_to_seconds(data.index)
    if t.size < 3:
        return True
    dt = np.diff(t)
    return bool(np.allclose(dt, dt.mean(), rtol=rtol, atol=0.0))


def get_sampling_rate(data: FrameOrSeries, check_uniform: bool = True) -> float:
    """Determine the sampling rate of a time series.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Time series with a time index. The values are ignored.
    check_uniform : bool, default True
        If ``True``, verify equidistant sampling via
        :func:`is_uniformly_sampled` before computing the rate. Set to
        ``False`` only if you knowingly want the mean rate of a non-uniform
        series (e.g. as a target rate for :func:`resample_uniform`).

    Returns
    -------
    float
        Sampling rate in Hz, computed as ``(n - 1) / (t[-1] - t[0])``.

    Raises
    ------
    ValueError
        If fewer than two samples are present, or if ``check_uniform`` is
        ``True`` and the time steps are not equidistant.

    Notes
    -----
    The rate is derived from the total span rather than from the first
    interval, which averages out floating-point noise in the index.
    """
    t = time_to_seconds(data.index)
    if t.size < 2:
        raise ValueError("At least two samples are required.")
    if check_uniform and not is_uniformly_sampled(data):
        raise ValueError(
            "Time series is not uniformly sampled. Use `resample_uniform` first."
        )
    return float((t.size - 1) / (t[-1] - t[0]))


def resample_uniform(
    data: FrameOrSeries,
    sampling_rate: float | None = None,
    method: Literal["interp", "fourier"] = "interp",
) -> FrameOrSeries:
    """Resample a time series onto an equidistant time grid.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Input time series. May be non-uniformly sampled if
        ``method="interp"``. The index must be monotonically increasing.
    sampling_rate : float, optional
        Target sampling rate in Hz. If ``None``, the mean rate of the input
        (``(n - 1) / duration``) is used, which preserves the sample count as
        closely as possible.
    method : {'interp', 'fourier'}, default 'interp'
        Resampling algorithm:

        * ``"interp"`` — piecewise linear interpolation via
          :func:`numpy.interp`. Robust and works on non-uniform input, but
          acts as a mild low-pass filter and can introduce aliasing when
          downsampling.
        * ``"fourier"`` — band-limited resampling via
          :func:`scipy.signal.resample`. More accurate for smooth signals but
          requires uniformly sampled input and implicitly assumes the signal
          is periodic within the record, which causes ringing at the edges
          otherwise.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Resampled data of the same type as ``data``, with a float index named
        ``"time"`` starting at ``0.0`` and step ``1 / sampling_rate``. The
        number of samples is ``floor(duration * sampling_rate) + 1``. For a
        ``Series`` input, the original ``name`` is retained.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or if ``method="fourier"`` is requested for
        non-uniformly sampled input.

    Notes
    -----
    The returned index is always relative to zero seconds; absolute timestamps
    of a ``DatetimeIndex`` input are not preserved. Keep the original start
    time separately if you need it.

    Examples
    --------
    >>> s = pd.Series([0.0, 1.0, 3.0], index=[0.0, 0.1, 0.3])
    >>> resample_uniform(s, sampling_rate=10.0).round(3)
    time
    0.0    0.0
    0.1    1.0
    0.2    2.0
    0.3    3.0
    dtype: float64
    """
    t = time_to_seconds(data.index)
    duration = t[-1] - t[0]
    if sampling_rate is None:
        sampling_rate = (t.size - 1) / duration

    n_new = int(np.floor(duration * sampling_rate)) + 1
    t_new = np.arange(n_new) / sampling_rate

    frame = data.to_frame() if isinstance(data, pd.Series) else data

    if method == "interp":
        out = np.column_stack(
            [np.interp(t_new, t, frame[col].to_numpy(dtype=float)) for col in frame]
        )
    elif method == "fourier":
        if not is_uniformly_sampled(data):
            raise ValueError("method='fourier' requires uniformly sampled input.")
        try:
            from scipy.signal import resample as _fourier_resample
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "method='fourier' needs scipy; install it or use method='interp'."
            ) from exc
        out = _fourier_resample(frame.to_numpy(dtype=float), n_new, axis=0)
    else:
        raise ValueError(f"Unknown method {method!r}.")

    result = pd.DataFrame(out, index=pd.Index(t_new, name="time"), columns=frame.columns)
    return result.iloc[:, 0].rename(data.name) if isinstance(data, pd.Series) else result    
