from typing import Union, Callable, Literal
from collections.abc import Iterable
from functools import partial

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Literal

from trimes._optional import ic

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



def get_sample(
    ts: Union[pd.DataFrame, pd.Series], time: Union[np.float64, np.array]
) -> Union[pd.DataFrame, pd.Series]:
    """Get first sample after 'time'.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time (np.float64 | np.array): Point(s) in time

    Returns:
        Union[pd.DataFrame, pd.Series]: Sample(s)
    """
    return ts.iloc[np.searchsorted(ts.index.values, time)]  # uses binary search


def get_sample_shifted(
    ts: Union[pd.DataFrame, pd.Series], time: np.float64 | list[np.float64], shift: int
) -> Union[pd.DataFrame, pd.Series]:
    """Get sample after 'time' but shifted by 'shift' samples.

    Example: If shift = -1, the first sample after 'time' is searched and then the sample before that is resturned (i.e. the first sample before 'time').

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time (np.float64 | list[np.float64]): point(s) in time
        shift (int): Samples are shifted by 'shift'

    Returns:
        Union[pd.DataFrame, pd.Series]: Sample(s)
    """
    return ts.iloc[np.searchsorted(ts.index.values, time) + shift]


def get_samples_around(
    ts: Union[pd.DataFrame, pd.Series],
    time: np.float64,
    num_samples_before: int = -1,
    num_samples_after: int = 0,
) -> Union[pd.DataFrame, pd.Series]:
    """Get samples around point(s) in time (between 'num_samples_before' and 'num_samples_after', these values are relative to the first sample after 'time'). By default, the samples before and after 'time' are returned.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time (np.float64): point(s) in time
        num_samples_before (int, optional): shift before
        num_samples_after (int, optional): shift after

    Returns:
        Union[pd.DataFrame, pd.Series]: Sample values
    """
    index = np.searchsorted(ts.index.values, time)
    return ts.iloc[index + num_samples_before : index + num_samples_after]


def get_sample_const(df: pd.DataFrame, time: list[np.float64]):
    return df.iloc[
        np.rint(
            (time - df.index.values[0]) / (df.index.values[1] - df.index.values[0])
        ).astype(int)
    ]


def interp_df(df: pd.DataFrame, time: np.array) -> pd.DataFrame:
    """Interpolate (linear) DataFrame at 'time'.

    Args:
        df (pd.DataFrame): time series
        time (np.array): point(s) in time

    Returns:
        pd.DataFrame: interpolated value(s)
    """
    return pd.DataFrame(
        interp_np_matrix(time, df.index.to_numpy(), df.to_numpy()),
        columns=df.columns,
        index=time,
    )


def interp_np_matrix(new_x: np.array, old_x: np.array, y: np.array) -> np.matrix:
    """Interpolate (linear) numpy matrix at points 'new_x'.

    Args:
      new_x (np.array): new x values
      old_x (np.array): old x values
      y (np.array): old values

    Returns:
        np.matrix: Interpolated values
    """
    interpolated = np.empty((len(new_x), y.shape[1]), dtype=np.float64)

    for y_col in range(y.shape[1]):
        interpolated[:, y_col] = np.interp(new_x, old_x, y[:, y_col])
    return interpolated


def interp_series(ts: pd.Series, time: np.array) -> pd.Series:
    """Interpolate time series 'ts' (using linear interpolation) at 'time'.

    Args:
        ts (pd.Series): time series
        time (np.array): point(s) in time

    Returns:
        pd.Series: Interpolated values
    """
    return pd.Series(np.interp(time, ts.index.values, ts.values), index=time)


def get_between(
    ts: Union[pd.DataFrame, pd.Series], tstart: float, tend: float
) -> Union[pd.DataFrame, pd.Series]:
    """Get values between 'tstart' and 'tend'. Returns values for 'tstart' < t < 'tend'.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        tstart (int): start time
        tend (int): end time

    Returns:
        Union[pd.DataFrame, pd.Series]: Values in range
    """
    indices = np.searchsorted(ts.index.to_numpy(), [tstart, tend])
    return ts.iloc[indices[0] : indices[1]]


def get_between_and_around(
    ts: Union[pd.DataFrame, pd.Series],
    tstart: float,
    tend: float,
    num_samples_start: int = -1,
    num_samples_end: int = 1,
) -> Union[pd.DataFrame, pd.Series]:
    """Get values between 'tstart' and 'tend'. Similar to 'get_between', but includes samples before 'tstart' (or at 'tstart') and after 'tend' (or at 'tend').

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        tstart (int): start time
        tend (int): end time
        num_samples_start (int, optional): negative values include samples before 'tstart'. Defaults to -1.
        num_samples_end (int, optional): positive values include samples after 'tend'. Defaults to 1.

    Returns:
        Union[pd.DataFrame, pd.Series]: Values in range and around range
    """
    indices = np.searchsorted(ts.index.to_numpy(), [tstart, tend])
    return ts.iloc[indices[0] + num_samples_start : indices[1] + num_samples_end]


def get_between_interp(
    ts: Union[pd.DataFrame, pd.Series], tstart: float, tend: float
) -> Union[pd.DataFrame, pd.Series]:
    """Get between 'tstart' and 'tend' and interpolate the values at 'tstart' and 'tend'.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        tstart (float): start time
        tend (float): end time

    Returns:
        Union[pd.DataFrame, pd.Series]: Values in range
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


def get_index(ts: Union[pd.DataFrame, pd.Series], time: np.array) -> int | ArrayLike:
    """Get first index after 'time'.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time (np.array): Point(s) in time.

    Returns:
        Union[pd.DataFrame, pd.Series]: index
    """
    return np.searchsorted(ts.index.values, time)


def resample(
    ts: Union[pd.DataFrame, pd.Series], new_time: Union[np.array, Iterable, float, int]
) -> Union[pd.DataFrame, pd.Series]:
    """Resample time series at 'new_time'.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        x (Union[np.array, Iterable, float, int]): New sample time.

    Returns:
        Union[pd.DataFrame, pd.Series]: Resampled values
    """
    if not isinstance(new_time, np.ndarray):
        new_time = np.arange(ts.index.values[0], ts.index.values[-1], new_time)
    if isinstance(ts, pd.DataFrame):
        return interp_df(ts, new_time)
    else:
        return interp_series(ts, new_time)


def get_delta(ts: Union[pd.DataFrame, pd.Series], tstart: np.float64, tend: np.float64):
    """Get difference between values at 'tend' and 'tstart'

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        tstart (np.float64): start time
        tend (np.float64): end time

    Returns:
        _type_: Difference
    """
    return get_sample(ts, tend) - get_sample(ts, tstart)


def get_delta_shift(
    ts: Union[pd.DataFrame, pd.Series],
    time: np.float64,
    num_samples_before: int,
    num_samples_after: int,
) -> Union[pd.DataFrame, pd.Series]:
    """Get difference between samples around 'time'. The samples are shifted by 'num_samples_before'/'num_samples_after' indeces (e.g. '0' would be the first sample after 'time').

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time (np.float64): point in time
        num_samples_before (int): sample shift before (usually negative)
        num_samples_after (int): sample shift after (usually positive)

    Returns:
        Union[pd.DataFrame, pd.Series]: Difference
    """

    index = np.searchsorted(ts.index.values, time)
    return ts.iloc[index + num_samples_after] - ts.iloc[index + num_samples_before]


def get_delta_around_event(
    ts: Union[pd.DataFrame, pd.Series],
    evemt_time: np.float64,
    num_samples_before: int = -2,
    num_samples_after: int = 1,
):
    """Returns the difference between time series values around an event. Uses 'get_delta_shifted' with default samples. The main reason to have this  function in addition to 'get_delta_shifted' is its descriptive name when used in the context of events in time domain simulations.
    The default values (-2 and 1) are chosen because numerical solvers sometimes behave unexpectedly around events. Hence, not the direct samples before and after the event are used, but a distance of one step is maintained.
    """
    return get_delta_shift(ts, evemt_time, num_samples_before, num_samples_after)


def get_delta_interp_df(
    df: pd.DataFrame, tstart: np.float64, tend: np.float64
) -> pd.DataFrame:
    """Get difference between interpolated (linear) values at 'tend' and 'tstart'

    Args:
        df (pd.DataFrame): time series
        tstart (np.float64): start time
        tend (np.float64): end time

    Returns:
        pd.DataFrame: Difference
    """
    values = interp_df(df, (tstart, tend))
    return values.iloc[1, :] - values.iloc[0, :]


def get_delta_interp_series(s: pd.Series, tstart: int, tend: int):
    """Get difference between interpolated (linear) values at 'tend' and 'tstart'

    Args:
        df (pd.Series): time series
        tstart (np.float64): start time
        tend (np.float64): end time

    Returns:
        pd.DataFrame: Difference
    """
    values = interp_series(s, (tstart, tend))
    return values.iloc[1] - values.iloc[0]


def get_duration(ts: pd.DataFrame | pd.Series):
    return ts.index.values[-1] - ts.index.values[0]


def create_pandas_series_or_frame_with_same_columns_and_index(
    data: np.array, original: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(data, columns=original.columns, index=original.index)
    else:
        return pd.Series(data, name=original.name, index=original.index)


def superpose_series(series: list[pd.Series]):
    return pd.concat(series, axis=1).sum(axis=1)


def apply_to_columns(
    ts: pd.DataFrame | pd.Series, func: Callable, **kwargs
) -> pd.DataFrame | pd.Series:
    """
    Apply 'func' to all columns of 'ts'. Provide additional arguments to 'func' through keyword arguments ('**kwargs').

    Args:
        ts (pd.DataFrame | pd.Series): time series
        func (Callable): function for which additional arguments can be provided.
        kwargs: keyword arguments of this function can be keyword or positional arguments of 'func'.

    Returns:
        pd.DataFrame | pd.Series: time series
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
