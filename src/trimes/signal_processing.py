"""Generic signal-processing helpers: edge extension and rolling averages."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trimes.base import FrameOrSeries, resample
from trimes.fourier import get_fourier_coef_real

__all__ = [
    "extend_ts",
    "extend_np",
    "get_angle",
    "average_rolling",
    "average_rolling_variable_window",
]


def extend_ts(ts: FrameOrSeries, n: int, mode: str = "wrap") -> FrameOrSeries:
    """Extend a time series by ``n`` samples at one end, index included.

    See :func:`extend_np` for the value semantics; this wrapper also
    extrapolates the time index at the same step.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series (1-D or 2-D).
    n : int
        Number of samples to add. Positive extends the start, negative the end.
    mode : {'wrap', 'reflect'}, default 'wrap'
        ``'wrap'`` copies the first/last ``n`` samples; ``'reflect'`` copies
        them flipped.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The extended series.
    """
    ts_extended_np = extend_np(ts.to_numpy(), n, mode)
    if n > 0:
        extension = ts.index[0] - ts.index[n] + ts.index[:n]
        index = np.concatenate((extension, ts.index))
    else:
        extension = ts.index[-1] - ts.index[n - 1] + ts.index[n:]
        index = np.concatenate((ts.index, extension))
    ts_extended = pd.DataFrame(ts_extended_np, index=index, columns=ts.columns)
    ts_extended.index.name = ts.index.name
    return ts_extended.squeeze()


def extend_np(x: np.ndarray, n: int, mode: str = "wrap") -> np.ndarray:
    """Extend an array by ``n`` samples at one end.

    Typical use: pad the start so a rolling window has data for the first
    ``n`` samples.

    Parameters
    ----------
    x : numpy.ndarray
        Array to extend (1-D or 2-D).
    n : int
        Number of samples to add. Positive extends the start, negative the end.
    mode : {'wrap', 'reflect'}, default 'wrap'
        ``'wrap'`` copies the first/last ``n`` samples; ``'reflect'`` copies
        them flipped.

    Returns
    -------
    numpy.ndarray
        The extended array.

    Raises
    ------
    ValueError
        If ``x`` has more than two dimensions.
    """
    ndim = x.ndim
    if ndim == 1:  # work with 2-D arrays
        x = x.reshape(-1, 1)
    elif ndim > 2:
        raise ValueError("x must be 1D or 2D array")
    if n > 0:
        extension = x[:n, :]
    else:
        extension = x[n:, :]
    if mode == "wrap":
        pass
    elif mode == "reflect":
        extension = np.flip(extension, axis=0)
    if n > 0:
        x = np.concatenate((extension, x), axis=0)
    else:
        x = np.concatenate((x, extension), axis=0)
    if ndim == 1:
        x = x.reshape(-1)
    return x


def get_angle(x: ArrayLike, **kwargs) -> float:
    """Phase angle of one period of a periodic signal, in radians.

    Found by maximising the real part of the fundamental Fourier coefficient
    over the starting angle.

    Parameters
    ----------
    x : array_like
        Exactly one period of the signal.
    **kwargs
        Forwarded to :func:`scipy.optimize.minimize_scalar`.

    Returns
    -------
    float
        Angle in ``[-pi, pi]``.

    Notes
    -----
    Needs scipy (the ``dsp`` extra). TODO: replace with
    ``np.angle`` of the Fourier coefficient, which is a direct formula.
    """
    try:
        from scipy.optimize import minimize_scalar
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "get_angle needs scipy; install it with: pip install 'trimes[dsp]'"
        ) from exc
    f = lambda y, args: -get_fourier_coef_real(args, angle=y)
    opt_res = minimize_scalar(f, args=(x), bounds=(-np.pi, np.pi), **kwargs)
    return opt_res.x


def average_rolling(
    ts: FrameOrSeries,
    samples_per_window: int,
    pad_mode: str = "constant",
    pad_width: int | tuple[int, int] | None = None,
    **kwargs,
) -> FrameOrSeries:
    """Rolling mean over a fixed window of ``samples_per_window`` samples.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    samples_per_window : int
        Window length in samples.
    pad_mode : str, default "constant"
        Padding mode for the leading samples, passed to :func:`numpy.pad`.
    pad_width : int or tuple of int, optional
        Padding width. Defaults to ``(samples_per_window, 0)``.
    **kwargs
        Forwarded to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Same shape and index as ``ts``; the leading window is filled per
        ``pad_mode``.
    """
    if pad_width is None:
        pad_width = (samples_per_window, 0)
    ts_average = ts.copy()
    ts_ndim = ts.ndim
    if ts_ndim == 1:
        ts_average = ts_average.to_frame()
        num_col = 1
        ts = ts.to_frame()
    else:
        num_col = ts.shape[1]
    for col in range(num_col):
        # not vectorised: DataFrame columns may have different dtypes
        ts_cumsum = np.cumsum(ts.iloc[:, col].to_numpy())
        avg = (
            ts_cumsum[samples_per_window:] - ts_cumsum[:-samples_per_window]
        ) / samples_per_window
        ts_average.iloc[:, col] = np.pad(
            avg,
            pad_width=pad_width,
            mode=pad_mode,
            **kwargs,
        )
    if ts_ndim == 1:
        ts_average = ts_average.squeeze()
    return ts_average


def average_rolling_variable_window(
    ts: FrameOrSeries,
    time_windows: ArrayLike,
    samples_per_window: int,
) -> FrameOrSeries:
    """Rolling mean over a window of varying duration.

    Each window is resampled to ``samples_per_window`` points before averaging,
    so it suits signals whose period drifts.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time_windows : array_like
        Per-sample window duration (same length as ``ts``).
    samples_per_window : int
        Number of points each window is resampled to.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Same shape and index as ``ts``; the leading window is wrap-extended.
    """
    time = ts.index.values
    index_first_window = (
        np.argmax(time - time[0] > time_windows) + 1
    )  # first window where time_window fits in ts
    ts_average = ts.copy()
    ts_ndim = ts.ndim
    if ts_ndim == 1:
        ts_average = ts_average.to_frame()
        num_col = 1
        ts = ts.to_frame()
    else:
        num_col = ts.shape[1]
    for col in range(num_col):
        avg_col = np.empty_like(ts.iloc[index_first_window:, col].to_numpy())
        for idx in range(index_first_window, ts.shape[0]):
            t_samples_window = np.linspace(
                time[idx] - time_windows[idx],
                time[idx],
                samples_per_window,
            )
            samples = resample(ts.iloc[:, col], t_samples_window)
            avg_col[idx - index_first_window] = np.mean(samples)
        ts_average.iloc[:, col] = extend_np(avg_col, index_first_window, "wrap")
    if ts_ndim == 1:
        ts_average = ts_average.squeeze()
    return ts_average
