"""Generic methods for signal processing"""

from typing import Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from icecream import ic


from trimes.base import (
    create_pandas_series_or_frame_with_same_columns_and_index,
    resample,
)
import trimes.fourier


def extend_ts(
    ts: Union[pd.DataFrame, pd.Series], n: int, mode: str = "wrap"
) -> Union[pd.DataFrame, pd.Series]:
    """Extend array by 'n' samples in the beginning or end of the array.

    See 'extend_np' for more details.

    Args:
        x (ArrayLike): array to be extended (1D or 2D)
        n (int): number of samples of the extension. Positive extends the array in the beginning, negative extends the array in the end.
        mode (str): 'wrap' simply copies the samples from the first/last n samples of 'x', 'reflect' flips the samples. Defaults to 'wrap'.

    Returns:
        Union[pd.DataFrame, pd.Series]: extended time series
    """
    ts_extended_np = extend_np(ts.to_numpy(), n, mode)
    if n > 0:
        extension = ts.index[0] - ts.index[n] + ts.index[:n]
        index = np.concat((extension, ts.index))
    else:
        extension = ts.index[-1] - ts.index[n - 1] + ts.index[n:]
        index = np.concat((ts.index, extension))
    ts_extended = pd.DataFrame(ts_extended_np, index=index, columns=ts.columns)
    ts_extended.index.name = ts.index.name
    return ts_extended.squeeze()


def extend_np(x: ArrayLike, n: int, mode: str = "wrap") -> ArrayLike:
    """Extend array by 'n' samples in the beginning or end of the array.

    A typical use case is to extend the array by 'n' samples in the beginning of the array, so that the rolling average can be calculated for the first 'n' samples.


    Args:
        x (ArrayLike): array to be extended (1D or 2D)
        n (int): number of samples of the extension. Positive extends the array in the beginning, negative extends the array in the end.
        mode (str): 'wrap' simply copies the samples from the first/last n samples of 'x', 'reflect' flips the samples. Defaults to 'wrap'.

    Raises:
        ValueError: if 'x' is not 1D or 2D.

    Returns:
        ArrayLike: extended array
    """
    ndim = x.ndim
    if ndim == 1:  # work with 2D arrays
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
    """Get angle of periodic time series 'x'.

    TODO: Just use np.angle(fourier_coeff) to get angle.

    Get angle by maximizing (minimizing negative) real part of fourier coefficient.

    Args:
        x (ArrayLike): One period of time series.

    Returns:
        float: angle [rad]
    """
    f = lambda y, args: -trimes.fourier.get_fourier_coef_real(args, angle=y)
    opt_res = minimize_scalar(f, args=(x), bounds=(-np.pi, np.pi), **kwargs)
    return opt_res.x


def average_rolling(
    ts: Union[pd.DataFrame, pd.Series],
    samples_per_window: int,
) -> Union[pd.DataFrame, pd.Series]:
    """Get rolling average of time series 'ts' using 'samples_per_window' samples.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        samples_per_window (int): number of samples per window

    Returns:
        Union[pd.DataFrame, pd.Series]: Averaged time series (same shape as 'ts', first window is extended)
    """
    ts_average = ts.copy()
    ts_ndim = ts.ndim
    if ts_ndim == 1:
        ts_average = ts_average.to_frame()
        num_col = 1
        ts = ts.to_frame()
    else:
        num_col = ts.shape[1]
    for col in range(
        num_col
    ):  # Cannot vectorize this loop because dtypes of columns can be different in DataFrames
        ts_cumsum = np.cumsum(ts.iloc[:, col].to_numpy())
        avg = (
            ts_cumsum[samples_per_window:] - ts_cumsum[:-samples_per_window]
        ) / samples_per_window
        ts_average.iloc[:, col] = extend_np(avg, samples_per_window, "wrap")
    if ts_ndim == 1:
        ts_average = ts_average.squeeze()
    return ts_average


def average_rolling_variable_window(
    ts: Union[pd.DataFrame, pd.Series],
    time_windows: ArrayLike,
    samples_per_window: int,
) -> Union[pd.DataFrame, pd.Series]:
    """Get rolling average of time series 'ts' using variable time windows that are resampled with 'samples_per_window' number of samples.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time_windows (np.array): duration of time windows (same length as rows in 'ts')
        samples_per_window (int): number of samples per window


    Returns:
        Union[pd.DataFrame, pd.Series]: Averaged time series (same shape as 'ts', first window is extended)
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
