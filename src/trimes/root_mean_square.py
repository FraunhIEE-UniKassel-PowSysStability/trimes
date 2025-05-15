from typing import Union

import pandas as pd
import numpy as np
from icecream import ic

from trimes.base import (
    create_pandas_series_or_frame_with_same_columns_and_index,
    resample,
)
from trimes.signal_processing import extend_np


def rms_rolling(
    ts: Union[pd.DataFrame, pd.Series],
    samples_per_window: int,
    normalize_magnitude: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the root mean square (RMS) of a time series using a rolling window.

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        samples_per_window (int): number of samples per window (e.g. for one period)
        normalize_magnitude (bool, optional): If true, results are multiplied by sqrt(2) (e.g. for per unit values). Defaults to False.

    Returns:
        Union[pd.DataFrame, pd.Series]: time series with RMS values
    """
    normalization_factor = np.sqrt(2) if normalize_magnitude else 1
    ts_squared = np.cumsum(np.power(ts.to_numpy(), 2), axis=0)
    func_rms = (
        lambda x: np.sqrt(
            (x[samples_per_window:] - x[:-samples_per_window]) / samples_per_window
        )
        * normalization_factor
    )
    rms = np.apply_along_axis(func_rms, 0, ts_squared)
    rms = extend_np(rms, samples_per_window, "wrap")
    return create_pandas_series_or_frame_with_same_columns_and_index(rms, ts)


def rms_rolling_variable_window(
    ts: Union[pd.DataFrame, pd.Series],
    time_windows: np.array,
    samples_per_window: int,
    normalize_magnitude: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the root mean square (RMS) of a time series using a rolling window with variable length.
    The length of the window is defined by the time_windows parameter. The time_windows parameter must be the

    Args:
        ts (Union[pd.DataFrame, pd.Series]): time series
        time_windows (np.array): Duration of windows (e.g. periods for variable frequencies). Same length as the number of rows in the time series.
        samples_per_window (int): Number of samples per window.
        normalize_magnitude (bool, optional): If true, results are multiplied by sqrt(2) (e.g. for per unit values). Defaults to False.

    Returns:
        Union[pd.DataFrame, pd.Series]: RMS values of the time series
    """
    normalization_factor = np.sqrt(2) if normalize_magnitude else 1
    time = ts.index.values
    index_first_window = (
        np.argmax(time - time[0] < time_windows) + 1
    )  # first window where time_window fits in ts
    rms = np.empty_like(ts.to_numpy()).reshape(ts.shape[0], -1)[index_first_window:, :]
    for idx in range(ts.shape[0] - index_first_window):
        t_samples_window = np.linspace(
            time[idx] - time_windows[idx], time[idx], samples_per_window
        )
        samples = resample(ts, t_samples_window)
        rms[idx, :] = (
            np.sqrt(np.sum(np.power(samples, 2), axis=0) / samples_per_window)
            * normalization_factor
        )
    rms = extend_np(rms, index_first_window, "wrap")
    return create_pandas_series_or_frame_with_same_columns_and_index(rms.squeeze(), ts)


def rms_min_max(
    ts: pd.DataFrame,
    filter_time_constant: float,
) -> pd.Series:
    """Get apprmation of RMS value of a 3-phase signal using difference between maxima and minima of the phases and low pass filtering to reduce the ripple.

    According to www.pscad.com/webhelp/EMTDC_Tools_Library/Meters/vm3ph2

    Args:
        ts (pd.DataFrame): time series (3 phases)
        filter_time_constant (float): low pass filter time constant in seconds, e.g. half a period of the fundamental frequency

    Returns:
        pd.Series: RMS value of the time series
    """
    ts_frame = ts.to_frame() if isinstance(ts, pd.Series) else ts
    rms = np.zeros(ts.shape[0], dtype=float)
    factor = np.pi / (3 * np.sqrt(3))
    first_row = ts_frame.to_numpy()[0, :]
    rms[0] = factor * (np.max(first_row) - np.min(first_row))
    for n, row_values in enumerate(ts_frame.to_numpy()[1:, :], 1):
        val = factor * (np.max(row_values) - np.min(row_values))
        rms[n] = rms[n - 1] + (val - rms[n - 1]) / filter_time_constant * (
            ts_frame.index.values[n] - ts_frame.index.values[n - 1]
        )
    return pd.Series(rms, index=ts.index)
