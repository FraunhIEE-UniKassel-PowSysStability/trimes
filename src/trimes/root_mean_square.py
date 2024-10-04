from typing import Union

import pandas as pd
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize_scalar
from icecream import ic

from trimes.base import create_pandas_series_or_frame_with_same_columns_and_index
from trimes.signal_processing import extend
from trimes.filter import pt1


def rms_rolling_2(
    ts: Union[pd.DataFrame, pd.Series],
    samples_per_window: int,
    mode: str = "wrap",
) -> Union[pd.DataFrame, pd.Series]:
    """

    From https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal

    Args:
        ts (Union[pd.DataFrame, pd.Series]): _description_
        samples_per_window (int): _description_

    Returns:
        Union[pd.DataFrame, pd.Series]: _description_
    """
    ts_squared = np.power(ts.to_numpy(), 2)
    window_averaging_weights = np.ones(samples_per_window) / float(samples_per_window)
    rms = np.sqrt(convolve1d(ts_squared, window_averaging_weights, mode=mode, axis=0))
    return create_pandas_series_or_frame_with_same_columns_and_index(rms, ts)


def rms_rolling(
    ts: Union[pd.DataFrame, pd.Series],
    samples_per_window: int,
    normalize_magnitude: bool = False,
) -> Union[pd.DataFrame, pd.Series]:

    normalization_factor = np.sqrt(2) if normalize_magnitude else 1
    ts_squared = np.cumsum(np.power(ts.to_numpy(), 2), axis=0)
    func_rms = (
        lambda x: np.sqrt(
            (x[samples_per_window:] - x[:-samples_per_window]) / samples_per_window
        )
        * normalization_factor
    )
    rms = np.apply_along_axis(func_rms, 0, ts_squared)
    rms = extend(rms, samples_per_window, "wrap")
    return create_pandas_series_or_frame_with_same_columns_and_index(rms, ts)


def rms_min_max(
    ts: Union[pd.DataFrame, pd.Series],
    filter_time_constant: float,
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(ts, pd.Series):
        ts = ts.to_frame()
    rms = np.zeros(ts.shape[0], dtype=float)

    factor = np.pi / (3 * np.sqrt(3))

    first_row = ts.to_numpy()[0, :]
    rms[0] = factor * (np.max(first_row) - np.min(first_row))

    for n, row_values in enumerate(ts.to_numpy()[1:, :], 1):
        val = factor * (np.max(row_values) - np.min(row_values))
        rms[n] = rms[n - 1] + (val - rms[n - 1]) / filter_time_constant * (
            ts.index.values[n] - ts.index.values[n - 1]
        )
    return pd.DataFrame(rms, index=ts.index)
