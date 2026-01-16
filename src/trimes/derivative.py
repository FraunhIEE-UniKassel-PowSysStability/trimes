import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from icecream import ic

from trimes.base import apply_to_columns
from trimes.signal_processing import average_rolling


def diff_moving_avg(
    ts: pd.DataFrame | pd.Series,
    samples_per_window: int,
    sample_time: float,
    pad_mode: str = "constant",
    pad_width: int | tuple | None = None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Calculate derivate using difference between time steps and moving average filter.

    Args:
        ts (pd.DataFrame | pd.Series): time series
        samples_per_window (int): for moving average
        sample_time (float): used for calculation of derivative
        pad_mode (str, optional): pad mode (similar to how it is used in numpy). Defaults to "constant". TODO enhance description
        pad_width (int | tuple | None, optional): with for padding. Defaults to None.

    Returns:
        pd.DataFrame | pd.Series: derivative
    """
    index = ts.index.copy(deep=True)
    ts = apply_to_columns(ts, np.diff).divide(sample_time)
    ts.index = index[1:]
    return apply_to_columns(
        ts,
        average_rolling,
        samples_per_window=samples_per_window,
        pad_mode=pad_mode,
        pad_width=pad_width,
        **kwargs,
    )


def savgol_derivative(
    ts: pd.DataFrame | pd.Series,
    samples_per_window: int,
    sample_time: float,
    polyorder=2,
    mode: str = "interp",
) -> pd.DataFrame | pd.Series:
    """Apply a Savitzky-Golay filter to get derivative. See also SciPy docs.

    Args:
        ts (pd.DataFrame | pd.Series): time series
        samples_per_window (int): of filter
        sample_time (float): of ts
        polyorder (int, optional): of filter. Defaults to 2.
        mode (str, optional): of filter. Defaults to "interp".

    Returns:
        pd.DataFrame | pd.Series: Derivative
    """
    if samples_per_window % 2 == 0:  # must be odd
        samples_per_window += 1
    return apply_to_columns(
        ts,
        savgol_filter,
        window_length=samples_per_window,
        polyorder=polyorder,
        deriv=1,
        delta=sample_time,
        mode=mode,
    )
