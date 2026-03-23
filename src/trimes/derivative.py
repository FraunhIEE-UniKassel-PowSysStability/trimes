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
    pad_width: int | tuple[int, int] | None = None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Calculate derivate using difference between time steps and moving average filter.

    numpy.pad is used for padding (see numpy.org/doc/stable/reference/generated/numpy.pad.html)

    Args:
        ts (pd.DataFrame | pd.Series): time series
        samples_per_window (int): for moving average
        sample_time (float): used for calculation of derivative
        pad_mode (str, optional): pad mode (see numpy docs). Defaults to "constant".
        pad_width (int | tuple | None, optional): width for padding (see numpy docs). Defaults to None.
        **kwargs: used in numpy.pad

    Returns:
        pd.DataFrame | pd.Series: derivative
    """
    index = ts.index.copy(deep=True)
    ts = apply_to_columns(ts, np.diff).divide(sample_time)
    ts.index = index[1:]
    if pad_width is None:
        pad_width = (samples_per_window, 0)
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
    pad_mode: str | None = None,
    pad_width: int | tuple[int, int] | None = None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Apply a Savitzky-Golay filter to get derivative. See also SciPy docs (docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).

    numpy.pad can be used for padding (see numpy.org/doc/stable/reference/generated/numpy.pad.html).

    Args:
        ts (pd.DataFrame | pd.Series): time series
        samples_per_window (int): of filter
        sample_time (float): of ts
        polyorder (int, optional): of filter. Defaults to 2.
        mode (str, optional): Used for padding if 'pad_mode' is None. See docs of SciPy's 'savgol_filter'. Defaults to "interp".
        pad_mode (str | None, , optional): If not None, numpy.pad (see numpy docs) is used instead of the SciPy padding. Defaults to None.
        pad_width (int | tuple[int, int] | None): Defaults to None.
        **kwargs: used in numpy.pad

    Returns:
        pd.DataFrame | pd.Series: Derivative
    """
    if samples_per_window % 2 == 0:  # must be odd
        samples_per_window += 1
    ts_der = apply_to_columns(
        ts,
        savgol_filter,
        window_length=samples_per_window,
        polyorder=polyorder,
        deriv=1,
        delta=sample_time,
        mode=mode,
    )
    if pad_mode:
        length_that_was_padded = int((samples_per_window - 1) / 2)
        ts_der = ts_der.iloc[length_that_was_padded:-length_that_was_padded, :]
        if pad_width is None:
            pad_width = (samples_per_window, 0)
        ts_der = apply_to_columns(
            ts_der, np.pad, pad_width=pad_width, mode=pad_mode, **kwargs
        )
    return ts_der
