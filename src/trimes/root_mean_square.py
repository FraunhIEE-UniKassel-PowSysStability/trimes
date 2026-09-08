"""Rolling RMS of a time series -- fixed window, variable window, and a
min/max approximation for three-phase signals.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trimes.base import (
    FrameOrSeries,
    create_pandas_series_or_frame_with_same_columns_and_index,
    resample,
)

__all__ = ["rms_rolling", "rms_rolling_variable_window", "rms_min_max"]


def rms_rolling(
    ts: FrameOrSeries,
    samples_per_window: int,
    normalize_magnitude: bool = False,
    pad_width: int | tuple[int, int] | None = None,
    pad_mode: str = "constant",
    **kwargs,
) -> FrameOrSeries:
    """Rolling RMS over a fixed window of ``samples_per_window`` samples.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    samples_per_window : int
        Window length in samples (e.g. one fundamental period).
    normalize_magnitude : bool, default False
        If True, multiply the result by ``sqrt(2)`` (peak-referenced per-unit
        values instead of true RMS).
    pad_width : int or tuple of int, optional
        Padding for the leading samples. Defaults to ``(samples_per_window, 0)``.
    pad_mode : str, default "constant"
        Padding mode, passed to :func:`numpy.pad`.
    **kwargs
        Forwarded to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Same shape and index as ``ts``; the leading window is filled per
        ``pad_mode``.
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
    if pad_width is None:
        pad_width = (samples_per_window, 0)
    rms = np.apply_along_axis(
        np.pad, 0, rms, pad_width=pad_width, mode=pad_mode, **kwargs
    )
    return create_pandas_series_or_frame_with_same_columns_and_index(rms, ts)


def rms_rolling_variable_window(
    ts: FrameOrSeries,
    time_windows: ArrayLike,
    samples_per_window: int,
    normalize_magnitude: bool = False,
    pad_width: int | tuple[int, int] | None = None,
    pad_mode: str = "constant",
) -> FrameOrSeries:
    """Rolling RMS over a window of varying duration.

    Each window is resampled to ``samples_per_window`` points before the RMS is
    taken, so it suits signals whose period drifts.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    time_windows : array_like
        Per-sample window duration (same length as ``ts``), e.g. the local
        fundamental period.
    samples_per_window : int
        Number of points each window is resampled to.
    normalize_magnitude : bool, default False
        If True, multiply the result by ``sqrt(2)``.
    pad_width : int or tuple of int, optional
        Padding for the leading samples that have no full window. Defaults to
        ``(index_of_first_full_window, 0)``.
    pad_mode : str, default "constant"
        Padding mode, passed to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        RMS values, same shape and index as ``ts``.
    """
    normalization_factor = np.sqrt(2) if normalize_magnitude else 1
    time = ts.index.values
    index_first_window = int(
        np.argmax(time - time[0] > time_windows) + 1
    )  # first sample with a full window behind it
    ts_ndim = ts.ndim
    frame = ts.to_frame() if ts_ndim == 1 else ts
    rms = frame.copy()
    if pad_width is None:
        pad_width = (index_first_window, 0)
    for col in range(frame.shape[1]):
        rms_col = np.empty(frame.shape[0] - index_first_window, dtype=float)
        for idx in range(index_first_window, frame.shape[0]):
            t_samples_window = np.linspace(
                time[idx] - time_windows[idx], time[idx], samples_per_window
            )
            samples = resample(frame.iloc[:, col], t_samples_window).to_numpy()
            rms_col[idx - index_first_window] = (
                np.sqrt(np.mean(np.square(samples))) * normalization_factor
            )
        rms.iloc[:, col] = np.pad(rms_col, pad_width=pad_width, mode=pad_mode)
    return rms.squeeze() if ts_ndim == 1 else rms


def rms_min_max(
    ts: pd.DataFrame,
    filter_time_constant: float,
) -> pd.Series:
    """Approximate three-phase RMS from the phase-to-phase min/max spread.

    Takes ``max(phases) - min(phases)`` per sample, scales it to an RMS, and
    low-pass filters (first order) to suppress the residual ripple. Follows
    www.pscad.com/webhelp/EMTDC_Tools_Library/Meters/vm3ph2.

    Parameters
    ----------
    ts : pandas.DataFrame
        Three-phase time series (one column per phase).
    filter_time_constant : float
        Low-pass time constant in seconds, e.g. half a fundamental period.

    Returns
    -------
    pandas.Series
        The filtered RMS estimate.
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
