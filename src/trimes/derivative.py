"""Numerical time derivatives of noisy time series (needs scipy -- the
``dsp`` extra).
"""

import numpy as np

try:
    from scipy.signal import savgol_filter
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.derivative needs scipy; install it with: pip install 'trimes[dsp]'"
    ) from _exc

from trimes.base import FrameOrSeries, apply_to_columns
from trimes.signal_processing import average_rolling

__all__ = ["diff_moving_avg", "savgol_derivative"]


def diff_moving_avg(
    ts: FrameOrSeries,
    samples_per_window: int,
    sample_time: float,
    pad_mode: str = "constant",
    pad_width: int | tuple[int, int] | None = None,
    **kwargs,
) -> FrameOrSeries:
    """Derivative as a finite difference smoothed by a moving average.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    samples_per_window : int
        Window length of the moving-average smoothing.
    sample_time : float
        Sample step, used to scale the finite difference.
    pad_mode : str, default "constant"
        Padding mode for the leading samples, passed to :func:`numpy.pad`.
    pad_width : int or tuple of int, optional
        Padding width. Defaults to ``(samples_per_window, 0)``.
    **kwargs
        Forwarded to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The smoothed derivative.
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
    ts: FrameOrSeries,
    samples_per_window: int,
    sample_time: float,
    polyorder: int = 2,
    mode: str = "interp",
    pad_mode: str | None = None,
    pad_width: int | tuple[int, int] | None = None,
    **kwargs,
) -> FrameOrSeries:
    """Derivative via a Savitzky-Golay filter.

    Wraps :func:`scipy.signal.savgol_filter` with ``deriv=1``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    samples_per_window : int
        Filter window length; incremented by one if even (it must be odd).
    sample_time : float
        Sample step of ``ts`` (the ``delta`` of the filter).
    polyorder : int, default 2
        Polynomial order of the filter.
    mode : str, default "interp"
        Edge handling of :func:`scipy.signal.savgol_filter`, used when
        ``pad_mode`` is None.
    pad_mode : str, optional
        If given, discard the filter's edge samples and re-pad with
        :func:`numpy.pad` in this mode instead.
    pad_width : int or tuple of int, optional
        Padding width when ``pad_mode`` is used. Defaults to
        ``(samples_per_window, 0)``.
    **kwargs
        Forwarded to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The derivative.
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
