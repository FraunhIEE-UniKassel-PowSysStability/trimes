"""Polynomial regression for time series."""

import numpy as np
import pandas as pd

from trimes.base import FrameOrSeries

__all__ = ["polyfit"]


def polyfit(
    ts: FrameOrSeries, degree: int, t_offset: float | None = None
) -> np.ndarray:
    """Least-squares polynomial fit of ``ts`` against its time index.

    Thin wrapper around :func:`numpy.polyfit` (``x = ts.index``,
    ``y = ts.to_numpy()``).

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series. For a DataFrame each column is fitted independently.
    degree : int
        Degree of the fitting polynomial.
    t_offset : float, optional
        Added to the time index before fitting (``x = ts.index + t_offset``),
        e.g. to recentre the fit and improve conditioning.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients, highest power first -- shape ``(degree + 1,)``
        for a Series, ``(degree + 1, n_columns)`` for a DataFrame.

    Notes
    -----
    ``numpy.polynomial.Polynomial.fit`` is the currently recommended API but
    did not handle the DataFrame (multi-column ``y``) case cleanly, so the
    legacy :func:`numpy.polyfit` is kept here.
    """
    x = ts.index if t_offset is None else ts.index + t_offset
    return np.polyfit(x, ts.to_numpy(), degree)
