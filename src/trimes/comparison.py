"""Compare a time series against a reference or an envelope.

Element-wise arithmetic and comparison operators with optional resampling to a
common grid, plus time-weighted error metrics (via :mod:`trimes.metrics`) that
can be restricted to the spans where a condition holds.
"""

import operator
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trimes.base import FrameOrSeries, get_between, resample
from trimes.metrics import get_metric_time_series, integral_abs_error

__all__ = [
    "subtract",
    "add",
    "apply_operator_series",
    "apply_operator_df",
    "greater_than_series",
    "less_than_series",
    "outside_envelope",
    "get_time_bool",
    "comparison_series",
    "comparison_df",
    "envelope_comparison_series",
    "envelope_comparison_df",
]


def subtract(
    ts: FrameOrSeries,
    reference: pd.Series | pd.DataFrame,
    extend: bool = False,
    resample_ts: bool = False,
    resample_reference: bool = False,
) -> pd.Series | pd.DataFrame:
    """Subtract ``reference`` from ``ts``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    reference : pandas.Series or pandas.DataFrame
        Reference to subtract.
    extend : bool, default False
        If False (the default), the result is clipped to the time span of
        ``ts``. If True, it keeps pandas' native behaviour of extending to the
        union of both spans.
    resample_ts : bool, default False
        Resample ``ts`` onto ``reference``'s index before subtracting.
    resample_reference : bool, default False
        Resample ``reference`` onto ``ts``'s index before subtracting. Without
        either resample flag, the two must already share a sample grid.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The difference.
    """
    if not extend:
        reference = _avoid_extension(ts, reference)
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    return ts.sub(reference, axis=0)


def add(
    ts: FrameOrSeries,
    reference: pd.Series | pd.DataFrame,
    extend: bool = False,
    resample_ts: bool = False,
    resample_reference: bool = False,
) -> pd.Series | pd.DataFrame:
    """Add ``reference`` to ``ts``. See :func:`subtract` for the arguments."""
    if not extend:
        reference = _avoid_extension(ts, reference)
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    return ts.add(reference, axis=0)


def _avoid_extension(ts, reference):
    """Clip ``reference`` to the time span of ``ts`` if it reaches beyond it."""
    t_start = ts.index.values[0]
    t_end = ts.index.values[-1]
    if reference.index.values[0] < t_start or reference.index.values[-1] > t_end:
        reference = get_between(reference, t_start, t_end)
    return reference


def _align_sample_time_of_ts_and_reference(
    ts, reference, resample_ts: bool, resample_reference: bool
):
    """Optionally resample one of ``ts`` / ``reference`` onto the other's index."""
    if resample_ts:
        ts = resample(ts, reference.index.values)
    elif resample_reference:
        reference = resample(reference, ts.index.values)
    return ts, reference


def apply_operator_series(
    ts: pd.Series,
    reference: pd.Series,
    operator: Callable,
    resample_ts: bool = False,
    resample_reference: bool = False,
) -> np.ndarray:
    """Apply a binary operator element-wise to ``ts`` and ``reference``.

    Parameters
    ----------
    ts : pandas.Series
        Time series.
    reference : pandas.Series
        Reference to compare against.
    operator : callable
        A binary operator, e.g. from the built-in :mod:`operator` module
        (``operator.gt``, ``operator.lt``, ...).
    resample_ts, resample_reference : bool, default False
        Align the sample grids first (see :func:`subtract`).

    Returns
    -------
    numpy.ndarray
        The element-wise result (a boolean array for comparison operators).
    """
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    return operator(ts.to_numpy(), reference.to_numpy())


def apply_operator_df(
    ts: pd.DataFrame,
    reference: pd.Series,
    operator: Callable,
    resample_ts: bool = False,
    resample_reference: bool = False,
) -> np.ndarray:
    """:func:`apply_operator_series` for a DataFrame ``ts``.

    Returns a 2-D array whose columns correspond to those of ``ts``, each
    compared against the same ``reference``.
    """
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    reference_np = reference.to_numpy()
    ts_np = ts.to_numpy()
    return np.transpose([operator(col, reference_np) for col in ts_np.T])


def greater_than_series(
    ts: pd.Series, reference: pd.Series, resample_ts: bool = False
) -> np.ndarray:
    """Boolean array: where ``ts > reference``."""
    return apply_operator_series(ts, reference, operator.gt, resample_ts)


def less_than_series(
    ts: pd.Series, reference: pd.Series, resample_ts: bool = False
) -> np.ndarray:
    """Boolean array: where ``ts < reference``."""
    return apply_operator_series(ts, reference, operator.lt, resample_ts)


def outside_envelope(
    ts: pd.Series,
    envelope: pd.DataFrame,
    resample_ts: bool = False,
    resample_envelope: bool = False,
) -> np.ndarray:
    """Boolean array: where ``ts`` leaves the envelope.

    Parameters
    ----------
    ts : pandas.Series
        Time series.
    envelope : pandas.DataFrame
        Column 0 is the upper boundary, column 1 the lower boundary.
    resample_ts, resample_envelope : bool, default False
        Align the sample grids first (see :func:`subtract`).

    Returns
    -------
    numpy.ndarray
        True where ``ts`` is above the upper or below the lower boundary.
    """
    ts, envelope = _align_sample_time_of_ts_and_reference(
        ts, envelope, resample_ts, resample_envelope
    )
    greater = apply_operator_series(ts, envelope.iloc[:, 0], operator.gt)
    smaller = apply_operator_series(ts, envelope.iloc[:, 1], operator.lt)
    return np.logical_or(greater, smaller)


def get_time_bool(
    b: ArrayLike,
    sample_time: float | ArrayLike,
) -> float:
    """Total time for which ``b`` is True.

    Parameters
    ----------
    b : array_like
        Boolean array.
    sample_time : float or array_like
        A constant sample step, or the per-sample time vector.

    Returns
    -------
    float
        The accumulated duration.
    """
    if np.isscalar(sample_time):
        return b.sum() * sample_time
    else:
        return np.sum(np.diff(sample_time)[b[:-1]])


def comparison_series(
    ts: pd.Series,
    reference: pd.Series,
    operator_or_bool: Callable | ArrayLike | None = None,
    metric: Callable = integral_abs_error,
    sample_time: float | ArrayLike | None = None,
    resample_ts: bool = False,
    resample_reference: bool = False,
) -> float:
    """Time-weighted error of ``ts`` vs ``reference``, optionally masked.

    Parameters
    ----------
    ts : pandas.Series
        Time series under test.
    reference : pandas.Series
        Reference series.
    operator_or_bool : callable or array_like, optional
        Either a binary operator applied to ``ts`` and ``reference`` to build a
        mask, or a ready-made boolean mask. Only the masked spans contribute to
        the error; elsewhere ``ts`` is treated as equal to ``reference``. If
        omitted, the whole series contributes.
    metric : callable, default :func:`~trimes.metrics.integral_abs_error`
        An sklearn-style metric.
    sample_time : float or array_like, optional
        Sample step(s); defaults to ``diff(reference.index)``.
    resample_ts, resample_reference : bool, default False
        Align the sample grids first (see :func:`subtract`).

    Returns
    -------
    float
        The metric value (``0.0`` if the mask selects nothing).
    """
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    if operator_or_bool is not None:
        if callable(operator_or_bool):
            mask = apply_operator_series(
                ts, reference, operator_or_bool, resample_ts=False
            )
        else:
            mask = operator_or_bool
        if np.any(mask):
            ts_diff = reference.copy()
            ts_diff[mask] = ts[mask]
            return get_metric_time_series(
                ts_diff, reference, metric, sample_time=sample_time
            )
        else:
            return 0.0
    else:
        return get_metric_time_series(ts, reference, metric, sample_time=sample_time)


def comparison_df(
    ts: pd.DataFrame,
    reference: pd.Series,
    operator_or_bool: Callable | ArrayLike | None = None,
    metric: Callable = integral_abs_error,
    resample_ts: bool = False,
    resample_reference: bool = False,
    sample_time: float | ArrayLike | None = None,
) -> ArrayLike:
    """:func:`comparison_series` for a DataFrame ``ts`` (one error per column)."""
    ts, reference = _align_sample_time_of_ts_and_reference(
        ts, reference, resample_ts, resample_reference
    )
    if operator_or_bool is not None:
        if callable(operator_or_bool):
            masks = apply_operator_df(
                ts, reference, operator_or_bool, resample_ts=False
            )
        else:
            masks = operator_or_bool
        errors = np.zeros(masks.shape[1])
        for col, mask in enumerate(masks.T):
            if np.any(mask):
                ts_diff = reference.copy()
                ts_diff[mask] = ts.iloc[:, col][mask]
                errors[col] = get_metric_time_series(
                    ts_diff, reference, metric, sample_time=sample_time
                )
        return errors
    else:
        return get_metric_time_series(ts, reference, metric, sample_time=sample_time)


def envelope_comparison_series(
    ts: pd.Series,
    envelope: pd.DataFrame,
    metric: Callable = integral_abs_error,
    resample_ts: bool = False,
) -> float:
    """:func:`comparison_series` against an envelope.

    The error is the sum of the metric on the spans where ``ts`` exceeds the
    upper boundary and the spans where it drops below the lower boundary.

    Parameters
    ----------
    ts : pandas.Series
        Time series under test.
    envelope : pandas.DataFrame
        Column 0 is the upper boundary, column 1 the lower boundary.
    metric : callable, default :func:`~trimes.metrics.integral_abs_error`
        An sklearn-style metric.
    resample_ts : bool, default False
        Resample ``ts`` onto the envelope's index first.

    Returns
    -------
    float
        The combined boundary-violation error.
    """
    error_upper = comparison_series(
        ts, envelope.iloc[:, 0], operator.gt, metric=metric, resample_ts=resample_ts
    )
    error_lower = comparison_series(
        ts, envelope.iloc[:, 1], operator.lt, metric=metric, resample_ts=resample_ts
    )
    return error_upper + error_lower


def envelope_comparison_df(
    ts: pd.DataFrame,
    envelope: pd.DataFrame,
    metric: Callable = integral_abs_error,
    resample_ts: bool = False,
) -> ArrayLike:
    """:func:`envelope_comparison_series` for a DataFrame ``ts`` (one error per column)."""
    error_upper = comparison_df(
        ts, envelope.iloc[:, 0], operator.gt, metric=metric, resample_ts=resample_ts
    )
    error_lower = comparison_df(
        ts, envelope.iloc[:, 1], operator.lt, metric=metric, resample_ts=resample_ts
    )
    return error_upper + error_lower
