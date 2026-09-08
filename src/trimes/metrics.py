"""Error metrics and similarity measures for time series.

The ``get_metric_*`` helpers adapt time-domain data to the call convention of
:mod:`sklearn.metrics` (``metric(y_true, y_pred, sample_weight=..., multioutput=...)``),
weighting each sample by its time step so that the result is an integral over
time rather than a plain sample average. :func:`integral_abs_error` and
:func:`integral_squared_error` follow the same convention and can be passed
wherever an sklearn metric is expected.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from trimes.base import FrameOrSeries, resample

__all__ = [
    "get_metric_time_series",
    "get_metric_np",
    "integral_abs_error",
    "integral_squared_error",
    "get_cosine_similarity",
    "get_cosine_similarity_matrix",
    "get_cosine_similarity_matrix_as_df",
]

# numpy renamed np.trapz -> np.trapezoid in 2.0; support both (numpy >= 1.24).
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def get_metric_time_series(
    ts: FrameOrSeries,
    reference: pd.Series,
    metric: Callable,
    sample_time: float | ArrayLike | None = None,
    resample_ts: bool = False,
) -> float | ArrayLike:
    """Evaluate a time-weighted error metric of ``ts`` against ``reference``.

    Also serves as an interface to :mod:`sklearn.metrics` (e.g.
    :func:`sklearn.metrics.root_mean_squared_error`).

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series under test. Each column of a DataFrame is scored against
        the same ``reference``.
    reference : pandas.Series
        Reference series. Its index provides the time steps used as sample
        weights.
    metric : callable
        An sklearn-style metric ``metric(y_true, y_pred, sample_weight=...,
        multioutput=...)``.
    sample_time : float or array_like, optional
        Sample step(s) of ``ts``. Defaults to ``diff(reference.index)``. A
        scalar is broadcast.
    resample_ts : bool, default False
        If True, resample ``ts`` onto ``reference``'s index first; otherwise
        the two must already share a sample grid.

    Returns
    -------
    float or numpy.ndarray
        The metric value, or one value per column of ``ts``.

    Notes
    -----
    The last sample of both series is dropped, because there are ``n - 1``
    time steps to weight ``n`` samples (a trapezoidal-style convention).
    """
    if resample_ts:
        ts = resample(ts, reference.index.values)
    if sample_time is None:
        sample_weight = np.diff(reference.index.values)
    elif np.isscalar(sample_time):
        sample_weight = np.broadcast_to(sample_time, len(reference) - 1)
    else:
        sample_weight = sample_time
    return get_metric_np(
        ts.to_numpy()[:-1],
        reference.to_numpy()[:-1],
        metric,
        sample_weight=sample_weight,
    )


def get_metric_np(
    ts: ArrayLike,
    reference: ArrayLike,
    metric: Callable,
    sample_weight: ArrayLike | None = None,
    **kwargs,
) -> float | ArrayLike:
    """Evaluate an sklearn-style metric on numpy arrays.

    Parameters
    ----------
    ts : array_like
        Values under test, shape ``(n,)`` or ``(n, n_columns)``.
    reference : array_like
        Reference values, shape ``(n,)``. Broadcast across the columns of
        ``ts`` when ``ts`` is 2-D.
    metric : callable
        An sklearn-style metric (see :func:`get_metric_time_series`).
    sample_weight : array_like, optional
        Per-sample weights (the sample step, for a time integral).
    **kwargs
        Forwarded to ``metric``.

    Returns
    -------
    float or numpy.ndarray
        Scalar for 1-D input, one value per column for 2-D input.
    """
    if ts.ndim > 1:
        multioutput = "raw_values"
        # one reference column per column of 'ts' (not a hard-coded 2)
        reference = np.broadcast_to(np.asarray(reference)[:, None], ts.shape)
    else:
        multioutput = "uniform_average"
    return metric(
        reference, ts, sample_weight=sample_weight, multioutput=multioutput, **kwargs
    )


def integral_abs_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
    multioutput: str | ArrayLike = "uniform_average",
) -> float | ArrayLike:
    """Integral of ``|y_pred - y_true|`` over time (the area between the curves).

    An sklearn-style metric: same signature as
    :func:`sklearn.metrics.root_mean_squared_error`, so it can be passed to
    :func:`get_metric_time_series`.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted values.
    sample_weight : array_like, optional
        Per-sample weights (the sample step).
    multioutput : {'uniform_average', 'raw_values'} or array_like, default 'uniform_average'
        Column aggregation, following the sklearn convention.

    Returns
    -------
    float or numpy.ndarray
    """
    weighted_error = _weighted_error(y_true, y_pred, sample_weight)
    integrated_error = _trapezoid(np.abs(weighted_error), axis=0)
    return _handle_sklearn_multioutput(multioutput, integrated_error)


def integral_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
    multioutput: str | ArrayLike = "uniform_average",
) -> float | ArrayLike:
    """Integral of ``(y_pred - y_true) ** 2`` over time.

    Companion of :func:`integral_abs_error`; see it for the argument meanings.
    """
    weighted_error = _weighted_error(y_true, y_pred, sample_weight)
    integrated_error = _trapezoid(np.square(weighted_error), axis=0)
    return _handle_sklearn_multioutput(multioutput, integrated_error)


def _weighted_error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike
) -> ArrayLike:
    """Weighted error ``(y_pred - y_true) * sample_weight``.

    ``y_true`` and ``sample_weight`` have shape ``(n,)``; ``y_pred`` may be
    ``(n,)`` or ``(n, m)``. The result matches ``y_pred``.
    """
    return ((y_pred - y_true).T * sample_weight).T


def _handle_sklearn_multioutput(
    multioutput: str | None | ArrayLike, metric_result: ArrayLike
):
    """Apply the sklearn ``multioutput`` aggregation to a per-column result."""
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return metric_result
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(metric_result, weights=multioutput)


def get_cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity of two vectors.

    Depends only on the angle between ``a`` and ``b``, not their magnitudes.
    They must have the same length (e.g. two variables of a time-domain
    simulation).

    Parameters
    ----------
    a, b : array_like
        Vectors of equal length.

    Returns
    -------
    float
        Value in ``[-1, 1]``.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_cosine_similarity_matrix(a: np.ndarray) -> np.ndarray:
    """Symmetric matrix of pairwise cosine similarities between the columns of ``a``.

    Parameters
    ----------
    a : numpy.ndarray
        Each column is one signal (e.g. one variable over time).

    Returns
    -------
    numpy.ndarray
        Square matrix; entry ``[i, j]`` is the similarity of columns ``i`` and
        ``j``, with ``1`` on the diagonal.

    Notes
    -----
    TODO: replace the explicit loop with
    :func:`sklearn.metrics.pairwise.cosine_similarity`.
    """
    cosine_similarity = np.zeros((a.shape[1], a.shape[1]), dtype=float)
    for col_num_1 in range(a.shape[1]):
        for col_num_2 in range(0, col_num_1):
            cosine_similarity[col_num_1, col_num_2] = get_cosine_similarity(
                a[:, col_num_1], a[:, col_num_2]
            )
            cosine_similarity[col_num_2, col_num_1] = cosine_similarity[
                col_num_1, col_num_2
            ]
    np.fill_diagonal(cosine_similarity, 1)
    return cosine_similarity


def get_cosine_similarity_matrix_as_df(a: pd.DataFrame) -> pd.DataFrame:
    """:func:`get_cosine_similarity_matrix` with ``a``'s column names as labels.

    Parameters
    ----------
    a : pandas.DataFrame
        Each column is one signal.

    Returns
    -------
    pandas.DataFrame
        The similarity matrix, indexed and columned by ``a.columns``.
    """
    cosine_similarity = get_cosine_similarity_matrix(a.to_numpy())
    return pd.DataFrame(cosine_similarity, columns=a.columns, index=a.columns)
