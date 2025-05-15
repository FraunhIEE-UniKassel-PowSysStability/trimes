from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from icecream import ic

from trimes.base import resample


def get_metric_time_series(
    ts: pd.Series | pd.DataFrame,
    reference: pd.Series,
    metric: Callable,
    sample_time: float | ArrayLike | None = None,
    resample_ts: bool = False,
) -> float | ArrayLike:
    """Get metric (e.g. root mean squared error) of a time series.

    Serves also as an interface to the metrics of the scikit-learn package (see sklearn.metrics module).

    Args:
        ts (pd.Series | pd.DataFrame): time series.

        reference (pd.Series): reference to compare to.

        metric (Callable): A callable as defined in sklearn.metrics, see for example sklearn.metrics.root_mean_squared_error (must have same arguments).

        sample_time (float | ArrayLike | None, optional): Sample time of ts. Defaults to None (will be calculated from 'ts.index').

        resample_ts (bool, optional): If True, 'ts' wil be resampled according to 'reference'. Defaults to False (sampling time of 'ts' and 'reference' must be equal).

    Returns:
        float | ArrayLike: results for metric (for each column if 'ts' is a DataFrame)
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
    """Get metric from time series data in numpy format.

    Serves also as an interface to the metrics of the scikit-learn package (see sklearn.metrics module).

    Args:
        ts (ArrayLike): time series

        reference (ArrayLike): reference to compare to.

        metric (Callable): A callable as defined in sklearn.metrics, see for example sklearn.metrics.root_mean_squared_error (must have same arguments).

        sample_weight (ArrayLike | None, optional): Sample weights (sampling time for time series). Defaults to None.

    Returns:
        float | ArrayLike: metric result
    """
    if ts.ndim > 1:
        multioutput = "raw_values"
        reference = np.broadcast_to(reference, (ts.ndim, len(sample_weight))).T
    else:
        multioutput = "uniform_average"
    # return None
    return metric(
        reference, ts, sample_weight=sample_weight, multioutput=multioutput, **kwargs
    )


def integral_abs_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
    multioutput="uniform_average",
) -> float | ArrayLike:
    """Integral absolute error (area).

    Metric similar to metrics defined in the scikit-learn package (see for example sklearn.metrics.root_mean_squared_error).

    Args:
        y_true (ArrayLike): true values

        y_pred (ArrayLike): predicted values

        sample_weight (_type_, optional): Sample weights. Defaults to None.

        multioutput (str, optional): See scikit-learn package. Defaults to "uniform_average".

    Returns:
        float | ArrayLike: result
    """
    weighted_error = _weighted_error(y_true, y_pred, sample_weight)
    integrated_error = np.trapezoid(np.abs(weighted_error), axis=0)
    return _handle_sklearn_multioutput(multioutput, integrated_error)


def integral_squared_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
) -> float | ArrayLike:
    """Integral squared error.

    Metric similar to metrics defined in the scikit-learn package (see for example sklearn.metrics.root_mean_squared_error).

    Args:
        y_true (ArrayLike): true values

        y_pred (ArrayLike): predicted values

        sample_weight (_type_, optional): Sample weights. Defaults to None.

        multioutput (str, optional): See scikit-learn package. Defaults to "uniform_average".

    Returns:
        float | ArrayLike: result
    """
    weighted_error = _weighted_error(y_true, y_pred, sample_weight)
    integrated_error = np.trapezoid(np.square(weighted_error), axis=0)
    return _handle_sklearn_multioutput(multioutput, integrated_error)


def _weighted_error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike
) -> ArrayLike:
    """Weighte error (difference) between 'y_pred' and 'y_true'.

    Args:
        y_true (ArrayLike): dimension (n,)
        y_pred (ArrayLike): dimension (n,) or (n,m)
        sample_weight (ArrayLike): dimension (n,)

    Returns:
        ArrayLike: dimension (n,) or (n,m)
    """
    return ((y_pred - y_true).T * sample_weight).T


def _handle_sklearn_multioutput(
    multioutput: str | None | ArrayLike, metric_result: ArrayLike
):
    """See scikit-learn package and its metrics module (e.g. sklearn.metrics.root_mean_squared_error)."""
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return metric_result
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(metric_result, weights=multioutput)


def get_cosine_similarity(a: np.array, b: np.array) -> float:
    """Get cosine similarity of vectors a and b.

    Cosine similarity is a measure of similarity that does not depend on the magnitudes of the vectors, but only on their angle.
    Vectors a and b must be of same length (for example results of two variables of a time domain simulation).

    Args:
        a (np.array): vector
        b (np.array): vector

    Returns:
        float: cosine similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_cosine_similarity_matrix(a: np.array) -> np.array:
    """Get cosine similarity matrix of columns in a.

    Returns a symmetric matrix with cosine similarity values between the columns in a (e.g. indices [x,y] are the similarity between columns x and y).

    Args:
        a (np.array): matrix to calculate similarity of columns (e.g. each column is a result of a variable in time domain)

    Returns:
        np.array: similarity matrix

    TODO: use scikitlearn instead
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
    """Get cosine similarity of columns in a pandas DataFrame.

    See also get_cosine_similarity_matrix. This method adds columns (headers) and indices to the similarity matrix using the columns of a.

    Args:
        a (pd.DataFrame): DataFrame with matrix to calculate similarity of columns (e.g. each column is a result of a variable in time domain)

    Returns:
        pd.DataFrame: DataFrame with similarity matrix and columns/indices from a
    """
    cosine_similarity = get_cosine_similarity_matrix(a.to_numpy())
    return pd.DataFrame(cosine_similarity, columns=a.columns, index=a.columns)
