from collections.abc import Callable
import operator
from functools import partial

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from icecream import ic
import sklearn.metrics


from trimes.base import resample, get_between, get_duration
from trimes.metrics import get_metric_time_series, integral_abs_error


def subtract(
    ts: pd.Series,
    reference: pd.Series | pd.DataFrame,
    extend: bool = False,
    resample_ts: bool = False,
) -> pd.Series | pd.DataFrame:
    """Difference between 'ts' and 'reference' (subtract 'reference' from 'ts').

    Args:
        ts (pd.Series | pd.DataFrame): time series

        reference (pd.Series): reference time series

        extend (bool, optional): extend the result beyond the boundaries of 'ts' if the boundaries of 'reference' are beyond 'ts' (which is the default behavior of the 'sub' method of pandas but might not be desired). Defaults to False (i.e. the boundaries of 'reference' - and therefore also of the returned time series - are limited to the boundaries of 'ts').

        resample_ts (bool): If True, 'ts' is resampled with the index of 'reference'. If False, the sampling time of 'ts' and 'reference' must be equal. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: difference
    """
    if not extend:
        t_start = ts.index.values[0]
        t_end = ts.index.values[-1]
        cutting_ts_is_required = False
        if reference.index.values[0] < t_start or reference.index.values[-1] > t_end:
            cutting_ts_is_required = True
        if cutting_ts_is_required:
            reference = get_between(reference, t_start, t_end)
    if resample_ts:
        ts = resample(ts, reference.index.values)
    return ts.sub(reference, axis=0)


def sklearn_metric(
    ts: pd.Series | pd.DataFrame,
    reference: pd.Series,
    fun: Callable,
    resample_ts: bool = False,
    sample_time: float | ArrayLike | None = None,
    **kwargs,
) -> np.array:
    """Wrapper for scikit-learn metrics for time series.

    Apply any metric provided by the callable 'fun' to 'reference' (desired time series) and 'ts' (actual time series).

    The function allows to use metrics from scikit-learn (see scikit-learn.org/1.5/api/sklearn.metrics.html) to be used for time series (see example below).

    Args:
        reference (pd.Series): time series

        ts (pd.Series | pd.DataFrame): time series

        fun (Callable): Callable to calculate the metric (e.g. from sklearn.metrics module). Input arguments are 'reference' ('y_true' in sklearn), 'ts' ('y_pred' in sklearn). Keyword argument is 'sample_weight'.

        resample_ts (bool, optional): If True, 'ts' is resampled with the index of 'reference'. If False, the sampling time of 'ts' and 'reference' must be equal. Defaults to False.

    Returns:
        np.array: metric

    Examples:
        - Get mean absolute percentage error of DataFrame 'df_ts' using the reference 'series_reference':

            from sklearn.metrics import mean_absolute_percentage_error
            metric(series_reference, df_ts, mean_absolute_percentage_error)


    """
    if resample_ts:
        ts = resample(ts, reference.index.values)
    if sample_time is None:
        sample_weight = np.diff(reference.index.values)
    elif np.isscalar(sample_time):
        sample_weight = None
    else:
        sample_weight = sample_time
    time_span = get_duration(reference)
    # ic(sample_weight)
    ic(time_span)

    if isinstance(ts, pd.Series):
        return (
            fun(
                reference.to_numpy()[1:],
                ts.to_numpy()[1:],
                sample_weight=sample_weight,
                **kwargs,
            )
            * time_span
        )
    else:
        ref_np = reference.to_numpy()
        return np.array(
            [
                fun(
                    ref_np[1:],
                    ts.to_numpy(),
                    sample_weight=sample_weight,
                    **kwargs,
                )
                * time_span
            ]
        )


def apply_operator_series(
    ts: pd.Series, reference: pd.Series, operator: Callable, resample_ts: bool = False
) -> np.array:
    """Apply operator (e.g. gretaer than/less than) to series.

    Args:
        ts (pd.Series): time series

        reference (pd.Series): reference to compare to

        operator (Callable): operator (e.g. from built-in operator module)

        resample_ts (bool, optional): resample_ts (bool, optional): If True, 'ts' wil be resampled according to 'reference'. Defaults to False (sampling time of 'ts' and 'reference' must be equal).

    Returns:
        np.array: boolean
    """
    if resample_ts:
        ts = resample(ts, reference.index.values)
    return operator(ts.to_numpy(), reference.to_numpy())


def apply_operator_df(
    ts: pd.DataFrame,
    reference: pd.Series,
    operator: Callable,
    resample_ts: bool = False,
) -> np.array:
    """Apply operator (e.g. gretaer than/less than) to DataFrame. See 'apply_operator_series'."""
    if resample_ts:
        ts = resample(ts, reference.index.values)
    reference_np = reference.to_numpy()
    ts_np = ts.to_numpy()
    return np.transpose([operator(col, reference_np) for col in ts_np.T])


def greater_than_series(
    ts: pd.Series, reference: pd.Series, resample_ts: bool = False
) -> np.array:
    return apply_operator_series(ts, reference, resample_ts)


def less_than_series(
    ts: pd.Series, reference: pd.Series, resample_ts: bool = False
) -> np.array:
    return apply_operator_series(ts, reference, resample_ts)


def comparison_series(
    ts: pd.Series,
    reference: pd.Series,
    operator: Callable,
    metric: Callable = integral_abs_error,
    sample_time: float | ArrayLike | None = None,
    resample_ts: bool = False,
) -> np.float64:
    """Compare time series to a reference (and calculate error metric).

    Args:
        ts (pd.Series | pd.DataFrame): time series.

        reference (pd.Series): reference to compare to.


        operator (Callable): _description_

        metric (Callable): A callable as defined in sklearn.metrics, see for example sklearn.metrics.root_mean_squared_error (must have same arguments).

        sample_time (float | ArrayLike | None, optional): Sample time of ts. Defaults to None (will be calculated from 'ts.index').

        resample_ts (bool, optional): If True, 'ts' wil be resampled according to 'reference'. Defaults to False (sampling time of 'ts' and 'reference' must be equal).

    Returns:
        np.float64: results of metric
    """
    if resample_ts:
        ts = resample(ts, reference.index.values)
    mask = apply_operator_series(ts, reference, operator, resample_ts=False)
    if np.any(mask):
        ts_diff = reference.copy()
        ts_diff[mask] = ts[mask]
        return get_metric_time_series(
            ts_diff, reference, metric, sample_time=sample_time
        )
    else:
        return 0.0


def comparison_df(
    ts: pd.DataFrame,
    reference: pd.Series,
    operator: Callable,
    metric: Callable = integral_abs_error,
    resample_ts: bool = False,
    sample_time: float | ArrayLike | None = None,
) -> ArrayLike:
    """Compare time series to a reference (and calculate error metric). See 'comparison_series'."""
    if resample_ts:
        ts = resample(ts, reference.index.values)
    masks = apply_operator_df(ts, reference, operator, resample_ts=False)
    errors = np.zeros(masks.shape[1])
    for col, mask in enumerate(masks.T):
        if np.any(mask):
            ts_diff = reference.copy()
            ts_diff[mask] = ts.iloc[:, col][mask]
            errors[col] = get_metric_time_series(
                ts_diff, reference, metric, sample_time=sample_time
            )
    return errors


def envelope_comparison_series(
    ts: pd.Series,
    envelope: pd.DataFrame,
    metric: Callable = integral_abs_error,
    resample_ts: bool = False,
) -> float:
    """'comparison_series' for a reference that is an envelope.

    Args:
        ts (pd.Series): time series
        envelope (pd.DataFrame): first column is upper boundary, second column is lower boundary.

        metric (Callable, optional): see 'comparison_series'. Defaults to integral_abs_error.

        resample_ts (bool, optional): If True, 'ts' wil be resampled according to 'reference'. Defaults to False (sampling time of 'ts' and 'reference' must be equal).

    Returns:
        float: metric result
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
    """See envelope_comparison_series (only difference: 'ts' is a DataFrame)."""
    error_upper = comparison_df(
        ts, envelope.iloc[:, 0], operator.gt, metric=metric, resample_ts=resample_ts
    )
    error_lower = comparison_df(
        ts, envelope.iloc[:, 1], operator.lt, metric=metric, resample_ts=resample_ts
    )
    return error_upper + error_lower
