"""Digital filters for time series (needs scipy -- the ``dsp`` extra)."""

from functools import partial

try:
    from scipy.signal import lfilter
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.filter needs scipy; install it with: pip install 'trimes[dsp]'"
    ) from _exc

from trimes.base import FrameOrSeries, apply_to_columns

__all__ = ["pt1_filter"]


def pt1_filter(ts: FrameOrSeries, T: float, dt: float) -> FrameOrSeries:
    """First-order (PT1) low-pass filter, applied column-wise.

    Implements the discrete recursion ``y[n] = alpha * y[n-1] + (1 - alpha) * x[n]``
    with ``alpha = T / (T + dt)``.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Uniformly sampled input.
    T : float
        Filter time constant, in the same time unit as ``dt``.
    dt : float
        Sample step.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The filtered signal, same shape and index as ``ts``.
    """
    alpha = T / (T + dt)
    b = [1 - alpha]  # numerator coefficients
    a = [1, -alpha]  # denominator coefficients
    lfilter_part = partial(lfilter, b, a)
    return apply_to_columns(ts, lfilter_part)
