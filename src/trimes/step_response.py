"""Step-response and recovery characteristics of a time series.

numpy + pandas only (unlike :mod:`trimes.control`, which wraps python-control).
Works on any :class:`pandas.Series` with a numeric (seconds) index, uniformly
sampled or not -- e.g. simulation output with extra samples at events.

Times are sample times: a level counts as reached at the first sample that
reaches it, so the resolution is the sampling interval. Characteristics that
are never reached are returned as ``numpy.inf`` (so a comparison against a
maximum fails), undefined ones as ``numpy.nan``.

Functions
---------
crossing_time   first time the series reaches a level
rise_time       time between reaching two fractions of a step
settling_time   time until the series stays inside a band around its final value
recovery_time   time until the series stays on one side of a level
overshoot       largest excursion beyond the final value, as a fraction of the step
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "crossing_time",
    "rise_time",
    "settling_time",
    "recovery_time",
    "overshoot",
]

Direction = Literal["up", "down"]


def _after(ts: pd.Series, start: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Time and values of the samples at or after ``start`` (default: first sample)."""
    t = ts.index.to_numpy(dtype=float)
    y = ts.to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("empty series")
    if start is None:
        start = float(t[0])
    keep = t >= start
    return t[keep], y[keep], float(start)


def _time_to_stay(t: np.ndarray, ok: np.ndarray, start: float) -> float:
    """Time from ``start`` after which ``ok`` holds for every remaining sample."""
    if t.size == 0:
        return np.nan
    if ok.all():
        return 0.0
    last_bad = int(np.flatnonzero(~ok)[-1])
    if last_bad == t.size - 1:
        return np.inf
    return float(t[last_bad + 1] - start)


def crossing_time(
    ts: pd.Series,
    level: float,
    *,
    start: float | None = None,
    direction: Direction = "up",
) -> float:
    """First sample time at or after ``start`` at which ``ts`` has reached ``level``.

    Parameters
    ----------
    ts : pandas.Series
        Time series with a numeric index in seconds.
    level : float
        Level to reach.
    start : float, optional
        Ignore samples before this time. Default: the first sample.
    direction : {'up', 'down'}, default 'up'
        ``'up'``: first sample ``>= level``; ``'down'``: first sample ``<= level``.

    Returns
    -------
    float
        Absolute time of the crossing (not relative to ``start``); ``inf`` if
        the level is never reached.
    """
    t, y, _ = _after(ts, start)
    reached = y >= level if direction == "up" else y <= level
    return float(t[np.argmax(reached)]) if reached.any() else np.inf


def rise_time(
    ts: pd.Series,
    *,
    initial: float,
    final: float,
    limits: tuple[float, float] = (0.1, 0.9),
    start: float | None = None,
) -> float:
    """Time between reaching ``limits[0]`` and ``limits[1]`` of the step
    ``initial -> final``.

    With ``limits[0] == 0`` the rise time is measured from ``start`` (the step
    instant), e.g. ``limits=(0, 0.9)`` for "time to reach 90 % of the step".

    Parameters
    ----------
    ts : pandas.Series
        The step response.
    initial, final : float
        Values before and after the step.
    limits : (float, float), default (0.1, 0.9)
        Fractions of the step, ``0 <= limits[0] < limits[1]``.
    start : float, optional
        Step instant; samples before it are ignored. Default: the first sample.

    Returns
    -------
    float
        Rise time in seconds; ``inf`` if ``limits[1]`` is never reached,
        ``nan`` if the step is zero.
    """
    lo, hi = limits
    if not 0.0 <= lo < hi:
        raise ValueError(f"limits must satisfy 0 <= lo < hi, got {limits}")
    step = final - initial
    if step == 0.0:
        return np.nan
    _, _, t0 = _after(ts, start)
    direction: Direction = "up" if step > 0 else "down"
    t_lo = t0 if lo == 0.0 else crossing_time(ts, initial + lo * step, start=t0, direction=direction)
    t_hi = crossing_time(ts, initial + hi * step, start=t0, direction=direction)
    if np.isinf(t_hi):
        return np.inf
    return float(t_hi - t_lo)


def settling_time(
    ts: pd.Series,
    *,
    final: float,
    band: float,
    start: float | None = None,
) -> float:
    """Time from ``start`` until ``ts`` stays inside ``final ± band``.

    Parameters
    ----------
    ts : pandas.Series
        The response.
    final : float
        Value the response settles to.
    band : float
        Half-width of the settling band, absolute (in the unit of ``ts``).
    start : float, optional
        Step instant; samples before it are ignored. Default: the first sample.

    Returns
    -------
    float
        Time from ``start`` to the first sample after the last one outside the
        band; ``0`` if no sample is outside, ``inf`` if the last one is.
    """
    if band < 0.0:
        raise ValueError("band must be >= 0")
    t, y, t0 = _after(ts, start)
    return _time_to_stay(t, np.abs(y - final) <= band, t0)


def recovery_time(
    ts: pd.Series,
    level: float,
    *,
    start: float | None = None,
    direction: Direction = "up",
) -> float:
    """Time from ``start`` until ``ts`` stays at or above (``'up'``) or at or
    below (``'down'``) ``level``.

    E.g. active-power recovery after a fault: the time after clearing from
    which power stays above 90 % of its pre-fault value. Brief excursions back
    across the level restart the clock, so a spike that touches the level does
    not count as recovery.

    Returns
    -------
    float
        Time from ``start`` to the first sample after the last one on the
        wrong side; ``0`` if no sample is, ``inf`` if the last one is.
    """
    t, y, t0 = _after(ts, start)
    ok = y >= level if direction == "up" else y <= level
    return _time_to_stay(t, ok, t0)


def overshoot(
    ts: pd.Series,
    *,
    initial: float,
    final: float,
    start: float | None = None,
) -> float:
    """Largest excursion beyond ``final`` in the direction of the step, as a
    fraction of ``|final - initial|``.

    Returns
    -------
    float
        ``0`` if the response never passes ``final``; ``nan`` if the step is zero.
    """
    step = final - initial
    if step == 0.0:
        return np.nan
    _, y, _ = _after(ts, start)
    excursion = np.max(np.sign(step) * (y - final), initial=0.0)
    return float(max(excursion, 0.0) / abs(step))
