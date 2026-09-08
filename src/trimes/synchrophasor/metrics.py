"""Phase 5: synchrophasor evaluation metrics.

The error functions take an estimator ``trace`` -- a DataFrame with a time
index and at least a ``phasor`` column (``frequency`` and ``rocof`` optional)
-- and an :class:`~trimes.synchrophasor.signal.AnalyticSignal`, evaluate the
signal's truth at the trace's own reporting instants, and compare.

numpy + pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trimes.synchrophasor.signal import AnalyticSignal
from trimes.synchrophasor.util import wrap_pi

__all__ = [
    "ERROR_COLUMNS",
    "errors",
    "tve",
    "fe",
    "rfe",
    "summary",
    "response_time",
    "delay_time",
    "overshoot",
    "compliance",
]

ERROR_COLUMNS = ["tve", "fe", "rfe", "amplitude_error", "phase_error_deg"]


def _truth_at(trace: pd.DataFrame, truth: AnalyticSignal, nominal) -> pd.DataFrame:
    times = trace.index.to_numpy(dtype=float)
    return truth.ground_truth(times, nominal)


def errors(
    trace: pd.DataFrame, truth: AnalyticSignal, *, nominal: float | None = None
) -> pd.DataFrame:
    """Per-frame errors of ``trace`` against ``truth``.

    Returns a DataFrame with the same index as ``trace`` and columns
    :data:`ERROR_COLUMNS`: ``tve`` (percent), ``fe`` (Hz), ``rfe`` (Hz/s),
    ``amplitude_error`` (signal units), ``phase_error_deg`` (wrapped degrees).
    Columns the trace does not provide are ``nan``. ``tve`` is ``nan`` where the
    truth phasor magnitude is zero.
    """
    gt = _truth_at(trace, truth, nominal)
    n = len(trace)
    nan = np.full(n, np.nan)

    truth_phasor = gt["phasor"].to_numpy()
    if "phasor" in trace:
        with np.errstate(divide="ignore", invalid="ignore"):
            tve_col = 100.0 * np.abs(
                trace["phasor"].to_numpy() - truth_phasor
            ) / np.abs(truth_phasor)
    else:
        tve_col = nan

    fe_col = (
        np.abs(trace["frequency"].to_numpy() - gt["frequency"].to_numpy())
        if "frequency" in trace
        else nan
    )
    rfe_col = (
        np.abs(trace["rocof"].to_numpy() - gt["rocof"].to_numpy())
        if "rocof" in trace
        else nan
    )
    amp_col = (
        trace["amplitude"].to_numpy() - gt["amplitude"].to_numpy()
        if "amplitude" in trace
        else nan
    )
    if "phase" in trace:
        phase_col = np.degrees(
            wrap_pi(trace["phase"].to_numpy() - gt["phase"].to_numpy())
        )
    else:
        phase_col = nan

    return pd.DataFrame(
        {
            "tve": tve_col,
            "fe": fe_col,
            "rfe": rfe_col,
            "amplitude_error": amp_col,
            "phase_error_deg": phase_col,
        },
        index=trace.index,
    )


def tve(trace, truth, *, nominal=None) -> pd.Series:
    """Total vector error per frame, in percent."""
    return errors(trace, truth, nominal=nominal)["tve"]


def fe(trace, truth, *, nominal=None) -> pd.Series:
    """Frequency error per frame, in Hz."""
    return errors(trace, truth, nominal=nominal)["fe"]


def rfe(trace, truth, *, nominal=None) -> pd.Series:
    """ROCOF error per frame, in Hz/s."""
    return errors(trace, truth, nominal=nominal)["rfe"]


def _transition_windows(truth: AnalyticSignal, half_width: float):
    """(start, stop) windows to exclude: ``metadata['exclusion']`` if present,
    otherwise ``+- half_width`` around each transition time."""
    explicit = truth.metadata.get("exclusion")
    if explicit:
        return [tuple(map(float, w)) for w in explicit]
    return [(t - half_width, t + half_width) for t in truth.transition_times]


def _keep_mask(times: np.ndarray, windows) -> np.ndarray:
    keep = np.ones(times.shape, dtype=bool)
    for lo, hi in windows:
        keep &= ~((times >= lo) & (times <= hi))
    return keep


def summary(
    trace: pd.DataFrame,
    truth: AnalyticSignal,
    *,
    nominal: float | None = None,
    exclude_around_transitions: float = 0.0,
) -> dict:
    """Aggregate per-frame errors into scalar max / rms metrics.

    Parameters
    ----------
    exclude_around_transitions : float, default 0.0
        Half-width in seconds of the window dropped around each transition. If
        the signal carries ``metadata["exclusion"]`` (e.g. a frequency ramp),
        those explicit windows are used instead and this value is ignored.

    Returns
    -------
    dict
        ``max_tve`` / ``rms_tve`` (percent), ``max_fe`` / ``rms_fe`` (Hz),
        ``max_rfe`` / ``rms_rfe`` (Hz/s), ``frames`` (kept) and
        ``excluded_frames``.
    """
    frame = errors(trace, truth, nominal=nominal)
    times = frame.index.to_numpy(dtype=float)
    windows = _transition_windows(truth, exclude_around_transitions)
    keep = _keep_mask(times, windows) if windows else np.ones(times.shape, bool)
    kept = frame[keep]

    def _max(col):
        v = kept[col].to_numpy()
        v = v[np.isfinite(v)]
        return float(np.max(np.abs(v))) if v.size else np.nan

    def _rms(col):
        v = kept[col].to_numpy()
        v = v[np.isfinite(v)]
        return float(np.sqrt(np.mean(v**2))) if v.size else np.nan

    return {
        "max_tve": _max("tve"),
        "rms_tve": _rms("tve"),
        "max_fe": _max("fe"),
        "rms_fe": _rms("fe"),
        "max_rfe": _max("rfe"),
        "rms_rfe": _rms("rfe"),
        "frames": int(keep.sum()),
        "excluded_frames": int((~keep).sum()),
    }


def response_time(
    trace: pd.DataFrame,
    truth: AnalyticSignal,
    *,
    band: float = 1.0,
    quantity: str = "tve",
    nominal: float | None = None,
) -> float:
    """Seconds from the first to the last frame whose ``quantity`` error
    exceeds ``band`` (plus one reporting interval).

    With ``quantity="tve"`` and ``band=1.0`` this is the standard's response
    time for a step change. Returns ``0.0`` if the band is never exceeded.
    """
    frame = errors(trace, truth, nominal=nominal)
    over = frame.index[frame[quantity].abs() > band]
    if len(over) == 0:
        return 0.0
    times = frame.index.to_numpy(dtype=float)
    interval = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    return float(over[-1] - over[0] + interval)


def _step_endpoints(trace, truth, quantity, nominal):
    times = trace.index.to_numpy(dtype=float)
    gt = truth.ground_truth(times, nominal)[quantity].to_numpy()
    values = trace[quantity].to_numpy()
    return times, values, float(gt[0]), float(gt[-1])


def overshoot(
    trace: pd.DataFrame,
    truth: AnalyticSignal,
    *,
    quantity: str = "amplitude",
    nominal: float | None = None,
) -> float:
    """Peak excursion past the final truth value, relative to the step size.

    ``nan`` if the step in ``quantity`` is ~0.
    """
    _, values, initial, final = _step_endpoints(trace, truth, quantity, nominal)
    size = abs(final - initial)
    if size < 1e-12:
        return float("nan")
    if final >= initial:
        return float((np.nanmax(values) - final) / size)
    return float((final - np.nanmin(values)) / size)


def delay_time(
    trace: pd.DataFrame,
    truth: AnalyticSignal,
    *,
    quantity: str = "amplitude",
    nominal: float | None = None,
) -> float:
    """Time from the step instant to the trace first reaching the step midpoint.

    The step instant is ``truth.metadata["step_time"]`` (or the first
    transition time). ``nan`` if there is no step or the midpoint is never
    reached.
    """
    step_time = truth.metadata.get("step_time")
    if step_time is None:
        transitions = truth.transition_times
        if not transitions:
            return float("nan")
        step_time = transitions[0]
    step_time = float(step_time)

    times, values, initial, final = _step_endpoints(trace, truth, quantity, nominal)
    midpoint = 0.5 * (initial + final)
    after = times >= step_time
    if not np.any(after):
        return float("nan")
    reached = np.where(
        (values[after] - midpoint) * np.sign(final - initial) >= 0.0
    )[0]
    if reached.size == 0:
        return float("nan")
    return float(times[after][reached[0]] - step_time)


def compliance(summary: dict, test_metadata: dict) -> pd.DataFrame:
    """Pass / fail table of a :func:`summary` against a condition's limits.

    Parameters
    ----------
    summary : dict
        Output of :func:`summary`.
    test_metadata : dict
        A condition signal's ``metadata`` (needs a ``limits`` sub-dict, as
        stamped by :mod:`trimes.synchrophasor.conditions`).

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``value``, ``limit``, ``pass``; one row per limit
        that applies. Empty if the class is exempt (no limits).
    """
    limits = test_metadata.get("limits") or {}
    pairs = {"tve": "max_tve", "fe": "max_fe", "rfe": "max_rfe"}
    rows = []
    for metric, key in pairs.items():
        if limits.get(metric) is None:
            continue
        value = summary.get(key, np.nan)
        limit = float(limits[metric])
        rows.append(
            {
                "metric": metric,
                "value": value,
                "limit": limit,
                "pass": bool(np.isfinite(value) and value <= limit),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "value", "limit", "pass"])
