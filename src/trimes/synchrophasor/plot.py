"""Phase 6: synchrophasor plotting helpers.

Needs matplotlib (the ``viz`` extra); imported lazily so this module can be
imported without it, failing only when a plot is actually drawn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trimes.synchrophasor.metrics import errors
from trimes.synchrophasor.signal import AnalyticSignal

__all__ = ["plot_waveform", "plot_errors", "plot_sweep_heatmap"]

_ERROR_ROWS = (
    ("tve", "TVE [%]", "tve"),
    ("fe", "FE [Hz]", "fe"),
    ("rfe", "RFE [Hz/s]", "rfe"),
)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "trimes.synchrophasor.plot needs matplotlib; "
            "install it with: pip install 'trimes[viz]'"
        ) from exc
    return plt


def plot_waveform(signal: AnalyticSignal, *, ax=None, truth: bool = True):
    """Plot the sampled waveform, optionally with the amplitude-truth envelope."""
    plt = _pyplot()
    ax = ax or plt.gca()
    t = signal.time
    ax.plot(t, np.asarray(signal.signal.to_numpy(), dtype=float), lw=0.8, label="signal")
    if truth:
        amp = signal.amplitude(t)
        (line,) = ax.plot(t, amp, "--", lw=1.0, label="amplitude truth")
        ax.plot(t, -amp, "--", lw=1.0, color=line.get_color())
    for boundary in signal.transition_times:
        ax.axvline(boundary, color="grey", lw=0.8, alpha=0.6)
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right")
    return ax


def plot_errors(
    trace: pd.DataFrame,
    truth: AnalyticSignal,
    *,
    nominal: float | None = None,
    limits: dict | None = None,
    exclude: float | None = None,
    axes=None,
):
    """Plot ``|TVE|`` / ``|FE|`` / ``|RFE|`` over time with limit lines and
    exclusion shading.

    Parameters
    ----------
    limits : dict, optional
        ``{"tve": ..., "fe": ..., "rfe": ...}`` -- drawn as horizontal dashed
        lines. Defaults to ``truth.metadata["limits"]`` if present.
    exclude : float, optional
        Half-width in seconds of shading around each transition. If the signal
        carries ``metadata["exclusion"]`` those explicit windows are shaded
        instead.
    """
    plt = _pyplot()
    frame = errors(trace, truth, nominal=nominal)
    if limits is None:
        limits = truth.metadata.get("limits") or {}

    if axes is None:
        _, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axes = np.atleast_1d(axes)

    windows = truth.metadata.get("exclusion")
    if windows is None and exclude:
        windows = [(t - exclude, t + exclude) for t in truth.transition_times]

    for ax, (col, ylabel, lim_key) in zip(axes, _ERROR_ROWS):
        ax.plot(frame.index, frame[col].abs())
        if limits.get(lim_key) is not None:
            ax.axhline(float(limits[lim_key]), ls="--", color="tab:red", label="limit")
        for lo, hi in windows or []:
            ax.axvspan(lo, hi, color="grey", alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("time [s]")
    return axes


def plot_sweep_heatmap(table: pd.DataFrame, *, ax=None, cbar_label="metric"):
    """Heatmap of a ``parameter x method`` results table (rows x columns)."""
    plt = _pyplot()
    ax = ax or plt.gca()
    data = np.asarray(table.to_numpy(), dtype=float)
    im = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index])
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    return ax
