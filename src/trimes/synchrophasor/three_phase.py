"""Phase 3: three-phase expansion with symmetrical-component content.

:func:`three_phase` turns a single-phase positive-sequence
:class:`~trimes.analytic.AnalyticSignal` into an ``abc`` set with
optional negative- and zero-sequence content, per-phase harmonics and
per-phase noise. The amplitude-invariant (Fortescue / Clarke) convention is
used: for a balanced set each phase peak equals ``|V+|``, so the returned
positive-sequence truth is unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from trimes.analytic import AnalyticSignal, TWO_PI

__all__ = ["three_phase"]

_SHIFT = TWO_PI / 3.0
# phase -> its 120-degree offset sign for the positive sequence
_PHASE_OFFSET = {"a": 0, "b": -1, "c": 1}


def three_phase(
    positive_seq_signal: AnalyticSignal,
    *,
    negative: float = 0.0,
    zero: float = 0.0,
    harmonics: dict | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[dict, AnalyticSignal]:
    """Expand a positive-sequence signal into a three-phase set.

    Parameters
    ----------
    positive_seq_signal : AnalyticSignal
        The positive-sequence dynamics; its ``amplitude`` and ``total_phase``
        define the fundamental of every phase.
    negative, zero : float, default 0.0
        Negative- and zero-sequence content as a fraction of the
        positive-sequence amplitude.
    harmonics : dict, optional
        ``{order: relative_amplitude}`` added to every phase (rotating at
        ``order`` times the fundamental, keeping the phase's 120-degree offset).
    noise : float, default 0.0
        Per-phase white-noise standard deviation as a fraction of the peak
        amplitude; each phase gets an independent realisation
        (``seed``, ``seed + 1``, ``seed + 2``).
    seed : int, default 0
        Base RNG seed.

    Returns
    -------
    abc : dict of pandas.Series
        ``{"a": ..., "b": ..., "c": ...}`` sampled phase signals, sharing the
        input's time index.
    truth : AnalyticSignal
        The positive-sequence truth (the input, with ``metadata`` extended by
        ``negative_sequence`` and ``zero_sequence``).
    """
    asig = positive_seq_signal
    t = asig.time
    psi = np.asarray(asig.total_phase_fn(t), dtype=float)
    amp = np.broadcast_to(
        np.asarray(asig.amplitude_fn(t), dtype=float), t.shape
    ).astype(float)
    peak = float(np.max(np.abs(amp))) if amp.size else 1.0

    abc: dict[str, pd.Series] = {}
    for offset, (label, k) in enumerate(_PHASE_OFFSET.items()):
        x = amp * np.cos(psi + k * _SHIFT)
        if negative:
            x = x + float(negative) * amp * np.cos(psi - k * _SHIFT)
        if zero:
            x = x + float(zero) * amp * np.cos(psi)
        for order, relative in (harmonics or {}).items():
            x = x + float(relative) * amp * np.cos(order * (psi + k * _SHIFT))
        if noise:
            rng = np.random.default_rng(seed + offset)
            x = x + rng.normal(0.0, float(noise) * peak, t.size)
        abc[label] = pd.Series(x, index=asig.signal.index, name=label)

    truth = dataclasses.replace(
        asig,
        metadata={
            **asig.metadata,
            "negative_sequence": float(negative),
            "zero_sequence": float(zero),
        },
    )
    return abc, truth
