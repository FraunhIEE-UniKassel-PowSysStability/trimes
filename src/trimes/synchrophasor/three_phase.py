"""Phase 3: three-phase expansion with symmetrical-component content.

:func:`three_phase` turns a single-phase positive-sequence
:class:`~trimes.analytic.AnalyticSignal` into an ``abc`` set with
optional negative- and zero-sequence content, per-phase harmonics,
interharmonic tones at arbitrary frequencies and per-phase noise. The amplitude-invariant (Fortescue / Clarke) convention is
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


def _profile(value, t: np.ndarray) -> np.ndarray:
    """Broadcast a scalar, per-sample array or callable of the time grid to ``t``."""
    evaluated = value(t) if callable(value) else value
    return np.broadcast_to(np.asarray(evaluated, dtype=float), t.shape)


# phase -> its 120-degree offset sign for the positive sequence
_PHASE_OFFSET = {"a": 0, "b": -1, "c": 1}


def _scalar_or_label(value):
    """A float for a constant profile, the string ``"time-varying"`` otherwise."""
    if callable(value) or np.ndim(value) > 0:
        return "time-varying"
    return float(value)


def three_phase(
    positive_seq_signal: AnalyticSignal,
    *,
    negative=0.0,
    zero=0.0,
    harmonics: dict | None = None,
    extra_tones: dict | None = None,
    noise: float = 0.0,
    seed: int = 0,
    dc_offset=None,
) -> tuple[dict, AnalyticSignal]:
    """Expand a positive-sequence signal into a three-phase set.

    Parameters
    ----------
    positive_seq_signal : AnalyticSignal
        The positive-sequence dynamics; its ``amplitude`` and ``total_phase``
        define the fundamental of every phase.
    negative, zero : float, array_like or callable, default 0.0
        Negative- and zero-sequence content as a fraction of the
        positive-sequence amplitude. Besides a constant, each may be an array
        with one value per sample or a callable that maps the time grid to such
        an array, which switches the content on and off during the record.
    harmonics : dict, optional
        ``{order: relative_amplitude}`` added to every phase (rotating at
        ``order`` times the fundamental, keeping the phase's 120-degree offset).
        Each amplitude accepts the same forms as ``negative``.
    extra_tones : dict, optional
        ``{frequency_hz: relative_amplitude}`` tones at arbitrary (non-integer,
        possibly out-of-band) frequencies, added as a balanced positive-sequence
        set: every phase keeps its 120-degree offset, so the tone carries no
        negative- or zero-sequence content of its own and an interharmonic test
        is not mixed with an unbalance test. The frequency is absolute and does
        not follow the fundamental, unlike ``harmonics``. Each amplitude is a
        fraction of the (time-varying) fundamental amplitude and accepts the
        same forms as ``negative``, so a tone can be switched on during the
        record. The phase of the tone is a cosine at the first sample.
    noise : float, default 0.0
        Per-phase white-noise standard deviation as a fraction of the peak
        amplitude; each phase gets an independent realisation
        (``seed``, ``seed + 1``, ``seed + 2``).
    seed : int, default 0
        Base RNG seed.
    dc_offset : float, array_like or callable, optional
        Common DC offset added to every phase, as a fraction of the peak
        positive-sequence amplitude. A scalar is constant, an array gives one
        value per sample, and a callable maps the time grid to such an array
        (a time-varying offset such as a ramp). Being identical in all three
        phases, it is zero-sequence content and leaves the positive-sequence
        truth unchanged.

    Returns
    -------
    abc : dict of pandas.Series
        ``{"a": ..., "b": ..., "c": ...}`` sampled phase signals, sharing the
        input's time index.
    truth : AnalyticSignal
        The positive-sequence truth (the input, with ``metadata`` extended by
        everything this function injected: ``negative_sequence``,
        ``zero_sequence``, ``harmonics``, ``extra_tones``, ``dc_offset`` and
        ``noise``). A profile that varies over the record is recorded as the
        string ``"time-varying"`` rather than as a number, so the keys say what
        is present even where they cannot say how much.
    """
    asig = positive_seq_signal
    t = asig.time
    psi = np.asarray(asig.total_phase_fn(t), dtype=float)
    amp = np.broadcast_to(
        np.asarray(asig.amplitude_fn(t), dtype=float), t.shape
    ).astype(float)
    peak = float(np.max(np.abs(amp))) if amp.size else 1.0

    negative_fraction = _profile(negative, t)
    zero_fraction = _profile(zero, t)
    harmonic_fractions = {
        order: _profile(relative, t) for order, relative in (harmonics or {}).items()
    }
    tone_fractions = {
        float(frequency_hz): _profile(relative, t)
        for frequency_hz, relative in (extra_tones or {}).items()
    }
    tone_phase = {
        frequency_hz: TWO_PI * frequency_hz * (t - t[0] if t.size else t)
        for frequency_hz in tone_fractions
    }

    offset = np.zeros(t.shape)
    if dc_offset is not None:
        fraction = dc_offset(t) if callable(dc_offset) else dc_offset
        fraction = np.broadcast_to(np.asarray(fraction, dtype=float), t.shape)
        offset = fraction * peak

    abc: dict[str, pd.Series] = {}
    for phase_index, (label, k) in enumerate(_PHASE_OFFSET.items()):
        x = amp * np.cos(psi + k * _SHIFT)
        if np.any(negative_fraction):
            x = x + negative_fraction * amp * np.cos(psi - k * _SHIFT)
        if np.any(zero_fraction):
            x = x + zero_fraction * amp * np.cos(psi)
        for order, relative in harmonic_fractions.items():
            x = x + relative * amp * np.cos(order * (psi + k * _SHIFT))
        for frequency_hz, relative in tone_fractions.items():
            x = x + relative * amp * np.cos(tone_phase[frequency_hz] + k * _SHIFT)
        if dc_offset is not None:
            x = x + offset
        if noise:
            rng = np.random.default_rng(seed + phase_index)
            x = x + rng.normal(0.0, float(noise) * peak, t.size)
        abc[label] = pd.Series(x, index=asig.signal.index, name=label)

    truth = dataclasses.replace(
        asig,
        metadata={
            **asig.metadata,
            "negative_sequence": _scalar_or_label(negative),
            "zero_sequence": _scalar_or_label(zero),
            # Everything this function injects is recorded, so a record carries a
            # trace of its own content. Without the four keys below, a set with
            # harmonics, interharmonic tones, a DC offset and noise in it looks
            # from its metadata exactly like a clean one, which leaves an
            # analysis of the waveform nothing to be checked against.
            "harmonics": {
                int(order): _scalar_or_label(relative)
                for order, relative in (harmonics or {}).items()
            },
            "extra_tones": {
                float(frequency_hz): _scalar_or_label(relative)
                for frequency_hz, relative in (extra_tones or {}).items()
            },
            "dc_offset": _scalar_or_label(dc_offset) if dc_offset is not None else 0.0,
            "noise": float(noise),
        },
    )
    return abc, truth
