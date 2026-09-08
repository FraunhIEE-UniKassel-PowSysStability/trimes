"""Single-bin Fourier coefficients over a sliding window.

A lightweight (numpy + pandas) alternative to :mod:`trimes.fft` for the common
power-systems task of tracking one harmonic of a periodic signal sample by
sample. For full spectra use :mod:`trimes.fft` instead.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trimes.base import (
    FrameOrSeries,
    create_pandas_series_or_frame_with_same_columns_and_index,
    resample,
)

__all__ = [
    "get_fourier_coef_rolling",
    "get_fourier_coef_real",
    "get_fourier_coef_imag",
]


def get_fourier_coef_rolling(
    ts: FrameOrSeries,
    samples_per_window: int,
    k: int = 1,
    time_windows: ArrayLike | None = None,
    pad_width: int | tuple[int, int] | None = None,
    pad_mode: str = "constant",
) -> FrameOrSeries:
    """Rolling complex Fourier coefficient of harmonic ``k``, one value per sample.

    Parameters
    ----------
    ts : pandas.Series or pandas.DataFrame
        Time series.
    samples_per_window : int
        Samples per window, i.e. per fundamental period.
    k : int, default 1
        Harmonic number.
    time_windows : array_like, optional
        Per-sample window duration (same length as ``ts``), for signals whose
        period varies. When given, each window is resampled to
        ``samples_per_window`` points. Defaults to a fixed-width window.
    pad_width : int or tuple of int, optional
        Padding applied to the leading samples that have no full window.
        Defaults to ``(index_of_first_full_window, 0)``.
    pad_mode : str, default "constant"
        Padding mode, passed to :func:`numpy.pad`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Complex coefficients (RMS-scaled by ``sqrt(2)``), same type, shape and
        index as ``ts``; the leading samples are filled per ``pad_mode``.
    """
    is_series = ts.ndim == 1
    frame = ts.to_frame() if is_series else ts

    fourier_coef_real = np.empty_like(frame.to_numpy()).reshape(frame.shape[0], -1)
    fourier_coef_imag = np.empty_like(frame.to_numpy()).reshape(frame.shape[0], -1)

    time = frame.index.values
    if time_windows is None:
        index_first_window = samples_per_window
        get_samples_for_window = lambda idx, col: frame.iloc[
            idx - samples_per_window : idx, col
        ].to_numpy()
    else:
        index_first_window = int(
            np.argmax(time - time[0] > time_windows) + 1
        )  # first window where time_window fits in ts
        get_samples_for_window = lambda idx, col: resample(
            frame.iloc[:, col],
            np.linspace(
                time[idx] - time_windows[idx],
                time[idx] - time_windows[idx] / samples_per_window,
                samples_per_window,
            ),
        ).to_numpy()
    if pad_width is None:
        pad_width = (index_first_window, 0)
    for col in range(frame.shape[1]):
        fc_real_col = np.empty_like(frame.iloc[index_first_window:, col].to_numpy())
        fc_imag_col = np.empty_like(frame.iloc[index_first_window:, col].to_numpy())
        for idx in range(index_first_window, frame.shape[0]):
            samples = get_samples_for_window(idx, col)
            fc_real_col[idx - index_first_window] = np.sqrt(2) * get_fourier_coef_real(
                samples, k=k
            )
            fc_imag_col[idx - index_first_window] = np.sqrt(2) * get_fourier_coef_imag(
                samples, k=k
            )
        fourier_coef_real[:, col] = np.pad(
            fc_real_col,
            pad_width=pad_width,
            mode=pad_mode,
        )
        fourier_coef_imag[:, col] = np.pad(
            fc_imag_col,
            pad_width=pad_width,
            mode=pad_mode,
        )
    result = create_pandas_series_or_frame_with_same_columns_and_index(
        fourier_coef_real + 1j * fourier_coef_imag, frame
    )
    return result.iloc[:, 0].rename(ts.name) if is_series else result


def get_fourier_coef_real(x: ArrayLike, k: int = 1, angle: float = 0.0) -> float:
    """Real part of the ``k``-th Fourier coefficient of one period ``x``.

    ``x`` is assumed to span exactly one period of the fundamental, so the
    ``k``-th harmonic completes ``k`` full cycles over ``len(x)`` samples.

    Parameters
    ----------
    x : array_like
        One period of samples.
    k : int, default 1
        Harmonic number.
    angle : float, default 0.0
        Starting angle of the reference cosine, in radians.

    Returns
    -------
    float
    """
    n = len(x)
    cos = np.cos(np.linspace(angle, k * 2 * np.pi + angle - k * 2 * np.pi / n, n))
    return np.mean(cos * x)


def get_fourier_coef_imag(x: ArrayLike, k: int = 1, angle: float = 0.0) -> float:
    """Imaginary part of the ``k``-th Fourier coefficient of one period ``x``.

    See :func:`get_fourier_coef_real`; ``angle`` is the starting angle of the
    reference sine.
    """
    n = len(x)
    sin = -np.sin(np.linspace(angle, k * 2 * np.pi + angle - k * 2 * np.pi / n, n))
    return np.mean(sin * x)
