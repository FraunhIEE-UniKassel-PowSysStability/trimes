from typing import Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from icecream import ic

from trimes.base import (
    create_pandas_series_or_frame_with_same_columns_and_index,
    resample,
)


def get_fourier_coef_rolling(
    ts: pd.DataFrame | pd.Series,
    samples_per_window: int,
    k: int = 1,
    time_windows: None | ArrayLike = None,
    pad_width: int | tuple[int, int] | None = None,
    pad_mode: str = "constant",
) -> tuple[ArrayLike, ArrayLike]:
    """Get rolling fourier coefficients for each sample of a time series.

    Args:
        ts (pd.DataFrame | pd.Series): time series
        samples_per_window (int): number of samples per window (e.g. number of samples of one period)
        k (int, optional): harmonic number. Defaults to 1.
        time_windows (None | ArrayLike, optional): If specified, the time period of the rolling windows is adapted. Same length as rows in 'ts'. Defaults to None.
        pad_mode (str, optional): pad mode (see numpy docs). Defaults to "constant".
        pad_width (int | tuple | None, optional): width for padding (see numpy docs). Defaults to None.

    Returns:
        tuple[ArrayLike, ArrayLike]: Fourier coefficients (real and imaginary) for each sample of the time series (values of first window are extended)
    """
    ts_ndim = ts.ndim
    if ts_ndim == 1:  # work with 2D array
        num_col = 1
        ts = ts.to_frame()
    else:
        num_col = ts.shape[1]

    fourier_coef_real = np.empty_like(ts.to_numpy()).reshape(ts.shape[0], -1)
    fourier_coef_imag = np.empty_like(ts.to_numpy()).reshape(ts.shape[0], -1)

    time = ts.index.values
    if time_windows is None:
        index_first_window = samples_per_window
        get_samples_for_window = lambda idx, col: ts.iloc[
            idx - samples_per_window : idx, col
        ].to_numpy()
    else:
        index_first_window = int(
            np.argmax(time - time[0] > time_windows) + 1
        )  # first window where time_window fits in ts
        get_samples_for_window = lambda idx, col: resample(
            ts.iloc[:, col],
            np.linspace(time[idx] - time_windows[idx], time[idx], samples_per_window),
        ).to_numpy()
    if pad_width is None:
        pad_width = (index_first_window, 0)
    for col in range(num_col):
        fc_real_col = np.empty_like(ts.iloc[index_first_window:, col].to_numpy())
        fc_imag_col = np.empty_like(ts.iloc[index_first_window:, col].to_numpy())
        for idx in range(index_first_window, ts.shape[0]):
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
    return create_pandas_series_or_frame_with_same_columns_and_index(
        fourier_coef_real + 1j * fourier_coef_imag, ts
    )


def get_fourier_coef_real(x: ArrayLike, k: int = 1, angle: float = 0.0) -> float:
    """Get the real part of the Fourier coefficient.

    Args:
        x (ArrayLike): samples
        k (int, optional): harmonic number. Defaults to 1.
        angle (float, optional): starting angle of cosine. Defaults to 0.0.

    Returns:
        float: Real part of Fourier coefficient
    """
    cos = np.cos(
        np.linspace(angle, k * 2 * np.pi + angle - 2 * np.pi / len(x), k * len(x))
    )
    return np.mean(cos * x)


def get_fourier_coef_imag(x: ArrayLike, k: int = 1, angle: float = 0.0) -> float:
    """Get the imaginary part of the Fourier coefficient.

    Args:
        x (ArrayLike): samples
        k (int, optional): harmonic number. Defaults to 1.
        angle (float, optional): starting angle of sine. Defaults to 0.0.

    Returns:
        float: Imaginary part of Fourier coefficient
    """
    sin = -np.sin(
        np.linspace(angle, k * 2 * np.pi + angle - 2 * np.pi / len(x), k * len(x))
    )
    return np.mean(sin * x)
