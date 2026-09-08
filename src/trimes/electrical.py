"""Electric-power quantities from time-domain signals.

Instantaneous and window-averaged active / reactive / apparent power for
three-phase systems, plus power from symmetrical components. Inputs are the
sampled ``abc`` waveforms (as arrays or DataFrames with one column per phase),
not phasors.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trimes.base import resample
from trimes.fourier import get_fourier_coef_rolling
from trimes.signal_processing import average_rolling, average_rolling_variable_window
from trimes.transforms import abc_2_symmetrical_components

__all__ = [
    "get_apparent_power_symmetrical_components",
    "get_active_power_symmetrical_components",
    "get_reactive_power_symmetrical_components",
    "get_apparent_power_using_symmetrical_components_and_fourier_coefficients",
    "get_instantaneous_active_power_3ph",
    "get_instantaneous_reactive_power_3ph",
    "get_instantaneous_reactive_power_using_phase_shift_3ph",
    "get_instantaneous_magnitude_3ph",
    "get_active_power_average_rolling_3ph",
    "get_active_power_average_rolling_variable_window_3ph",
    "get_reactive_power_average_rolling_3ph",
    "get_reactive_power_average_rolling_variable_window_3ph",
]


def get_apparent_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
) -> complex:
    """Complex power ``factor * u * conj(i)`` from sequence components.

    Parameters
    ----------
    u, i : array_like
        Voltage and current sequence components (positive / negative / zero).
    factor : float, default 3
        Scaling factor (``3`` for a per-phase-to-total conversion).

    Returns
    -------
    complex
        Apparent power (real part active, imaginary part reactive).
    """
    return factor * u * np.conj(i)


def get_active_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
) -> float:
    """Active power from voltage and current sequence components."""
    return factor * (np.dot(np.real(u), np.real(i)) + np.dot(np.imag(u), np.imag(i)))


def get_reactive_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
) -> float:
    """Reactive power from voltage and current sequence components."""
    return factor * (-np.dot(np.real(u), np.imag(i)) + np.dot(np.imag(u), np.real(i)))


def get_apparent_power_using_symmetrical_components_and_fourier_coefficients(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    samples_per_window: int,
    time_windows: ArrayLike | None = None,
    return_symmetrical_components: bool = False,
) -> ArrayLike:
    """Rolling apparent power of the symmetrical components.

    For each window: take the fundamental Fourier coefficient of every phase,
    transform to symmetrical components, then form ``3 * U * conj(I)``.

    Parameters
    ----------
    u_abc, i_abc : pandas.DataFrame
        Three-phase voltages and currents.
    samples_per_window : int
        Samples per window (fundamental period) for the Fourier coefficients.
    time_windows : array_like, optional
        Per-sample window duration for a drifting period; ``None`` uses a fixed
        window.
    return_symmetrical_components : bool, default False
        Also return the voltage and current sequence components.

    Returns
    -------
    array_like
        The apparent power, or ``(power, u_sequence, i_sequence)`` when
        ``return_symmetrical_components`` is True.
    """
    if time_windows is not None:
        u_fourier_coef = get_fourier_coef_rolling(
            u_abc, samples_per_window, time_windows=time_windows
        )
        i_fourier_coef = get_fourier_coef_rolling(
            i_abc, samples_per_window, time_windows=time_windows
        )
    else:
        u_fourier_coef = get_fourier_coef_rolling(u_abc, samples_per_window)
        i_fourier_coef = get_fourier_coef_rolling(i_abc, samples_per_window)

    u_sym_comp = abc_2_symmetrical_components(u_fourier_coef)
    i_sym_comp = abc_2_symmetrical_components(i_fourier_coef)

    apparent_power = get_apparent_power_symmetrical_components(u_sym_comp, i_sym_comp)
    if not return_symmetrical_components:
        return apparent_power
    return apparent_power, u_sym_comp, i_sym_comp


def get_instantaneous_active_power_3ph(
    u_abc: ArrayLike, i_abc: ArrayLike
) -> ArrayLike:
    """Instantaneous three-phase active power ``u_a i_a + u_b i_b + u_c i_c``.

    Parameters
    ----------
    u_abc, i_abc : array_like
        Three-phase voltages and currents (one column per phase).

    Returns
    -------
    array_like
        Instantaneous power, one value per sample.
    """
    return np.sum(u_abc * i_abc, axis=1)


def get_instantaneous_reactive_power_3ph(
    u_abc: ArrayLike, i_abc: ArrayLike
) -> ArrayLike:
    """Instantaneous three-phase reactive power.

    Follows PowerFactory's *Technical Reference - Common Result Variables for
    Terminals and Elements (EMT Simulation)*.

    Parameters
    ----------
    u_abc, i_abc : array_like
        Three-phase voltages and currents (one column per phase).

    Returns
    -------
    array_like
        Instantaneous reactive power, one value per sample.
    """
    return (
        1
        / np.sqrt(3)
        * (
            (u_abc[:, 1] - u_abc[:, 2]) * i_abc[:, 0]
            + (u_abc[:, 2] - u_abc[:, 0]) * i_abc[:, 1]
            + (u_abc[:, 0] - u_abc[:, 1]) * i_abc[:, 2]
        )
    )


def get_instantaneous_reactive_power_using_phase_shift_3ph(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    periods: float | ArrayLike,
) -> pd.Series:
    """Instantaneous reactive power via a quarter-period current phase shift.

    Follows *The Measurement of Reactive Energy in Polluted Distribution Power
    Systems: An Analysis of the Performance of Commercial Static Meters*.

    Parameters
    ----------
    u_abc, i_abc : pandas.DataFrame
        Three-phase voltages and currents.
    periods : float or array_like
        Fundamental period in seconds -- a scalar, or one value per sample if
        the period varies.

    Returns
    -------
    pandas.Series
        Instantaneous reactive power, indexed by the shifted time.
    """
    t_i_abc = i_abc.index.values + periods / 4
    i_abc_shifted = resample(i_abc, t_i_abc)
    return pd.Series(
        np.sum(u_abc.to_numpy()[-len(t_i_abc) :] * i_abc_shifted, axis=1), index=t_i_abc
    )


def get_instantaneous_magnitude_3ph(abc: ArrayLike) -> ArrayLike:
    """Instantaneous magnitude of a three-phase signal.

    Follows IEEE Std 1459-2010, *Standard Definitions for the Measurement of
    Electric Power Quantities Under Sinusoidal, Nonsinusoidal, Balanced, or
    Unbalanced Conditions*.

    Parameters
    ----------
    abc : array_like
        Three-phase signal (one column per phase).

    Returns
    -------
    array_like
        Magnitude, one value per sample.
    """
    return np.sqrt(
        1
        / 9
        * (
            (abc[:, 0] - abc[:, 1]) ** 2
            + (abc[:, 1] - abc[:, 2]) ** 2
            + (abc[:, 2] - abc[:, 0]) ** 2
        )
    )


def get_active_power_average_rolling_3ph(
    u_abc: ArrayLike, i_abc: ArrayLike, samples_per_window: int
) -> pd.DataFrame | pd.Series:
    """Active power averaged over a fixed rolling window.

    Parameters
    ----------
    u_abc, i_abc : array_like
        Three-phase voltages and currents.
    samples_per_window : int
        Window length in samples (e.g. one fundamental period).

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The averaged active power.
    """
    p_df = pd.DataFrame(get_instantaneous_active_power_3ph(u_abc, i_abc))
    return average_rolling(p_df, samples_per_window=samples_per_window)


def get_active_power_average_rolling_variable_window_3ph(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    time_windows: ArrayLike,
    samples_per_window: int,
) -> pd.DataFrame | pd.Series:
    """Active power averaged over a rolling window of varying duration.

    Parameters
    ----------
    u_abc, i_abc : pandas.DataFrame
        Three-phase voltages and currents.
    time_windows : array_like
        Per-sample window duration (same length as the inputs).
    samples_per_window : int
        Number of points each window is resampled to.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The averaged active power.
    """
    p_df = pd.DataFrame(
        get_instantaneous_active_power_3ph(u_abc.to_numpy(), i_abc.to_numpy()),
        index=u_abc.index,
    )
    return average_rolling_variable_window(
        p_df, time_windows, samples_per_window=samples_per_window
    )


def get_reactive_power_average_rolling_3ph(
    u_abc: ArrayLike, i_abc: ArrayLike, samples_per_window: int
) -> pd.DataFrame | pd.Series:
    """Reactive power averaged over a fixed rolling window.

    See :func:`get_active_power_average_rolling_3ph` for the arguments.
    """
    p_df = pd.DataFrame(get_instantaneous_reactive_power_3ph(u_abc, i_abc))
    return average_rolling(p_df, samples_per_window=samples_per_window)


def get_reactive_power_average_rolling_variable_window_3ph(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    time_windows: ArrayLike,
    samples_per_window: int,
) -> pd.DataFrame | pd.Series:
    """Reactive power averaged over a rolling window of varying duration.

    See :func:`get_active_power_average_rolling_variable_window_3ph` for the
    arguments.
    """
    p_df = pd.DataFrame(
        get_instantaneous_reactive_power_3ph(u_abc.to_numpy(), i_abc.to_numpy()),
        index=u_abc.index,
    )
    return average_rolling_variable_window(
        p_df, time_windows, samples_per_window=samples_per_window
    )
