"""
Calculate quantities of electric power systems.
"""

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from icecream import ic

from trimes.root_mean_square import rms_rolling, rms_rolling_variable_window
from trimes.base import resample
from trimes.fourier import (
    get_fourier_coef_rolling,
)
from trimes.transforms import abc_2_symmetrical_components
from trimes.signal_processing import (
    average_rolling_variable_window,
    average_rolling,
)


def get_apparent_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
) -> complex:
    """Get apparent power from symmetrical components of voltages and currents.

    Args:
        u (ArrayLike): Voltage pos., neg. and zero sequence components
        i (ArrayLike): Current pos., neg. and zero sequence components
        factor (float, optional): Multiplication factor. Defaults to 3.

    Returns:
        ArrayLike: Apparent power
    """
    return factor * u * np.conj(i)


def get_active_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
):
    return factor * (np.dot(np.real(u), np.real(i)) + np.dot(np.imag(u), np.imag(i)))


def get_reactive_power_symmetrical_components(
    u: ArrayLike, i: ArrayLike, factor: float = 3
):
    return factor * (-np.dot(np.real(u), np.imag(i)) + np.dot(np.imag(u), np.real(i)))


def get_apparent_power_using_symmetrical_components_and_fourier_coefficients(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    samples_per_window: int,
    time_windows: ArrayLike | None = None,
    return_symmetrical_components: bool = False,
) -> ArrayLike:
    """Get apparent power of symmetrical components

    Steps:
    - calculate fourier coefficients
    - calculate symmetrical components of v and i
    - calculate apparent power

    Args:
        u_abc (pd.DataFrame): three-phase voltages

        i_abc (pd.DataFrame): three-phase currents

        samples_per_window (int): Samples per window (period) to calculate fourier coefficients

        time_windows (ArrayLike | None, optional): Time windows for fourier coefficient calculation (voltages and currents are resampled for each time window). Defaults to None (no resampling).

        return_symmetrical_components (bool, optional): If True, the sym. comp. of voltages and currents are returned. Defaults to False.

    Returns:
        ArrayLike: Apparent power of sym. components (+ sym. comp. of v and i if 'return_symmetrical_components' is True)
    """
    n = 5
    if time_windows is not None:
        u_fourier_coef_real, u_fourier_coef_imag = get_fourier_coef_rolling(
            u_abc,
            samples_per_window,
            time_windows=time_windows,
        )

        i_fourier_coef_real, i_fourier_coef_imag = get_fourier_coef_rolling(
            i_abc,
            samples_per_window,
            time_windows=time_windows,
        )
    else:
        u_fourier_coef_real, u_fourier_coef_imag = get_fourier_coef_rolling(
            u_abc,
            samples_per_window,
        )
        i_fourier_coef_real, i_fourier_coef_imag = get_fourier_coef_rolling(
            i_abc,
            samples_per_window,
        )

    u_phasors_abc = u_fourier_coef_real + u_fourier_coef_imag * 1j
    i_phasors_abc = i_fourier_coef_real + i_fourier_coef_imag * 1j

    u_sym_comp = np.apply_along_axis(abc_2_symmetrical_components, 1, u_phasors_abc)
    i_sym_comp = np.apply_along_axis(abc_2_symmetrical_components, 1, i_phasors_abc)
    if not return_symmetrical_components:
        return get_apparent_power_symmetrical_components(u_sym_comp, i_sym_comp)
    else:
        return (
            get_apparent_power_symmetrical_components(u_sym_comp, i_sym_comp),
            u_sym_comp,
            i_sym_comp,
        )


def get_instantaneous_active_power_3ph(u_abc: ArrayLike, i_abc: ArrayLike) -> ArrayLike:
    """u_a*i_a + u_b*i_b + u_c*i_c

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents

    Returns:
        ArrayLike: _description_
    """
    return np.sum(u_abc * i_abc, axis=1)


def get_instantaneous_reactive_power_3ph(
    u_abc: ArrayLike, i_abc: ArrayLike
) -> ArrayLike:
    """From PowerFactory's 'Technical Reference - Common Result Variables for Terminals and Elements (EMT Simulation)'.

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents

    Returns:
        ArrayLike: instantaneous reactive power.
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
) -> ArrayLike:
    """
    Calculate instantaneous reactive power using 90 degree phase shift of the current.

    From 'The Measurement of Reactive Energy in Polluted Distribution Power Systems: An Analysis of the Performance of Commercial Static Meters'

    Args:
        u_abc (pd.DataFrame): three-phase voltages
        i_abc (pd.DataFrame): three-phase currents
        periods (ArrayLike): Duration of one period in seconds or array of durations (same length as u_abc/i_abc) if periods are not constant.

    Returns:
        ArrayLike: instantaneous reactive power.
    """
    t_i_abc = i_abc.index.values + periods / 4
    i_abc_shifted = resample(i_abc, t_i_abc)
    return pd.Series(
        np.sum(u_abc.to_numpy()[-len(t_i_abc) :] * i_abc_shifted, axis=1), index=t_i_abc
    )


def get_instantaneous_magnitude_3ph(abc: ArrayLike) -> ArrayLike:
    """Calculate the instantaneous magnitude of a three-phase signal.

        According to “IEEE Std 1459-2010 Standard Definitions for the Measurement
    of Electric Power Quantities Under Sinusoidal, Nonsinusoidal, Balanced,
    or Unbalanced Conditions,” 2010.

        Args:
            abc (ArrayLike): three-phase signal

        Returns:
            ArrayLike: magnitude
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
    """Calculate the active power using a rolling window of constant number of samples.

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents
        samples_per_window (int): number of samples per window (e.g. number of samples per period)

    Returns:
        pd.DataFrame | pd.Series: active power
    """
    p_df = pd.DataFrame(get_instantaneous_active_power_3ph(u_abc, i_abc))
    return average_rolling(p_df, samples_per_window=samples_per_window)


def get_active_power_average_rolling_variable_window_3ph(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    time_windows: ArrayLike,
    samples_per_window: int,
) -> pd.DataFrame | pd.Series:
    """Calculate the active power using a rolling window of variable duration.

    Voltages and currents are resampled for each time window.

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents
        time_windows (ArrayLike): time periods for the rolling window (same length as u_abc/i_abc)
        samples_per_window (int): number of samples per window

    Returns:
        pd.DataFrame | pd.Series: active power
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
    """Calculate the reactive power using a rolling window of constant number of samples.

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents
        samples_per_window (int): number of samples per window (e.g. number of samples per period)

    Returns:
        pd.DataFrame | pd.Series: reactive power
    """
    p_df = pd.DataFrame(get_instantaneous_reactive_power_3ph(u_abc, i_abc))
    return average_rolling(p_df, samples_per_window=samples_per_window)


def get_reactive_power_average_rolling_variable_window_3ph(
    u_abc: pd.DataFrame,
    i_abc: pd.DataFrame,
    time_windows: np.array,
    samples_per_window: int,
) -> pd.DataFrame | pd.Series:
    """Calculate the reactive power using a rolling window of variable duration.

    Voltages and currents are resampled for each time window.

    Args:
        u_abc (ArrayLike): three-phase voltages
        i_abc (ArrayLike): three-phase currents
        time_windows (ArrayLike): time periods for the rolling window (same length as u_abc/i_abc)
        samples_per_window (int): number of samples per window

    Returns:
        pd.DataFrame | pd.Series: reactive power
    """
    p_df = pd.DataFrame(
        get_instantaneous_reactive_power_3ph(u_abc.to_numpy(), i_abc.to_numpy()),
        index=u_abc.index,
    )
    return average_rolling_variable_window(
        p_df, time_windows, samples_per_window=samples_per_window
    )
