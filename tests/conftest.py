"""Shared fixtures for the trimes test suite."""

import numpy as np
import pandas as pd
import pytest

# Use a non-interactive matplotlib backend so plot helpers can be exercised
# without a display and without popping up windows.
import matplotlib

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figures a test may have opened."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def sample_time() -> float:
    return 1e-4


@pytest.fixture
def fundamental() -> float:
    return 50.0


@pytest.fixture
def time_grid(sample_time) -> np.ndarray:
    """0.1 s (five fundamental periods) sampled at 10 kHz, starting at 0."""
    return np.arange(0.0, 0.1, sample_time)


@pytest.fixture
def samples_per_period(sample_time, fundamental) -> int:
    return int(round(1.0 / fundamental / sample_time))


@pytest.fixture
def sine_series(time_grid, fundamental) -> pd.Series:
    """Unit-amplitude 50 Hz sine, index named ``time``."""
    values = np.sin(2 * np.pi * fundamental * time_grid)
    return pd.Series(values, index=pd.Index(time_grid, name="time"), name="u")


@pytest.fixture
def three_phase_df(time_grid, fundamental) -> pd.DataFrame:
    """Balanced 50 Hz three-phase set of unit amplitude."""
    w = 2 * np.pi * fundamental * time_grid
    return pd.DataFrame(
        {
            "a": np.cos(w),
            "b": np.cos(w - 2 / 3 * np.pi),
            "c": np.cos(w + 2 / 3 * np.pi),
        },
        index=pd.Index(time_grid, name="time"),
    )


@pytest.fixture
def ramp_series() -> pd.Series:
    """y = 2 t on a unit grid of 50 samples."""
    t = np.arange(50.0)
    return pd.Series(2.0 * t, index=pd.Index(t, name="time"), name="ramp")
