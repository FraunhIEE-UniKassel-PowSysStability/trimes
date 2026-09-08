"""Phase 6: synchrophasor plotting helpers (smoke tests, Agg backend)."""

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from trimes.synchrophasor import conditions
from trimes.synchrophasor.plot import plot_errors, plot_sweep_heatmap, plot_waveform


@pytest.fixture
def ramp():
    return conditions.frequency_ramp(10_000.0, "P")


def _trace(sig):
    times = np.arange(0.05, sig.duration - 0.05, 0.02)
    gt = sig.ground_truth(times)
    rng = np.random.default_rng(0)
    n = len(times)
    return pd.DataFrame(
        {
            "phasor": gt["phasor"].to_numpy() * (1 + 0.003 * rng.standard_normal(n)),
            "frequency": gt["frequency"].to_numpy() + 0.003 * rng.standard_normal(n),
            "rocof": gt["rocof"].to_numpy() + 0.05 * rng.standard_normal(n),
        },
        index=pd.Index(times, name="time"),
    )


def test_plot_waveform_runs(ramp):
    fig, ax = plt.subplots()
    out = plot_waveform(ramp, ax=ax)
    assert out is ax
    assert ax.lines  # signal + envelope drawn


def test_plot_errors_draws_three_rows_with_limits(ramp):
    axes = plot_errors(_trace(ramp), ramp)
    assert len(axes) == 3
    # limit line present on the TVE axis (from metadata["limits"])
    assert any(line.get_linestyle() == "--" for line in axes[0].lines)


def test_plot_errors_shades_ramp_exclusion(ramp):
    axes = plot_errors(_trace(ramp), ramp)
    assert axes[0].patches  # axvspan patches for the exclusion windows


def test_plot_sweep_heatmap_runs():
    table = pd.DataFrame(
        np.random.default_rng(0).random((4, 3)),
        index=["fdev=-2", "fdev=-1", "fdev=1", "fdev=2"],
        columns=["dft", "ipdft", "eipdft"],
    )
    fig, ax = plt.subplots()
    out = plot_sweep_heatmap(table, ax=ax)
    assert out is ax
    assert ax.images
