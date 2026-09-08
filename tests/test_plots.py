"""Smoke tests for the plotting helpers.

These check that the helpers run against the Agg backend and return the
expected object types; they do not assert on rendered pixels.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from trimes import plots


class TestIsolatedPoints:
    def test_flags_finite_sample_between_two_nans(self):
        y = np.array([np.nan, 1.0, np.nan, 2.0, 3.0, np.nan])
        mask = plots._isolated_points(y)
        np.testing.assert_array_equal(mask, [False, True, False, False, False, False])

    def test_no_isolated_points_in_dense_signal(self):
        assert not plots._isolated_points(np.arange(10.0)).any()

    def test_all_nan(self):
        assert not plots._isolated_points(np.full(5, np.nan)).any()


class TestColorHelpers:
    def test_default_colormap_is_ten_hex_strings(self):
        cmap = plots.get_pyplot_default_colormap_hex()
        assert len(cmap) == 10

    def test_color_cycler_start_index_offsets(self):
        base = plots.get_pyplot_default_color_cycler(0)
        shifted = plots.get_pyplot_default_color_cycler(2)
        first_shifted = next(iter(shifted))
        by_hand = list(iter(base))[2]
        assert first_shifted == by_hand


class TestPlotHelpers:
    def test_plot_2y_returns_two_axes(self):
        idx = np.linspace(0.0, 1.0, 50)
        df1 = pd.DataFrame({"a": np.sin(idx)}, index=idx)
        df2 = pd.DataFrame({"b": np.cos(idx)}, index=idx)
        ax1, ax2 = plots.plot_2y(df1, df2)
        assert ax1 is not ax2

    def test_add_vertical_line_to_plot(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plots.add_vertical_line_to_plot(0.5)
        assert any(line.get_xdata()[0] == 0.5 for line in ax.get_lines())

    def test_fill_between_runs(self):
        idx = np.linspace(0.0, 1.0, 20)
        lo = pd.Series(-np.ones_like(idx), index=idx)
        hi = pd.Series(np.ones_like(idx), index=idx)
        fig, ax = plt.subplots()
        plots.fill_between(lo, hi, ax=ax)
        assert ax.collections
