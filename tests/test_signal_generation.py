import numpy as np
import pandas as pd
import pytest

from trimes import signal_generation as sg
from trimes.signal_generation import PeriodicSignal


class TestPeriodicSignal:
    def test_constant_frequency_signal_shape_and_amplitude(self):
        t = np.linspace(0.0, 1.0, 1001)
        ps = PeriodicSignal(t, func=np.sin, f=2.0, mag=3.0, offset=1.0)
        y = ps.get_signal()
        assert y.shape == t.shape
        assert y.max() == pytest.approx(4.0, abs=1e-3)
        assert y.min() == pytest.approx(-2.0, abs=1e-3)

    def test_initial_angle_and_phase(self):
        t = np.linspace(0.0, 1.0, 1001)
        ps = PeriodicSignal(t, func=np.cos, f=1.0, phi=np.pi / 2)
        # cos(2 pi t + pi/2) starts at 0
        assert ps.get_signal()[0] == pytest.approx(0.0, abs=1e-12)

    def test_get_signal_series_index_name(self):
        t = np.linspace(0.0, 0.1, 100)
        series = PeriodicSignal(t, f=50.0).get_signal_series()
        assert isinstance(series, pd.Series)
        assert series.index.name == "time"

    def test_frequency_ramp_via_tuple(self):
        t = np.linspace(0.0, 1.0, 1001)
        ps = PeriodicSignal(t, f=(1.0, 5.0))
        assert np.size(ps.f) == t.size
        assert ps.get_angle_over_time().shape == t.shape

    def test_get_signal_n_phases(self):
        t = np.linspace(0.0, 0.1, 200)
        df = PeriodicSignal(t, f=50.0).get_signal_n_phases(3)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (200, 3)

    def test_signal_attributes_over_time(self):
        t = np.linspace(0.0, 0.1, 50)
        attrs = PeriodicSignal(t, f=50.0, mag=2.0).get_signal_attributes_over_time()
        assert list(attrs.columns) == ["f", "mag", "offset"]
        np.testing.assert_allclose(attrs["mag"].to_numpy(), 2.0)


class TestConcatenation:
    def _signals(self, step=1e-3):
        return [
            [PeriodicSignal(np.arange(0.0, 0.1, step), f=50.0)],
            [PeriodicSignal(np.arange(0.1, 0.2, step), f=50.0)],
        ]

    def test_single_phase_returns_series(self):
        res = sg.superimpose_and_concat_periodic_signals(self._signals())
        assert isinstance(res, pd.Series)
        assert res.shape == (200,)

    def test_multi_phase_returns_dataframe(self):
        res = sg.superimpose_and_concat_periodic_signals(self._signals(), num_phases=3)
        assert isinstance(res, pd.DataFrame)
        assert res.shape == (200, 3)

    def test_angle_continuation_avoids_jump(self):
        step = 1e-4
        res = sg.superimpose_and_concat_periodic_signals(
            self._signals(step), angle_continuation=True
        )
        # No discontinuity at the concatenation boundary: the step across it is
        # no larger than the largest step elsewhere in the signal.
        diffs = np.abs(np.diff(res.to_numpy()))
        assert diffs[999] <= diffs.max()

    def test_angle_continuation_reproduces_unsplit_signal(self):
        # Two contiguous constant-frequency segments concatenated with angle
        # continuation must match one PeriodicSignal over the whole range.
        step = 1e-4
        t_full = np.arange(0.0, 0.2, step)
        full = PeriodicSignal(t_full, f=50.0).get_signal()
        res = sg.superimpose_and_concat_periodic_signals(
            self._signals(step), angle_continuation=True
        )
        np.testing.assert_allclose(res.to_numpy(), full, atol=1e-9)

    def test_ragged_input_raises(self):
        bad = [
            [PeriodicSignal(np.arange(0.0, 0.1, 1e-3), f=50.0)],
            [
                PeriodicSignal(np.arange(0.1, 0.2, 1e-3), f=50.0),
                PeriodicSignal(np.arange(0.1, 0.2, 1e-3), f=100.0),
            ],
        ]
        with pytest.raises(ValueError):
            sg.superimpose_and_concat_periodic_signals(bad)


class TestLinearTimeSeries:
    def test_interpolates_between_breakpoints(self):
        ts = sg.linear_time_series(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 5.0]), 0.5
        )
        assert isinstance(ts, pd.Series)
        np.testing.assert_allclose(ts.to_numpy(), [0.0, 5.0, 10.0, 7.5, 5.0])
        assert ts.index.name == "time"

    def test_make_monotonously_increasing(self):
        got = sg.make_monotonously_increasing([0.0, 0.0, 1.0, 1.0], 0.1)
        np.testing.assert_allclose(got, [0.0, 0.1, 1.0, 1.1])

    def test_get_interpolated_list_returns_dataframe(self):
        got = sg.get_interpolated(
            np.array([0.0, 2.0]), [np.array([0.0, 2.0]), np.array([0.0, 4.0])], 1.0
        )
        assert isinstance(got, pd.DataFrame)
        assert got.shape == (3, 2)


class TestDiscretize:
    def test_matches_docstring_example(self):
        ts = pd.Series([20.0, 50.0, 60.0], index=[0.0, 2.0, 3.0])
        got = sg.discretize(ts, delta_t=1e-3)
        np.testing.assert_allclose(got.to_numpy(), [20.0, 20.0, 50.0, 50.0, 60.0])
        np.testing.assert_allclose(got.index.to_numpy(), [0.0, 1.999, 2.0, 2.999, 3.0])

    def test_dataframe_input(self):
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=[0.0, 1.0])
        got = sg.discretize(df)
        assert isinstance(got, pd.DataFrame)


class TestMirrorAndDict:
    def test_mirror_y(self):
        s = pd.Series([1.0, 2.0, 3.0])
        np.testing.assert_allclose(sg.mirror_y(s, 0.0).to_numpy(), [-1.0, -2.0, -3.0])

    def test_mirror_y_inplace_returns_frame(self):
        s = pd.Series([1.0, 2.0, 3.0])
        got = sg.mirror_y(s, 5.0, inplace=True)
        assert isinstance(got, pd.DataFrame)
        np.testing.assert_allclose(got[1].to_numpy(), [9.0, 8.0, 7.0])

    def test_create_ts_from_dict_pads_short_columns_with_last_value(self):
        d = {"time": np.array([0.0, 1.0, 2.0, 3.0]), "a": np.array([1.0, 2.0])}
        df = sg.create_ts_from_dict_with_varying_length(d)
        np.testing.assert_allclose(df["a"].to_numpy(), [1.0, 2.0, 2.0, 2.0])
        assert df.index.name == "time"


class TestNoise:
    def test_normal_noise_is_reproducible_with_seed(self):
        s = pd.Series(np.zeros(1000), index=np.arange(1000.0))
        a = sg.add_normal_noise(s, scale=2.0, seed=7)
        b = sg.add_normal_noise(s, scale=2.0, seed=7)
        pd.testing.assert_series_equal(a, b)
        assert a.std() == pytest.approx(2.0, rel=0.1)

    def test_uniform_noise_bounds(self):
        s = pd.Series(np.zeros(1000), index=np.arange(1000.0))
        got = sg.add_uniform_noise(s, low=-1.0, high=1.0, seed=3)
        assert got.min() >= -1.0
        assert got.max() < 1.0

    def test_different_seeds_differ(self):
        s = pd.Series(np.zeros(100), index=np.arange(100.0))
        a = sg.add_normal_noise(s, 1.0, seed=1)
        b = sg.add_normal_noise(s, 1.0, seed=2)
        assert not np.allclose(a.to_numpy(), b.to_numpy())
