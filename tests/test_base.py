import numpy as np
import pandas as pd
import pytest

from trimes import base


@pytest.fixture
def simple_series() -> pd.Series:
    return pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=[0.0, 1.0, 2.0, 3.0, 4.0])


class TestSampling:
    def test_get_sample_returns_first_sample_after_time(self, simple_series):
        assert base.get_sample(simple_series, 2.5) == 3.0

    def test_get_sample_exact_hit(self, simple_series):
        assert base.get_sample(simple_series, 2.0) == 2.0

    def test_get_sample_shifted(self, simple_series):
        # first sample after 2.5 is index 3; shift -1 -> index 2 -> value 2.0
        assert base.get_sample_shifted(simple_series, 2.5, -1) == 2.0

    def test_get_samples_around_default(self, simple_series):
        around = base.get_samples_around(simple_series, 2.5)
        assert around.to_numpy().tolist() == [2.0]

    def test_get_index(self, simple_series):
        assert base.get_index(simple_series, 2.5) == 3
        np.testing.assert_array_equal(
            base.get_index(simple_series, [0.0, 2.5, 10.0]), [0, 3, 5]
        )

    def test_get_sample_const(self):
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]}, index=[0.0, 1.0, 2.0, 3.0])
        got = base.get_sample_const(df, np.array([0.0, 2.0]))
        assert got["a"].tolist() == [10.0, 30.0]


class TestInterpolation:
    def test_interp_series_linear(self):
        s = pd.Series([0.0, 10.0], index=[0.0, 1.0])
        got = base.interp_series(s, np.array([0.25, 0.5]))
        np.testing.assert_allclose(got.to_numpy(), [2.5, 5.0])
        np.testing.assert_allclose(got.index.to_numpy(), [0.25, 0.5])

    def test_interp_df_linear(self):
        df = pd.DataFrame({"a": [0.0, 2.0, 4.0, 6.0]}, index=[0.0, 1.0, 2.0, 3.0])
        got = base.interp_df(df, np.array([0.5, 1.5]))
        np.testing.assert_allclose(got["a"].to_numpy(), [1.0, 3.0])

    def test_interp_np_matrix_matches_numpy(self):
        old_x = np.array([0.0, 1.0, 2.0])
        y = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        got = base.interp_np_matrix(np.array([0.5, 1.5]), old_x, y)
        np.testing.assert_allclose(got, [[0.5, 15.0], [1.5, 25.0]])


class TestRanges:
    def test_get_between_is_exclusive_of_endpoints_between_samples(self, simple_series):
        got = base.get_between(simple_series, 1.0, 3.0)
        assert got.to_numpy().tolist() == [1.0, 2.0]

    def test_get_between_and_around_includes_neighbours(self, simple_series):
        got = base.get_between_and_around(simple_series, 1.5, 2.5)
        assert got.to_numpy().tolist() == [1.0, 2.0, 3.0]

    def test_get_between_interp_interpolates_edges(self, simple_series):
        got = base.get_between_interp(simple_series, 0.5, 3.5)
        np.testing.assert_allclose(got.index.to_numpy()[[0, -1]], [0.5, 3.5])
        np.testing.assert_allclose(got.to_numpy()[[0, -1]], [0.5, 3.5])

    def test_get_between_interp_does_not_mutate_source_index(self, simple_series):
        original = simple_series.index.to_numpy().copy()
        base.get_between_interp(simple_series, 0.5, 3.5)
        np.testing.assert_array_equal(simple_series.index.to_numpy(), original)


class TestDeltas:
    def test_get_delta(self, simple_series):
        assert base.get_delta(simple_series, 1.0, 3.0) == 2.0

    def test_get_delta_shift(self, simple_series):
        assert base.get_delta_shift(simple_series, 2.5, -1, 0) == 1.0

    def test_get_delta_interp_series(self):
        s = pd.Series([0.0, 10.0, 20.0], index=[0.0, 1.0, 2.0])
        assert base.get_delta_interp_series(s, 0.5, 1.5) == pytest.approx(10.0)

    def test_get_duration(self, simple_series):
        assert base.get_duration(simple_series) == 4.0


class TestResample:
    def test_resample_with_scalar_step(self, simple_series):
        got = base.resample(simple_series, 0.5)
        np.testing.assert_allclose(got.iloc[:3].to_numpy(), [0.0, 0.5, 1.0])

    def test_resample_dataframe(self):
        df = pd.DataFrame({"a": [0.0, 2.0, 4.0]}, index=[0.0, 1.0, 2.0])
        got = base.resample(df, np.array([0.5, 1.5]))
        np.testing.assert_allclose(got["a"].to_numpy(), [1.0, 3.0])


class TestHelpers:
    def test_create_pandas_series_or_frame_series(self):
        original = pd.Series([1.0, 2.0], index=[0.0, 1.0], name="x")
        got = base.create_pandas_series_or_frame_with_same_columns_and_index(
            np.array([3.0, 4.0]), original
        )
        assert isinstance(got, pd.Series)
        assert got.name == "x"

    def test_create_pandas_series_or_frame_frame(self):
        original = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=[0.0])
        got = base.create_pandas_series_or_frame_with_same_columns_and_index(
            np.array([[3.0, 4.0]]), original
        )
        assert list(got.columns) == ["a", "b"]

    def test_superpose_series(self):
        a = pd.Series([1.0, 2.0], index=[0.0, 1.0])
        b = pd.Series([10.0, 20.0], index=[0.0, 1.0])
        np.testing.assert_allclose(base.superpose_series([a, b]).to_numpy(), [11.0, 22.0])

    def test_apply_to_columns_series_roundtrips_to_series(self):
        s = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
        got = base.apply_to_columns(s, np.cumsum)
        assert isinstance(got, pd.Series)
        np.testing.assert_allclose(got.to_numpy(), [1.0, 3.0, 6.0])

    def test_apply_to_columns_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[0.0, 1.0])
        got = base.apply_to_columns(df, np.cumsum)
        np.testing.assert_allclose(got["a"].to_numpy(), [1.0, 3.0])
        np.testing.assert_allclose(got["b"].to_numpy(), [3.0, 7.0])


class TestUniformSampling:
    def test_is_uniformly_sampled_true_for_arange(self):
        s = pd.Series(np.zeros(100), index=np.arange(0.0, 10.0, 0.1))
        assert base.is_uniformly_sampled(s) is True

    def test_is_uniformly_sampled_false_for_gap(self):
        idx = np.concatenate([np.arange(0.0, 1.0, 0.1), [2.0, 3.0]])
        s = pd.Series(np.zeros(idx.size), index=idx)
        assert base.is_uniformly_sampled(s) is False

    def test_is_uniformly_sampled_trivial_for_short_series(self):
        assert base.is_uniformly_sampled(pd.Series([1.0, 2.0], index=[0.0, 5.0]))

    def test_get_sampling_rate(self):
        s = pd.Series(np.zeros(11), index=np.linspace(0.0, 1.0, 11))
        assert base.get_sampling_rate(s) == pytest.approx(10.0)

    def test_get_sampling_rate_rejects_nonuniform(self):
        idx = np.concatenate([np.arange(0.0, 1.0, 0.1), [2.0, 3.0]])
        s = pd.Series(np.zeros(idx.size), index=idx)
        with pytest.raises(ValueError):
            base.get_sampling_rate(s)

    def test_get_sampling_rate_requires_two_samples(self):
        with pytest.raises(ValueError):
            base.get_sampling_rate(pd.Series([1.0], index=[0.0]))


class TestTimeToSeconds:
    def test_numeric_index_is_anchored_at_zero(self):
        got = base.time_to_seconds(pd.Index([5.0, 6.0, 7.5]))
        np.testing.assert_allclose(got, [0.0, 1.0, 2.5])

    def test_timedelta_index(self):
        idx = pd.timedelta_range(0, periods=3, freq="500ms")
        np.testing.assert_allclose(base.time_to_seconds(idx), [0.0, 0.5, 1.0])

    def test_datetime_index_matches_docstring(self):
        idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        np.testing.assert_allclose(base.time_to_seconds(idx), [0.0, 1.0])

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_datetime_index_is_resolution_agnostic(self, unit):
        idx = pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:03"]
        ).as_unit(unit)
        np.testing.assert_allclose(base.time_to_seconds(idx), [0.0, 1.0, 3.0])

    def test_datetime_index_keeps_sub_second_resolution(self):
        idx = pd.DatetimeIndex(
            pd.Timestamp("2024-01-01") + pd.to_timedelta([0, 100, 250], unit="us")
        )
        np.testing.assert_allclose(
            base.time_to_seconds(idx), [0.0, 100e-6, 250e-6], rtol=0, atol=1e-12
        )


class TestResampleUniform:
    def test_interp_method_on_nonuniform_input(self):
        s = pd.Series([0.0, 1.0, 3.0], index=[0.0, 0.1, 0.3])
        got = base.resample_uniform(s, sampling_rate=10.0)
        np.testing.assert_allclose(got.to_numpy(), [0.0, 1.0, 2.0, 3.0])
        assert got.index.name == "time"

    def test_default_rate_preserves_sample_count(self):
        s = pd.Series(np.arange(11.0), index=np.linspace(0.0, 1.0, 11))
        got = base.resample_uniform(s)
        assert len(got) == 11

    def test_fourier_method_requires_uniform_input(self):
        s = pd.Series([0.0, 1.0, 3.0], index=[0.0, 0.1, 0.3])
        with pytest.raises(ValueError):
            base.resample_uniform(s, sampling_rate=10.0, method="fourier")

    def test_unknown_method_raises(self):
        s = pd.Series(np.arange(5.0), index=np.arange(5.0))
        with pytest.raises(ValueError):
            base.resample_uniform(s, method="nope")

    def test_dataframe_roundtrips_type(self):
        df = pd.DataFrame({"a": np.arange(11.0)}, index=np.linspace(0.0, 1.0, 11))
        got = base.resample_uniform(df, sampling_rate=10.0)
        assert isinstance(got, pd.DataFrame)
        assert list(got.columns) == ["a"]
