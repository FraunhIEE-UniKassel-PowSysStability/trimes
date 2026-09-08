import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import root_mean_squared_error

from trimes import metrics


class TestIntegralErrors:
    def test_integral_abs_error_trapezoids_the_weighted_error(self):
        y_true = np.zeros(3)
        y_pred = np.ones(3)
        weight = np.array([1.0, 1.0, 1.0])
        # weighted error is 1 at each of 3 points; trapezoid over unit spacing -> 2
        assert metrics.integral_abs_error(y_true, y_pred, weight) == pytest.approx(2.0)

    def test_integral_squared_error(self):
        y_true = np.zeros(4)
        y_pred = np.full(4, 2.0)
        weight = np.ones(4)
        # weighted error is 2 everywhere, squared 4, trapezoid over 4 points -> 12
        assert metrics.integral_squared_error(y_true, y_pred, weight) == pytest.approx(
            12.0
        )

    def test_multioutput_raw_values(self):
        y_true = np.zeros((3, 2))
        y_pred = np.ones((3, 2))
        weight = np.ones(3)
        got = metrics.integral_abs_error(
            y_true, y_pred, weight, multioutput="raw_values"
        )
        np.testing.assert_allclose(got, [2.0, 2.0])


class TestGetMetricTimeSeries:
    def test_zero_error_for_identical_series(self):
        idx = np.arange(11.0)
        ref = pd.Series(np.sin(idx), index=idx)
        assert metrics.get_metric_time_series(
            ref.copy(), ref, metrics.integral_abs_error
        ) == pytest.approx(0.0)

    def test_matches_manual_area(self):
        idx = np.arange(11.0)
        ref = pd.Series(np.zeros(11), index=idx)
        ts = pd.Series(np.ones(11), index=idx)
        # 10 unit-wide intervals, constant error 1 -> area 9 (last sample dropped)
        assert metrics.get_metric_time_series(
            ts, ref, metrics.integral_abs_error
        ) == pytest.approx(9.0)

    def test_interface_to_sklearn_metric(self):
        idx = np.arange(0.0, 1.0, 0.01)
        ref = pd.Series(np.zeros_like(idx), index=idx)
        ts = pd.Series(np.full_like(idx, 3.0), index=idx)
        got = metrics.get_metric_time_series(ts, ref, root_mean_squared_error)
        assert got == pytest.approx(3.0, rel=1e-6)

    def test_resampling_aligns_series(self):
        ref = pd.Series(np.zeros(21), index=np.linspace(0.0, 1.0, 21))
        ts = pd.Series(np.ones(11), index=np.linspace(0.0, 1.0, 11))
        got = metrics.get_metric_time_series(
            ts, ref, metrics.integral_abs_error, resample_ts=True
        )
        # unit error integrated over [0, 1] with the last sample dropped
        assert got == pytest.approx(0.95, rel=1e-6)

    def test_resampling_raises_without_flag_on_mismatched_length(self):
        ref = pd.Series(np.zeros(21), index=np.linspace(0.0, 1.0, 21))
        ts = pd.Series(np.ones(11), index=np.linspace(0.0, 1.0, 11))
        with pytest.raises(Exception):
            metrics.get_metric_time_series(ts, ref, metrics.integral_abs_error)


class TestCosineSimilarity:
    def test_orthogonal_vectors(self):
        assert metrics.get_cosine_similarity(
            np.array([1.0, 0.0]), np.array([0.0, 1.0])
        ) == pytest.approx(0.0)

    def test_parallel_vectors(self):
        assert metrics.get_cosine_similarity(
            np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
        ) == pytest.approx(1.0)

    def test_forty_five_degrees(self):
        assert metrics.get_cosine_similarity(
            np.array([1.0, 0.0]), np.array([1.0, 1.0])
        ) == pytest.approx(1 / np.sqrt(2))

    def test_similarity_matrix_is_symmetric_with_unit_diagonal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(50, 4))
        m = metrics.get_cosine_similarity_matrix(a)
        assert m.shape == (4, 4)
        np.testing.assert_allclose(np.diag(m), 1.0)
        np.testing.assert_allclose(m, m.T)

    def test_similarity_matrix_as_df_keeps_labels(self):
        df = pd.DataFrame(
            np.random.default_rng(1).normal(size=(30, 3)), columns=["x", "y", "z"]
        )
        got = metrics.get_cosine_similarity_matrix_as_df(df)
        assert list(got.columns) == ["x", "y", "z"]
        assert list(got.index) == ["x", "y", "z"]
