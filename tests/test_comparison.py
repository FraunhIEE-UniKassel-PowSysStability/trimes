import operator

import numpy as np
import pandas as pd
import pytest

from trimes import comparison as cmp


@pytest.fixture
def idx() -> np.ndarray:
    return np.arange(0.0, 1.0001, 0.001)


@pytest.fixture
def zeros(idx) -> pd.Series:
    return pd.Series(np.zeros_like(idx), index=idx, name="ts")


@pytest.fixture
def ones(idx) -> pd.Series:
    return pd.Series(np.ones_like(idx), index=idx, name="ref")


class TestArithmetic:
    def test_subtract(self, zeros, ones):
        got = cmp.subtract(zeros, ones)
        np.testing.assert_allclose(got.to_numpy(), -1.0)

    def test_add(self, zeros, ones):
        got = cmp.add(zeros, ones)
        np.testing.assert_allclose(got.to_numpy(), 1.0)

    def test_subtract_does_not_extend_beyond_ts(self, idx):
        ts = pd.Series(np.zeros(101), index=np.linspace(0.2, 0.8, 101))
        ref = pd.Series(np.ones_like(idx), index=idx)
        got = cmp.subtract(ts, ref, resample_reference=True)
        assert got.index.min() >= 0.2 - 1e-9
        assert got.index.max() <= 0.8 + 1e-9

    def test_resample_reference_aligns_grids(self):
        ts = pd.Series(np.zeros(11), index=np.linspace(0.0, 1.0, 11))
        ref = pd.Series(np.ones(21), index=np.linspace(0.0, 1.0, 21))
        got = cmp.subtract(ts, ref, resample_reference=True)
        assert len(got) == 11
        np.testing.assert_allclose(got.to_numpy(), -1.0)


class TestOperators:
    def test_greater_than_series(self, idx):
        ts = pd.Series(np.linspace(-1.0, 1.0, idx.size), index=idx)
        ref = pd.Series(np.zeros_like(idx), index=idx)
        mask = cmp.greater_than_series(ts, ref)
        assert mask.dtype == bool
        assert mask[-1] and not mask[0]

    def test_less_than_series(self, idx):
        ts = pd.Series(np.linspace(-1.0, 1.0, idx.size), index=idx)
        ref = pd.Series(np.zeros_like(idx), index=idx)
        assert cmp.less_than_series(ts, ref)[0]

    def test_outside_envelope(self, idx):
        ts = pd.Series(np.sin(2 * np.pi * idx) * 2.0, index=idx)
        envelope = pd.DataFrame(
            {"upper": np.ones_like(idx), "lower": -np.ones_like(idx)}, index=idx
        )
        mask = cmp.outside_envelope(ts, envelope)
        assert mask.any()
        # inside points near a zero crossing are not flagged
        assert not mask[0]

    def test_apply_operator_df(self, idx):
        ts = pd.DataFrame(
            {"a": np.linspace(-1, 1, idx.size), "b": np.linspace(1, -1, idx.size)},
            index=idx,
        )
        ref = pd.Series(np.zeros_like(idx), index=idx)
        got = cmp.apply_operator_df(ts, ref, operator.gt)
        assert got.shape == (idx.size, 2)


class TestGetTimeBool:
    def test_scalar_sample_time(self):
        b = np.array([True, True, False, True])
        assert cmp.get_time_bool(b, 0.5) == pytest.approx(1.5)

    def test_array_sample_time(self):
        b = np.array([True, True, False, False])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        # diff(t)[b[:-1]] -> intervals starting at the first two samples
        assert cmp.get_time_bool(b, t) == pytest.approx(2.0)


class TestComparisonSeries:
    def test_zero_metric_for_identical(self, idx):
        ref = pd.Series(np.sin(2 * np.pi * idx), index=idx)
        assert cmp.comparison_series(ref.copy(), ref) == pytest.approx(0.0)

    def test_metric_matches_area(self, zeros, ones):
        # constant unit error over ~1 s
        assert cmp.comparison_series(zeros, ones) == pytest.approx(1.0, abs=5e-3)

    def test_operator_restricts_region(self, idx):
        ref = pd.Series(np.zeros_like(idx), index=idx)
        ts = pd.Series(np.where(idx < 0.5, 1.0, -1.0), index=idx)
        # only where ts > ref contributes
        err_gt = cmp.comparison_series(ts, ref, operator.gt)
        assert err_gt == pytest.approx(0.5, abs=1e-2)

    def test_no_mask_hit_returns_zero(self, zeros, ones):
        assert cmp.comparison_series(zeros, ones, operator.gt) == 0.0


class TestComparisonDf:
    def test_columns_evaluated_independently(self, idx):
        ref = pd.Series(np.zeros_like(idx), index=idx)
        ts = pd.DataFrame(
            {"a": np.ones_like(idx), "b": np.full_like(idx, 2.0)}, index=idx
        )
        got = cmp.comparison_df(ts, ref)
        assert got.shape == (2,)
        assert got[1] == pytest.approx(2 * got[0], rel=1e-3)


class TestEnvelopeComparison:
    def test_signal_inside_envelope_has_zero_error(self, idx):
        ts = pd.Series(0.5 * np.sin(2 * np.pi * idx), index=idx)
        envelope = pd.DataFrame(
            {"upper": np.ones_like(idx), "lower": -np.ones_like(idx)}, index=idx
        )
        assert cmp.envelope_comparison_series(ts, envelope) == pytest.approx(0.0)

    def test_signal_outside_envelope_has_positive_error(self, idx):
        ts = pd.Series(3.0 * np.sin(2 * np.pi * idx), index=idx)
        envelope = pd.DataFrame(
            {"upper": np.ones_like(idx), "lower": -np.ones_like(idx)}, index=idx
        )
        assert cmp.envelope_comparison_series(ts, envelope) > 0.0
