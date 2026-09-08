import numpy as np
import pandas as pd
import pytest

from trimes import transforms as tf


@pytest.fixture
def balanced_abc() -> np.ndarray:
    """Balanced positive-sequence three-phase set, shape (n, 3)."""
    theta = np.linspace(0.0, 4 * np.pi, 200, endpoint=False)
    return np.column_stack(
        [
            np.cos(theta),
            np.cos(theta - 2 / 3 * np.pi),
            np.cos(theta + 2 / 3 * np.pi),
        ]
    )


class TestPark:
    def test_park_inverse_roundtrip(self):
        abc = np.array([1.0, -0.5, -0.5])
        for phi in (0.0, 0.7, -2.1):
            back = tf.dq0_2_abc(tf.abc_2_dq0(abc, phi), phi)
            np.testing.assert_allclose(back, abc, atol=1e-12)

    def test_park_of_balanced_set_is_dc_when_aligned(self):
        # abc aligned with rotating frame -> constant d, zero q and zero-sequence
        n = 100
        theta = np.linspace(0.0, 4 * np.pi, n, endpoint=False)
        d = np.empty(n)
        q = np.empty(n)
        for k, th in enumerate(theta):
            abc = np.array(
                [np.cos(th), np.cos(th - 2 / 3 * np.pi), np.cos(th + 2 / 3 * np.pi)]
            )
            d[k], q[k], _ = tf.abc_2_dq0(abc, th)
        np.testing.assert_allclose(d, 1.0, atol=1e-12)
        np.testing.assert_allclose(q, 0.0, atol=1e-12)


class TestClarke:
    def test_zero_sequence_vanishes_for_balanced_set(self, balanced_abc):
        ab0 = tf.abc_2_ab0(balanced_abc.T)
        np.testing.assert_allclose(ab0[2], 0.0, atol=1e-12)

    def test_amplitude_invariance(self, balanced_abc):
        ab0 = tf.abc_2_ab0(balanced_abc.T)
        np.testing.assert_allclose(ab0[0] ** 2 + ab0[1] ** 2, 1.0, atol=1e-12)


class TestSymmetricalComponents:
    def test_balanced_positive_sequence_phasors(self):
        a = np.exp(2j / 3 * np.pi)
        # a-phase reference, b lags 120 deg, c lags 240 deg
        abc = np.array([[1.0 + 0j, a**2, a]])
        sc = tf.abc_2_symmetrical_components(abc)[0]
        np.testing.assert_allclose(sc[0], 0.0, atol=1e-12)  # zero sequence
        np.testing.assert_allclose(sc[1], 1.0, atol=1e-12)  # positive sequence
        np.testing.assert_allclose(sc[2], 0.0, atol=1e-12)  # negative sequence

    def test_negative_sequence_phasors(self):
        a = np.exp(2j / 3 * np.pi)
        abc = np.array([[1.0 + 0j, a, a**2]])  # reversed rotation
        sc = tf.abc_2_symmetrical_components(abc)[0]
        np.testing.assert_allclose(sc[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(sc[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(sc[2], 1.0, atol=1e-12)

    def test_zero_sequence_phasors(self):
        abc = np.array([[2.0 + 0j, 2.0 + 0j, 2.0 + 0j]])
        sc = tf.abc_2_symmetrical_components(abc)[0]
        np.testing.assert_allclose(sc, [2.0, 0.0, 0.0], atol=1e-12)

    def test_frobenius_matrices_have_expected_shape(self):
        assert tf.frobenius_matrix_symcomp().shape == (3, 3)
        assert tf.inverse_frobenius_matrix_symcomp().shape == (3, 3)

    def test_forward_then_inverse_is_the_identity(self, balanced_abc):
        roundtrip = tf.symmetrical_components_2_abc(
            tf.abc_2_symmetrical_components(balanced_abc)
        )
        np.testing.assert_allclose(roundtrip, balanced_abc, atol=1e-12)

    def test_inverse_matrix_is_the_true_inverse_of_the_forward(self):
        identity = tf.frobenius_matrix_symcomp() @ tf.inverse_frobenius_matrix_symcomp()
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)


class TestTransformTimeSeries:
    def test_apply_clarke_to_frame(self, balanced_abc):
        df = pd.DataFrame(balanced_abc, columns=["a", "b", "c"])
        transform = lambda row: tf.abc_2_ab0(row.to_numpy())
        got = tf.transform_time_series(df, transform)
        assert got.shape == df.shape
        np.testing.assert_allclose(got.iloc[:, 2].to_numpy(), 0.0, atol=1e-12)
