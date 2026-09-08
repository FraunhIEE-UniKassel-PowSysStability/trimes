import numpy as np
import pandas as pd
import pytest

from trimes import electrical as el


@pytest.fixture
def balanced(three_phase_df):
    return three_phase_df


class TestInstantaneousPower:
    def test_active_power_sum_of_products(self):
        u = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        i = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        got = el.get_instantaneous_active_power_3ph(u, i)
        np.testing.assert_allclose(got, [6.0, 6.0])

    def test_reactive_power_zero_for_in_phase_currents(self):
        u = np.array([[1.0, -0.5, -0.5], [0.5, 0.5, -1.0]])
        got = el.get_instantaneous_reactive_power_3ph(u, u)
        np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_instantaneous_magnitude_of_balanced_set_is_constant(self, balanced):
        mag = el.get_instantaneous_magnitude_3ph(balanced.to_numpy())
        np.testing.assert_allclose(mag, mag[0], rtol=1e-6)
        # unit phase peak amplitude -> constant magnitude of 1/sqrt(2)
        assert mag[0] == pytest.approx(1 / np.sqrt(2), rel=1e-6)


class TestSymmetricalComponentPower:
    def test_apparent_power_definition(self):
        u = np.array([1.0 + 0j, 2.0 + 0j, 0.0 + 0j])
        i = np.array([1.0 - 1j, 0.5 + 0j, 0.0 + 0j])
        got = el.get_apparent_power_symmetrical_components(u, i, factor=3)
        np.testing.assert_allclose(got, 3 * u * np.conj(i))

    def test_active_and_reactive_split(self):
        u = np.array([1.0 + 0j])
        i = np.array([1.0 + 1j])
        p = el.get_active_power_symmetrical_components(u, i, factor=1)
        q = el.get_reactive_power_symmetrical_components(u, i, factor=1)
        assert p == pytest.approx(1.0)
        assert q == pytest.approx(-1.0)


class TestRollingActivePower:
    def test_active_power_of_balanced_load(self, three_phase_df, samples_per_period):
        u = three_phase_df
        # current in phase with voltage, amplitude 2 -> P = 3 * (1*2)/2 = 3
        i = three_phase_df * 2.0
        got = el.get_active_power_average_rolling_3ph(
            u.to_numpy(), i.to_numpy(), samples_per_period
        )
        settled = got.to_numpy().ravel()[2 * samples_per_period :]
        np.testing.assert_allclose(settled, 3.0, atol=1e-6)
