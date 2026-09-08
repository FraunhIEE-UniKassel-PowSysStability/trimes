"""Phase 4: IEC/IEEE 60255-118-1:2018 test conditions and limit tables.

The limit values asserted here are from the 2018 standard, Tables 2-9.
"""

import math

import numpy as np
import pytest

from trimes.synchrophasor import conditions
from trimes.synchrophasor import limits as lim

FS = 10_000.0  # waveform sampling rate


class TestLimitTables:
    def test_steady_state_tve_is_one_percent(self):
        for cond in ("frequency_range", "magnitude", "harmonic"):
            assert lim.P_CLASS_LIMITS[cond]["tve"] == 1.0
            assert lim.M_CLASS_LIMITS[cond]["tve"] == 1.0

    def test_frequency_range_fe_rfe(self):
        # Table 3
        assert lim.P_CLASS_LIMITS["frequency_range"]["fe"] == 0.005
        assert lim.P_CLASS_LIMITS["frequency_range"]["rfe"] == 0.4
        assert lim.M_CLASS_LIMITS["frequency_range"]["fe"] == 0.005
        assert lim.M_CLASS_LIMITS["frequency_range"]["rfe"] == 0.1

    def test_frequency_range_hz_is_fs_dependent_for_m(self):
        # Table 2: P always +-2; M +-2 (Fs<10), +-Fs/5 (10<=Fs<25), +-5 (Fs>=25)
        assert lim.frequency_range_hz("P", 50) == 2.0
        assert lim.frequency_range_hz("M", 8) == 2.0
        assert lim.frequency_range_hz("M", 15) == 3.0
        assert lim.frequency_range_hz("M", 25) == 5.0

    def test_harmonic_levels_and_fe(self):
        assert lim.P_CLASS_LIMITS["harmonic"]["level"] == 0.01
        assert lim.M_CLASS_LIMITS["harmonic"]["level"] == 0.10
        # Table 3: P fe 0.005 / rfe 0.4 (all Fs); M fe 0.005 (Fs<=20) else 0.025, no rfe
        assert lim.harmonic_error_limits("P", 50) == {"tve": 1.0, "fe": 0.005, "rfe": 0.4}
        assert lim.harmonic_error_limits("M", 20) == {"tve": 1.0, "fe": 0.005}
        assert lim.harmonic_error_limits("M", 50) == {"tve": 1.0, "fe": 0.025}

    def test_magnitude_ranges(self):
        assert lim.P_CLASS_LIMITS["magnitude"]["range_v"] == (0.8, 1.2)
        assert lim.M_CLASS_LIMITS["magnitude"]["range_v"] == (0.1, 1.2)
        assert lim.P_CLASS_LIMITS["magnitude"]["range_i"] == (0.1, 2.0)

    def test_out_of_band_is_m_only(self):
        # Table 2/3: P has no OOB requirement; M TVE 1.3 %, FE 0.01 Hz, no RFE
        assert lim.P_CLASS_LIMITS["out_of_band"] is None
        assert lim.M_CLASS_LIMITS["out_of_band"]["tve"] == 1.3
        assert lim.M_CLASS_LIMITS["out_of_band"]["fe"] == 0.01
        assert lim.M_CLASS_LIMITS["out_of_band"]["rfe"] is None
        assert lim.error_limits("out_of_band", "P") == {}
        assert lim.error_limits("out_of_band", "M") == {"tve": 1.3, "fe": 0.01}

    def test_modulation_table_5_values(self):
        # Table 5, rounded values -- what the standard says to use for compliance
        assert lim.modulation_error_limits("P", 10) == pytest.approx(
            {"tve": 3.0, "fe": 0.03, "rfe": 0.6}
        )
        assert lim.modulation_error_limits("P", 50) == pytest.approx(
            {"tve": 3.0, "fe": 0.06, "rfe": 2.3}
        )
        assert lim.modulation_error_limits("M", 15) == pytest.approx(
            {"tve": 3.0, "fe": 0.18, "rfe": 5.1}
        )
        assert lim.modulation_error_limits("M", 50) == pytest.approx(
            {"tve": 3.0, "fe": 0.30, "rfe": 14.0}
        )

    def test_modulation_fr_caps(self):
        # Table 4: P Fr = min(Fs/10, 2); M Fr = min(Fs/5, 5)
        assert lim.modulation_fr_hz("P", 10) == pytest.approx(1.0)
        assert lim.modulation_fr_hz("P", 50) == pytest.approx(2.0)
        assert lim.modulation_fr_hz("M", 20) == pytest.approx(4.0)
        assert lim.modulation_fr_hz("M", 50) == pytest.approx(5.0)

    def test_modulation_formula_matches_table_off_grid(self):
        # non-standard Fs -> formula: fe = 0.06*Fr, rfe = 0.18*pi*Fr**2 for M
        el = lim.modulation_error_limits("M", 17.5)  # Fr = 3.5
        assert el["fe"] == pytest.approx(0.06 * 3.5)
        assert el["rfe"] == pytest.approx(0.18 * math.pi * 3.5**2)

    def test_ramp_limits_and_exclusion(self):
        # Table 6/7: TVE 1 %, FE 0.01 Hz, RFE 0.4 (P) / 0.2 (M); exclusion max(n/Fs, n/f0)
        assert lim.P_CLASS_LIMITS["frequency_ramp"]["rfe"] == 0.4
        assert lim.M_CLASS_LIMITS["frequency_ramp"]["rfe"] == 0.2
        assert lim.ramp_exclusion_s("P", 30, 60) == pytest.approx(2.0 / 30.0)
        assert lim.ramp_exclusion_s("M", 30, 60) == pytest.approx(7.0 / 30.0)
        assert lim.ramp_exclusion_s("P", 200, 50) == pytest.approx(2.0 / 50.0)  # n/f0 wins

    def test_step_response_delay_overshoot(self):
        # Table 8: P response 2/f0, M larger of 7/Fs or 7/f0; delay 1/(4 Fs)
        assert lim.step_response_time_s("P", 50, 50) == pytest.approx(0.04)
        assert lim.step_response_time_s("M", 30, 60) == pytest.approx(7.0 / 30.0)
        assert lim.step_delay_time_s(50) == pytest.approx(1.0 / 200.0)
        assert lim.P_CLASS_LIMITS["magnitude_step"]["overshoot"] == 0.05
        assert lim.M_CLASS_LIMITS["magnitude_step"]["overshoot"] == 0.10

    def test_step_sizes(self):
        assert lim.P_CLASS_LIMITS["magnitude_step"]["step"] == 0.1
        assert lim.P_CLASS_LIMITS["phase_step"]["step_deg"] == 10.0  # pi/18 rad


class TestSteadyStateConditions:
    def test_frequency_range_default_is_full_deviation(self):
        p = conditions.frequency_range(FS, "P", duration=1.0)
        assert p.metadata["fdev"] == 2.0
        assert p.frequency(np.array([0.5]))[0] == pytest.approx(52.0)
        assert p.metadata["limits"] == {"tve": 1.0, "fe": 0.005, "rfe": 0.4}

    def test_frequency_range_m_uses_fs_dependent_range(self):
        m = conditions.frequency_range(FS, "M", reporting_rate=15.0, duration=1.0)
        assert m.metadata["fdev"] == pytest.approx(3.0)

    def test_frequency_range_sweep_spans_the_range(self):
        sweep = conditions.frequency_range_sweep(FS, "M", step=1.0, reporting_rate=50.0)
        fdevs = sorted(s.metadata["fdev"] for s in sweep)
        assert fdevs[0] == -5.0 and fdevs[-1] == 5.0 and 0.0 in fdevs

    def test_magnitude_sweep_voltage_p_class(self):
        sweep = conditions.magnitude_sweep(FS, "P", kind="V", step=0.1)
        levels = sorted(s.metadata["level"] for s in sweep)
        assert levels[0] == pytest.approx(0.8) and levels[-1] == pytest.approx(1.2)

    def test_single_harmonic_uses_class_level_and_fe_limit(self):
        h = conditions.single_harmonic(FS, "M", order=7, reporting_rate=50.0)
        assert h.metadata["order"] == 7 and h.metadata["level"] == 0.10
        assert h.metadata["limits"] == {"tve": 1.0, "fe": 0.025}

    def test_harmonic_sweep_covers_orders_2_to_50(self):
        sweep = conditions.harmonic_sweep(FS, "P", duration=0.1)
        assert [s.metadata["order"] for s in sweep] == list(range(2, 51))

    def test_out_of_band_default_tone_is_at_the_passband_edge(self):
        oob = conditions.out_of_band(FS, "M", reporting_rate=50.0)
        assert oob.metadata["interharmonic_hz"] == pytest.approx(75.0)  # 50 + 50/2
        assert oob.metadata["in_band"] is False
        assert oob.metadata["limits"] == {"tve": 1.3, "fe": 0.01}

    def test_out_of_band_p_class_is_exempt(self):
        oob = conditions.out_of_band(FS, "P")
        assert oob.metadata["exempt"] is True
        assert oob.metadata["limits"] == {}


class TestDynamicConditions:
    def test_modulation_fm_and_limits_track_reporting_rate(self):
        m10 = conditions.modulation(FS, "P", reporting_rate=10.0, duration=1.0)
        assert m10.metadata["fm"] == pytest.approx(1.0)
        assert m10.metadata["limits"] == pytest.approx({"tve": 3.0, "fe": 0.03, "rfe": 0.6})

        m50 = conditions.modulation(FS, "M", reporting_rate=50.0, duration=1.0)
        assert m50.metadata["fm"] == pytest.approx(5.0)
        assert m50.metadata["limits"] == pytest.approx({"tve": 3.0, "fe": 0.30, "rfe": 14.0})

    def test_modulation_sweep_from_0p1_to_fr(self):
        sweep = conditions.modulation_sweep(
            FS, "M", reporting_rate=50.0, step=0.5, duration=1.0
        )
        fms = [s.metadata["fm"] for s in sweep]
        assert fms[0] == pytest.approx(0.1) and max(fms) <= 5.0

    def test_frequency_ramp_spans_full_range_with_exclusion(self):
        r = conditions.frequency_ramp(FS, "P", reporting_rate=50.0, settle_s=0.2)
        assert r.metadata["range_hz"] == 2.0
        assert r.metadata["rocof"] == pytest.approx(1.0)
        # ramps from f0-2 to f0+2
        mid = 0.2 + r.metadata["ramp_time"] / 2.0
        np.testing.assert_allclose(
            r.frequency(np.array([0.1, mid, r.duration - 0.1])), [48.0, 50.0, 52.0],
            atol=1e-6,
        )
        half = 2.0 / 50.0  # max(2/Fs, 2/f0)
        assert r.metadata["exclusion"][0] == pytest.approx((0.2 - half, 0.2 + half))
        assert r.metadata["limits"] == {"tve": 1.0, "fe": 0.01, "rfe": 0.4}

    def test_frequency_ramp_direction(self):
        down = conditions.frequency_ramp(FS, "P", direction=-1, settle_s=0.1)
        assert down.frequency(np.array([0.05]))[0] == pytest.approx(52.0)

    def test_magnitude_step_carries_response_limits(self):
        s = conditions.magnitude_step(FS, "M", step_time=0.5, reporting_rate=30.0)
        assert s.metadata["transition_times"] == [0.5]
        assert s.metadata["response_time_s"] == pytest.approx(7.0 / 30.0)
        assert s.metadata["delay_time_s"] == pytest.approx(1.0 / 120.0)
        assert s.metadata["overshoot_limit"] == 0.10

    def test_phase_step_default_is_10_degrees(self):
        s = conditions.phase_step(FS, "P", reporting_rate=50.0)
        assert s.metadata["step_deg"] == 10.0
        assert s.metadata["response_time_s"] == pytest.approx(2.0 / 50.0)
