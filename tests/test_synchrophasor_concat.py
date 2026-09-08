"""Phase 2: concat_segments -- phase-continuous concatenation."""

import numpy as np
import pandas as pd
import pytest

from trimes.synchrophasor import concat_segments, steady_state
from trimes.synchrophasor.util import TWO_PI, wrap_pi
from trimes.signal_generation import PeriodicSignal

FS = 5000.0
ATOL = 1e-9


class TestContinuity:
    def test_phase_is_continuous_without_a_jump(self):
        sig = concat_segments(
            [(0.1, {"frequency": 50.0}), (0.1, {"frequency": 50.0})], fs=FS
        )
        # same frequency both sides -> the two segment models agree at the boundary
        tb = 0.1
        left = sig.total_phase(np.array([tb - 1e-12]))[0]
        right = sig.total_phase(np.array([tb + 1e-12]))[0]
        assert abs(right - left) < ATOL
        assert sig.metadata["transition_times"] == pytest.approx([0.1])

    def test_frequency_change_keeps_phase_c0(self):
        sig = concat_segments(
            [(0.1, {"frequency": 50.0}), (0.1, {"frequency": 52.0})], fs=FS
        )
        tb = 0.1
        # left model extrapolated to tb vs right model at tb: equal (C0)
        eps = 1e-7
        left_rate = (
            sig.total_phase(np.array([tb - eps]))[0]
            - sig.total_phase(np.array([tb - 2 * eps]))[0]
        ) / eps
        # continuity: value from the right equals left extrapolated by one eps step
        left_val = sig.total_phase(np.array([tb - eps]))[0] + left_rate * eps
        right_val = sig.total_phase(np.array([tb + 1e-12]))[0]
        assert abs(right_val - left_val) < 1e-6

    def test_waveform_is_consistent_with_truth(self):
        sig = concat_segments(
            [(0.1, {"frequency": 50.0}), (0.1, {"rocof": 2.0})], fs=FS
        )
        t = sig.time
        np.testing.assert_allclose(
            sig.signal.to_numpy(),
            sig.amplitude(t) * np.cos(sig.total_phase(t)),
            atol=1e-12,
        )


class TestJumps:
    def test_phase_jump_matches(self):
        sig = concat_segments(
            [(0.1, {"frequency": 50.0}), (0.1, {"frequency": 50.0})],
            fs=FS,
            jumps={1: {"phase_deg": 30.0}},
        )
        tb = 0.1
        delta = (
            sig.total_phase(np.array([tb + 1e-12]))[0]
            - sig.total_phase(np.array([tb - 1e-12]))[0]
        )
        assert np.degrees(delta) == pytest.approx(30.0, abs=1e-6)

    def test_magnitude_jump_persists(self):
        sig = concat_segments(
            [(0.1, {}), (0.1, {}), (0.1, {})],
            fs=FS,
            jumps={1: {"magnitude": 1.2}},
        )
        np.testing.assert_allclose(sig.amplitude(np.array([0.05])), 1.0, atol=ATOL)
        np.testing.assert_allclose(sig.amplitude(np.array([0.15])), 1.2, atol=ATOL)
        np.testing.assert_allclose(sig.amplitude(np.array([0.25])), 1.2, atol=ATOL)

    def test_jump_by_time_key(self):
        sig = concat_segments(
            [(0.1, {}), (0.1, {})], fs=FS, jumps={0.1: {"phase_deg": 10.0}}
        )
        assert sig.metadata["transition_times"] == pytest.approx([0.1])

    def test_jump_on_non_interior_boundary_raises(self):
        with pytest.raises(ValueError, match="interior boundary"):
            concat_segments([(0.1, {}), (0.1, {})], fs=FS, jumps={0: {"phase_deg": 1}})


class TestSpecForms:
    def test_periodic_signal_segment(self):
        ps = PeriodicSignal(np.arange(0.0, 0.1, 1 / FS), f=50.0, mag=1.0)
        sig = concat_segments([(0.1, ps), (0.1, {"frequency": 51.0})], fs=FS)
        np.testing.assert_allclose(sig.frequency(np.array([0.05])), 50.0, atol=ATOL)
        np.testing.assert_allclose(sig.frequency(np.array([0.15])), 51.0, atol=ATOL)

    def test_analytic_signal_segment(self):
        seg = steady_state(FS, frequency=49.0, amplitude=2.0, duration=0.1)
        sig = concat_segments([(0.1, seg), (0.1, {"frequency": 50.0})], fs=FS)
        np.testing.assert_allclose(sig.frequency(np.array([0.05])), 49.0, atol=ATOL)
        np.testing.assert_allclose(sig.amplitude(np.array([0.05])), 2.0, atol=ATOL)

    def test_ramp_spec_end_frequency(self):
        sig = concat_segments([(1.0, {"frequency": (50.0, 52.0)})], fs=FS)
        np.testing.assert_allclose(sig.frequency(np.array([0.0, 0.5, 1.0])), [50.0, 51.0, 52.0], atol=1e-9)

    def test_callable_spec(self):
        def seg(duration):
            f = 50.0
            return (
                lambda t: np.ones_like(np.asarray(t, float)),
                lambda t: np.full_like(np.asarray(t, float), f),
                lambda t: TWO_PI * f * np.asarray(t, float),
            )

        sig = concat_segments([(0.1, seg), (0.1, {"frequency": 55.0})], fs=FS)
        np.testing.assert_allclose(sig.frequency(np.array([0.05])), 50.0, atol=ATOL)

    def test_bad_spec_raises(self):
        with pytest.raises(TypeError):
            concat_segments([(0.1, "nonsense")], fs=FS)


class TestModulationSegment:
    def test_modulation_segment_truth(self):
        sig = concat_segments(
            [(2.0, {"am_depth": 0.1, "pm_depth": 0.1, "modulation_frequency": 2.0})],
            fs=FS,
            nominal=50.0,
        )
        q = np.linspace(0.0, 2.0, 51)
        wm = TWO_PI * 2.0
        np.testing.assert_allclose(
            sig.frequency(q), 50.0 + 0.1 * 2.0 * np.sin(wm * q), atol=1e-9
        )
        np.testing.assert_allclose(
            sig.phase(q, 50.0), wrap_pi(-0.1 * (np.cos(wm * q) - 1.0)), atol=1e-9
        )
