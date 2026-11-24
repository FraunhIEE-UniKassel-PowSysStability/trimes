"""Phase 3: three_phase -- abc expansion with symmetrical components."""

import numpy as np
import pandas as pd
import pytest

from trimes.synchrophasor import steady_state, three_phase
from trimes.transforms import abc_2_ab0, abc_2_symmetrical_components

FS = 5000.0


@pytest.fixture
def positive_seq():
    return steady_state(FS, nominal=50.0, amplitude=2.0, duration=0.1)


class TestBalanced:
    def test_returns_abc_and_positive_sequence_truth(self, positive_seq):
        abc, truth = three_phase(positive_seq)
        assert list(abc) == ["a", "b", "c"]
        for s in abc.values():
            assert isinstance(s, pd.Series)
            assert s.index.equals(positive_seq.signal.index)
        # truth is the (unchanged) positive-sequence signal
        np.testing.assert_allclose(
            truth.amplitude(truth.time), positive_seq.amplitude(positive_seq.time)
        )

    def test_amplitude_invariant_convention(self):
        # a fine grid so the sampled peak resolves the true peak
        fine = steady_state(200_000.0, nominal=50.0, amplitude=2.0, duration=0.05)
        abc, truth = three_phase(fine)
        # each phase peak == |V+| == the positive-sequence amplitude
        for s in abc.values():
            assert s.abs().max() == pytest.approx(2.0, rel=1e-4)

    def test_phases_are_120_degrees_apart(self, positive_seq):
        abc, _ = three_phase(positive_seq)
        stack = np.column_stack([abc[p].to_numpy() for p in "abc"])
        seq = abc_2_symmetrical_components(stack.astype(complex))
        # real cosines -> energy split between + and - image; but |0-seq| ~ 0
        assert np.allclose(seq[:, 0], 0.0, atol=1e-9)


class TestUnbalanced:
    def test_zero_sequence_appears(self, positive_seq):
        abc, _ = three_phase(positive_seq, zero=0.2)
        # a+b+c == 3 * zero-sequence term, non-zero
        total = abc["a"].to_numpy() + abc["b"].to_numpy() + abc["c"].to_numpy()
        assert np.abs(total).max() > 0.1

    def test_balanced_set_has_no_zero_sequence(self, positive_seq):
        abc, _ = three_phase(positive_seq)
        total = abc["a"].to_numpy() + abc["b"].to_numpy() + abc["c"].to_numpy()
        np.testing.assert_allclose(total, 0.0, atol=1e-9)

    def test_metadata_records_sequence_content(self, positive_seq):
        _, truth = three_phase(positive_seq, negative=0.1, zero=0.05)
        assert truth.metadata["negative_sequence"] == pytest.approx(0.1)
        assert truth.metadata["zero_sequence"] == pytest.approx(0.05)
        # the input signal's metadata is not mutated
        assert "negative_sequence" not in positive_seq.metadata

    def test_noise_is_per_phase_and_reproducible(self, positive_seq):
        abc1, _ = three_phase(positive_seq, noise=0.01, seed=5)
        abc2, _ = three_phase(positive_seq, noise=0.01, seed=5)
        for p in "abc":
            pd.testing.assert_series_equal(abc1[p], abc2[p])
        # different phases got different realisations
        assert not np.allclose(abc1["a"].to_numpy(), abc1["b"].to_numpy())

    def test_harmonics_added_to_every_phase(self, positive_seq):
        abc_clean, _ = three_phase(positive_seq)
        abc_h, _ = three_phase(positive_seq, harmonics={5: 0.1})
        for p in "abc":
            assert not np.allclose(
                abc_clean[p].to_numpy(), abc_h[p].to_numpy()
            )


class TestDcOffset:
    def test_scalar_offset_is_added_to_every_phase(self, positive_seq):
        clean, _ = three_phase(positive_seq)
        shifted, _ = three_phase(positive_seq, dc_offset=0.1)
        # the fraction refers to the peak amplitude (2.0)
        for p in "abc":
            np.testing.assert_allclose(
                shifted[p].to_numpy() - clean[p].to_numpy(), 0.2, atol=1e-12
            )

    def test_offset_is_zero_sequence_and_keeps_the_truth(self, positive_seq):
        clean, truth_clean = three_phase(positive_seq)
        shifted, truth = three_phase(positive_seq, dc_offset=0.1)
        stack = np.column_stack([shifted[p].to_numpy() for p in "abc"])
        clean_stack = np.column_stack([clean[p].to_numpy() for p in "abc"])
        seq = abc_2_symmetrical_components(stack.astype(complex))
        seq_clean = abc_2_symmetrical_components(clean_stack.astype(complex))
        np.testing.assert_allclose(seq[:, 0] - seq_clean[:, 0], 0.2, atol=1e-12)
        np.testing.assert_allclose(seq[:, 1], seq_clean[:, 1], atol=1e-12)
        np.testing.assert_allclose(
            truth.phasor(truth.time), truth_clean.phasor(truth_clean.time)
        )

    def test_array_offset_is_applied_per_sample(self, positive_seq):
        n = positive_seq.time.size
        ramp = np.linspace(0.0, 0.5, n)
        clean, _ = three_phase(positive_seq)
        shifted, _ = three_phase(positive_seq, dc_offset=ramp)
        np.testing.assert_allclose(
            shifted["b"].to_numpy() - clean["b"].to_numpy(), ramp * 2.0, atol=1e-12
        )

    def test_callable_offset_receives_the_time_grid(self, positive_seq):
        clean, _ = three_phase(positive_seq)
        shifted, _ = three_phase(positive_seq, dc_offset=lambda t: 5.0 * t)
        t = positive_seq.time
        np.testing.assert_allclose(
            shifted["c"].to_numpy() - clean["c"].to_numpy(), 5.0 * t * 2.0, atol=1e-12
        )

    def test_default_is_unchanged(self, positive_seq):
        a, _ = three_phase(positive_seq)
        b, _ = three_phase(positive_seq, dc_offset=None)
        for p in "abc":
            pd.testing.assert_series_equal(a[p], b[p])

    def test_noise_realisations_are_unchanged_by_the_offset_option(self, positive_seq):
        a, _ = three_phase(positive_seq, noise=0.01, seed=5)
        b, _ = three_phase(positive_seq, noise=0.01, seed=5, dc_offset=0.0)
        for p in "abc":
            np.testing.assert_allclose(a[p].to_numpy(), b[p].to_numpy(), atol=1e-12)


class TestTimeVaryingContent:
    """Sequence content and harmonics may switch on during the record."""

    @staticmethod
    def _switch(t0, level):
        return lambda t: np.where(np.asarray(t) >= t0, level, 0.0)

    def test_negative_sequence_switched_on_midway_only_appears_afterwards(self, positive_seq):
        t0 = 0.05
        abc, truth = three_phase(positive_seq, negative=self._switch(t0, 0.2))
        reference, _ = three_phase(positive_seq)
        t = positive_seq.time
        before = t < t0
        for p in "abc":
            np.testing.assert_allclose(
                abc[p].to_numpy()[before], reference[p].to_numpy()[before], atol=1e-12
            )
        stack = np.column_stack([abc[p].to_numpy() for p in "abc"])
        seq = abc_2_symmetrical_components(stack.astype(complex))
        assert np.abs(seq[~before, 2]).max() > 0.05
        assert truth.metadata["negative_sequence"] == "time-varying"

    def test_constant_callable_matches_the_scalar(self, positive_seq):
        scalar, _ = three_phase(positive_seq, negative=0.1, zero=0.05, harmonics={5: 0.02})
        varying, _ = three_phase(
            positive_seq,
            negative=lambda t: np.full(np.shape(t), 0.1),
            zero=np.full(positive_seq.time.shape, 0.05),
            harmonics={5: lambda t: np.full(np.shape(t), 0.02)},
        )
        for p in "abc":
            np.testing.assert_allclose(scalar[p].to_numpy(), varying[p].to_numpy(), atol=1e-12)

    def test_harmonic_switched_on_midway_is_a_negative_sequence_fifth(self, positive_seq):
        abc, _ = three_phase(positive_seq, harmonics={5: self._switch(0.05, 0.1)})
        reference, _ = three_phase(positive_seq)
        after = positive_seq.time >= 0.05
        difference = np.column_stack(
            [(abc[p] - reference[p]).to_numpy()[after] for p in "abc"]
        )
        assert np.abs(difference).max() == pytest.approx(0.2, rel=0.05)

        def rotation_direction(values):
            alpha_beta = abc_2_ab0(values.T)
            angle = np.unwrap(np.arctan2(alpha_beta[1], alpha_beta[0]))
            return np.sign(np.mean(np.diff(angle)))

        fundamental = np.column_stack([reference[p].to_numpy()[after] for p in "abc"])
        # the fundamental is positive sequence and a physical 5th harmonic
        # rotates against it, so it is negative sequence
        assert rotation_direction(fundamental) == 1.0
        assert rotation_direction(difference) == -1.0
        # no zero-sequence content
        assert np.abs(abc_2_ab0(difference.T)[2]).max() < 1e-9

    def test_scalar_metadata_is_unchanged(self, positive_seq):
        _, truth = three_phase(positive_seq, negative=0.1, zero=0.02)
        assert truth.metadata["negative_sequence"] == 0.1
        assert truth.metadata["zero_sequence"] == 0.02

    def test_metadata_records_everything_that_was_injected(self, positive_seq):
        """A record whose metadata omits its own content cannot be checked."""
        _, truth = three_phase(
            positive_seq,
            negative=0.1,
            zero=0.02,
            harmonics={5: 0.05, 7: 0.02},
            extra_tones={71.0: 0.1},
            dc_offset=0.03,
            noise=0.01,
        )
        assert truth.metadata["harmonics"] == {5: 0.05, 7: 0.02}
        assert truth.metadata["extra_tones"] == {71.0: 0.1}
        assert truth.metadata["dc_offset"] == 0.03
        assert truth.metadata["noise"] == 0.01

    def test_metadata_of_a_clean_set_says_so(self, positive_seq):
        _, truth = three_phase(positive_seq)
        assert truth.metadata["harmonics"] == {}
        assert truth.metadata["extra_tones"] == {}
        assert truth.metadata["dc_offset"] == 0.0
        assert truth.metadata["noise"] == 0.0

    def test_a_time_varying_profile_is_labelled_not_numbered(self, positive_seq):
        switch = lambda t: np.where(np.asarray(t) >= 0.05, 0.05, 0.0)
        _, truth = three_phase(
            positive_seq, harmonics={5: switch}, extra_tones={71.0: switch}, dc_offset=switch
        )
        assert truth.metadata["harmonics"][5] == "time-varying"
        assert truth.metadata["extra_tones"][71.0] == "time-varying"
        assert truth.metadata["dc_offset"] == "time-varying"


class TestExtraTones:
    """Interharmonic and out-of-band tones at an absolute frequency."""

    #: One second, so a spectrum resolves 1 Hz and 71 Hz lands on a bin.
    @pytest.fixture
    def long_seq(self):
        return steady_state(FS, nominal=50.0, amplitude=2.0, duration=1.0)

    @staticmethod
    def _tone(abc, reference):
        """The added content of every phase, as an (n, 3) array."""
        return np.column_stack([(abc[p] - reference[p]).to_numpy() for p in "abc"])

    @staticmethod
    def _peak_frequency(tone):
        spectrum = np.abs(np.fft.rfft(tone))
        return np.fft.rfftfreq(tone.size, 1.0 / FS)[np.argmax(spectrum)]

    def test_tone_sits_at_the_requested_absolute_frequency(self, long_seq):
        abc, _ = three_phase(long_seq, extra_tones={71.0: 0.1})
        reference, _ = three_phase(long_seq)
        tone = self._tone(abc, reference)[:, 0]
        assert self._peak_frequency(tone) == pytest.approx(71.0)
        # a fraction of the fundamental amplitude, which is 2.0 here
        assert np.abs(tone).max() == pytest.approx(0.2, rel=1e-6)

    def test_tone_is_balanced_positive_sequence(self, long_seq):
        abc, _ = three_phase(long_seq, extra_tones={71.0: 0.1})
        reference, _ = three_phase(long_seq)
        tone = self._tone(abc, reference)
        alpha_beta = abc_2_ab0(tone.T)
        assert np.abs(alpha_beta[2]).max() < 1e-12
        angle = np.unwrap(np.arctan2(alpha_beta[1], alpha_beta[0]))
        assert np.sign(np.mean(np.diff(angle))) == 1.0

    def test_tone_does_not_follow_the_fundamental(self):
        """Unlike a harmonic, the tone keeps its frequency off nominal."""
        off_nominal = steady_state(FS, nominal=50.0, frequency=45.0, amplitude=1.0, duration=1.0)
        abc, _ = three_phase(off_nominal, extra_tones={71.0: 0.1})
        reference, _ = three_phase(off_nominal)
        tone = self._tone(abc, reference)[:, 0]
        assert self._peak_frequency(tone) == pytest.approx(71.0)

    def test_tone_may_be_switched_on_midway(self, long_seq):
        t0 = 0.5
        profile = lambda t: np.where(np.asarray(t) >= t0, 0.1, 0.0)
        abc, _ = three_phase(long_seq, extra_tones={71.0: profile})
        reference, _ = three_phase(long_seq)
        tone = self._tone(abc, reference)
        before = long_seq.time < t0
        assert np.abs(tone[before]).max() < 1e-12
        assert np.abs(tone[~before]).max() == pytest.approx(0.2, rel=1e-6)

    def test_default_is_unchanged(self, positive_seq):
        abc, _ = three_phase(positive_seq)
        plain, _ = three_phase(positive_seq, extra_tones=None)
        empty, _ = three_phase(positive_seq, extra_tones={})
        for p in "abc":
            np.testing.assert_array_equal(abc[p].to_numpy(), plain[p].to_numpy())
            np.testing.assert_array_equal(abc[p].to_numpy(), empty[p].to_numpy())
