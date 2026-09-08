"""Phase 3: three_phase -- abc expansion with symmetrical components."""

import numpy as np
import pandas as pd
import pytest

from trimes.synchrophasor import steady_state, three_phase
from trimes.transforms import abc_2_symmetrical_components

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
