# trimes test suite

Pytest-based tests for the `trimes` package.

## Running

```bash
pip install -e ".[test]"      # pytest, scikit-learn + all optional extras
pytest
```

Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.
`conftest.py` forces the non-interactive matplotlib `Agg` backend and closes
figures after every test, so the plotting helpers can be exercised headless.

## Layout

One test module per source module (`test_<module>.py`). `test_package.py`
checks that every submodule imports.

The `test_synchrophasor*.py` files cover `trimes.synchrophasor` (see
`scratchpad/trimes-benchmark-spec.md`):

- `test_synchrophasor.py` -- `AnalyticSignal`, `build_analytic_signal`, the
  generators, the `PeriodicSignal` bridges, and subprocess checks that
  `import trimes` (and every submodule but the `dsp`/`viz`/`control` ones)
  needs only numpy + pandas, and that the `dsp` modules raise a helpful error
  without scipy.
- `test_synchrophasor_concat.py` -- Phase 2 `concat_segments` (continuity to
  1e-9, jump sizes, all spec forms).
- `test_synchrophasor_three_phase.py` -- Phase 3 `three_phase`.
- `test_synchrophasor_conditions.py` -- Phase 4 test-condition generators and
  the `P_CLASS_LIMITS` / `M_CLASS_LIMITS` tables.
- `test_synchrophasor_metrics.py` -- Phase 5 `errors` / `summary` /
  `response_time` / `delay_time` / `overshoot` / `compliance`.
- `test_synchrophasor_plot.py` -- Phase 6 plotting helpers (Agg smoke tests).

## Bugs found and fixed

- `base.time_to_seconds` divided a `DatetimeIndex` by a hard-coded `1e9`
  (nanoseconds), but pandas 2.x stores `pd.to_datetime([...])` as
  `datetime64[us]`, so the result was off by 1000x. Now it takes the
  difference to the first sample in the index's native resolution and converts
  via `TimedeltaIndex.total_seconds()`, which is unit-agnostic.
  (`test_base.py::TestTimeToSeconds`)
- `fourier.get_fourier_coef_real` / `get_fourier_coef_imag` built a length
  `k * len(x)` reference wave and multiplied it by `x`, which raised a
  broadcasting `ValueError` for every harmonic `k > 1`. The reference wave is
  now `len(x)` samples long and completes `k` cycles over that span.
  (`test_fourier.py::TestFourierCoefficients`)
- `transforms.inverse_frobenius_matrix_symcomp` carried a spurious `1/3`
  factor, so `symmetrical_components_2_abc(abc_2_symmetrical_components(x))`
  returned `x / 3` instead of `x`. (`test_transforms.py::TestSymmetricalComponents`)
- `metrics.get_metric_np` broadcast the reference to a hard-coded 2 columns
  (`ts.ndim`), breaking any comparison of a 3+ column frame; now uses
  `ts.shape`. (`test_metrics.py`)
- `base.interp_np_matrix` / `interp_series` called the private
  `np.core.multiarray.interp`; now `np.interp`.
