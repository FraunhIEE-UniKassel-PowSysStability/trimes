# trimes

*trimes* (transient time series) is a python package for transient time series data in pandas format. The application is actually for all time series data where the time vector has a numerical format (e.g numpy's float64) - as opposed to the frequently used *DateTime* format. To the best of our knowledge, there is currently no other python package focusing on transient time series data as described and the mentioned  *DateTime* format is not convenient for transient time series.

trimes provides functionality for pandas DataFrames (in the format mentioned above) for the following use cases:

- get data points 
- interpolation
- resampling
- regression 
- signal generation (harmonics, symmetrical components)
- comparison of times series (difference, boundaries, envelopes)
- metrics (e.g. root mean squared error)
- step response analysis
- plotting
- FFT / spectra (`trimes.fft`) and a single-tone frequency / amplitude / phase estimator (`trimes.phasor`)
- analytic test signals with closed-form ground truth (`trimes.analytic`), the IEC/IEEE 60255-118-1:2018 test matrix and TVE / FE / RFE compliance metrics (`trimes.synchrophasor`)

and more.

Most helpers are also available as methods via the `.trimes` accessor on a
Series or DataFrame, e.g. `df.trimes.resample(1e-3)`.

Have a look at the [documentation](https://fraunhiee-unikassel-powsysstability.github.io/trimes/docs/index.html) to get started.

## Installation

```shell
pip install trimes
```

The core (`import trimes`, the `.trimes` accessor, `trimes.base`,
`trimes.analytic`, `trimes.phasor`, `trimes.synchrophasor`) needs only numpy and
pandas. Heavier functionality lives behind optional extras -- `trimes[viz]`
(matplotlib), `trimes[dsp]` (scipy), `trimes[control]` (python-control), or
`trimes[all]`.
