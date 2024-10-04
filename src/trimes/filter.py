import numpy as np


def pt1(x, Tf, sample_time):
    """Implement a first-order low-pass filter.

    The input data is x, the filter's cutoff frequency is omega_c
    [rad/s] and the sample time is T [s].  The output is y.
    """
    y = np.empty(x.shape[0], dtype=float)
    y[0] = x[0]
    for k in np.arange(1, x.shape[0]):
        y[k] = y[k - 1] + (x[k] - y[k - 1]) / Tf * sample_time
    return y
