from functools import partial

import numpy as np
import pandas as pd

try:
    from scipy.signal import lfilter
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on env
    raise ModuleNotFoundError(
        "trimes.filter needs scipy; install it with: pip install 'trimes[dsp]'"
    ) from _exc

from trimes.base import (
    apply_to_columns,
)


def pt1_filter(ts: np.ndarray, T: float, dt: float) -> np.ndarray:
    alpha = T / (T + dt)
    b = [1 - alpha]  # Numerator coefficients
    a = [1, -alpha]  # Denominator coefficients
    lfilter_part = partial(lfilter, b, a)
    return apply_to_columns(
        ts,
        lfilter_part,
    )
