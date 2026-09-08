"""trimes -- transient time series in pandas format.

``import trimes`` needs only numpy and pandas. Heavier functionality lives in
submodules behind optional extras:

* ``trimes[viz]``      -- matplotlib (``trimes.plots``, ``PeriodicSignal.plot``)
* ``trimes[dsp]``      -- scipy (``trimes.fft``, ``trimes.filter``, ``trimes.derivative``)
* ``trimes[control]``  -- python-control + icecream (``trimes.control``)
"""

from trimes.base import *  # noqa: F401,F403  (re-export, guarded by base.__all__)
from trimes.base import __all__ as _base_all
from trimes.synchrophasor import AnalyticSignal

__all__ = [*_base_all, "AnalyticSignal"]
