"""trimes -- transient time series in pandas format.

``import trimes`` needs only numpy and pandas. It re-exports the
:mod:`trimes.base` helpers and :class:`trimes.analytic.AnalyticSignal`, and
registers the ``.trimes`` accessor (``df.trimes.resample(...)`` etc.).

Every submodule is reachable as an attribute without an explicit import
(``trimes.fft.rfft``); the import happens lazily on first access, so the
optional dependencies are only pulled in when actually used:

* ``trimes[viz]``      -- matplotlib (``trimes.plots``, ``PeriodicSignal.plot``)
* ``trimes[dsp]``      -- scipy (``trimes.fft``, ``trimes.filter``, ``trimes.derivative``)
* ``trimes[control]``  -- python-control + icecream (``trimes.control``)
"""

import importlib
from importlib import metadata

from trimes.base import *  # noqa: F401,F403  (re-export, guarded by base.__all__)
from trimes.base import __all__ as _base_all
from trimes.analytic import AnalyticSignal
from trimes import accessor as _accessor  # noqa: F401  (registers `.trimes`)

try:
    __version__ = metadata.version("trimes")
except metadata.PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [*_base_all, "AnalyticSignal", "__version__"]

# Submodules exposed via lazy attribute access (PEP 562), so that e.g.
# `trimes.fft` works after a bare `import trimes` without importing scipy up
# front.
_SUBMODULES = frozenset(
    {
        "accessor",
        "analytic",
        "base",
        "comparison",
        "control",
        "derivative",
        "electrical",
        "fft",
        "filter",
        "fourier",
        "metrics",
        "phasor",
        "plots",
        "regression",
        "root_mean_square",
        "signal_generation",
        "signal_processing",
        "step_response",
        "synchrophasor",
        "transforms",
    }
)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache: __getattr__ runs only once per name
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES, *__all__})
