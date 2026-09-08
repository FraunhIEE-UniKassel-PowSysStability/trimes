"""Import-level checks for the trimes package."""

import importlib

import pytest

SUBMODULES = [
    "trimes._optional",
    "trimes.base",
    "trimes.comparison",
    "trimes.control",
    "trimes.derivative",
    "trimes.electrical",
    "trimes.fft",
    "trimes.filter",
    "trimes.fourier",
    "trimes.metrics",
    "trimes.phasor",
    "trimes.plots",
    "trimes.regression",
    "trimes.root_mean_square",
    "trimes.signal_generation",
    "trimes.signal_processing",
    "trimes.synchrophasor",
    "trimes.synchrophasor.concat",
    "trimes.synchrophasor.conditions",
    "trimes.synchrophasor.generators",
    "trimes.synchrophasor.limits",
    "trimes.synchrophasor.metrics",
    "trimes.synchrophasor.plot",
    "trimes.synchrophasor.signal",
    "trimes.synchrophasor.three_phase",
    "trimes.synchrophasor.util",
    "trimes.transforms",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    importlib.import_module(name)


def test_base_names_reexported_from_package():
    import trimes

    for attr in ("resample", "interp_series", "get_between", "is_uniformly_sampled"):
        assert hasattr(trimes, attr)


def test_analytic_signal_is_reexported_from_package():
    import trimes

    assert trimes.AnalyticSignal is not None
