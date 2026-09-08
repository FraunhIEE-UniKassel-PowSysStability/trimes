"""Import-level checks for the trimes package."""

import importlib
import subprocess
import sys

import pytest

SUBMODULES = [
    "trimes._optional",
    "trimes.analytic",
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


def test_import_trimes_stays_light():
    """``import trimes`` must not pull in optional extras or the whole
    synchrophasor benchmark layer -- it runs on numpy + pandas alone."""
    code = (
        "import sys, trimes; "
        "heavy = [m for m in "
        "('scipy', 'matplotlib', 'control', 'sklearn', 'trimes.synchrophasor') "
        "if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"import trimes pulled in: {out.stdout.strip()}"
