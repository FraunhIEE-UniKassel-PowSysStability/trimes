"""Import-level checks for the trimes package."""

import importlib
import subprocess
import sys

import pytest

SUBMODULES = [
    "trimes._optional",
    "trimes.accessor",
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


def test_lazy_submodule_attribute_access():
    """Submodules resolve as attributes of the package without an explicit import."""
    import trimes

    assert trimes.fft.rfft.__name__ == "rfft"
    assert trimes.transforms.abc_2_dq0.__name__ == "abc_2_dq0"
    # cached on the package after first access
    assert "fft" in vars(trimes)
    assert "fft" in dir(trimes)

    with pytest.raises(AttributeError):
        trimes.does_not_exist


def test_lazy_access_does_not_eagerly_import_scipy():
    """Touching a numpy-only submodule via attribute access stays scipy-free."""
    code = (
        "import sys, trimes; "
        "trimes.transforms; "
        "print('scipy' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
