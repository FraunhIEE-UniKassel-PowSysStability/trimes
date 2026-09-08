"""IEC/IEEE 60255-118-1:2018 test parameters and compliance limits.

All numbers here are taken from the 2018 standard (Tables 2-9 and the
associated equations). Several limits depend on the phasor **reporting rate**
``Fs`` (not the waveform sampling rate) and are therefore given as functions
rather than constants; the module-level ``*_CLASS_LIMITS`` dicts hold the
reporting-rate-independent values and the full-range references.

Sign / equation conventions (standard, Clause 6):

* off-nominal tone -- Eq (12): ``Xm cos(2*pi*fin*t)``
* harmonic -- Eq (15): fundamental + ``Xm*kx*cos(2*pi*n*f0*t)``
* out-of-band -- Eq (18): fundamental + ``Xm*kx*cos(2*pi*fi*t)``
* modulation -- Eq (25): ``Xm[1 + kx*cos(wm t)] * cos[2*pi*f0*t + ka*cos(wm t - pi)]``
  so ``total_phase = 2*pi*f0*t - ka*cos(wm t)`` and
  ``f = f0 + ka*fm*sin(wm t)``
* frequency ramp -- Eq (33): ``Xm cos(2*pi*f0*t + pi*Rf*t**2)``
* step -- Eq (41): ``Xm[1 + kx*u(t)] * cos[2*pi*f0*t + ka*u(t)]``
"""

from __future__ import annotations

NOMINAL_HZ = 50.0
DEFAULT_REPORTING_RATE_HZ = 50.0


# --------------------------------------------------------------------------- #
# Reference tables (reporting-rate-independent values / full-range references)
# --------------------------------------------------------------------------- #
P_CLASS_LIMITS: dict[str, dict | None] = {
    "frequency_range": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.4,
        "range_hz": 2.0,  # +-2 Hz, all Fs (Table 2)
    },
    "magnitude": {
        "tve": 1.0,
        "fe": None,
        "rfe": None,
        "range_v": (0.8, 1.2),  # 80-120 % rated voltage
        "range_i": (0.1, 2.0),  # 10-200 % rated current
    },
    "harmonic": {
        "tve": 1.0,
        "fe": 0.005,  # all Fs (Table 3)
        "rfe": 0.4,
        "level": 0.01,  # 1 % of each harmonic, orders 2..50
        "orders": (2, 50),
    },
    "out_of_band": None,  # P class: no out-of-band requirement (Table 2/3)
    "modulation": {
        "tve": 3.0,
        "am_depth": 0.1,  # kx
        "pm_depth": 0.1,  # ka
        "fr_cap_hz": 2.0,  # Fr = min(Fs/10, 2)
        "fr_cap_ratio": 0.1,
        # fe = 0.03 * Fr ; rfe = 0.18 * pi * Fr**2   (Table 5)
    },
    "frequency_ramp": {
        "tve": 1.0,
        "fe": 0.01,
        "rfe": 0.4,  # Table 7
        "rocof_hz_s": 1.0,
        "range_hz": 2.0,
        # exclusion_s = larger of 2/Fs or 2/f0  (Table 6)
        "exclusion_n": 2,
    },
    "magnitude_step": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.4,
        "step": 0.1,  # kx = +-0.1
        # response_time_s = 2/f0 ; delay_time_s = 1/(4*Fs) ; overshoot = 0.05
        "response_time_num": 2.0,
        "overshoot": 0.05,
    },
    "phase_step": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.4,
        "step_deg": 10.0,  # ka = +-pi/18
        "response_time_num": 2.0,
        "overshoot": 0.05,
    },
}

M_CLASS_LIMITS: dict[str, dict | None] = {
    "frequency_range": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.1,
        "range_hz": 5.0,  # full range; actual = f(Fs), see frequency_range_hz()
    },
    "magnitude": {
        "tve": 1.0,
        "fe": None,
        "rfe": None,
        "range_v": (0.1, 1.2),
        "range_i": (0.1, 2.0),
    },
    "harmonic": {
        "tve": 1.0,
        "fe": 0.025,  # Fs > 20; 0.005 for Fs <= 20, see harmonic_error_limits()
        "rfe": None,  # M class: no RFE requirement for the harmonic test
        "level": 0.10,
        "orders": (2, 50),
    },
    "out_of_band": {
        "tve": 1.3,
        "fe": 0.01,
        "rfe": None,  # M class: no RFE requirement for the OOB test
        "level": 0.10,  # 10 % interharmonic outside the passband, Fs >= 10
    },
    "modulation": {
        "tve": 3.0,
        "am_depth": 0.1,
        "pm_depth": 0.1,
        "fr_cap_hz": 5.0,  # Fr = min(Fs/5, 5)
        "fr_cap_ratio": 0.2,
        # fe = 0.06 * Fr ; rfe = 0.18 * pi * Fr**2
    },
    "frequency_ramp": {
        "tve": 1.0,
        "fe": 0.01,
        "rfe": 0.2,  # Table 7
        "rocof_hz_s": 1.0,
        "range_hz": 5.0,
        "exclusion_n": 7,  # exclusion_s = larger of 7/Fs or 7/f0
    },
    "magnitude_step": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.1,
        "step": 0.1,
        # response_time_s = larger of 7/Fs or 7/f0
        "response_time_num": 7.0,
        "overshoot": 0.10,
    },
    "phase_step": {
        "tve": 1.0,
        "fe": 0.005,
        "rfe": 0.1,
        "step_deg": 10.0,
        "response_time_num": 7.0,
        "overshoot": 0.10,
    },
}

_TABLES = {"P": P_CLASS_LIMITS, "M": M_CLASS_LIMITS}


# --------------------------------------------------------------------------- #
# Reporting-rate-dependent quantities (functions, per the standard's formulas)
# --------------------------------------------------------------------------- #
def frequency_range_hz(class_: str, reporting_rate: float | None = None) -> float:
    """Off-nominal frequency test range ``+- range_hz`` (Table 2).

    P class: ``2.0`` for all Fs. M class: ``2.0`` for ``Fs < 10``, ``Fs/5`` for
    ``10 <= Fs < 25``, ``5.0`` for ``Fs >= 25``.
    """
    if class_.upper() == "P":
        return 2.0
    fs = DEFAULT_REPORTING_RATE_HZ if reporting_rate is None else float(reporting_rate)
    if fs < 10.0:
        return 2.0
    if fs < 25.0:
        return fs / 5.0
    return 5.0


def modulation_fr_hz(class_: str, reporting_rate: float) -> float:
    """Maximum modulation frequency ``Fr`` (Table 4).

    P: ``min(Fs/10, 2)``. M: ``min(Fs/5, 5)``.
    """
    lim = _TABLES[class_.upper()]["modulation"]
    return min(float(reporting_rate) * lim["fr_cap_ratio"], lim["fr_cap_hz"])


# Table 5, verbatim: reporting_rate -> {"P": (Fr, FE, RFE), "M": (Fr, FE, RFE)}.
# NOTE (Table 5): "For test compliance, use the rounded values shown in the
# table, not the formulas."
_MODULATION_TABLE_5 = {
    10: {"P": (1.0, 0.03, 0.6), "M": (2.0, 0.12, 2.3)},
    12: {"P": (1.2, 0.04, 0.8), "M": (2.4, 0.14, 3.3)},
    15: {"P": (1.5, 0.05, 1.3), "M": (3.0, 0.18, 5.1)},
    20: {"P": (2.0, 0.06, 2.3), "M": (4.0, 0.24, 9.0)},
    25: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
    30: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
    50: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
    60: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
    100: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
    120: {"P": (2.0, 0.06, 2.3), "M": (5.0, 0.30, 14.0)},
}


def modulation_error_limits(class_: str, reporting_rate: float) -> dict:
    """``{tve, fe, rfe}`` for the modulation test at this reporting rate.

    Uses the rounded Table 5 values for the standard reporting rates
    (10, 12, 15, 20, 25, 30, 50, 60, 100, 120 Hz); for any other rate it falls
    back to the formulas ``fe = 0.03*Fr`` (P) / ``0.06*Fr`` (M) and
    ``rfe = 0.18*pi*Fr**2``.
    """
    import math

    cls = class_.upper()
    row = _MODULATION_TABLE_5.get(int(round(float(reporting_rate))))
    if row is not None and float(reporting_rate) == int(round(float(reporting_rate))):
        _fr, fe, rfe = row[cls]
        return {"tve": 3.0, "fe": fe, "rfe": rfe}

    fr = modulation_fr_hz(cls, reporting_rate)
    fe_coeff = 0.03 if cls == "P" else 0.06
    return {"tve": 3.0, "fe": fe_coeff * fr, "rfe": 0.18 * math.pi * fr * fr}


def harmonic_error_limits(class_: str, reporting_rate: float | None = None) -> dict:
    """``{tve, fe, rfe}`` for the single-harmonic test (Table 3).

    P: ``fe 0.005``, ``rfe 0.4`` for all Fs. M: ``fe 0.005`` for ``Fs <= 20``
    else ``0.025``; no RFE requirement.
    """
    if class_.upper() == "P":
        return {"tve": 1.0, "fe": 0.005, "rfe": 0.4}
    fs = DEFAULT_REPORTING_RATE_HZ if reporting_rate is None else float(reporting_rate)
    return {"tve": 1.0, "fe": 0.005 if fs <= 20.0 else 0.025}


def ramp_exclusion_s(
    class_: str, reporting_rate: float, nominal: float = NOMINAL_HZ
) -> float:
    """Ramp exclusion half-interval: ``larger of n/Fs or n/f0`` (Table 6).

    ``n`` is 2 (P) or 7 (M).
    """
    n = _TABLES[class_.upper()]["frequency_ramp"]["exclusion_n"]
    return max(n / float(reporting_rate), n / float(nominal))


def step_response_time_s(
    class_: str, reporting_rate: float, nominal: float = NOMINAL_HZ
) -> float:
    """Max phasor response time for a step (Table 8).

    P: ``2/f0``. M: ``larger of 7/Fs or 7/f0``.
    """
    if class_.upper() == "P":
        return 2.0 / float(nominal)
    return max(7.0 / float(reporting_rate), 7.0 / float(nominal))


def step_delay_time_s(reporting_rate: float) -> float:
    """Max |delay time| for a step: ``1/(4*Fs)`` (Table 8)."""
    return 1.0 / (4.0 * float(reporting_rate))


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #
def limits_for(condition: str, class_: str) -> dict | None:
    """The reference limit / parameter dict for a condition and class.

    ``None`` if the class is exempt from the condition. Reporting-rate-dependent
    entries hold the full-range value; use :func:`error_limits` (with a
    ``reporting_rate``) for the exact numbers.
    """
    table = _TABLES[class_.upper()]
    if condition not in table:
        raise KeyError(f"unknown test condition {condition!r}")
    entry = table[condition]
    return dict(entry) if entry is not None else None


def error_limits(
    condition: str, class_: str, reporting_rate: float | None = None
) -> dict:
    """The ``tve`` / ``fe`` / ``rfe`` limits that apply to a condition.

    Missing / not-required entries are omitted; an empty dict means the class is
    exempt. Reporting-rate-dependent conditions (``modulation``, ``harmonic``,
    ...) need ``reporting_rate`` for the exact values; without it the
    full-range reference is returned.
    """
    entry = limits_for(condition, class_)
    if entry is None:
        return {}

    if condition == "modulation" and reporting_rate is not None:
        return modulation_error_limits(class_, reporting_rate)
    if condition == "harmonic":
        return harmonic_error_limits(class_, reporting_rate)

    return {k: entry[k] for k in ("tve", "fe", "rfe") if entry.get(k) is not None}
