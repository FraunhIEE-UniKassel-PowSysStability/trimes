"""Shims for optional dependencies.

Core ``trimes`` (everything reachable from ``import trimes``) must run on
numpy + pandas alone. ``icecream`` is a debugging convenience only; when it is
not installed we fall back to a no-op ``ic`` so the imports still succeed.

Install the extras with ``pip install "trimes[control]"`` to get the real
``icecream`` and the ``control`` package.
"""

from __future__ import annotations

try:  # pragma: no cover - trivial import shim
    from icecream import ic
except ModuleNotFoundError:  # pragma: no cover - exercised only without icecream

    def ic(*args):
        """No-op stand-in for :func:`icecream.ic` when icecream is absent.

        Mirrors icecream's return contract: nothing in, ``None`` out; one
        argument in, that argument back; several arguments in, the tuple.
        """
        if not args:
            return None
        return args[0] if len(args) == 1 else args


__all__ = ["ic"]
