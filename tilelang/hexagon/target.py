"""Hexagon target normalization.

A normalized Hexagon target names its DSP architecture in ``mcpu``
(``hexagonv79``) and carries the capabilities compilation reads:

* ``max_vector_bits``: the HVX vector width, 1024. It is a kind default
  (``src/hexagon/target.cc``), so every Hexagon target has it.
* ``vtcm-capacity``: VTCM bytes available to a kernel.
* ``num-cores``: hardware threads available to the worker pool.

``vtcm-capacity`` and ``num-cores`` default only for architectures measured on
hardware. Elsewhere they stay unset unless given, and only the runtime checks
the actual grant. The architecture comes from ``mcpu`` when the target has one,
else from the ``HEXAGON_DSP_ARCH`` environment variable, else v79.
"""

from __future__ import annotations

import os
import re

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer

SUPPORTED_ARCHS = ("v73", "v75", "v79", "v81")
DEFAULT_ARCH = "v79"

# Capabilities measured on hardware, per architecture.
_MEASURED_CAPABILITIES = {
    # SM8750 (OnePlus 13): HAP grants 8 MiB of VTCM, QuRT reports 6 hardware threads.
    "v79": {"vtcm-capacity": 8 << 20, "num-cores": 6},
}

_ARCH_PATTERN = re.compile(r"^(?:hexagon)?v?(\d+)$")


def _canonical_arch(value) -> str | None:
    match = _ARCH_PATTERN.match(str(value).strip().lower())
    return f"v{match.group(1)}" if match else None


def hexagon_arch(target: TargetLike | None) -> str | None:
    """The DSP architecture of a Hexagon target (``"v79"``), or None if it names none."""

    if target is None:
        return None
    if not isinstance(target, Target):
        target = Target(target)
    if target.kind.name != "hexagon" or "mcpu" not in target.attrs:
        return None
    return _canonical_arch(target.attrs["mcpu"])


def normalize_hexagon_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name != "hexagon":
            return None
        parsed = target
    elif isinstance(target, dict):
        if target.get("kind") != "hexagon":
            return None
        parsed = Target(target)
    elif isinstance(target, str):
        text = target.strip()
        if text == "hexagon":
            parsed = Target(text)
        elif text.startswith("{"):
            try:
                parsed = Target(text)
            except Exception:
                return None
            if parsed.kind.name != "hexagon":
                return None
        else:
            return None
    else:
        return None

    config = dict(parsed.export())
    requested = config.get("mcpu") or os.environ.get("HEXAGON_DSP_ARCH") or DEFAULT_ARCH
    arch = _canonical_arch(requested)
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Hexagon architecture {requested!r} is not supported; use one of {', '.join(SUPPORTED_ARCHS)}, "
            f"e.g. {{'kind': 'hexagon', 'mcpu': 'hexagon{DEFAULT_ARCH}'}}"
        )
    config["mcpu"] = f"hexagon{arch}"
    for key, value in _MEASURED_CAPABILITIES.get(arch, {}).items():
        config.setdefault(key, value)
    return Target(config)


register_target_normalizer("hexagon", normalize_hexagon_target, override=True)
