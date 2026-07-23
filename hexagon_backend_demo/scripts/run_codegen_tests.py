#!/usr/bin/env python3
"""Run offline Hexagon codegen contracts without requiring pytest or a device."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tilelang-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.tilelang_root.resolve()
    sys.path.insert(0, str(root))

    q4 = load_module(
        root / "testing/python/hexagon/test_tilelang_hmx_intrin.py",
        "test_tilelang_hmx_intrin",
    )
    tests = [
        (name, getattr(q4, name))
        for name in dir(q4)
        if name.startswith("test_")
    ]
    for name, test in tests:
        print(f"[q4] {name}")
        test()

    import tilelang
    from tilelang.hexagon.qgemv import (
        Q8_0_TILED_LAYOUT,
        make_q8_0_dot_prim_func,
    )
    from tilelang.utils.target import determine_target

    layout = Q8_0_TILED_LAYOUT
    assert layout.source_tile_bytes == 1088
    assert layout.staged_tile_bytes == 1152
    assert layout.staged_weight_bytes(2048) == 64 * 1152
    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            make_q8_0_dot_prim_func(2048, symbol="demo_qgemv_q8_0"),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = str(artifact.kernel_source)
    assert "tl_hexagon_q8_0_dot_32x1(2048" in source
    assert "tl_hexagon_hmx" not in source
    print("[q8] layout and dot-atom lowering")
    print(f"passed {len(tests) + 1} offline codegen contracts")


if __name__ == "__main__":
    main()
