#!/usr/bin/env python3
"""Lower a generic TileLang matmul and show the Hexagon hardware calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tilelang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tilelang_root))
    import tilelang
    from examples.hexagon.example_matmul import make_matmul
    from tilelang.utils.target import determine_target

    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            make_matmul(128, 128, 256, 64, 64),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = str(artifact.kernel_source)
    print(source)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(source, encoding="utf-8")

    print("Hexagon lowering highlights")
    for line in source.splitlines():
        if any(
            token in line
            for token in (
                "tl_vtcm_base",
                "tl_vtcm_shared_high_water",
                "tl_hexagon_hmx_gemm",
                "for (int32_t blockIdx",
            )
        ):
            print(line)


if __name__ == "__main__":
    main()
