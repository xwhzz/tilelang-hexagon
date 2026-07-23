"""Compile the walkthrough sources with an existing v79 compile database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


def compile_replacement(
    entries: list[dict[str, str]],
    reference_suffix: str,
    source: Path,
    output: Path,
) -> None:
    entry = next(item for item in entries if item["file"].endswith(reference_suffix))
    args = shlex.split(entry["command"])
    replaced: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-o":
            replaced.extend(["-o", str(output)])
            index += 2
            continue
        if arg == entry["file"]:
            replaced.append(str(source))
        else:
            replaced.append(arg)
        index += 1
    subprocess.run(replaced, cwd=entry["directory"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-db", type=Path, required=True)
    parser.add_argument("--walkthrough-dir", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads(args.compile_db.read_text(encoding="utf-8"))
    root = args.walkthrough_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="tl-llama-walkthrough-") as temp:
        temp_dir = Path(temp)
        compile_replacement(
            entries,
            "tl_ggml_matmul.cc",
            root / "step3_tl_ggml_adapter.cc",
            temp_dir / "adapter.o",
        )
        compile_replacement(
            entries,
            "kernel_qmatmul_hmx_staged_32x256x2048.cc",
            root / "generated/kernel_walkthrough_q4_m32_n256_k2048.cc",
            temp_dir / "kernel.o",
        )
    print("Hexagon v79 compile: PASS (adapter + generated kernel)")


if __name__ == "__main__":
    main()
