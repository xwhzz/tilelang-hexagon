"""Linux ARM64 host build using the target's existing FastRPC runtime.

No target program is run by this module. Android remains the default build path.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
from pathlib import Path


def _check(argv, *, cwd, env, verbose):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    if verbose or p.returncode:
        print("[linux-host]", " ".join(map(str, argv)))
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
    if p.returncode:
        raise RuntimeError(f"Linux host build failed: {argv[0]} ({p.returncode})")
    return p.stdout


def build_linux_host(project_dir, name, sdk, env, verbose=False):
    """Build the project's POSIX agent or one-shot driver against BSP libraries."""
    root = Path(project_dir).resolve()
    out = root / "linux_aarch64"
    out.mkdir(exist_ok=True)
    sdk_root = Path(sdk.sdk_root)
    runtime_dir = os.environ.get("TILELANG_HEXAGON_LINUX_RPC_LIB_DIR")
    if not runtime_dir:
        raise RuntimeError("Set TILELANG_HEXAGON_LINUX_RPC_LIB_DIR to the target BSP's captured runtime libraries")
    runtime = Path(runtime_dir).resolve()
    rpc = runtime / "libcdsprpc.so"
    if not rpc.is_file():
        raise RuntimeError(f"Target FastRPC library missing: {rpc}")
    cc = shutil.which(os.environ.get("TILELANG_HEXAGON_LINUX_CC", "aarch64-linux-gnu-gcc"), path=env["PATH"])
    if not cc:
        raise RuntimeError("Linux ARM64 compiler not found")
    includes = [out, sdk_root / "incs", sdk_root / "incs/stddef", sdk_root / "utils/examples", sdk_root / "ipc/fastrpc/incs"]
    inc_flags = ["-I" + str(p) for p in includes]
    _check([sdk.qaic, "-mdll", "-o", str(out), *inc_flags, str(root / (name + ".idl"))], cwd=root, env=env, verbose=verbose)
    agent = (root / (name + "_agent.c")).is_file()
    exe = out / (name + ("_agent" if agent else "_test"))
    source = root / (name + ("_agent.c" if agent else "_host.c"))
    sources = [out / (name + "_stub.c"), sdk_root / "utils/examples/dsp_capabilities_utils.c", source]
    command = [
        cc,
        "-std=gnu11",
        "-O2",
        "-fPIC",
        "-ffunction-sections",
        "-fdata-sections",
        "-DVERIFY_PRINT_ERROR",
        *inc_flags,
        *map(str, sources),
        "-Wl,--gc-sections",
        "-L" + str(runtime),
        "-Wl,-rpath-link," + str(runtime),
        "-l:libcdsprpc.so",
        "-ldl",
        "-lpthread",
        "-o",
        str(exe),
    ]
    _check(command, cwd=root, env=env, verbose=verbose)
    # Check the host ABI against supplied BSP libraries before deployment.
    readelf = shutil.which("aarch64-linux-gnu-readelf", path=env["PATH"])
    if not readelf:
        raise RuntimeError("aarch64-linux-gnu-readelf not found")

    def inspect(path, *flags):
        return _check([readelf, *flags, str(path)], cwd=root, env=env, verbose=False)

    header = inspect(exe, "-h")
    program = inspect(exe, "-l")
    if "AArch64" not in header or "/lib/ld-linux-aarch64.so.1" not in program:
        raise RuntimeError("Expected an ARM64 Linux host executable")
    checked = set()
    queue = [exe]
    while queue:
        binary = queue.pop()
        if binary.name in checked:
            continue
        checked.add(binary.name)
        if "AArch64" not in inspect(binary, "-h"):
            raise RuntimeError(f"Wrong architecture: {binary}")
        for needed in re.findall(r"\(NEEDED\).*\[(.*?)\]", inspect(binary, "-d")):
            lib = runtime / needed
            if not lib.exists():
                raise RuntimeError(f"Target runtime dependency not captured: {needed}")
            queue.append(lib)
        current = None
        for line in inspect(binary, "--version-info").splitlines():
            m = re.search(r"File: (\S+)", line)
            if m:
                current = runtime / m.group(1)
            m = re.search(r"Name: (\S+)\s+Flags:.*Version:", line)
            if current and m:
                info = inspect(current, "--version-info")
                definitions = info.split("Version definition section", 1)
                definitions = definitions[1].split("Version needs section", 1)[0] if len(definitions) == 2 else ""
                available = set(re.findall(r"Name: (\S+)", definitions))
                if m.group(1) not in available:
                    raise RuntimeError(f"ABI requirement unavailable: {current.name} {m.group(1)}")
    exports = set()
    required = set()
    for binary in [exe, *(runtime / name for name in checked if name != exe.name)]:
        for line in inspect(binary, "--dyn-syms", "--wide").splitlines():
            columns = line.split()
            if len(columns) >= 8 and columns[0].endswith(":"):
                symbol = columns[7].replace("@@", "@")
                if columns[6] != "UND" and columns[4] in {"GLOBAL", "WEAK"}:
                    exports.add(symbol)
                    exports.add(symbol.split("@")[0])
                elif columns[6] == "UND" and columns[4] == "GLOBAL":
                    required.add(symbol)
    missing = required - exports
    if missing:
        raise RuntimeError(f"Target runtime lacks required dynamic symbols: {sorted(missing)}")
    return str(exe)


def validate_dsp_artifact(path, arch):
    """Reject an incorrectly targeted DSP library before any deployment."""
    header = Path(path).read_bytes()[:64]
    if header[:4] != b"\x7fELF" or len(header) < 52:
        raise RuntimeError("DSP artifact is not ELF")
    endian = "<" if header[5] == 1 else ">"
    machine = struct.unpack_from(endian + "H", header, 18)[0]
    flags = struct.unpack_from(endian + "I", header, 36 if header[4] == 1 else 48)[0]
    if machine != 164 or (flags & 255) != int(arch.removeprefix("v"), 16):
        raise RuntimeError(f"DSP artifact architecture mismatch: machine={machine}, flags={flags:#x}, expected={arch}")
