"""Q4 x half dequantization through the real header on the SDK simulator/device.

Opt in with TL_HEXAGON_SIMULATOR_TEST=1 and HEXAGON_TOOLS_ROOT. Host simulator
dependencies belong on LD_LIBRARY_PATH. No device or native TileLang import
is required by this test. Both architectures compile with ordinary flags:
the production helper must supply its own local IEEE feature boundary.
Both architectures require exact RNE across all finite scales and groups
with one exceptional scale in each lane position. The old
qfloat expression is retained only as diagnostic output, not as an oracle.
"""

import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest


SIMULATOR = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)
DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _tool(name):
    root = os.environ.get("HEXAGON_TOOLS_ROOT")
    path = Path(root) / "Tools/bin" / name if root else Path(shutil.which(name) or name)
    if not path.is_file():
        pytest.fail(f"{name} is required for the requested simulator test: {path}")
    return str(path)


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_q4_half_dequantization_on_simulator(arch, tmp_path, record_property):
    root = Path(__file__).resolve().parents[3]
    fixture = Path(__file__).with_name("hvx_dequant4_sim.cc")
    executable, output = tmp_path / f"dequant-{arch}", tmp_path / "output.bin"
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-I",
            str(root / "src"),
            str(fixture),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert "HVX_DEQUANT_SIM_OK" in simulated.stdout
    data = output.read_bytes()
    iterations, rows, cols = 248 + 32, 128, 128
    assert struct.unpack_from("<6I", data) == (0x51444846, int(arch[1:]), iterations, rows, cols, 4)
    actual = np.frombuffer(data, dtype="<u2", offset=24).reshape(iterations, 4, rows * cols)

    # Independent logical-coordinate indexing of native [nt, kt] Crouton
    # tiles. Every group contains every nibble four times, with row/column
    # variation: a lane or scale-group permutation cannot pass accidentally.
    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    offset = (row // 32 * (cols // 32) + col // 32) * 1024
    offset = offset + (col % 32 // 2) * 64 + (row % 32) * 2 + col % 2
    codes_unsigned = (row * 3 + col * 5) % 16
    codes_signed = np.where(codes_unsigned >= 8, codes_unsigned - 16, codes_unsigned)
    finite_bits = np.arange(65536, dtype=np.uint16)
    finite_bits = finite_bits[(finite_bits & 0x7C00) != 0x7C00]
    assert len(finite_bits) == 63488
    all_bits = finite_bits.reshape(248, 2, rows)
    mixed = np.empty((32, 2, rows), dtype=np.uint16)
    for lane in range(32):
        mixed[lane] = np.where((np.arange(rows)[None, :] + np.arange(2)[:, None]) % 2, 0x3C01, 0xBC03)
        mixed[lane, 0, (lane // 8) * 32 + lane] = 0x8001 if lane % 2 else 0x0001
        mixed[lane, 1, ((lane // 8 + 1) % 4) * 32 + 31 - lane] = 0x8000 if lane % 2 else 0x0000
    all_scales = np.concatenate((all_bits, mixed)).view(np.float16)
    legacy_mismatches = 0
    for iteration in range(iterations):
        scales = all_scales[iteration].T.repeat(64, axis=1).astype(np.float32)
        for signed, codes in enumerate((codes_signed, codes_unsigned)):
            # Each integer*half product is exactly representable in float32;
            # only the final conversion rounds. Compare half bits, including
            # negative zero, subnormal outputs, tie parity and infinity.
            with np.errstate(over="ignore"):
                expected = (codes.astype(np.float32) * scales).astype(np.float16).view(np.uint16)
            observed = actual[iteration, 2 * signed][offset]
            detail = f"{arch}, iteration={iteration}, signed={signed == 0}"
            np.testing.assert_array_equal(observed, expected, err_msg=detail)
            legacy = actual[iteration, 2 * signed + 1].reshape(rows, cols)
            legacy_mismatches += int(np.count_nonzero(legacy != expected))
    record_property("legacy_qfloat_rne_mismatches", legacy_mismatches)


def _device_dequant_kernel(signed):
    from tilelang.hexagon import language as T

    source = """
#include <tl_templates/hexagon/crouton.h>
static void tl_test_q4_half(const uint8_t *raw, const half *scale, half *out, half *old) {
  for (int nt = 0; nt < 4; ++nt) {
    half *dst = out + nt * 128 * 32;
    const void *panel = raw + nt * 128 * 16;
    const half *sfb = scale + nt * 32;
    if (nt == 0) {
      tl_crouton_dequant4<SIGNED>(dst, panel, sfb, 128, 64, 0, 16);
    } else {
      tl_crouton_dequant4<SIGNED>(dst, panel, sfb, 128, 64, 0, 3);
      tl_crouton_dequant4<SIGNED>(dst, panel, sfb, 128, 64, 3, 6);
      tl_crouton_dequant4<SIGNED>(dst, panel, sfb, 128, 64, 9, 7);
    }
  }
  // Independent pointwise old expression, without panel/Crouton accesses.
  const uint16_t positive[16] = {
    0x0000,0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,
    0x4800,0x4880,0x4900,0x4980,0x4a00,0x4a80,0x4b00,0x4b80};
  const uint16_t negative[8] = {0xc800,0xc700,0xc600,0xc500,0xc400,0xc200,0xc000,0xbc00};
  alignas(128) uint16_t code[64];
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < 64; ++col) {
      int nibble = (row * 3 + col * 5) & 15;
      code[col] = SIGNED && nibble >= 8 ? negative[nibble - 8] : positive[nibble];
    }
    HVX_Vector a = *reinterpret_cast<const HVX_Vector *>(code);
    for (int group = 0; group < 2; ++group) {
      HVX_Vector b = Q6_Vh_vsplat_R(reinterpret_cast<const uint16_t *>(scale)[group * 128 + row]);
      *reinterpret_cast<HVX_Vector *>(old + row * 128 + group * 64) =
        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
    }
  }
}
""".replace("SIGNED", "true" if signed else "false")

    @T.prim_func
    def kernel(
        W: T.Tensor((8192,), "uint8"),
        S: T.Tensor((256,), "float16"),
        Out: T.Tensor((16384,), "float16"),
        Old: T.Tensor((16384,), "float16"),
    ):
        with T.Kernel(1):
            T.import_source(source)
            raw = T.alloc_shared((8192,), "uint8")
            scales = T.alloc_shared((256,), "float16")
            tiles = T.alloc_shared((16384,), "float16")
            T.copy(W, raw)
            T.copy(S, scales)
            T.evaluate(
                T.call_extern(
                    "void",
                    "tl_test_q4_half",
                    T.access_ptr(raw[0], "r", 8192),
                    T.access_ptr(scales[0], "r", 256),
                    T.access_ptr(tiles[0], "w", 16384),
                    T.access_ptr(Old[0], "w", 16384),
                )
            )
            T.copy(tiles, Out)

    return kernel


@DEVICE
@pytest.mark.parametrize("signed", [True, False])
def test_q4_half_dequantization_on_device(signed, record_property):
    import torch
    import tilelang
    from tilelang.hexagon.intrinsics.panel import pack_panel

    rows = torch.arange(128)[:, None]
    cols = torch.arange(128)[None, :]
    codes = (rows * 3 + cols * 5) % 16
    if signed:
        codes = torch.where(codes >= 8, codes - 16, codes)
    # Distinct signs, subnormal/normal boundaries, midpoint products and
    # overflow. Scale groups differ so a group-stride error cannot pass.
    bits = torch.tensor(
        [0, -32768, 1, -32767, 0x3FF, -31745, 0x400, -31744, 0x3C00, -17408, 0x3C01, -17407, 0x3C03, -17405, 0x7BFF, -1025],
        dtype=torch.int16,
    )
    normal_bits = torch.tensor([0x400, -31744, 0x3C00, -17408, 0x3C01, -17407, 0x3C03, -17405, 0x7BFF, -1025], dtype=torch.int16)
    offset = (rows // 32 * 4 + cols // 32) * 1024
    offset = offset + (cols % 32 // 2) * 64 + (rows % 32) * 2 + cols % 2
    kernel = tilelang.compile(_device_dequant_kernel(signed), out_idx=[2, 3], target="hexagon")
    try:
        for case, patterns in (("mixed", bits), ("normal", normal_bits)):
            scale = patterns.view(torch.float16)[(rows + 3 * torch.arange(2)[None, :]) % len(patterns)]
            packed_scale = scale.t().contiguous().flatten()
            expected = (codes.float() * scale.repeat_interleave(64, -1).float()).half()
            actual, legacy = kernel(pack_panel(codes, 4), packed_scale)
            actual = actual[offset]
            torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0, msg=case)
            record_property(
                f"legacy_qfloat_rne_mismatches_{case}",
                int((legacy.reshape(128, 128).view(torch.int16) != expected.view(torch.int16)).sum()),
            )
    finally:
        kernel.adapter.close()
