"""Q4 decode attention must not depend on invalid V-cache rows."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import tilelang.language as T
from tilelang.hexagon.runtime.session import lower_kernels

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/hexagon/lfm"))
from decode import attend  # noqa: E402

SIMULATOR = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)
LENGTHS = (1, 2, 7, 16, 32, 63, 64, 65, 128)


def _tool(name):
    root = os.environ.get("HEXAGON_TOOLS_ROOT")
    path = Path(root) / "Tools/bin" / name if root else Path(shutil.which(name) or name)
    if not path.is_file():
        pytest.fail(f"{name} is required: {path}")
    return str(path)


def _attention_kernel():
    @T.prim_func
    def attention(
        Q: T.Tensor((1, 256), "float32"),
        K: T.Tensor((1, 1, 128, 64), "float32"),
        V: T.Tensor((1, 1, 128, 64), "float32"),
        Out: T.Tensor((1, 256), "float32"),
        n: T.int32,
    ):
        with T.Kernel(1):
            qg = T.alloc_fragment((4, 64), "float32")
            kb = T.alloc_fragment((64, 64), "float32")
            vb = T.alloc_fragment((64, 64), "float32")
            sc = T.alloc_fragment((4, 64), "float32")
            p = T.alloc_fragment((4, 64), "float32")
            o = T.alloc_fragment((4, 64), "float32")
            m = T.alloc_fragment((4,), "float32")
            mb = T.alloc_fragment((4,), "float32")
            l = T.alloc_fragment((4,), "float32")
            lb = T.alloc_fragment((4,), "float32")
            alpha = T.alloc_fragment((4,), "float32")
            attend(Q, 0, K, V, 0, 0, n - 1, Out, 0, 4, qg, kb, vb, sc, p, o, m, mb, l, lb, alpha)

    return attention


@SIMULATOR
@pytest.mark.parametrize("arch", ("v73", "v75", "v79"))
def test_attention_ignores_invalid_v_rows(arch, tmp_path):
    ((name, source, _, _),) = lower_kernels([(_attention_kernel(), [])], target={"kind": "hexagon", "mcpu": f"hexagon{arch}"})
    generated = tmp_path / "attention.cc"
    generated.write_text(source)
    # Replace only the resource provider; compile the actual generated body
    # and the repository's arithmetic helpers without rewriting either.
    stub = tmp_path / "include/tl_templates/hexagon/vtcm.h"
    stub.parent.mkdir(parents=True)
    stub.write_text("""#pragma once
#include <stdint.h>
alignas(128) static uint8_t arena[65536];
static uint8_t *tl_vtcm_arena() { return arena; }
static unsigned tl_vtcm_arena_bytes() { return sizeof(arena); }
static unsigned tl_vtcm_shared_high_water;
""")
    driver = tmp_path / "driver.cc"
    driver.write_text(
        """
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "attention.cc"
alignas(128) static float q[256], k[8192], values[8192], v[8192], saved[8192], out[288];
int main(int argc, char **argv) {
  if (argc != 3) return 1;
  SIM_ACQUIRE_HVX; SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *output = fopen(argv[2], "wb");
  if (!in || !output) return 2;
  if (fread(q, sizeof(q), 1, in) != 1 || fread(k, sizeof(k), 1, in) != 1 ||
      fread(values, sizeof(values), 1, in) != 1) return 3;
  fclose(in);
  const int lengths[] = {1,2,7,16,32,63,64,65,128};
  for (int n : lengths) for (int tail = 0; tail < 4; ++tail) {
    memcpy(v, values, sizeof(v));
    for (int i = n*64; i < 8192; ++i) {
      if (tail == 0) v[i] = 0.0f;
      if (tail == 1) v[i] = float((i*17+3)%127-63)/8.0f;
      if (tail >= 2) { uint32_t bits = tail == 2 ? 0x7fc01234u : 0x7f800000u; memcpy(v+i, &bits, 4); }
    }
    memcpy(saved, v, sizeof(v)); memset(out, 0xa5, sizeof(out));
    if (KERNEL(q, k, v, out, n)) return 4;
    if (memcmp(saved, v, sizeof(v))) return 5;
    for (int i = 256; i < 288; ++i) { uint32_t bits; memcpy(&bits, out+i, 4); if (bits != 0xa5a5a5a5u) return 6; }
    if (fwrite(out, 1024, 1, output) != 1) return 7;
  }
  return fclose(output) ? 8 : 0;
}
""".replace("KERNEL(", name + "(")
    )
    rng = np.random.default_rng(73128)
    q = rng.normal(0, 0.5, (4, 64)).astype(np.float32)
    k = rng.normal(0, 0.5, (128, 64)).astype(np.float32)
    v = rng.normal(0, 0.5, (128, 64)).astype(np.float32)
    inputs, output, executable = (tmp_path / x for x in ("input.bin", "output.bin", "attention"))
    inputs.write_bytes(q.tobytes() + k.tobytes() + v.tobytes())
    command = [
        _tool("hexagon-clang++"),
        f"-m{arch}",
        "-mhvx",
        "-mhvx-length=128B",
        "-O2",
        "-std=c++17",
        "-mllvm",
        "--disable-handle-qfp",
        "-mllvm",
        "-hexagon-insert=false",
        "-I",
        str(stub.parents[2]),
        "-I",
        str(ROOT / "src"),
        str(driver),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    ran = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(inputs), str(output)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    actual = np.frombuffer(output.read_bytes(), np.float32).reshape(len(LENGTHS), 4, 4, 64)
    assert np.isfinite(actual).all()
    for i, n in enumerate(LENGTHS):
        for tail in range(1, 4):
            np.testing.assert_array_equal(actual[i, 0].view(np.uint32), actual[i, tail].view(np.uint32))
        logits = (q.astype(np.float64) @ k[:n].astype(np.float64).T) * 0.125
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        reference = (probability / probability.sum(axis=1, keepdims=True)) @ v[:n].astype(np.float64)
        # Same tiny-model reference tolerance; history independence above is exact.
        np.testing.assert_allclose(actual[i, 0], reference, rtol=1e-3, atol=1e-3 * np.max(np.abs(reference)))
