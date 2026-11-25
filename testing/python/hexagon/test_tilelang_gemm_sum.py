"""End-to-end standard fp16 prefill timing on the tiny LFM3 model."""

import os
from pathlib import Path
import sys
from statistics import median

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
LFM = ROOT / "examples/hexagon/lfm"
sys.path.insert(0, str(LFM))


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires a configured Hexagon device and SDK")
def test_tiny_lfm3_standard_prefill_e2e_device(tmp_path):
    import bundle
    import convert
    from decode import Decoder
    from tiny import tiny_lfm3
    import tilelang.hexagon.runtime.session as hex_session

    c, weights = tiny_lfm3(seed=29)
    bundle_path = tmp_path / "tiny_lfm3_standard_prefill.bundle"
    # bundle_path = os.environ["LFM3_BUNDLE"]
    bundle.write(str(bundle_path), c.to_dict(), convert.build_tensors(c, weights), {})

    torch.manual_seed(41)
    prompt = torch.randint(0, c.vocab, (1024,), dtype=torch.int64).tolist()
    original_lower_kernels = hex_session.lower_kernels

    def lower_with_custom_prefill(kernels, target="hexagon"):
        return original_lower_kernels(
            kernels,
            target=target,
            change_kernel_name="prefill_kernel",
            change_kernel_path="/home/zzr/tilelang-hexagon/testing/python/hexagon/prefill_rewritten.c"
            #change_kernel_path="/home/zzr/tilelang-hexagon/testing/python/hexagon/prefill_standard.c"
            #change_kernel_path=None
        )

    hex_session.lower_kernels = lower_with_custom_prefill
    try:
        decoder = Decoder(
            str(bundle_path),
            max_pos=1024,
            max_tokens=1024,
            prefill_rows=128,
            processes=1,
        )
    finally:
        hex_session.lower_kernels = original_lower_kernels

    try:
        for _ in range(2):
            decoder.reset()
            decoder.prefill(prompt)

        samples = []
        for _ in range(7):
            decoder.reset()
            decoder.prefill(prompt)
            samples.append(decoder.last_run_seconds())
    finally:
        decoder.close()

    print("Standard fp16 prefill, tiny LFM3, median of 7 warmed device calls")
    print(f"prompt=1024 tokens, prefill_rows=128, layers={c.layers}, dim={c.dim}")
    print(f"full prefill={median(samples) * 1e3:.3f} ms")
