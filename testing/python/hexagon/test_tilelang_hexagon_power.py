"""Every kernel's FastRPC session votes for the DSP clocks."""

import os

import pytest

import tilelang
from tilelang.hexagon import language as T
from tilelang.hexagon.runtime import fastrpc

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _session_lines(source):
    skel = fastrpc.gen_dsp("tl_probe", "probe_kernel", source, [])
    assert "#include <tl_templates/hexagon/power.h>" in skel
    open_line = next(line for line in skel.splitlines() if "tl_probe_open(" in line)
    close_line = next(line for line in skel.splitlines() if "tl_probe_close(" in line)
    return open_line, close_line


def test_every_session_votes_for_clocks():
    open_line, close_line = _session_lines("int32_t probe_kernel() { return 0; }")
    assert "tl_power_vote();" in open_line
    assert "tl_power_unvote();" in close_line


def test_hmx_sessions_start_after_the_vote_and_end_before_its_removal():
    open_line, close_line = _session_lines("int32_t probe_kernel() { tl_hexagon_hmx_acc_release(0); return 0; }")
    assert open_line.index("tl_power_vote();") < open_line.index("tl_hmx_session_init();")
    assert close_line.index("tl_hmx_session_deinit();") < close_line.index("tl_power_unvote();")


def _clock_probe(with_hmx):
    """The cycles and microseconds of a fixed HVX loop; with an HMX gemm or not."""

    @T.prim_func
    def probe(X: T.Tensor((64, 64), "float16"), t: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            T.import_source("#include <HAP_perf.h>\n")
            x = T.alloc_shared((64, 64), "float16")
            y = T.alloc_shared((64, 64), "float16")
            T.copy(X, x)
            if with_hmx:
                a = T.alloc_shared((32, 32), "float16")
                b = T.alloc_shared((32, 32), "float16")
                c = T.alloc_shared((32, 32), "float16")
                T.fill(a, 0)
                T.fill(b, 0)
                T.gemm(a, b, c, clear_accum=True)
            cycles = T.call_extern("uint64", "HAP_perf_get_pcycles")
            us = T.call_extern("uint64", "HAP_perf_get_time_us")
            for _ in T.serial(20000):
                T.copy(x, y)
            t[0] = T.Cast("float32", T.call_extern("uint64", "HAP_perf_get_pcycles") - cycles)
            t[1] = T.Cast("float32", T.call_extern("uint64", "HAP_perf_get_time_us") - us)

    return probe


@DEVICE
def test_hvx_and_hmx_kernels_run_at_the_same_clock():
    import torch

    x = torch.randn(64, 64).half()
    mhz = {}
    for with_hmx in (False, True):
        kernel = tilelang.compile(_clock_probe(with_hmx), out_idx=[1], target="hexagon")
        try:
            # The best of a few calls: the clock may ramp up after the vote.
            mhz[with_hmx] = max(float(t[0] / t[1]) for t in (kernel(x) for _ in range(5)))
        finally:
            kernel.adapter.close()
    assert abs(mhz[False] / mhz[True] - 1) < 0.1, mhz
