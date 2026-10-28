"""The VTCM arena of a Hexagon tile program: one allocation, laid out in TIR."""

import importlib.util
import re
from pathlib import Path

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

ROOT = Path(__file__).resolve().parents[3]


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _offsets(source, *names):
    offsets = {}
    for name in names:
        match = re.search(rf"void\* {name} = (?:tl_vtcm_exclusive\()?\(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\)\)?;", source)
        assert match is not None, source
        offsets[name] = int(match.group(1))
    return offsets


def test_arena_starts_at_the_vtcm_base():
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float16")
            b = T.alloc_shared((256,), "float32")
            T.copy(A, a)
            for i in T.Parallel(256):
                b[i] = T.cast(a[i], "float32")
            T.copy(b, B)

    source = _source(kernel)
    # No reserved prefix: the planned arena is the kernel's VTCM usage.
    assert "uint8_t* buf_dyn_shmem = (uint8_t*)tl_vtcm_arena();" in source
    assert "tl_vtcm_shared_high_water = 1536u;" in source


def test_buffers_start_on_whole_hvx_vectors():
    @T.prim_func
    def kernel(A: T.Tensor((64,), "float16"), B: T.Tensor((8,), "float32"), C: T.Tensor((64,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float16")
            b = T.alloc_shared((8,), "float32")  # 32 bytes
            c = T.alloc_shared((64,), "float16")
            T.copy(A, a)
            T.copy(B, b)
            for i in T.Parallel(64):
                c[i] = a[i] * T.cast(b[i % 8], "float16")
            # a and b stay live next to c.
            for i in T.Parallel(64):
                c[i] = c[i] + a[i] * T.cast(b[i % 8], "float16")
            T.copy(c, C)

    offsets = _offsets(_source(kernel), "a", "b", "c")
    assert len(set(offsets.values())) == 3
    assert all(offset % 128 == 0 for offset in offsets.values()), offsets


def test_dma_buffer_is_reused_after_its_wait():
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16"), C: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            t = T.alloc_shared((256,), "float16")
            T.dma_copy(A, s)
            T.dma_wait()
            T.copy(s, B)
            T.fill(t, 2)
            T.copy(t, C)

    # The transfer into s completes at the wait, and s is dead after copying
    # it out, so t takes its place.
    offsets = _offsets(_source(kernel), "s", "t")
    assert offsets["s"] == offsets["t"]


def test_buffers_alone_in_their_memory_get_their_own_pointer():
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16"), C: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            t = T.alloc_shared((256,), "float16")
            u = T.alloc_shared((256,), "float16")
            T.dma_copy(A, s)
            T.dma_wait()
            T.copy(s, B)
            T.fill(t, 2)
            for i in T.Parallel(256):
                u[i] = t[i] * 3
            T.copy(u, C)

    # t reuses s's memory, so both stay offsets into the arena, which LLVM
    # sees as one object; u shares memory with no buffer, so LLVM sees it as
    # its own allocation (MarkExclusiveBuffers).
    source = _source(kernel)
    assert _offsets(source, "s", "t")["s"] == _offsets(source, "s", "t")["t"]
    assert re.search(r"void\* s = \(\(void\*\)", source)
    assert re.search(r"void\* t = \(\(void\*\)", source)
    assert re.search(r"void\* u = tl_vtcm_exclusive\(\(\(void\*\)", source)


def test_dma_destination_stays_live_until_its_wait():
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float16"), C: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            t = T.alloc_shared((256,), "float16")
            T.dma_copy(A, s)
            T.fill(t, 2)
            T.copy(t, C)
            T.dma_wait()

    # Nothing reads s, but the DMA writes it until the wait: t must not take
    # its place in between.
    offsets = _offsets(_source(kernel), "s", "t")
    assert offsets["s"] != offsets["t"]


def test_tile_operators_reuse_their_scratch_across_calls():
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_shared((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            for r in T.serial(2):
                for k in list(range(4)):
                    T.gemm(a, b, c, clear_accum=(r == 0 and k == 0))
            T.copy(c, C)

    source = _source(kernel)
    # Each call stages its tiles in 4 KB of its own, live only during the
    # call: inside the loop too, the four calls share one, next to a, b and c
    # (32 KB).
    assert source.count("stage") > 4
    high_water = int(re.search(r"tl_vtcm_shared_high_water = (\d+)u;", source).group(1))
    assert high_water <= 32768 + 4096 + 2048, high_water


def test_generated_source_is_deterministic():
    spec = importlib.util.spec_from_file_location("flash", ROOT / "examples/hexagon/example_flash_attention.py")
    flash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flash)
    # Two GEMMs give two internal buffers of one name and size; their places
    # must not depend on pointer values.
    sources = {_source(flash.make_flash(64, 256, 64, 64)) for _ in range(8)}
    assert len(sources) == 1
