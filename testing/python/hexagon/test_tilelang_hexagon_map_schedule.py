"""Keep the v79 single-input multiply policy separate from other maps."""

import os

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires a Hexagon device and SDK")
PRAGMA = "#pragma clang loop pipeline(disable)"


def _map(mode, n=2048, dtype="float16", coefficient=2):
    size = 2 * n if mode == "strided" else n

    @T.prim_func
    def kernel(A: T.Tensor((size,), dtype), Z: T.Tensor((size,), dtype), B: T.Tensor((size,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((size,), dtype)
            z = T.alloc_shared((size,), dtype)
            b = T.alloc_shared((size,), dtype)
            T.copy(A, a)
            T.copy(Z, z)
            if mode == "inplace":
                for i in T.Parallel(n):
                    a[i] = a[i] * T.cast(coefficient, dtype)
                T.copy(a, B)
            if mode == "strided":
                T.clear(b)
                for i in T.Parallel(n):
                    b[i * 2] = a[i * 2] * T.cast(coefficient, dtype)
                T.copy(b, B)
            if mode == "partial":
                T.clear(b)
                for i in T.Parallel(n // 2):
                    b[i] = a[i] * T.cast(coefficient, dtype)
                T.copy(b, B)
            if mode not in ("inplace", "strided", "partial"):
                for i in T.Parallel(n):
                    if mode == "multiply":
                        b[i] = a[i] * T.cast(coefficient, dtype)
                    if mode == "square":
                        b[i] = T.cast(T.cast(a[i], "float32") * T.cast(a[i], "float32"), dtype)
                    if mode == "square_half":
                        b[i] = a[i] * a[i]
                    if mode == "two_inputs":
                        b[i] = a[i] * z[i]
                    if mode == "add":
                        b[i] = a[i] + z[i]
                    if mode == "affine":
                        b[i] = T.cast(T.cast(a[i], "float32") * T.float32(0.5) + T.float32(1), dtype)
                    if mode == "axpy":
                        b[i] = T.cast(T.cast(a[i], "float32") * T.float32(0.5) + T.cast(z[i], "float32"), dtype)
                    if mode == "copy":
                        b[i] = a[i]
                    if mode == "relu":
                        b[i] = T.max(a[i], T.cast(0, dtype))
                    if mode == "heavy":
                        b[i] = a[i] * T.sigmoid(a[i])
                T.copy(b, B)

    return kernel


def _source(func, arch="v79"):
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("mode", ["multiply", "square", "square_half"])
def test_single_input_multiply_uses_one_scheduling_annotation(mode):
    source = _source(_map(mode))
    assert source.count(PRAGMA) == 1
    assert "unroll_count(2)" not in source
    following = source.split(PRAGMA)[1].lstrip()
    assert following.startswith("for (") and "half64" in following


@pytest.mark.parametrize("coefficient", [0.5, 1.5, -0.75])
def test_fixed_half_coefficients_share_the_policy(coefficient):
    assert _source(_map("multiply", coefficient=coefficient)).count(PRAGMA) == 1


@pytest.mark.parametrize("mode", ["add", "affine", "axpy", "copy", "relu", "heavy", "inplace", "strided", "partial", "two_inputs"])
def test_other_maps_keep_their_schedule(mode):
    source = _source(_map(mode))
    assert PRAGMA not in source
    if mode == "heavy":
        assert source.count("#pragma clang loop unroll_count(2)") == 1


@pytest.mark.parametrize("mode", ["multiply", "square"])
@pytest.mark.parametrize("arch", ["v73", "v75"])
def test_policy_uses_the_function_target(mode, arch):
    assert PRAGMA not in _source(_map(mode), arch)


@pytest.mark.parametrize("mode", ["multiply", "square"])
def test_float32_and_partial_vectors_keep_their_schedule(mode):
    assert PRAGMA not in _source(_map(mode, dtype="float32"))
    assert PRAGMA not in _source(_map(mode, n=2047))


@DEVICE
@pytest.mark.parametrize("mode,coefficient", [("multiply", 2), ("multiply", 1.5), ("square", 2), ("square_half", 2)])
def test_scheduled_map_values_and_buffer_guards(mode, coefficient):
    from tilelang.hexagon.runtime.session import Session

    # Normal inputs check arithmetic independently; exceptional values are a
    # separate comparison against the unchanged compiler schedule.
    n, guard = 2048, b"\xa5" * 128
    a = np.linspace(0.5, 1.5, n, dtype=np.float16)
    a[::2] *= -1
    z = a.copy()
    expected = ((a.astype(np.float32) * coefficient) if mode == "multiply" else a.astype(np.float32) ** 2).astype(np.float16)
    with Session([(_map(mode, n=n, coefficient=coefficient), [])], target="hexagon") as session:
        buffers = [session.buffer(a.nbytes + 256) for _ in range(3)]
        for buffer in buffers:
            buffer.write(b"\xa5" * (a.nbytes + 256))
        aa, zz, bb = [buffer.at(128, a.nbytes) for buffer in buffers]
        aa.write(a)
        zz.write(z)
        session.kernels[0](aa, zz, bb)
        assert bb.read() == expected.tobytes()
        assert aa.read() == a.tobytes() and zz.read() == z.tobytes()
        for buffer in buffers:
            assert buffer.read(128) == guard
            assert buffer.read(128, a.nbytes + 128) == guard
