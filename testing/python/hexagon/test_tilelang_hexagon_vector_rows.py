"""Invariant vector row pointers preserve offsets, scope and access semantics."""

import os
import re

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T
from tvm import tirx

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
ROW_POINTER = re.compile(r"float\* (\w+) = \(\(float\*\)(\w+)\) \+ ([^;]+);")


def _target(arch):
    return tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})


def _emit(functions, arch):
    return tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule(functions), _target(arch)).inspect_source()


def _body(source, start):
    """The complete C block, without relying on generated temporary names."""
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[opening + 1 : end - 1], end


def _slot_update(workers):
    # A small row-scaled residual map: broadcast lowering creates a serial
    # column loop, whereas an ordinary 1-D map carries an unrolled-kind loop.
    @T.prim_func
    def kernel(
        X: T.Tensor((workers, 2, 4, 512), "float32"),
        F: T.Tensor((workers, 3, 4, 512), "float32"),
        U: T.Tensor((workers, 3, 4), "float32"),
        O: T.Tensor((workers, 2, 4, 512), "float32"),
        S: T.Tensor((workers, 2, 4, 512), "float32"),
    ):
        with T.Kernel(workers, num_workers=workers) as w:
            x = T.alloc_shared((2, 4, 512), "float32")
            f = T.alloc_shared((3, 4, 512), "float32")
            u = T.alloc_shared((3, 4), "float32")
            squared = T.alloc_shared((2, 4, 512), "float32")
            T.copy(X[w, :, :, :], x)
            T.copy(F[w, :, :, :], f)
            T.copy(U[w, :, :], u)
            for batch in T.serial(3):
                for i, j in T.Parallel(4, 512):
                    value = x[batch % 2, i, j] + f[batch, i, j] * u[batch, i]
                    x[batch % 2, i, j] = value
                    squared[batch % 2, i, j] = value * value
            T.copy(x, O[w, :, :, :])
            T.copy(squared, S[w, :, :, :])

    return kernel


@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize("workers", [1, 4])
def test_dynamic_slot_rows_are_bound_inside_each_outer_iteration(arch, workers):
    target = _target(arch)
    with target, tvm.transform.PassContext(opt_level=3):
        source = str(
            tilelang.lower(_slot_update(workers), target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source
        )
    (outer,) = list(re.finditer(r"for \(int32_t (batch\w*) = 0;", source))
    body, _ = _body(source, outer.start())
    pointers = list(ROW_POINTER.finditer(body))
    assert len(pointers) == 3, source
    inner = body.index("for (", pointers[-1].end())
    inner_body, _ = _body(body, inner)
    for pointer in pointers:
        name, _, offset = pointer.groups()
        assert outer.group(1) in offset
        assert re.search(rf"\b{re.escape(name)}\b", inner_body), source
    assert not re.search(rf"\b{outer.group(1)}\b", inner_body), source
    assert "tl_exclusive(" not in body, "Derived row pointers must not acquire a new allocation identity"


def _direct(mode="plain", loops=1):
    buffer = tirx.decl_buffer((2048,), "float32", name="data")
    slot = tirx.Var("slot", "int32")
    length = tirx.Var("length", "int32")
    indices = tirx.decl_buffer((1,), "int32", name="indices")
    statements = []
    for number in range(loops):
        column = tirx.Var(f"column{number}", "int32")
        origin = slot * 256 + number * 256
        binding = None
        if mode == "load_offset":
            origin = indices[0] * 256
        elif mode == "local_offset":
            local = tirx.Var("local", "int32")
            binding = tirx.Bind(local, origin)
            origin = local
        elif mode == "unaligned":
            origin = slot * 257
        stride = 64 if mode == "strided" else 32
        ramp = tirx.Ramp(origin + column * stride, 1, 32)
        value = tirx.BufferLoad(buffer, [ramp]) + tirx.Broadcast(tirx.FloatImm("float32", 1), 32)
        if mode == "different_offsets":
            value = value + tirx.BufferLoad(buffer, [tirx.Ramp(origin + column * 32 + 32, 1, 32)])
        if mode == "conditional_expr":
            value = tirx.if_then_else(slot >= 0, value, tirx.Broadcast(tirx.FloatImm("float32", 0), 32))
        body = tirx.BufferStore(buffer, value, [ramp])
        if mode == "conditional_stmt":
            body = tirx.IfThenElse(slot >= 0, body, None)
        if binding is not None:
            body = tirx.SeqStmt([binding, body])
        extent = length if mode == "dynamic" else (0 if mode == "zero" else 4)
        kind = tirx.ForKind.UNROLLED if mode == "unrolled" else tirx.ForKind.SERIAL
        loop = tirx.For(column, 1 if mode == "nonzero_min" else 0, extent, kind, body, step=2 if mode == "step" else None)
        statements.append(loop)
    # An access after the loops must still use the original buffer and offset.
    statements.append(tirx.BufferStore(buffer, tirx.BufferLoad(buffer, [tirx.Ramp(slot * 256, 1, 32)]), [tirx.Ramp(1024, 1, 32)]))
    params = [buffer.data, slot, length, indices.data]
    if mode == "volatile":
        # This tests exclusion from row binding, not new volatile-vector support.
        statements.insert(0, tirx.AllocBuffer(buffer, annotations={"tirx.volatile": 1}))
        params = [slot, length, indices.data]
    return tirx.PrimFunc(params, tirx.SeqStmt(statements)).with_attr("global_symbol", "rows")


@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize(
    "mode",
    [
        "zero",
        "dynamic",
        "step",
        "conditional_stmt",
        "conditional_expr",
        "load_offset",
        "local_offset",
        "volatile",
        "different_offsets",
        "unaligned",
        "strided",
        "nonzero_min",
        "unrolled",
    ],
)
def test_unsafe_or_unsupported_rows_keep_the_original_addressing(arch, mode):
    source = _emit({"main": _direct(mode)}, arch)
    assert not ROW_POINTER.search(source), source
    if mode == "unaligned":
        assert "tl_hvx_vec_loadu<float32>" in source and "tl_hvx_vec_storeu(" in source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_row_bindings_do_not_leak_between_loops_or_functions(arch):
    functions = {name: _direct(loops=4).with_attr("global_symbol", name) for name in ["first", "second"]}
    source = _emit(functions, arch)
    for name in functions:
        function, _ = _body(source, source.index(f"int32_t {name}("))
        pointers = list(ROW_POINTER.finditer(function))
        assert len(pointers) == 4 and len({p.group(1) for p in pointers}) == 4, function
        previous_end = 0
        for index, pointer in enumerate(pointers):
            loop = function.index(f"for (int32_t column{index} = ", pointer.end())
            body, end = _body(function, loop)
            assert previous_end <= pointer.start() < loop
            assert re.search(rf"\b{pointer.group(1)}\b", body), body
            assert "slot" not in body
            assert not any(re.search(rf"\b{other.group(1)}\b", body) for other in pointers if other is not pointer)
            previous_end = end
        assert all(not re.search(rf"\b{p.group(1)}\b", function[previous_end:]) for p in pointers)
        assert "slot * 256" in function[previous_end:]
    assert "tl_exclusive(" not in source and "__restrict" not in source


@DEVICE
@pytest.mark.parametrize("workers", [1, 4])
def test_dynamic_slots_preserve_updates_and_guards(workers):
    from tilelang.hexagon.runtime.session import Session

    x = ((np.arange(workers * 2 * 4 * 512) % 61 - 30) / 16).astype(np.float32).reshape(workers, 2, 4, 512)
    f = ((np.arange(workers * 3 * 4 * 512) % 13 - 6) / 64).astype(np.float32).reshape(workers, 3, 4, 512)
    u = np.ldexp(np.ones((workers, 3, 4), dtype=np.float32), (np.arange(workers * 3 * 4) % 3 - 1).reshape(workers, 3, 4))
    expected = x.copy()
    squared = np.empty_like(x)
    # Slots 0 and 1 receive different inputs and respectively two and one updates.
    for batch in range(3):
        slot = batch % 2
        expected[:, slot] += f[:, batch] * u[:, batch, :, None]
        squared[:, slot] = expected[:, slot] * expected[:, slot]
    guard = b"\xa5" * 128
    with Session([(_slot_update(workers), [])], target="hexagon") as session:
        payloads = [x.tobytes(), f.tobytes(), u.tobytes(), bytes(x.nbytes), bytes(x.nbytes)]
        allocations = [session.buffer(len(payload) + 256) for payload in payloads]
        views = [allocation.at(128, len(payload)) for allocation, payload in zip(allocations, payloads)]
        for allocation, payload in zip(allocations, payloads):
            allocation.write(guard + payload + guard)
        session.kernels[0](*views)
        for allocation, expected_bytes in zip(allocations, [x.tobytes(), f.tobytes(), u.tobytes(), expected.tobytes(), squared.tobytes()]):
            assert allocation.read() == guard + expected_bytes + guard
