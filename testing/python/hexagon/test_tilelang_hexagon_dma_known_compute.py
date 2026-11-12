"""Known tensor helpers do not observe compiler-private DMA completion records."""

import functools

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.engine.lower import _prepare_device_codegen_mod
from tilelang.hexagon import language as T
from test_tilelang_hexagon_dma_load_only import _emit, _replace, _scope

I = tvm.tirx


@functools.lru_cache(None)
def _mixed(arch="v79"):
    @T.prim_func
    def mixed(
        Q: T.Tensor((64, 64), "float16"),
        K: T.Tensor((256, 64), "float16"),
        V: T.Tensor((256, 64), "float16"),
        O: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(1):
            q = T.alloc_shared((64, 64), "float16")
            kr = T.alloc_shared((64, 64), "float16")
            vr = T.alloc_shared((64, 64), "float16")
            k = T.alloc_shared((64, 64), "float16")
            v = T.alloc_shared((64, 64), "float16")
            s = T.alloc_shared((64, 64), "float16")
            y = T.alloc_shared((64, 64), "float16")
            r = T.alloc_shared((64,), "float32")
            T.copy(Q, q)
            for tile in T.Pipelined(4, num_stages=1):
                T.copy(K[tile * 64, 0], kr)
                T.copy(V[tile * 64, 0], vr)
                T.copy(kr, k)
                T.copy(vr, v)
                T.gemm(q, k, s, transpose_B=True, clear_accum=True)
                T.reduce_max(s, r, dim=1)
                for i, j in T.Parallel(64, 64):
                    s[i, j] = T.exp(T.float32(s[i, j]) - r[i])
                T.gemm(s, v, y, clear_accum=True)
            T.copy(y, O)

    target = determine_target({"kind": "hexagon", "mcpu": "hexagon" + arch}, return_object=True)
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(mixed, target=target, enable_host_codegen=False, enable_device_compile=False)
        final = _prepare_device_codegen_mod(artifact.device_mod)
    (func,) = final.functions.values()
    return func, target


def _policy(func, target, enabled):
    source = _emit(func, target)
    assert ("tl_hexagon_dma_compact_load_context<" in source) == enabled
    assert ("tl_hexagon_dma_async_context<" in source) != enabled
    return source


@pytest.mark.parametrize("arch", ["v73", "v75", "v79", "v81"])
def test_normal_mixed_hmx_hvx_pipeline(arch):
    func, target = _mixed(arch)
    source = _policy(func, target, arch != "v81")
    assert "tl_hexagon_hmx_gemm" in source and "tl_crouton_reduce" in source
    assert "tl_hvx_exp" in source and "tl_hvx_broadcast_lane" in source
    if arch != "v81":
        assert "tl_hexagon_dma_compact_load_context<2, 2>" in source
        assert "tl_hexagon_dma_compact_wait(tl_dma_async, 0)" in source


@pytest.mark.parametrize(
    "op,index,text",
    [
        ("tl.hexagon_hmx_gemm", 0, "false /* custom text */"),
        ("tl.hexagon_crouton_copy", 0, "true /* custom text */"),
        ("tl.hexagon_crouton_reduce", 0, "TL_REDUCE_MAX /* custom text */"),
        ("tl.hexagon_load_lanes", 0, "32 /* custom text */"),
        ("tl.hexagon_broadcast_lane", 1, "2 /* custom text */"),
    ],
)
def test_custom_template_text_keeps_generic_context(op, index, text):
    func, target = _mixed()

    def change(call):
        args = list(call.args)
        args[index] = I.StringImm(text)
        return I.Call(call.dtype, call.op, args)

    _policy(_replace(func, op, change), target, False)


@pytest.mark.parametrize("location", ["argument", "prefix", "sibling", "import"])
def test_opaque_calls_and_imports_still_disable_mixed_policy(location):
    func, target = _mixed()
    opaque = I.call_pure_extern("int32", "opaque_dma_observer")
    if location == "argument":

        def change(call):
            args = list(call.args)
            args[4] = opaque
            return I.Call(call.dtype, call.op, args)

        _policy(_replace(func, "tl.hexagon_hmx_gemm", change), target, False)
    elif location == "prefix":
        _policy(_scope(func, lambda body: I.SeqStmt([I.Evaluate(opaque), *body.seq])), target, False)
    elif location == "import":
        imported = I.AttrStmt(I.const(0), "pragma_import_c", I.StringImm("#define mixed_import 1"), func.body)
        _policy(func.with_body(imported), target, False)
    else:
        good = func.with_attr("global_symbol", "a_good")
        bad = func.with_body(I.SeqStmt([func.body, I.Evaluate(opaque)])).with_attr("global_symbol", "z_bad")
        source = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"a_good": good, "z_bad": bad}), target).inspect_source()
        assert "tl_hexagon_dma_compact_" not in source


@pytest.mark.parametrize("kind", ["nested-scope", "store", "wait-expression", "runtime-string"])
def test_mixed_scope_does_not_relax_dma_ownership_or_direction(kind):
    func, target = _mixed()
    if kind == "nested-scope":
        bad = _scope(func, lambda body: I.AttrStmt(I.const(0), "tl.tile_program_scope", 1, body))
    elif kind == "store":
        bad = _replace(func, "tl.hexagon_dma_submit", lambda c: I.Call(c.dtype, c.op, [*c.args[:-1], I.const(1)]))
    elif kind == "wait-expression":
        bad = _replace(func, "tl.hexagon_dma_wait", lambda c: I.Call(c.dtype, c.op, [c.args[0], I.const(1) + I.const(2)]))
    else:
        bad = _replace(
            func, "tl.hexagon_hmx_gemm", lambda c: I.Call(c.dtype, c.op, [*c.args[:7], I.StringImm("queue_address"), *c.args[8:]])
        )
    _policy(bad, target, False)
