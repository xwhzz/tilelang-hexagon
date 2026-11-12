"""Conditional HMX copies keep one pack instead of a synchronous staging copy."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T

I = tvm.tirx


def _program(conditional=True):
    @T.prim_func
    def probe(Q: T.Tensor((64, 64), "float16"), K: T.Tensor((3, 64, 64), "float16"), O: T.Tensor((3, 64, 64), "float16"), enabled: T.int32):
        with T.Kernel(1):
            q = T.alloc_shared((64, 64), "float16")
            k = T.alloc_shared((64, 64), "float16")
            c = T.alloc_shared((64, 64), "float16")
            for block in T.Pipelined(3, num_stages=1):
                if conditional:
                    if block == 0:
                        T.copy(Q, q)
                else:
                    T.copy(Q, q)
                T.copy(K[block, :, :], k)
                T.gemm(q, k, c, clear_accum=True)
                T.copy(c, O[block, :, :])

    return probe


def _copy_endpoints(call):
    if isinstance(call, I.Call) and call.op.name == "tl.tileop.copy":
        return tuple(arg.args[0].buffer.name for arg in call.args)
    return None


def _stage(func):
    result = tilelang.hexagon.transform.StageHMXOperandCopies()(tvm.IRModule.from_expr(func))
    return next(iter(result.functions.values()))


def _copies(func):
    copies = []
    I.stmt_functor.post_order_visit(func.body, lambda node: copies.append(_copy_endpoints(node)) if _copy_endpoints(node) else None)
    return copies


def _math(func):
    calls = []
    I.stmt_functor.post_order_visit(
        func.body,
        lambda node: calls.append(node) if isinstance(node, I.Call) and node.op.name in ("tl.tileop.gemm", "tl.tileop.reduce") else None,
    )
    return calls


def _replace_q_copy(func, replacement):
    def post(node):
        if isinstance(node, I.Evaluate) and _copy_endpoints(node.value) == ("Q", "q"):
            return replacement(node)

    return func.with_body(I.stmt_functor.ir_transform(func.body, None, post, ["tirx.Evaluate"]))


def _wrap_pipeline(func, wrapper):
    def post(node):
        if isinstance(node, I.For) and "num_stages" in node.annotations:
            return wrapper(node)

    return func.with_body(I.stmt_functor.ir_transform(func.body, None, post, ["tirx.For"]))


def test_conditional_copy_keeps_direct_pack_and_unconditional_staging():
    original = _program()
    staged = _stage(original)
    assert _copies(staged) == [("Q", "q"), ("K", "k_dma"), ("k_dma", "k"), ("c", "O")]
    assert tvm.ir.structural_equal(_math(original), _math(staged))


def test_condition_around_whole_pipeline_keeps_staging():
    func = _program(False)
    func = _wrap_pipeline(func, lambda loop: I.IfThenElse(func.params[-1] != 0, loop, None))
    copies = _copies(_stage(func))
    assert ("Q", "q_dma") in copies and ("q_dma", "q") in copies
    assert ("K", "k_dma") in copies


def test_both_conditional_branches_keep_direct_pack():
    func = _program(False)

    def branches(stmt):
        other = I.Evaluate(I.Call(stmt.value.dtype, stmt.value.op, stmt.value.args, annotations=stmt.value.annotations))
        return I.IfThenElse(func.params[-1] != 0, stmt, other)

    func = _replace_q_copy(func, branches)
    copies = _copies(_stage(func))
    assert copies.count(("Q", "q")) == 2
    assert not any("q_dma" in pair for pair in copies)
    assert ("K", "k_dma") in copies


@pytest.mark.parametrize("predicated", [False, True])
def test_block_realize_predicate_matches_planner_condition_rule(predicated):
    func = _program(False)
    predicate = func.params[-1] != 0 if predicated else I.const(True, "bool")
    func = _replace_q_copy(func, lambda stmt: I.SBlockRealize([], predicate, I.SBlock([], [], [], "q_copy", stmt)))
    copies = _copies(_stage(func))
    assert (("Q", "q") in copies) == predicated
    assert (("Q", "q_dma") in copies) == (not predicated)
    assert ("K", "k_dma") in copies


def test_inner_pipeline_resets_and_restores_outer_conditional_context():
    func = _program(False)

    def replace(stmt):
        inner = I.For(I.Var("inner", "int32"), 0, 2, I.ForKind.SERIAL, stmt, annotations={"num_stages": 1})
        # The inner pipeline may stage its own copy; the following outer copy
        # remains conditional. The later unconditional K copy stages normally.
        return I.IfThenElse(func.params[-1] != 0, I.SeqStmt([inner, stmt]), None)

    copies = _copies(_stage(_replace_q_copy(func, replace)))
    assert copies.count(("Q", "q_dma")) == 1 and copies.count(("q_dma", "q")) == 1
    assert copies.count(("Q", "q")) == 1
    assert ("K", "k_dma") in copies


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_normal_lowering_keeps_direct_q_pack_and_async_k(arch):
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagon" + arch})
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(_program(), target=target, enable_host_codegen=False, enable_device_compile=False)
    source = artifact.kernel_source
    assert "q_dma" not in source
    assert "tl_crouton_copy<true, false>" in source and "(&(Q[0]))" in source
    assert "tl_hexagon_dma_async_submit(" in source and "(&(K[" in source
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 0)" in source
    assert "tl_hexagon_hmx_gemm<" in source
