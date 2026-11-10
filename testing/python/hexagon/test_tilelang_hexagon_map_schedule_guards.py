"""Exercise scheduling guards on the vector TIR consumed by InterleaveMaps."""

import pytest

from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon.transform import InterleaveMaps

tir = tvm.tirx
POLICY = "hexagon.disable_software_pipelining"


def _map_ir(case="direct", arch="v79", vector_bits=1024):
    # Two complete half64 vectors. Deliberately bind distinct views of one
    # arena, so alias tests cannot pass merely by comparing buffer names.
    arena = tir.decl_buffer((512,), "uint8", name="arena", scope="shared.dyn")
    a = tir.decl_buffer((128,), "float16", name="a", scope="shared.dyn")
    b = tir.decl_buffer((128,), "float16", name="b", scope="shared.dyn")
    source = tir.call_intrin("handle", "tirx.handle_add_byte_offset", arena.data, 0)
    if case != "no_exclusive":
        source = tir.call_intrin("handle", tvm.ir.Op.get("tl.hexagon_exclusive_pointer"), source)
    offset = {"same_offset": 0, "overlap": 128}.get(case, 256)
    destination = tir.call_intrin("handle", "tirx.handle_add_byte_offset", arena.data, offset)
    i = tir.Var("i", "int32")
    index = tir.Ramp(i * 64, 1, 64)
    mask = tir.LT(tir.Ramp(0, 1, 64), tir.Broadcast(tir.const(32, "int32"), 64))
    load = tir.BufferLoad(a, [index], predicate=mask if case == "load_predicate" else None)
    value = tir.Mul(load, tir.Broadcast(tir.FloatImm("float16", 2), 64))
    prefix = []
    suffix = []
    if case.startswith("local_"):
        temporary = tir.decl_buffer((64,), "float16", name="temporary", scope="local")
        local_index = tir.Ramp(0, 1, 64)
        definition = tir.BufferStore(temporary, load, [local_index])
        prefix = [tir.AllocBuffer(temporary)]
        if case == "local_before_write":
            suffix = [definition]
        else:
            prefix.append(definition)
        if case == "local_rewrite":
            prefix.append(tir.BufferStore(temporary, tir.Broadcast(tir.FloatImm("float16", 1), 64), [local_index]))
        if case == "local_unused":
            unused = tir.decl_buffer((64,), "float16", name="unused", scope="local")
            prefix.extend([tir.AllocBuffer(unused), tir.BufferStore(unused, load, [local_index])])
        widened = tir.Cast("float32x64", tir.BufferLoad(temporary, [local_index]))
        value = tir.Cast("float16x64", tir.Mul(widened, widened))
    store = tir.BufferStore(b, value, [index], predicate=mask if case == "store_predicate" else None)
    body = tir.SeqStmt([*prefix, store, *suffix]) if prefix else store
    if case.startswith("local_"):
        body = tir.AttrStmt(tir.const(0), "lexical_alloc_scope", 1, body)
    if case == "extra_store":
        body = tir.SeqStmt([store, tir.BufferStore(b, load, [index])])
    annotations = {"hexagon.unroll_count": 4} if case == "annotated" else {}
    step = {"unit_step": 1, "nonunit_step": 2}.get(case)
    loop = tir.For(i, 0, 2, tir.ForKind.SERIAL, body, annotations=annotations, step=step)
    body = tir.SeqStmt([tir.Bind(a.data, source), tir.Bind(b.data, destination), tir.DeclBuffer(a), tir.DeclBuffer(b), loop])
    target = determine_target(
        {"kind": "hexagon", "mcpu": f"hexagon{arch}", "min_vector_bits": vector_bits, "max_vector_bits": vector_bits},
        return_object=True,
    )
    return tir.PrimFunc([arena.data], body, buffer_map={arena.data: arena}).with_attr("target", target)


def _check(func, matched):
    result = InterleaveMaps()(tvm.IRModule({"main": func}))["main"]
    loops = []
    tir.stmt_functor.post_order_visit(result.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    assert len(loops) == 1
    loop = loops[0]
    assert (POLICY in loop.annotations) == matched
    if matched:
        assert int(loop.annotations[POLICY]) == 1

    # Recognition may add only the scheduling annotation; no forwarding,
    # arithmetic rewrite, store removal, or changed existing hint is permitted.
    def remove_policy(node):
        if isinstance(node, tir.For) and POLICY in node.annotations:
            annotations = {key: value for key, value in node.annotations.items() if key != POLICY}
            return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, annotations, node.step)
        return None

    restored = result.with_body(tir.stmt_functor.ir_transform(result.body, None, remove_policy, ["tirx.For"]))
    tvm.ir.assert_structural_equal(restored, func)


@pytest.mark.parametrize("case", ["direct", "unit_step", "local_square"])
def test_complete_disjoint_maps_change_only_the_schedule(case):
    _check(_map_ir(case), True)


@pytest.mark.parametrize("case", ["same_offset", "overlap", "no_exclusive"])
def test_distinct_arena_views_need_a_nonoverlap_proof(case):
    _check(_map_ir(case), False)


@pytest.mark.parametrize("case", ["local_before_write", "local_rewrite", "local_unused", "extra_store"])
def test_local_forwarding_requires_single_assignment_and_one_output(case):
    # These are pass inputs only: an undefined/reassigned local must never be
    # accepted as the simple SSA map or executed by this regression test.
    _check(_map_ir(case), False)


@pytest.mark.parametrize("case", ["load_predicate", "store_predicate", "annotated"])
def test_predicates_and_existing_annotations_keep_their_schedule(case):
    _check(_map_ir(case), False)


@pytest.mark.parametrize("arch,vector_bits", [("v81", 1024), ("v79", 512), ("v79", 2048)])
def test_direct_vector_ir_still_requires_the_qualified_target(arch, vector_bits):
    _check(_map_ir(arch=arch, vector_bits=vector_bits), False)


def test_nonunit_outer_step_is_not_a_complete_contiguous_map():
    _check(_map_ir("nonunit_step"), False)
