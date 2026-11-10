"""Direct pass inputs: physical region reordering requires unit loop steps."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.layout import Fragment, Layout

tir = tvm.tirx


def _region_ir(step, axis):
    tiles = tir.decl_buffer((8, 8), "int32", name="tiles", scope="shared")
    i, j = tir.Var("i", "int32"), tir.Var("j", "int32")
    logical_order = Fragment((2, 4), forward_thread_fn=lambda i, j: 0, forward_index_fn=lambda i, j: i * 4 + j)
    store = tir.BufferStore(tiles, i * 10 + j, [i + 2, j + 1])
    inner = tir.For(j, 0, 4, tir.ForKind.PARALLEL, store, step=step if axis == 1 else None)
    outer = tir.For(
        i,
        0,
        2,
        tir.ForKind.PARALLEL,
        inner,
        annotations={"parallel_loop_layout": logical_order},
        step=step if axis == 0 else None,
    )
    block = tir.SBlock(
        [],
        [],
        [],
        "root",
        outer,
        alloc_buffers=[tiles],
        annotations={"layout_map": {tiles: Layout((8, 8), lambda i, j: (j, i))}},
    )
    target = determine_target({"kind": "hexagon", "mcpu": "hexagonv79"}, return_object=True)
    params = [step] if isinstance(step, tir.Var) else []
    return tir.PrimFunc(params, tir.SBlockRealize([], True, block)).with_attr("target", target)


def _iteration_shape(func):
    # Stop at this pass. These synthetic stepped inputs exercise recognition,
    # not a claim that the rest of the pipeline implements arbitrary For.step.
    with func.attrs["target"], tvm.transform.PassContext(config={"tirx.disable_vectorize": True}):
        result = tilelang.transform.LowerTileOp()(tvm.IRModule({"main": func}))["main"]
    loops = []
    tir.stmt_functor.post_order_visit(result.body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    return [int(loop.extent) for loop in reversed(loops)]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("step", [None, 1])
def test_unit_step_region_uses_physical_order(axis, step):
    # The 2x4 logical region is transposed physically, so contiguous traversal
    # is 4x2. This ensures the negative cases do not pass through a dead path.
    assert _iteration_shape(_region_ir(step, axis)) == [4, 2]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("step", ["nonunit", "unknown"])
def test_unproven_step_region_keeps_existing_loop_layout(axis, step):
    value = 2 if step == "nonunit" else tir.Var("step", "int32")
    # The explicit input Fragment flattens logical i*4+j into one dimension;
    # it must not be replaced by the transpose's physical 4x2 traversal.
    assert _iteration_shape(_region_ir(value, axis)) == [8]
