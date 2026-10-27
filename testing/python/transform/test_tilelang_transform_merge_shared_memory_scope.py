"""MergeSharedMemoryAllocations places the merged buffer in its owning scope.

On GPU targets that scope is the outermost thread_extent. Targets that lower the
launch grid to loops mark the tile program body with tl.tile_program_scope
instead, so every program (and every worker running one) gets its own arena.
"""

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm.script import tirx as T


@T.prim_func(private=True)
def grid_as_loop(A: T.handle("float32", "global"), B: T.handle("float32", "global")):
    A_flat = T.decl_buffer((256,), data=A)
    B_flat = T.decl_buffer((256,), data=B)
    for bx in range(2):
        with T.attr(0, "tl.tile_program_scope", 1):
            x = T.alloc_buffer((128,), "float32", scope="shared.dyn")
            y = T.alloc_buffer((128,), "float32", scope="shared.dyn")
            for i in range(128):
                x[i] = A_flat[bx * 128 + i]
            for i in range(128):
                y[i] = x[i] * T.float32(2)
            for i in range(128):
                B_flat[bx * 128 + i] = y[i]


def _nodes(stmt, kind):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(stmt, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def test_merged_buffer_is_allocated_in_the_tile_program_scope():
    mod = tilelang.transform.MergeSharedMemoryAllocations()(tvm.IRModule({"main": grid_as_loop}))
    body = mod["main"].body
    (scope,) = [attr for attr in _nodes(body, tvm.tirx.AttrStmt) if attr.attr_key == "tl.tile_program_scope"]
    allocations = _nodes(body, tvm.tirx.AllocBuffer)
    assert [alloc.buffer.name for alloc in allocations] == ["buf_dyn_shmem"]
    assert _nodes(scope.body, tvm.tirx.AllocBuffer) == allocations


if __name__ == "__main__":
    tilelang.testing.main()
