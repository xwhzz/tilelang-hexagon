"""Automatic one-stage selection and ordering of the actual injected TIR."""

import operator

import pytest

import tilelang as tl
from tilelang import tvm
import tilelang.language as T

X = tvm.tirx


def _case(extent=5, minimum=0, grouped=False, conditional=False, recurrence=False, consumer_conditional=False):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32"), C: T.Tensor((32,), "int32")):
        sa = T.alloc_buffer((1,), "int32", scope="shared")
        sb = T.alloc_buffer((1,), "int32", scope="shared")
        state = T.alloc_buffer((1,), "int32", scope="local")
        state[0] = 0
        for i in T.serial(
            minimum,
            minimum + extent,
            annotations={
                "software_pipeline_stage": [0, 0, 1],
                "software_pipeline_order": [0, 1, 2],
                "software_pipeline_async_producers": [1, 1, 0],
                "software_pipeline_async_producer_groups": [0, 0 if grouped else 1, -1],
                "software_pipeline_async_stages": [0],
                "tl_pipelined_num_stages": T.int32(1),
            },
        ):
            with T.sblock("produce_a"):
                T.reads(A[i])
                T.writes(sa[0])
                if not conditional or i % 2 == 0:
                    sa[0] = A[i]
            with T.sblock("produce_b"):
                T.reads(B[i])
                T.writes(sb[0])
                if not conditional or i % 2 == 0:
                    sb[0] = B[i]
            with T.sblock("consume"):
                T.reads(sa[0], sb[0], state[0])
                T.writes(C[i], state[0])
                if not (conditional or consumer_conditional) or i % 2 == 0:
                    if recurrence:
                        C[i] = state[0] + sa[0] + sb[0]
                        state[0] = C[i]
                    else:
                        C[i] = sa[0] + sb[0]

    return kernel.with_attr("target", tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"}))


def _lower(func):
    mod = tl.transform.InjectSoftwarePipeline()(tvm.IRModule.from_expr(func))
    return tl.transform.Simplify()(mod)["kernel"]


def _rewrite_loop(func, change):
    def rewrite(node):
        if isinstance(node, X.For) and "software_pipeline_stage" in node.annotations:
            return change(node)
        return None

    return func.with_body(X.stmt_functor.ir_transform(func.body, None, rewrite))


def _annotations(func, update=None, remove=()):
    def change(loop):
        ann = dict(loop.annotations)
        for key in remove:
            ann.pop(key, None)
        ann.update(update or {})
        return X.For(loop.loop_var, loop.min, loop.extent, loop.kind, loop.body, loop.thread_binding, ann, step=loop.step)

    return _rewrite_loop(func, change)


def _assert_fallback(func):
    # A function-level imported source explicitly disables automatic selection.
    # Compare the actual negative control against that unchanged emitter.
    actual = _lower(func)
    fallback = _lower(func.with_attr("pragma_import_c", "/* force fallback */")).without_attr("pragma_import_c")
    if func.attrs and "pragma_import_c" in func.attrs:
        fallback = fallback.with_attr("pragma_import_c", func.attrs["pragma_import_c"])
    tvm.ir.assert_structural_equal(actual, fallback, map_free_vars=True)


class _FIFO:
    """Producer writes become visible only after a generated group wait."""

    def __init__(self):
        self.env, self.mem = {}, {}
        self.queue, self.uncommitted = [], []
        self.producer = False
        self.commits = self.empty_commits = 0

    def key(self, buffer, indices):
        indices = tuple(self.expr(i) for i in indices)
        shape = tuple(self.expr(i) for i in buffer.shape)
        assert len(indices) == len(shape) and all(0 <= i < n for i, n in zip(indices, shape))
        return buffer.name, indices

    def pending(self, key):
        return any(key == k for group in [*self.queue, self.uncommitted] for k, _ in group)

    def expr(self, expr):
        if isinstance(expr, X.IntImm):
            return int(expr)
        if isinstance(expr, X.Var):
            return self.env[expr]
        if isinstance(expr, X.Cast):
            return self.expr(expr.value)
        if isinstance(expr, X.Not):
            return not self.expr(expr.a)
        if isinstance(expr, X.BufferLoad):
            key = self.key(expr.buffer, expr.indices)
            assert not self.pending(key), f"read before retirement: {key}"
            return self.mem[key]
        binary = dict(
            Add=operator.add,
            Sub=operator.sub,
            Mul=operator.mul,
            FloorDiv=operator.floordiv,
            FloorMod=operator.mod,
            LT=operator.lt,
            LE=operator.le,
            EQ=operator.eq,
            NE=operator.ne,
            GT=operator.gt,
            GE=operator.ge,
            And=operator.and_,
            Or=operator.or_,
        )
        if type(expr).__name__ in binary:
            return binary[type(expr).__name__](self.expr(expr.a), self.expr(expr.b))
        if isinstance(expr, X.Call):
            if expr.op.name == "tirx.bitwise_and":
                return self.expr(expr.args[0]) & self.expr(expr.args[1])
            if expr.op.name == "tirx.if_then_else":
                return self.expr(expr.args[1] if self.expr(expr.args[0]) else expr.args[2])
        raise AssertionError(f"Unsupported expression: {expr}")

    def stmt(self, stmt):
        if isinstance(stmt, X.SeqStmt):
            for child in stmt.seq:
                self.stmt(child)
        elif isinstance(stmt, X.SBlockRealize):
            if self.expr(stmt.predicate):
                for var, value in zip(stmt.block.iter_vars, stmt.iter_values):
                    self.env[var.var] = self.expr(value)
                self.stmt(stmt.block)
        elif isinstance(stmt, X.SBlock):
            previous = self.producer
            self.producer = str(stmt.name_hint).startswith("produce_")
            self.stmt(stmt.body)
            self.producer = previous
        elif isinstance(stmt, X.For):
            start, extent = self.expr(stmt.min), self.expr(stmt.extent)
            for value in range(start, start + extent):
                self.env[stmt.loop_var] = value
                self.stmt(stmt.body)
        elif isinstance(stmt, X.IfThenElse):
            if self.expr(stmt.condition):
                self.stmt(stmt.then_case)
            elif stmt.else_case is not None:
                self.stmt(stmt.else_case)
        elif isinstance(stmt, X.Bind):
            self.env[stmt.var] = self.expr(stmt.value)
        elif isinstance(stmt, X.BufferStore):
            key = self.key(stmt.buffer, stmt.indices)
            assert not self.pending(key), f"reuse before retirement: {key}"
            value = self.expr(stmt.value)
            if self.producer:
                self.uncommitted.append((key, value))
            else:
                self.mem[key] = value
        elif isinstance(stmt, X.Evaluate):
            if isinstance(stmt.value, X.IntImm):
                return
            assert isinstance(stmt.value, X.Call)
            if stmt.value.op.name == "tirx.ptx_commit_group":
                self.commits += 1
                self.empty_commits += not self.uncommitted
                self.queue.append(self.uncommitted)
                self.uncommitted = []
            elif stmt.value.op.name == "tirx.ptx_wait_group":
                pending = self.expr(stmt.value.args[0])
                assert pending >= 0
                while len(self.queue) > pending:
                    self.mem.update(self.queue.pop(0))
            else:
                raise AssertionError(f"Unsupported call: {stmt.value}")
        elif not isinstance(stmt, (X.AllocBuffer, X.DeclBuffer)):
            raise AssertionError(f"Unsupported statement: {stmt}")


@pytest.mark.parametrize("extent", [0, 1, 2, 5])
@pytest.mark.parametrize("minimum", [0, 3])
@pytest.mark.parametrize(
    "grouped,conditional,recurrence,consumer_conditional",
    [
        (True, False, False, False),
        (False, False, False, False),
        (True, True, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_compact_pipeline_preserves_fifo_and_empty_groups(extent, minimum, grouped, conditional, recurrence, consumer_conditional):
    lowered = _lower(_case(extent, minimum, grouped, conditional, recurrence, consumer_conditional))
    fifo = _FIFO()
    for i in range(32):
        fifo.mem["A", (i,)], fifo.mem["B", (i,)], fifo.mem["C", (i,)] = 100 + i, 200 + i, -1
    fifo.stmt(lowered.body)
    total = 0
    for i in range(32):
        expected = -1
        if minimum <= i < minimum + extent and (not (conditional or consumer_conditional) or i % 2 == 0):
            expected = 300 + 2 * i
            if recurrence:
                total += expected
                expected = total
        assert fifo.mem["C", (i,)] == expected
        assert fifo.mem["A", (i,)] == 100 + i and fifo.mem["B", (i,)] == 200 + i
    assert not fifo.queue and not fifo.uncommitted
    assert fifo.commits == extent * (1 if grouped else 2)
    if conditional:
        assert fifo.empty_commits == sum(i % 2 for i in range(minimum, minimum + extent)) * (1 if grouped else 2)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_compact_pipeline_has_one_consumer_body_without_private_annotation(arch):
    func = _case().with_attr("target", tvm.target.Target({"kind": "hexagon", "mcpu": "hexagon" + arch}))
    lowered = _lower(func)
    stores = []
    X.stmt_functor.post_order_visit(
        lowered.body, lambda node: stores.append(node) if isinstance(node, X.BufferStore) and node.buffer.name == "C" else None
    )
    assert len(stores) == 1
    # A stale diagnostic annotation no longer controls selection.
    stale = _lower(_annotations(func, {"tl_hexagon_compact_probe": 0}))
    stale_stores = []
    X.stmt_functor.post_order_visit(
        stale.body, lambda node: stale_stores.append(node) if isinstance(node, X.BufferStore) and node.buffer.name == "C" else None
    )
    assert len(stale_stores) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"software_pipeline_stage": [0, 0, 2]},
        {"software_pipeline_stage": [1, 1, 2]},
        {"software_pipeline_async_producer_groups": [-1, -1, -1]},
        {"software_pipeline_async_producer_groups": [0, 1, 0]},
        {"software_pipeline_async_producer_groups": [0, 0, -1], "software_pipeline_order": [0, 2, 1]},
        {"tl_pipelined_num_stages": X.IntImm("int32", 2)},
    ],
)
def test_compact_pipeline_rejects_other_stages_or_unproven_groups(change):
    _assert_fallback(_annotations(_case(), change))


def test_compact_pipeline_preserves_unsupported_multiple_async_stage_error():
    func = _annotations(_case(), {"software_pipeline_stage": [0, 1, 1]})
    for candidate in [func, func.with_attr("pragma_import_c", "/* force fallback */")]:
        with pytest.raises(tvm.error.InternalError, match="dependency on multiple async stages"):
            _lower(candidate)


@pytest.mark.parametrize("target", [{"kind": "llvm"}, {"kind": "cuda", "arch": "sm_80"}])
def test_compact_pipeline_keeps_other_targets(target):
    _assert_fallback(_case().with_attr("target", tvm.target.Target(target)))


def test_compact_pipeline_requires_nonnegative_extent():
    func = _case()
    extent = X.Var("extent", "int32")
    func = _rewrite_loop(
        func,
        lambda loop: X.For(loop.loop_var, loop.min, extent, loop.kind, loop.body, loop.thread_binding, loop.annotations, step=loop.step),
    )
    func = X.PrimFunc([*func.params, extent], func.body, func.ret_type, func.buffer_map, func.attrs)
    _assert_fallback(func)


@pytest.mark.parametrize("scope", ["function", "root_block", "outside", "inside", "enclosing_async", "enclosing_async_block"])
def test_compact_pipeline_respects_imported_source_and_inherited_async_scopes(scope):
    func = _case()
    if scope == "function":
        func = func.with_attr("pragma_import_c", "/* imported source */")
    elif scope in ("root_block", "enclosing_async_block"):
        root = func.body
        block = root.block
        block = X.SBlock(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            block.body,
            block.init,
            block.alloc_buffers,
            block.match_buffers,
            {
                **dict(block.annotations),
                **({"pragma_import_c": "/* imported */"} if scope == "root_block" else {"async_commit_queue_scope": 0}),
            },
        )
        func = func.with_body(X.SBlockRealize(root.iter_values, root.predicate, block))
    elif scope == "outside":
        func = func.with_body(X.AttrStmt(0, "pragma_import_c", "/* imported */", func.body))
    elif scope == "inside":

        def wrap_import(loop):
            body = list(loop.body.seq)
            realize = body[0]
            block = realize.block
            block = X.SBlock(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                X.AttrStmt(0, "pragma_import_c", "/* imported */", block.body),
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
            )
            body[0] = X.SBlockRealize(realize.iter_values, realize.predicate, block)
            return X.For(
                loop.loop_var, loop.min, loop.extent, loop.kind, X.SeqStmt(body), loop.thread_binding, loop.annotations, step=loop.step
            )

        func = _rewrite_loop(func, wrap_import)
    else:
        func = func.with_body(X.AttrStmt(0, "async_commit_queue_scope", 0, func.body))
    _assert_fallback(func)


@pytest.mark.parametrize("kind", ["extern", "pure_extern", "packed", "manual_wait", "raw_commit", "return", "global_var"])
def test_compact_pipeline_rejects_manual_or_opaque_calls(kind):
    calls = {
        "extern": X.call_extern("int32", "opaque"),
        "pure_extern": X.call_pure_extern("int32", "opaque"),
        "packed": X.call_packed("opaque"),
        "manual_wait": X.Call("int32", tvm.ir.Op.get("tl.hexagon_dma_wait"), [X.IntImm("int32", 0)]),
        "raw_commit": X.Call("int32", tvm.ir.Op.get("tirx.ptx_commit_group"), []),
        "return": X.Call("int32", tvm.ir.Op.get("tirx.ret"), [X.IntImm("int32", 0)]),
        "global_var": X.Call("int32", tvm.ir.GlobalVar("unknown"), []),
    }
    func = _case()

    def change(loop):
        body = list(loop.body.seq)
        producer = body[0]
        block = producer.block
        block = X.SBlock(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            X.SeqStmt([X.Evaluate(calls[kind]), block.body]),
            block.init,
            block.alloc_buffers,
            block.match_buffers,
            block.annotations,
        )
        body[0] = X.SBlockRealize(producer.iter_values, producer.predicate, block)
        return X.For(
            loop.loop_var, loop.min, loop.extent, loop.kind, X.SeqStmt(body), loop.thread_binding, loop.annotations, step=loop.step
        )

    func = _rewrite_loop(func, change)
    if kind == "global_var":
        # The existing pipeline analyzer cannot resolve this call target.
        for candidate in [func, func.with_attr("pragma_import_c", "/* force fallback */")]:
            with pytest.raises(tvm.error.InternalError, match="bad optional access"):
                _lower(candidate)
    else:
        _assert_fallback(func)


def test_compact_pipeline_does_not_rewrite_either_nested_pipeline():
    func = _case()
    outer = X.Var("outer", "int32")

    def nest(loop):
        return X.For(
            outer,
            0,
            2,
            X.ForKind.SERIAL,
            X.SeqStmt([X.SBlockRealize([], True, X.SBlock([], [], [], "inner_pipeline", loop)), X.Evaluate(0)]),
            annotations={
                "software_pipeline_stage": [0, 1],
                "software_pipeline_order": [0, 1],
                "software_pipeline_async_producers": [0, 0],
                "software_pipeline_async_producer_groups": [-1, -1],
                "tl_pipelined_num_stages": X.IntImm("int32", 1),
            },
        )

    _assert_fallback(_rewrite_loop(func, nest))


@pytest.mark.parametrize("extent", [0, 1, 2, 5])
@pytest.mark.parametrize("minimum", [0, 3])
def test_compact_pipeline_replays_scalar_bindings(extent, minimum):
    func = _case(extent, minimum)

    def bind(loop):
        base, offset = X.Var("base", "int32"), X.Var("offset", "int32")
        body = X.SeqStmt(
            [X.Bind(base, loop.loop_var * 2), X.Bind(offset, base // 2), X.stmt_functor.substitute(loop.body, {loop.loop_var: offset})]
        )
        return X.For(loop.loop_var, loop.min, loop.extent, loop.kind, body, loop.thread_binding, loop.annotations, step=loop.step)

    lowered = _lower(_rewrite_loop(func, bind))
    assert not X.analysis.undefined_vars(lowered.body, [*lowered.params, *(b.data for b in lowered.buffer_map.values())])
    fifo = _FIFO()
    for i in range(32):
        fifo.mem["A", (i,)], fifo.mem["B", (i,)], fifo.mem["C", (i,)] = 100 + i, 200 + i, -1
    fifo.stmt(lowered.body)
    assert all(fifo.mem["C", (i,)] == (300 + 2 * i if minimum <= i < minimum + extent else -1) for i in range(32))
    assert not fifo.queue and not fifo.uncommitted


def _typed_copy(func, annotations, op="tl.tileop.copy"):
    def change(node):
        if isinstance(node, X.BufferStore) and node.buffer.name == "sa":
            call = T.copy(node.value, X.BufferLoad(node.buffer, node.indices), annotations=annotations)
            if op != "tl.tileop.copy":
                call = X.Call(call.dtype, tvm.ir.Op.get(op), call.args, annotations=call.annotations)
            return X.Evaluate(call)
        return None

    return func.with_body(X.stmt_functor.ir_transform(func.body, None, change))


@pytest.mark.parametrize("cache_bypass", [0, 1])
def test_compact_pipeline_preserves_typed_copy_cache_annotation(cache_bypass):
    lowered = _lower(_typed_copy(_case(), {"hexagon_dma_src_l2_bypass": cache_bypass}))
    stores, copies = [], []

    def collect(node):
        if isinstance(node, X.BufferStore) and node.buffer.name == "C":
            stores.append(node)
        if isinstance(node, X.Call) and node.op == tvm.ir.Op.get("tl.tileop.copy"):
            copies.append(node)

    X.stmt_functor.post_order_visit(lowered.body, collect)
    assert len(stores) == 1 and copies
    assert all(int(copy.annotations["hexagon_dma_src_l2_bypass"]) == cache_bypass for copy in copies)


@pytest.mark.parametrize("flag", ["is_async_copy", "force_cp_async", "is_tma_copy"])
def test_compact_pipeline_rejects_manual_copy_annotations(flag):
    _assert_fallback(_typed_copy(_case(), {flag: 1}))


@pytest.mark.parametrize("op", ["tl.tileop.async_copy", "tl.tileop.dma_copy"])
def test_compact_pipeline_rejects_explicit_async_copy_operators(op):
    _assert_fallback(_typed_copy(_case(), {"coalesced_width": 1}, op))
