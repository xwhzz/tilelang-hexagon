"""Opt-in prologue copies keep the automatic pipeline's FIFO schedule."""

import pytest

from tilelang import tvm
import tilelang.language as T

from test_tilelang_transform_hexagon_compact_pipeline import _FIFO, _annotations, _case, _lower, _rewrite_loop

X = tvm.tirx
POLICY = "hexagon.prologue_copy_policy"
CANDIDATE = "tl.hexagon.prologue_copy_candidate"


def _copies(func, mixed=False, explicit=None):
    def change(node):
        if isinstance(node, X.BufferStore) and node.buffer.name in ("sa", "sb"):
            annotations = {"hexagon_dma_src_l2_bypass": 0} if mixed and node.buffer.name == "sb" else {}
            annotations.update(explicit or {})
            src = X.BufferRegion(node.value.buffer, [tvm.ir.Range.from_min_extent(index, 1) for index in node.value.indices])
            dst = X.BufferRegion(node.buffer, [tvm.ir.Range.from_min_extent(index, 1) for index in node.indices])
            return X.Evaluate(T.copy(src, dst, annotations=annotations))
        return None

    return func.with_body(X.stmt_functor.ir_transform(func.body, None, change))


def _candidates(func):
    found = []
    X.stmt_functor.post_order_visit(
        func.body,
        lambda node: found.append(node) if isinstance(node, X.Call) and CANDIDATE in node.annotations else None,
    )
    return found


class _CopyFIFO(_FIFO):
    def __init__(self):
        super().__init__()
        self.sync = 0

    def stmt(self, stmt):
        if isinstance(stmt, X.Evaluate) and isinstance(stmt.value, X.Call) and stmt.value.op.name == "tl.tileop.copy":
            call = stmt.value
            src, dst = [region.args[0] for region in call.args[:2]]
            assert all(int(extent) == 1 for region in call.args[:2] for extent in region.args[2:])
            key, value = self.key(dst.buffer, dst.indices), self.expr(src)
            assert not self.pending(key), f"reuse before retirement: {key}"
            # These tests use ordinary, in-bounds same-dtype scalar regions.
            # The actual target lowerer is covered separately by source tests.
            selected = CANDIDATE in call.annotations and "hexagon_dma_src_l2_bypass" not in call.annotations
            if selected:
                self.mem[key] = value
                self.sync += 1
            else:
                self.uncommitted.append((key, value))
        else:
            super().stmt(stmt)


def _run(func):
    fifo = _CopyFIFO()
    for i in range(32):
        fifo.mem["A", (i,)], fifo.mem["B", (i,)], fifo.mem["C", (i,)] = 100 + i, 200 + i, -1
    fifo.stmt(func.body)
    return fifo


@pytest.mark.parametrize("extent", [0, 1, 2, 5])
@pytest.mark.parametrize("minimum", [0, 3])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_sync_prologue_preserves_fifo_and_empty_groups(extent, minimum, grouped, conditional, mixed):
    func = _copies(_case(extent, minimum, grouped, conditional), mixed)
    baseline = _run(_lower(func))
    lowered = _lower(_annotations(func, {POLICY: "sync_if_supported"}))
    actual = _run(lowered)
    assert actual.mem == baseline.mem
    assert not actual.queue and not actual.uncommitted
    assert actual.commits == baseline.commits == extent * (1 if grouped else 2)
    selected = extent > 0 and (not conditional or minimum % 2 == 0)
    assert actual.sync == (1 if mixed else 2) * selected
    extra_empty = (0 if mixed else 1) if grouped else (1 if mixed else 2)
    assert actual.empty_commits == baseline.empty_commits + extra_empty * selected
    assert all(int(call.annotations["no_implicit_async_commit_wait"]) == 1 for call in _candidates(lowered))


@pytest.mark.parametrize("extent", [0, 1, 2, 5])
def test_preserve_is_exact_default_injected_ir(extent):
    func = _copies(_case(extent=extent, minimum=3))
    default, preserve = _lower(func), _lower(_annotations(func, {POLICY: "preserve"}))
    tvm.ir.assert_structural_equal(default, preserve, map_free_vars=True)
    assert default.script() == preserve.script()
    assert not _candidates(default)


@pytest.mark.parametrize("value", ["", "sync", "SYNC_IF_SUPPORTED", 1, True])
def test_invalid_policy_is_rejected(value):
    with pytest.raises(tvm.error.InternalError, match="hexagon.prologue_copy_policy expects"):
        _lower(_annotations(_copies(_case()), {POLICY: value}))


@pytest.mark.parametrize("location", ["function", "block", "attr", "loop", "call"])
def test_dma_only_anywhere_disables_whole_function(location):
    func = _copies(_case())
    key = "tl.hexagon_dma_only"
    var = next(iter(func.buffer_map.values())).data
    if location == "function":
        func = func.with_attr(key, [var])
    elif location == "loop":
        func = _annotations(func, {key: [var]})
    elif location == "attr":
        func = func.with_body(X.AttrStmt(var, key, 1, func.body))
    else:

        def mark(node):
            if location == "block" and isinstance(node, X.SBlock):
                ann = dict(node.annotations)
                ann[key] = [var]
                return X.SBlock(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    node.body,
                    node.init,
                    node.alloc_buffers,
                    node.match_buffers,
                    ann,
                )
            if location == "call" and isinstance(node, X.Call) and node.op.name == "tl.tileop.copy":
                ann = dict(node.annotations)
                ann[key] = [var]
                return X.Call(node.dtype, node.op, node.args, annotations=ann)
            return None

        func = func.with_body(X.stmt_functor.ir_transform(func.body, None, mark))
    baseline = _lower(func)
    actual = _lower(_annotations(func, {POLICY: "sync_if_supported"}))
    assert not _candidates(actual)
    tvm.ir.assert_structural_equal(baseline, actual, map_free_vars=True)


@pytest.mark.parametrize("reason", ["import", "target", "stages", "manual", "tma"])
def test_ineligible_pipeline_preserves_old_lowering(reason):
    func = _copies(_case(), explicit={"is_async_copy": 1} if reason == "manual" else {"is_tma_copy": 1} if reason == "tma" else {})
    if reason == "import":
        func = func.with_attr("pragma_import_c", "/* opaque source */")
    elif reason == "target":
        func = func.with_attr("target", tvm.target.Target("llvm"))
    elif reason == "stages":
        func = _annotations(func, {"tl_pipelined_num_stages": X.IntImm("int32", 2)})
    baseline, actual = _lower(func), _lower(_annotations(func, {POLICY: "sync_if_supported"}))
    assert not _candidates(actual)
    tvm.ir.assert_structural_equal(baseline, actual, map_free_vars=True)


def test_fifo_oracle_rejects_missing_retirement():
    lowered = _lower(_annotations(_copies(_case()), {POLICY: "sync_if_supported"}))

    def corrupt(node):
        if isinstance(node, X.Call) and node.op.name == "tirx.ptx_wait_group":
            return X.Call(node.dtype, node.op, [X.IntImm("int32", 99)])
        return None

    corrupted = lowered.with_body(X.stmt_functor.ir_transform(lowered.body, None, corrupt))
    with pytest.raises(AssertionError, match="before retirement"):
        _run(corrupted)


@pytest.mark.parametrize("scope", ["nested", "inherited_async", "opaque"])
def test_sync_policy_keeps_original_scope_guards(scope):
    func = _copies(_case())
    if scope == "inherited_async":
        func = func.with_body(X.AttrStmt(0, "async_commit_queue_scope", 0, func.body))
    elif scope == "nested":

        def nest(loop):
            return X.For(
                X.Var("outer", "int32"),
                0,
                2,
                X.ForKind.SERIAL,
                X.SBlockRealize([], True, X.SBlock([], [], [], "inner", loop)),
                annotations={
                    "software_pipeline_stage": [0],
                    "software_pipeline_order": [0],
                    "software_pipeline_async_producers": [0],
                    "software_pipeline_async_producer_groups": [-1],
                    "tl_pipelined_num_stages": X.IntImm("int32", 1),
                },
            )

        func = _rewrite_loop(func, nest)
    else:

        def add_opaque(node):
            if isinstance(node, X.SBlock) and node.name_hint == "produce_a":
                return X.SBlock(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    X.SeqStmt([X.Evaluate(X.call_pure_extern("int32", "unknown_copy_observer")), node.body]),
                    node.init,
                    node.alloc_buffers,
                    node.match_buffers,
                    node.annotations,
                )
            return None

        func = func.with_body(X.stmt_functor.ir_transform(func.body, None, add_opaque))
    baseline = _lower(func)
    actual = _lower(_annotations(func, {POLICY: "sync_if_supported"}))
    assert not _candidates(actual)
    tvm.ir.assert_structural_equal(baseline, actual, map_free_vars=True)
