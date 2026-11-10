"""Keep a two-run lane permutation after a mixed-precision elementwise map."""

from tvm import arith, ir, tirx
from tvm.tirx.stmt_functor import post_order_visit, substitute
from tilelang.ir import get_stmt_span, stamp_stmt_spans


def _uses(expr, var):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.Var) and node.same_as(var):
            found = True

    post_order_visit(expr, visit)
    return found


def _pure_index(expr):
    indirect = []
    post_order_visit(expr, lambda node: indirect.append(node) if isinstance(node, (tirx.Call, tirx.BufferLoad)) else None)
    return not indirect


@tirx.functor.mutator
class _LatePermutation(tirx.PyStmtExprMutator):
    def __init__(self, owned):
        super().__init__()
        self.owned = owned
        self.analyzer = arith.Analyzer()

    def _dense(self, buffer):
        if not self.analyzer.can_prove(buffer.elem_offset == 0):
            return False
        stride = 1
        for i in reversed(range(len(buffer.shape))):
            if not isinstance(buffer.shape[i], tirx.IntImm) or buffer.shape[i].value <= 0:
                return False
            if buffer.strides and not self.analyzer.can_prove(buffer.strides[i] == stride):
                return False
            stride *= buffer.shape[i]
        return True

    def _load(self, load, loop, lane, parity, output):
        # Only distinct compiler-owned allocations: pointer parameters, views,
        # in-place stores and unknown alias relationships keep their old path.
        if load.buffer.data not in self.owned or load.buffer.data.same_as(output.data):
            return None
        if load.predicate is not None or not self._dense(load.buffer) or not all(_pure_index(i) for i in load.indices):
            return None
        indices = [self.analyzer.simplify(substitute(i, {loop.loop_var: 2 * lane + parity})) for i in load.indices]
        vector = False
        result = []
        for i, index in enumerate(indices):
            base = self.analyzer.simplify(substitute(index, {lane: 0}))
            if _uses(index, lane):
                if i != len(indices) - 1 or not self.analyzer.can_prove(index == base + lane):
                    return None
                if not self.analyzer.can_prove(base >= 0) or not self.analyzer.can_prove(base + 31 < load.buffer.shape[i]):
                    return None
                result.append(tirx.Ramp(base, 1, 32))
                vector = True
            else:
                if not self.analyzer.can_prove(index >= 0) or not self.analyzer.can_prove(index < load.buffer.shape[i]):
                    return None
                result.append(index)
        value = tirx.BufferLoad(load.buffer, result)
        return value if vector else tirx.Broadcast(value, 32)

    def _value(self, value, loop, lane, parity, output):
        # Rebuild the exact typed expression tree, not an algebraically
        # equivalent reassociation. Narrowing remains after both multiplies.
        if value.dtype != "float32":
            return None
        if isinstance(value, tirx.BufferLoad):
            return self._load(value, loop, lane, parity, output)
        if isinstance(value, tirx.FloatImm):
            return tirx.Broadcast(value, 32)
        if isinstance(value, tirx.Mul):
            a = self._value(value.a, loop, lane, parity, output)
            b = self._value(value.b, loop, lane, parity, output)
            return tirx.Mul(a, b) if a is not None and b is not None else None
        if isinstance(value, tirx.Cast):
            child = self._value(value.value, loop, lane, parity, output)
            return tirx.Cast("float32x32", child) if child is not None else None
        return None

    def _match(self, loop):
        if (
            loop.kind != tirx.ForKind.VECTORIZED
            or not isinstance(loop.extent, tirx.IntImm)
            or loop.extent.value != 64
            or not self.analyzer.can_prove(loop.min == 0)
            or (loop.step is not None and not self.analyzer.can_prove(loop.step == 1))
            or loop.annotations
            or loop.thread_binding is not None
        ):
            return None
        store = loop.body
        if (
            not isinstance(store, tirx.BufferStore)
            or store.predicate is not None
            or store.buffer.dtype != "float16"
            or not store.indices
            or not all(_pure_index(i) for i in store.indices)
            or store.buffer.data not in self.owned
            or not self._dense(store.buffer)
            or not isinstance(store.value, tirx.Cast)
            or not isinstance(store.value.value, tirx.Mul)
        ):
            return None
        indices = []
        for i, index in enumerate(store.indices):
            base = self.analyzer.simplify(substitute(index, {loop.loop_var: 0}))
            if i == len(store.indices) - 1:
                if not self.analyzer.can_prove(index == base + loop.loop_var):
                    return None
                if not self.analyzer.can_prove(base >= 0) or not self.analyzer.can_prove(base + 63 < store.buffer.shape[i]):
                    return None
                indices.append(tirx.Ramp(base, 1, 64))
            else:
                if (
                    _uses(index, loop.loop_var)
                    or not self.analyzer.can_prove(index >= 0)
                    or not self.analyzer.can_prove(index < store.buffer.shape[i])
                ):
                    return None
                indices.append(index)
        lane = tirx.Var("lane", loop.loop_var.dtype)
        self.analyzer.bind(lane, ir.Range(0, 32))
        halves = [self._value(store.value.value, loop, lane, p, store.buffer) for p in (0, 1)]
        if any(value is None for value in halves):
            return None
        # A uniform map has no permutation to remove. Require a contiguous
        # run, and different even/odd expressions (e.g. two distinct rows).
        loads = []
        post_order_visit(halves[0], lambda n: loads.append(n) if isinstance(n, tirx.BufferLoad) and n.dtype == "float32x32" else None)
        if not loads or ir.structural_equal(halves[0], halves[1]):
            return None
        # Concatenate before narrowing so that the final permutation consumes
        # a full HVX register. Shuffling two half32 (64-byte) values makes the
        # SDK scalarize lane extraction. This Cast still acts on each result
        # after its original FP32 expression; no arithmetic is reassociated.
        joined = tirx.Shuffle(halves, list(range(64)))
        narrowed = tirx.Cast("float16x64", joined)
        order = [index for i in range(32) for index in (i, i + 32)]
        result = tirx.BufferStore(store.buffer, tirx.Shuffle([narrowed], order), indices)
        stamp_stmt_spans(result, get_stmt_span(store))
        return result

    def visit_for_(self, op):
        # The bounds used below describe ordinary unit-step iteration. Do not
        # infer child access bounds through a different outer-loop contract.
        if op.step is not None and not self.analyzer.can_prove(op.step == 1):
            return op
        self.analyzer.bind(op.loop_var, ir.Range.from_min_extent(op.min, op.extent))
        result = self._match(op)
        if result is not None:
            return result
        body = self.visit_stmt(op.body)
        if body.same_as(op.body):
            return op
        result = tirx.For(op.loop_var, op.min, op.extent, op.kind, body, op.thread_binding, op.annotations, op.step)
        stamp_stmt_spans(result, get_stmt_span(op))
        return result


def LateLanePermutation():
    """Before DecoupleTypeCast, move a proven parity split after FP16 narrowing.

    Only v79/128-byte HVX, full 64-lane stores and pure FP32 multiplication
    trees are handled. Outer loops and rejected maps are unchanged.
    """

    def transform(func, _mod, _ctx):
        target = func.attrs.get("target") if func.attrs is not None else None
        if (
            _ctx.config.get("tirx.disable_vectorize", False)
            or target is None
            or target.kind.name != "hexagon"
            or str(target.attrs.get("mcpu", "")) != "hexagonv79"
            or int(target.attrs.get("min_vector_bits", 0)) != 1024
            or int(target.attrs.get("max_vector_bits", 0)) != 1024
            or not target.attrs.get("supports_unaligned_vector_access", False)
        ):
            return func
        owned = set()
        volatile = []

        def allocations(node):
            if isinstance(node, tirx.AllocBuffer):
                owned.add(node.buffer.data)
            elif isinstance(node, tirx.SBlock):
                owned.update(buffer.data for buffer in node.alloc_buffers)
            elif isinstance(node, tirx.AttrStmt) and node.attr_key == "volatile_scope":
                volatile.append(node)

        post_order_visit(func.body, allocations)
        if volatile:
            return func
        mutator = _LatePermutation(owned)
        body = mutator.visit_stmt(func.body)
        return func.with_body(body)

    return tirx.transform.prim_func_pass(transform, opt_level=0, name="tl.hexagon.LateLanePermutation")
