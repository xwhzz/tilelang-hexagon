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
        self.rewritten_stores = []
        self.parents = []

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
        self.rewritten_stores.append(result)
        return result

    def _contiguous_run(self, store, loop):
        # Check the active iteration axis, not the allocation's total width.
        # Each varying input must advance by exactly one FP32 vector.
        loads = []
        post_order_visit(store.value, lambda n: loads.append(n) if isinstance(n, tirx.BufferLoad) else None)
        contiguous = False
        for load in loads:
            offset, scale = 0, 1
            for index, shape in reversed(list(zip(load.indices, load.buffer.shape))):
                offset += (index.base if isinstance(index, tirx.Ramp) else index) * scale
                scale *= int(shape)
            delta = self.analyzer.simplify(substitute(offset, {loop.loop_var: loop.loop_var + 1}) - offset)
            if self.analyzer.can_prove(delta == 0):
                continue
            if load.dtype != "float32x32" or not self.analyzer.can_prove(delta == 32):
                return False
            contiguous = True
        return contiguous

    def _profitable_run(self, vector_loop, store):
        # The initial qualified scope needs at least 64 consecutive column
        # vectors. Small active regions keep their existing lowering, even
        # when their backing allocation is large.
        if not self.parents or not self.parents[-1].body.same_as(vector_loop):
            return False
        inner = self.parents[-1]

        def long_run(loop):
            return (
                loop.kind in (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED)
                and isinstance(loop.extent, tirx.IntImm)
                and loop.extent.value >= 64
                and self.analyzer.can_prove(loop.min == 0)
                and (loop.step is None or self.analyzer.can_prove(loop.step == 1))
                and not loop.annotations
                and loop.thread_binding is None
                and self._contiguous_run(store, loop)
            )

        if long_run(inner):
            return True
        if len(self.parents) < 2 or not self.parents[-2].body.same_as(inner):
            return False
        # A singleton pair axis disappears in normal simplification, so it
        # already exposes the long run without reordering any iterations.
        if (
            inner.kind in (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED)
            and self.analyzer.can_prove(inner.extent == 1)
            and self.analyzer.can_prove(inner.min == 0)
            and (inner.step is None or self.analyzer.can_prove(inner.step == 1))
            and not inner.annotations
            and inner.thread_binding is None
            and long_run(self.parents[-2])
        ):
            return True
        # Do not introduce permutation under a short inner loop unless the
        # complete independent-nest proof also permits the required swap.
        candidate = tirx.For(
            inner.loop_var, inner.min, inner.extent, inner.kind, store, inner.thread_binding, inner.annotations, inner.step
        )
        return self._interchange(self.parents[-2], candidate) is not None

    def _interchange(self, outer, inner):
        # Only reorder the independent store just produced above. A longer
        # contiguous input run belongs inside a short parity-pair loop, so the
        # SDK can pipeline more than a handful of dependent multiplies.
        if not isinstance(inner, tirx.For) or not isinstance(inner.body, tirx.BufferStore):
            return None
        store = inner.body
        if not any(store.same_as(value) for value in self.rewritten_stores):
            return None
        for loop in (outer, inner):
            if (
                loop.kind not in (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED)
                or not isinstance(loop.extent, tirx.IntImm)
                or not self.analyzer.can_prove(loop.min == 0)
                or (loop.step is not None and not self.analyzer.can_prove(loop.step == 1))
                or loop.annotations
                or loop.thread_binding is not None
            ):
                return None
        if outer.extent.value < 64 or not 1 <= inner.extent.value <= 4:
            return None
        # A dense allocation may have holes between the written row groups.
        # Flatten the store base and prove non-overlapping 64-element runs:
        # origin + outer * outer_pitch + inner * inner_pitch + lane.
        flat, stride = 0, 1
        for index, shape in reversed(list(zip(store.indices, store.buffer.shape))):
            base = index.base if isinstance(index, tirx.Ramp) else index
            flat += base * stride
            stride *= int(shape)
        if stride > (1 << 31) - 1 or not self.analyzer.can_prove(flat >= 0) or not self.analyzer.can_prove(flat + 63 < stride):
            return None
        zero = {outer.loop_var: 0, inner.loop_var: 0}
        origin = self.analyzer.simplify(substitute(flat, zero))
        pitches = []
        for loop in (outer, inner):
            at_one = {**zero, loop.loop_var: 1}
            pitch = self.analyzer.simplify(substitute(flat, at_one) - origin)
            if not isinstance(pitch, tirx.IntImm):
                return None
            pitches.append(pitch.value)
        outer_pitch, inner_pitch = pitches
        if inner_pitch < 64 or outer_pitch < (inner.extent.value - 1) * inner_pitch + 64:
            return None
        if not self.analyzer.can_prove(flat == origin + outer.loop_var * outer_pitch + inner.loop_var * inner_pitch):
            return None

        if not self._contiguous_run(store, outer):
            return None

        def with_body(loop, body):
            result = tirx.For(loop.loop_var, loop.min, loop.extent, loop.kind, body, loop.thread_binding, loop.annotations, loop.step)
            stamp_stmt_spans(result, get_stmt_span(loop))
            return result

        return with_body(inner, with_body(outer, store))

    def visit_for_(self, op):
        # The bounds used below describe ordinary unit-step iteration. Do not
        # infer child access bounds through a different outer-loop contract.
        if op.step is not None and not self.analyzer.can_prove(op.step == 1):
            return op
        self.analyzer.bind(op.loop_var, ir.Range.from_min_extent(op.min, op.extent))
        result = self._match(op)
        if result is not None:
            if self._profitable_run(op, result):
                return result
            self.rewritten_stores.pop()
            return op
        self.parents.append(op)
        try:
            body = self.visit_stmt(op.body)
        finally:
            self.parents.pop()
        interchanged = self._interchange(op, body)
        if interchanged is not None:
            return interchanged
        if body.same_as(op.body):
            return op
        result = tirx.For(op.loop_var, op.min, op.extent, op.kind, body, op.thread_binding, op.annotations, op.step)
        stamp_stmt_spans(result, get_stmt_span(op))
        return result


def LateLanePermutation():
    """Before DecoupleTypeCast, move a proven parity split after FP16 narrowing.

    Only v79/128-byte HVX, full 64-lane stores and pure FP32 multiplication
    trees with at least 64 consecutive 32-column iterations are handled.
    A short inner parity-pair loop requires a proven independent interchange,
    preserving both loop kinds; rejected maps keep their existing lowering.
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
