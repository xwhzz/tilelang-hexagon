"""Fold a private integer GEMM followed by a column-wise narrowing map."""

from collections import Counter

import tvm_ffi
from tvm import arith, ir, tirx, transform

from tilelang.hexagon.intrinsics.crouton import CroutonLayout
from tilelang.hexagon.intrinsics.panel import PanelLayout
from tilelang.hexagon.op.gemm.gemm_hvx import _strides


def _call(stmt, name):
    return isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and getattr(stmt.value.op, "name", None) == name


@tirx.functor.visitor
class _Uses(tirx.PyStmtExprVisitor):
    """Count each logical use, including opaque pointer escapes and buffer aliases."""

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.count = Counter()
        self.private = set()
        self.views = set()
        self.layouts = {}

    def visit_sblock_(self, block):
        self.private.update(b.data for b in block.alloc_buffers)
        # An opaque block may advertise access through a region.
        self.views.update(r.buffer.data for r in [*block.reads, *block.writes])
        for key, layout in block.annotations.get("layout_map", {}).items():
            self.layouts[key.data if isinstance(key, tirx.Buffer) else key] = layout
        for view in block.match_buffers:
            self.views.update((view.buffer.data, view.source.buffer.data))
        super().visit_sblock_(block)

    def visit_for_(self, loop):
        if loop.step is None or self.analyzer.can_prove_equal(loop.step, 1):
            self.analyzer.bind(loop.loop_var, ir.Range.from_min_extent(loop.min, loop.extent))
        super().visit_for_(loop)

    def visit_decl_buffer_(self, decl):
        self.views.add(decl.buffer.data)
        super().visit_decl_buffer_(decl)

    def visit_buffer_load_(self, load):
        self.count[load.buffer.data] += 1
        super().visit_buffer_load_(load)
        if load.predicate is not None:
            self.visit_expr(load.predicate)

    def visit_buffer_store_(self, store):
        self.count[store.buffer.data] += 1
        for index in store.indices:
            self.visit_expr(index)
        self.visit_expr(store.value)
        if store.predicate is not None:
            self.visit_expr(store.predicate)

    def visit_var_(self, var):
        self.count[var] += 1  # Includes escaped pointers.


def _match(call, loop, analyzer):
    """Return the first output, bias and shift elements of an exact rectangular map."""
    ci = call.args[2].args[0]
    m, n = call.args[5:7]

    def parallel(loop, extent):
        return (
            isinstance(loop, tirx.For)
            and loop.kind == tirx.ForKind.PARALLEL
            and (loop.step is None or analyzer.can_prove_equal(loop.step, 1))
            and all(key in ("parallel_loop_layout", "parallel_loop_requires_padding_guard") for key in loop.annotations)
            and analyzer.can_prove_equal(loop.min, 0)
            and analyzer.can_prove_equal(loop.extent, extent)
        )

    if parallel(loop, m) and parallel(loop.body, n):
        i, j = loop.loop_var, loop.body.loop_var
        loop = loop.body.body
    elif parallel(loop, m * n):
        # LayoutInference may have linearized a map of shared buffers.
        i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
        analyzer.bind(i, ir.Range(0, m))
        analyzer.bind(j, ir.Range(0, n))
        loop = tirx.stmt_functor.substitute(loop.body, {loop.loop_var: i * n + j})
    else:
        return None
    if not isinstance(loop, tirx.BufferStore) or loop.predicate is not None:
        return None
    cast = loop.value
    if not (isinstance(cast, tirx.Cast) and cast.dtype == "int16" and loop.buffer.dtype == "int16"):
        return None
    shr = cast.value
    if not (isinstance(shr, tirx.Call) and getattr(shr.op, "name", None) == "tirx.shift_right" and shr.dtype == "int32"):
        return None
    add, shift = shr.args
    if not isinstance(add, tirx.Add) or not all(isinstance(x, tirx.BufferLoad) and x.predicate is None for x in (add.a, add.b, shift)):
        return None
    acc, bias = (add.a, add.b) if add.a.buffer.same_as(ci.buffer) else (add.b, add.a)
    if not acc.buffer.same_as(ci.buffer) or len(acc.indices) < 2:
        return None
    expected = [*ci.indices[:-2], ci.indices[-2] + i, ci.indices[-1] + j]
    if not all(analyzer.can_prove_equal(x, y) for x, y in zip(acc.indices, expected, strict=True)):
        return None

    def first(buf, indices, offsets, extents):
        if len(indices) < len(offsets):
            return None
        offsets = [0] * (len(indices) - len(offsets)) + offsets
        extents = [1] * (len(indices) - len(extents)) + extents
        origin = [analyzer.simplify(tirx.stmt_functor.substitute(x, {i: 0, j: 0})) for x in indices]
        # Only hoist affine, loop-invariant addresses. An indirect index can
        # depend on a buffer written by the map.
        for x, base, step, size, bound in zip(indices, origin, offsets, extents, buf.shape, strict=True):
            if not (analyzer.can_prove(base >= 0) and analyzer.can_prove(base + size <= bound)):
                return None
            if not analyzer.can_prove_equal(x, base + step):
                return None
            if not isinstance(base, (tirx.IntImm, tirx.Var)):
                nodes = []
                tirx.stmt_functor.post_order_visit(base, nodes.append)
                if any(isinstance(node, (tirx.BufferLoad, tirx.Call)) for node in nodes):
                    return None
        return tirx.BufferLoad(buf, origin)

    if any(x.dtype != "int32" or len(x.indices) != 1 for x in (bias, shift)):
        return None
    firsts = (
        first(loop.buffer, loop.indices, [i, j], [m, n]),
        first(bias.buffer, bias.indices, [j], [n]),
        first(shift.buffer, shift.indices, [j], [n]),
    )
    return firsts if all(x is not None for x in firsts) else None


def fuse_hmx_requant(mod, *, prepare_local=False):
    """Fuse a contiguous map after layout inference, or normalize its private local before validation."""
    build_gemm = ir.Op.get("tl.tileop.gemm").get_attr("TLOpBuilder")
    rejection = tvm_ffi.get_global_func("tl.hexagon.GemmHMXIntRejection")
    vtcm = tvm_ffi.get_global_func("tl.hexagon.IsVTCMBuffer")
    row_major = tvm_ffi.get_global_func("tl.hexagon.RowMajor")

    def rewrite_func(func, _mod, _ctx):
        if not isinstance(func.body, tirx.SBlockRealize):
            return func
        layouts = func.body.block.annotations.get("layout_map", {})
        analyzer = arith.Analyzer()
        uses = None
        promoted = {}

        def match_pair(call, stmt):
            nonlocal uses
            c = call.args[2].args[0].buffer
            scopes = ("local",) if prepare_local else ("local.fragment", "shared", "shared.dyn")
            if c.scope() not in scopes or c.dtype != "int32" or not analyzer.can_prove_equal(call.args[9], 1):
                return None
            if uses is None:
                uses = _Uses(analyzer)
                uses.visit_stmt(func.body)
                uses.private.difference_update(b.data for b in func.buffer_map.values())
            if c.data not in uses.private or uses.count[c.data] != 2:
                return None
            firsts = _match(call, stmt, analyzer)
            if firsts is None:
                return None
            dst, bias, shift = (x.buffer for x in firsts)
            a, b = (call.args[i].args[0].buffer for i in (0, 1))
            inputs = (a, b, bias, shift)
            if not vtcm(bias) or not vtcm(shift) or uses.views.intersection(buf.data for buf in (c, dst, *inputs)):
                return None
            if any(dst.data.same_as(buf.data) for buf in (c, *inputs)):
                return None
            if (
                dst.data not in uses.private
                and not all(buf.data in uses.private for buf in inputs)
                and not func.attrs.get("tirx.noalias", False)
            ):
                return None

            # Before inference explicit layouts are indexed by data Var. After
            # inference use the actual physical layouts chosen for this function.
            buffers = (dst, bias, shift)
            physical = {buf: uses.layouts[buf.data] for buf in buffers if buf.data in uses.layouts} if prepare_local else layouts
            for buf in buffers:
                strides = _strides(buf, compact=buf in physical)
                if not row_major(buf, physical) or not analyzer.can_prove_equal(strides[-1], 1):
                    return None
                if (
                    buf.scope() == "global"
                    and buf in physical
                    and not all(analyzer.can_prove_equal(x, y) for x, y in zip(strides, _strides(buf), strict=True))
                ):
                    return None
            if not analyzer.can_prove(_strides(dst, compact=dst in physical)[-2] >= call.args[6]):
                return None

            if prepare_local:
                # Parallel-indexed thread locals have no distributed ownership.
                # Only the private accumulator of this pair qualifies as a fragment.
                al, bl = uses.layouts.get(a.data), uses.layouts.get(b.data)
                if al is not None and not (isinstance(al, CroutonLayout) and al.bits == 8 and not al.column_tiles and not al.transposed):
                    return None
                if (bl is not None and not (isinstance(bl, PanelLayout) and bl.lane == 4)) or c.data in uses.layouts:
                    return None
                frag = tirx.decl_buffer(c.shape, c.dtype, name=c.name, scope="local.fragment")
                region = call.args[2]
                args = list(call.args)
                args[2] = tirx.Call(region.dtype, region.op, [frag[tuple(region.args[0].indices)], *region.args[1:]])
                if not rejection(build_gemm(args, call.annotations)):
                    promoted[c.data] = frag.data
                return None

            if rejection(build_gemm(call.args, call.annotations)):
                return None
            annotations = dict(call.annotations)
            annotations["hexagon.requant_output"] = firsts
            return tirx.Evaluate(tirx.Call(call.dtype, call.op, call.args, annotations, call.span))

        def rewrite(seq):
            out = []
            for stmt in seq.seq:
                if out and isinstance(stmt, tirx.For) and _call(out[-1], "tl.tileop.gemm"):
                    fused = match_pair(out[-1].value, stmt)
                    if fused is not None:
                        out[-1] = fused
                        continue
                out.append(stmt)
            if len(out) == len(seq.seq):
                return seq
            return out[0] if len(out) == 1 else tirx.SeqStmt(out)

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.SeqStmt"])
        if promoted:
            body = tirx.stmt_functor.substitute(body, promoted)
        return func.with_body(body)

    name = "PrepareLocalHMXRequant" if prepare_local else "FuseHMXRequant"
    return transform.Sequential([tirx.transform.prim_func_pass(rewrite_func, opt_level=0, name="tl.hexagon." + name)])(mod)


def normalize_local_hmx_requant(mod):
    return fuse_hmx_requant(mod, prepare_local=True)
