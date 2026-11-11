"""A parity permutation moves after arithmetic, not across an arithmetic edge."""

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tvm import tirx


def _case(mode="map", arch=79, bits=1024, active=64):
    lane = tirx.Var("lane", "int32")
    outer = tirx.Var("outer", "int32")
    source = tirx.decl_buffer((2, 2048), "float32", name="source", scope="shared.dyn")
    rows = tirx.decl_buffer((2,), "float32", name="rows", scope="shared.dyn")
    columns = tirx.decl_buffer((4096 if mode == "strided" else 2048,), "float32", name="columns", scope="shared.dyn")
    output = tirx.decl_buffer((4096,), "float16", name="output", scope="shared.dyn")
    row = lane % 2
    column = outer * 32 + lane // 2
    value = (source[row, column] * rows[row]) * columns[column * (2 if mode == "strided" else 1)]
    if mode == "add":
        value = source[row, column] + rows[row]
    elif mode == "cast":
        value = tirx.Cast("float32", tirx.Cast("float16", value))
    elif mode == "alias":
        output = tirx.decl_buffer((64,), "float16", data=source.data, scope="shared.dyn")
    elif mode == "oob":
        value = source[row, column + 1] * rows[row]
    elif mode == "unknown_call":
        value = source[row, column] * tirx.call_extern("float32", "unknown_math", rows[row])
    elif mode == "constant":
        value = source[row, column] * tirx.FloatImm("float32", 0.75)
    elif mode == "rank_zero":
        output = tirx.decl_buffer((), "float16", name="output", scope="shared.dyn")
    store = tirx.BufferStore(output, tirx.Cast("float16", value), [] if mode == "rank_zero" else [outer * 64 + lane])
    body = tirx.IfThenElse(lane < 60, store, None) if mode == "predicate" else store
    if mode == "extra":
        body = tirx.SeqStmt([store, tirx.Evaluate(tirx.call_extern("int32", "opaque"))])
    loop = tirx.For(
        lane,
        0,
        63 if mode == "tail" else 64,
        tirx.ForKind.VECTORIZED,
        body,
        annotations={"unroll_explicit": 1} if mode == "annotation" else None,
        step=2 if mode == "step" else None,
    )
    loop = tirx.For(outer, 0, active, tirx.ForKind.SERIAL, loop)
    buffers = [source, rows, columns, output]
    owned = buffers[1:] if mode == "external" else buffers
    # AllocBuffer is a lexical declaration. Only objects allocated here are
    # eligible; pointer parameters do not acquire a noalias promise implicitly.
    body = tirx.SeqStmt([*(tirx.AllocBuffer(b) for b in owned), loop])
    if mode == "volatile":
        body = tirx.AttrStmt(source.data, "volatile_scope", 1, body)
    func = tirx.PrimFunc([], body).with_attr(
        "target",
        tvm.target.Target(
            {
                "kind": "hexagon",
                "mcpu": f"hexagonv{arch}",
                "min_vector_bits": bits,
                "max_vector_bits": bits,
            }
        ),
    )
    return func, lane, store, buffers


def _apply(func):
    return tilelang.hexagon.transform.LateLanePermutation()(tvm.IRModule({"main": func}))["main"]


def _nodes(body, cls):
    result = []
    tirx.stmt_functor.post_order_visit(body, lambda n: result.append(n) if isinstance(n, cls) else None)
    return result


@pytest.mark.parametrize(
    "mode",
    [
        "tail",
        "step",
        "strided",
        "alias",
        "external",
        "add",
        "cast",
        "predicate",
        "extra",
        "annotation",
        "oob",
        "unknown_call",
        "rank_zero",
        "volatile",
    ],
)
def test_rejected_maps_are_unchanged(mode):
    func, *_ = _case(mode)
    tvm.ir.assert_structural_equal(func, _apply(func))


@pytest.mark.parametrize("arch,bits", [(73, 1024), (75, 1024), (81, 1024), (79, 512)])
def test_other_targets_are_unchanged(arch, bits):
    func, *_ = _case(arch=arch, bits=bits)
    tvm.ir.assert_structural_equal(func, _apply(func))


def test_disabled_vectorization_keeps_scalar_fallback():
    func, *_ = _case()
    with tvm.transform.PassContext(config={"tirx.disable_vectorize": True}):
        tvm.ir.assert_structural_equal(func, _apply(func))


def test_target_requiring_aligned_vectors_keeps_existing_legalization():
    func, *_ = _case()
    func = func.with_attr(
        "target",
        tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79", "supports_unaligned_vector_access": False}),
    )
    tvm.ir.assert_structural_equal(func, _apply(func))


def _eval(expr, arrays, variables):
    if isinstance(expr, tirx.IntImm):
        return expr.value
    if isinstance(expr, tirx.FloatImm):
        return np.float32(expr.value)
    if isinstance(expr, tirx.Var):
        return variables[expr]
    if isinstance(expr, tirx.BufferLoad):
        return arrays[expr.buffer.data][tuple(_eval(index, arrays, variables) for index in expr.indices)]
    if isinstance(expr, tirx.Cast):
        return np.asarray(_eval(expr.value, arrays, variables)).astype(str(expr.dtype).split("x")[0])
    if isinstance(expr, tirx.Broadcast):
        return np.repeat(_eval(expr.value, arrays, variables), int(expr.lanes))
    if isinstance(expr, tirx.Ramp):
        return _eval(expr.base, arrays, variables) + np.arange(int(expr.lanes)) * _eval(expr.stride, arrays, variables)
    if isinstance(expr, tirx.Shuffle):
        joined = np.concatenate([_eval(value, arrays, variables) for value in expr.vectors])
        return joined[[_eval(index, arrays, variables) for index in expr.indices]]
    operations = {tirx.Add: np.add, tirx.Mul: np.multiply, tirx.FloorDiv: np.floor_divide, tirx.FloorMod: np.mod}
    for cls, operation in operations.items():
        if isinstance(expr, cls):
            return operation(_eval(expr.a, arrays, variables), _eval(expr.b, arrays, variables))
    raise AssertionError(type(expr))


def test_lane_mapping_and_round_order():
    func, lane, original, buffers = _case()
    result = _apply(func)
    outer = _nodes(func.body, tirx.For)[-1].loop_var
    stores = _nodes(result.body, tirx.BufferStore)
    assert len(stores) == 1
    shuffled = stores[0].value
    assert isinstance(shuffled, tirx.Shuffle)
    assert len(shuffled.vectors) == 1
    half = shuffled.vectors[0]
    assert isinstance(half, tirx.Cast) and half.dtype == "float16x64"
    assert isinstance(half.value, tirx.Shuffle)
    for row in half.value.vectors:
        assert isinstance(row, tirx.Mul) and isinstance(row.a, tirx.Mul)
        assert row.dtype == row.a.dtype == "float32x32"
    rng = np.random.default_rng(202)
    for _ in range(16):
        arrays = {buffer.data: rng.uniform(-2, 2, tuple(int(n) for n in buffer.shape)).astype(buffer.dtype) for buffer in buffers}
        expected = np.array([_eval(original.value, arrays, {outer: 0, lane: i}) for i in range(64)], dtype=np.float16)
        actual = _eval(shuffled, arrays, {outer: 0})
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert len(_nodes(result.body, tirx.For)) == 1


def test_no_cross_function_allocation_state():
    first, *_ = _case()
    second, *_ = _case("external")
    module = tvm.IRModule({"first": first, "second": second})
    result = tilelang.hexagon.transform.LateLanePermutation()(module)
    assert _nodes(result["first"].body, tirx.Shuffle)
    tvm.ir.assert_structural_equal(second, result["second"])


def test_constant_multiply_without_model_specific_scales():
    func, *_ = _case("constant")
    result = _apply(func)
    assert len(_nodes(result.body, tirx.Shuffle)) == 2
    assert sum(str(n.dtype).startswith("float") for n in _nodes(_nodes(result.body, tirx.BufferStore)[0].value, tirx.Mul)) == 2


@pytest.mark.parametrize("inputs", [1, 2])
def test_shuffle_inputs_are_evaluated_once(inputs):
    output = tirx.decl_buffer((64,), "float16", name="output")
    # The two expressions intentionally have the same spelling. A source CSE
    # cache must not merge two side-effecting calls, or duplicate a single one.
    vectors = [tirx.call_extern("float16x64", "next_vector") for _ in range(inputs)]
    order = list(reversed(range(64)))
    value = tirx.Shuffle(vectors, order)
    func = tirx.PrimFunc([output.data], tirx.BufferStore(output, value, [tirx.Ramp(0, 1, 64)]))
    func = func.with_attr("global_symbol", "shuffle_once")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"})
    module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"main": func}), target)
    source = module.inspect_source()
    assert source.count("next_vector()") == inputs
    assert "__builtin_shufflevector" in source


@pytest.mark.parametrize("lazy", [False, True])
def test_shuffle_call_stays_in_its_conditional_arm(lazy):
    output = tirx.decl_buffer((64,), "float16", name="output")
    condition = tirx.Var("condition", "bool")
    vector = tirx.call_extern("float16x64", "next_vector")
    shuffled = tirx.Shuffle([vector], list(reversed(range(64))))
    zero = tirx.Broadcast(tirx.FloatImm("float16", 0), 64)
    value = tirx.if_then_else(condition, shuffled, zero) if lazy else tirx.Select(condition, shuffled, zero)
    func = tirx.PrimFunc([output.data, condition], tirx.BufferStore(output, value, [tirx.Ramp(0, 1, 64)]))
    func = func.with_attr("global_symbol", "conditional_shuffle")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"})
    module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"main": func}), target)
    source = module.inspect_source()
    if lazy:
        assert source.count("next_vector()") == 1
        assert source.index("if (condition)") < source.index("next_vector()") < source.index("} else {")
    else:
        assert source.count("next_vector()") == 1
        assert source.index("condition ?") < source.index("next_vector()")


@pytest.mark.parametrize("operator", ["and", "or"])
def test_shuffle_call_stays_in_short_circuit_operand(operator):
    output = tirx.decl_buffer((1,), "int32", name="output")
    condition = tirx.Var("condition", "bool")
    vector = tirx.call_extern("float16x64", "next_vector")
    shuffled = tirx.Shuffle([vector], list(reversed(range(64))))
    predicate = tirx.call_extern("bool", "test_vector", shuffled)
    value = tirx.And(condition, predicate) if operator == "and" else tirx.Or(condition, predicate)
    func = tirx.PrimFunc([output.data, condition], tirx.BufferStore(output, tirx.Cast("int32", value), [0]))
    func = func.with_attr("global_symbol", "short_circuit_shuffle")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"})
    module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"main": func}), target)
    source = module.inspect_source()
    assert source.count("next_vector()") == 1
    assert source.index("condition &&" if operator == "and" else "condition ||") < source.index("next_vector()")


@pytest.mark.parametrize("dtype,lanes", [("float16", 16), ("float16", 32), ("float32", 24), ("int32", 32), ("bool", 32)])
def test_other_shuffles_keep_existing_codegen(dtype, lanes):
    output = tirx.decl_buffer((lanes,), dtype, name="output")
    input_buffer = tirx.decl_buffer((lanes,), dtype, name="input_buffer")
    vector = tirx.BufferLoad(input_buffer, [tirx.Ramp(0, 1, lanes)])
    value = tirx.Shuffle([vector], list(reversed(range(lanes))))
    func = tirx.PrimFunc([input_buffer.data, output.data], tirx.BufferStore(output, value, [tirx.Ramp(0, 1, lanes)]))
    func = func.with_attr("global_symbol", "partial_shuffle")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"})
    build = tvm.get_global_func("target.build.tilelang_hexagon")
    if dtype == "bool":
        # Bool vectors have no native vocabulary in the existing printer.
        # This optimization must not pretend to add that support.
        with pytest.raises(tvm.error.InternalError, match="Cannot convert type boolx32"):
            build(tvm.IRModule({"main": func}), target)
        return
    module = build(tvm.IRModule({"main": func}), target)
    assert "__builtin_shufflevector" not in module.inspect_source()


def _nested_case(mode="map", columns=64, pairs=2, outer_kind=tirx.ForKind.UNROLLED, inner_kind=tirx.ForKind.SERIAL):
    column, pair, lane = (tirx.Var(name, "int32") for name in ("column", "pair", "lane"))
    source = tirx.decl_buffer((pairs * 2, columns * 32), "float32", name="source", scope="shared.dyn")
    rows = tirx.decl_buffer((pairs * 2,), "float32", name="rows", scope="shared.dyn")
    weights = tirx.decl_buffer((columns * 32,), "float32", name="weights", scope="shared.dyn")
    output = tirx.decl_buffer((columns, 16, 64), "float16", name="output", scope="shared.dyn")
    r, c = pair * 2 + lane % 2, column * 32 + lane // 2
    value = (source[r, c] * rows[r]) * weights[c]
    out_column = 0 if mode == "overlap" else column % 2 if mode == "non_affine" else column
    store = tirx.BufferStore(output, tirx.Cast("float16", value), [out_column, pair + 2, lane])
    vector = tirx.For(lane, 0, 64, tirx.ForKind.VECTORIZED, store)
    if mode == "extra":
        vector = tirx.SeqStmt([vector, tirx.Evaluate(tirx.call_extern("int32", "opaque"))])
    inner = tirx.For(pair, 0, pairs, inner_kind, vector)
    outer = tirx.For(column, 0, columns, outer_kind, inner, annotations={"unroll_explicit": 0} if mode == "annotation" else None)
    buffers = [source, rows, weights, output]
    func = tirx.PrimFunc([], tirx.SeqStmt([*(tirx.AllocBuffer(b) for b in buffers), outer]))
    func = func.with_attr("target", tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv79"}))
    return func, outer, inner, store, lane, buffers


@pytest.mark.parametrize("kinds", [(tirx.ForKind.UNROLLED, tirx.ForKind.SERIAL), (tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED)])
def test_interchange_preserves_holey_store_domain_and_loop_kinds(kinds):
    func, outer, inner, _, _, _ = _nested_case(outer_kind=kinds[0], inner_kind=kinds[1])
    result = _apply(func)
    loops = _nodes(result.body, tirx.For)
    assert len(loops) == 2
    # Post-order visits the new inner loop first. The original loop objects'
    # variable/domain/kind move together; the output still leaves row holes.
    assert loops[0].loop_var.same_as(outer.loop_var) and loops[0].kind == outer.kind
    assert loops[1].loop_var.same_as(inner.loop_var) and loops[1].kind == inner.kind
    store = _nodes(result.body, tirx.BufferStore)[0]
    assert isinstance(store.value, tirx.Shuffle)
    assert store.indices[0].same_as(outer.loop_var)
    assert tvm.arith.Analyzer().can_prove(store.indices[1] == inner.loop_var + 2)
    addresses = [i * 1024 + (j + 2) * 64 + k for i in range(64) for j in range(2) for k in range(64)]
    assert len(addresses) == len(set(addresses)) == 8192


@pytest.mark.parametrize(
    "mode,columns,pairs",
    [("overlap", 64, 2), ("non_affine", 64, 2), ("extra", 64, 2), ("annotation", 64, 2), ("map", 4, 2), ("map", 64, 5)],
)
def test_interchange_rejects_dependencies_and_unqualified_shapes(mode, columns, pairs):
    func, outer, inner, *_ = _nested_case(mode, columns, pairs)
    result = _apply(func)
    tvm.ir.assert_structural_equal(func, result)


def test_interchange_keeps_each_typed_result_and_unwritten_element():
    func, outer, inner, original, lane, buffers = _nested_case(columns=64)
    result = _apply(func)
    transformed = _nodes(result.body, tirx.BufferStore)[0]
    rng = np.random.default_rng(20211)
    arrays = {b.data: rng.uniform(-2, 2, tuple(int(n) for n in b.shape)).astype(b.dtype) for b in buffers}
    expected = arrays[buffers[-1].data].copy()
    actual = expected.copy()
    for i in range(64):
        for j in range(2):
            for k in range(64):
                expected[i, j + 2, k] = _eval(original.value, arrays, {outer.loop_var: i, inner.loop_var: j, lane: k})
    for j in range(2):
        for i in range(64):
            actual[i, j + 2] = _eval(transformed.value, arrays, {outer.loop_var: i, inner.loop_var: j})
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("active", [1, 4, 31, 32, 63])
def test_small_active_run_in_large_allocation_is_unchanged(active):
    func, *_ = _case(active=active)
    tvm.ir.assert_structural_equal(func, _apply(func))


def test_standalone_vector_store_keeps_existing_lowering():
    func, *_ = _case()
    outer = func.body.seq[-1]
    func = func.with_body(tirx.SeqStmt([*func.body.seq[:-1], tirx.stmt_functor.substitute(outer.body, {outer.loop_var: 0})]))
    tvm.ir.assert_structural_equal(func, _apply(func))


def test_singleton_pair_axis_exposes_the_long_active_column_run():
    func, _, inner, *_ = _nested_case(pairs=1)
    body = tirx.stmt_functor.substitute(func.body, {inner.loop_var: 0})
    result = _apply(func.with_body(body))
    assert _nodes(result.body, tirx.Shuffle)
