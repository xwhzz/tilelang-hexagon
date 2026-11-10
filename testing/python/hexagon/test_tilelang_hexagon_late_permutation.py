"""A parity permutation moves after arithmetic, not across an arithmetic edge."""

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tvm import tirx


def _case(mode="map", arch=79, bits=1024):
    lane = tirx.Var("lane", "int32")
    source = tirx.decl_buffer((2, 32), "float32", name="source", scope="shared.dyn")
    rows = tirx.decl_buffer((2,), "float32", name="rows", scope="shared.dyn")
    columns = tirx.decl_buffer((64 if mode == "strided" else 32,), "float32", name="columns", scope="shared.dyn")
    output = tirx.decl_buffer((64,), "float16", name="output", scope="shared.dyn")
    row = lane % 2
    column = lane // 2
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
    store = tirx.BufferStore(output, tirx.Cast("float16", value), [] if mode == "rank_zero" else [lane])
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
        expected = np.array([_eval(original.value, arrays, {lane: i}) for i in range(64)], dtype=np.float16)
        actual = _eval(shuffled, arrays, {})
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert not _nodes(result.body, tirx.For)


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
    assert len(_nodes(result.body, tirx.Mul)) == 2


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
