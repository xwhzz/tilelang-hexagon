"""The load-only queue policy requires closed, statement-only managed use."""

import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.engine.lower import _prepare_device_codegen_mod

I = tvm.tirx
PREFIX = "tl_hexagon_dma_async_context<1, 1"
COMPACT_PREFIX = "tl_hexagon_dma_compact_load_context<1, 1"


def _normal(arch="v79"):
    @T.prim_func
    def copy(A: T.Tensor((256,), "uint8"), O: T.Tensor((256,), "uint8")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "uint8")
            T.dma_copy(A, s)
            T.dma_wait()
            T.copy(s, O)

    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(copy, target=target, enable_host_codegen=False, enable_device_compile=False)
    # Exercise the final IR that the C emitter actually consumes.
    with target:
        final_mod = _prepare_device_codegen_mod(artifact.device_mod)
    (func,) = final_mod.functions.values()
    return func, target


def _emit(func, target):
    name = str(func.attrs["global_symbol"])
    module = tvm.IRModule({name: func})
    return tvm.get_global_func("target.build.tilelang_hexagon")(module, target).inspect_source()


def _scope(func, change):
    def post(node):
        if isinstance(node, I.AttrStmt) and node.attr_key == "tl.tile_program_scope":
            return I.AttrStmt(node.node, node.attr_key, node.value, change(node.body))

    return func.with_body(I.stmt_functor.ir_transform(func.body, None, post, ["tirx.AttrStmt"]))


def _replace(func, name, change):
    def post(node):
        if isinstance(node, I.Call) and getattr(node.op, "name", "") == name:
            return change(node)

    return func.with_body(I.stmt_functor.ir_transform(func.body, None, post, ["tirx.Call"]))


def _assert_policy(func, target, enabled):
    source = _emit(func, target)
    assert (COMPACT_PREFIX + "> tl_dma_async") in source if enabled else (PREFIX + "> tl_dma_async") in source
    if not enabled:
        assert "tl_hexagon_dma_compact_" not in source
    return source


@pytest.mark.parametrize("arch", ["v73", "v75", "v79", "v81"])
@pytest.mark.parametrize("direction", [0, 3])
def test_normal_load_policy_is_explicit_and_architecture_scoped(arch, direction):
    func, target = _normal(arch)
    func = _replace(func, "tl.hexagon_dma_submit", lambda c: I.Call(c.dtype, c.op, [*c.args[:-1], I.const(direction)]))
    _assert_policy(func, target, arch in {"v73", "v75", "v79"})


@pytest.mark.parametrize("direction", [1, 2, 4, -1, "dynamic"])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_store_or_dynamic_direction_keeps_generic_completion(direction, arch):
    func, target = _normal(arch)
    value = I.Var("direction", "int32") if direction == "dynamic" else I.const(direction)
    if direction == "dynamic":
        func = I.PrimFunc([*func.params, value], func.body, func.ret_type, func.buffer_map, func.attrs)
    func = _replace(func, "tl.hexagon_dma_submit", lambda c: I.Call(c.dtype, c.op, [*c.args[:-1], value]))
    _assert_policy(func, target, False)


@pytest.mark.parametrize("kind", ["extern", "pure", "raw", "outside"])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_opaque_calls_keep_the_whole_function_generic(kind, arch):
    func, target = _normal(arch)
    call = I.call_pure_extern("int32", "opaque") if kind == "pure" else I.call_extern("int32", "asm" if kind == "raw" else "opaque")
    if kind == "outside":
        func = func.with_body(I.SeqStmt([func.body, I.Evaluate(call)]))
    else:
        func = _scope(func, lambda body: I.SeqStmt([*body.seq, I.Evaluate(call)]))
    _assert_policy(func, target, False)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_hidden_store_and_shared_expression_use_are_not_overlooked(arch):
    func, target = _normal(arch)

    def hidden(body):
        submit = next(
            s.value
            for s in body.seq
            if isinstance(s, I.Evaluate) and isinstance(s.value, I.Call) and s.value.op.name == "tl.hexagon_dma_submit"
        )
        store = I.Call(submit.dtype, submit.op, [*submit.args[:-1], I.const(1)])
        return I.SeqStmt([*body.seq, I.IfThenElse(I.const(False), I.Evaluate(store), None)])

    _assert_policy(_scope(func, hidden), target, False)

    def use_as_value(body):
        # Deliberately reuse the SAME Call object in Evaluate and BufferStore.
        # A set of statement-node pointers alone would miss the second use.
        wait = next(
            s.value
            for s in body.seq
            if isinstance(s, I.Evaluate) and isinstance(s.value, I.Call) and s.value.op.name == "tl.hexagon_dma_wait"
        )
        output = I.decl_buffer((1,), "int32", data=func.params[1])
        return I.SeqStmt([*body.seq, I.BufferStore(output, wait, [0])])

    _assert_policy(_scope(func, use_as_value), target, False)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_nested_queues_keep_generic_policy(arch):
    func, target = _normal(arch)
    nested = _scope(func, lambda body: I.AttrStmt(I.const(0), "tl.tile_program_scope", 1, body))
    _assert_policy(nested, target, False)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("descriptors,groups", [(2, 1), (3, 3), (4, 2), (256, 256)])
def test_capacity_does_not_change_the_load_only_proof(arch, descriptors, groups):
    func, target = _normal(arch)
    func = _replace(func, "tl.hexagon_dma_queue", lambda c: I.Call(c.dtype, c.op, [I.const(descriptors), I.const(groups)]))
    source = _emit(func, target)
    assert f"tl_hexagon_dma_compact_load_context<{descriptors}, {groups}> tl_dma_async" in source


def make_load_only_pipeline(stages=3):
    @T.prim_func
    def copy(A: T.Tensor((16, 256), "uint8"), O: T.Tensor((16, 256), "uint8")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "uint8")
            for tile in T.Pipelined(16, num_stages=stages):
                T.copy(A[tile, 0:256], s)
                for i in T.Parallel(256):
                    O[tile, i] = s[i] + T.uint8(1)

    return copy


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("stages", [2, 3])
def test_normal_multibuffer_pipeline_keeps_opaque_and_arch_guards(arch, stages):
    from tilelang.hexagon.runtime.session import lower_kernels

    [(name, source, params, outputs)] = lower_kernels(
        [(make_load_only_pipeline(stages), [1])], target={"kind": "hexagon", "mcpu": f"hexagon{arch}"}
    )
    # The parity builtin in the two-stage pipeline is still outside the
    # existing call whitelist. This capacity extension does not broaden it.
    policy = "compact_load" if stages == 3 else "async"
    helper = "compact" if stages == 3 else "async"
    assert f"tl_hexagon_dma_{policy}_context<{stages}, {stages}>" in source
    assert f"tl_hexagon_dma_{helper}_wait(tl_dma_async, 0)" in source
    assert f"tl_hexagon_dma_{helper}_wait(tl_dma_async, {stages - 1})" in source


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_load_only_pipeline_wraps_and_preserves_resident_guards():
    import torch
    from tilelang.hexagon.runtime.session import Session, lower_kernels

    target = determine_target("hexagon", return_object=True)
    func = make_load_only_pipeline()
    [(name, source, params, outputs)] = lower_kernels([(func, [1])], target=target)
    arch = str(target.attrs["mcpu"])
    assert ("tl_hexagon_dma_compact_load_context<3, 3>" in source) == (arch in {"hexagonv73", "hexagonv75", "hexagonv79"})
    payload = _pipeline_payload()
    expected = (payload.to(torch.int32) + 1).to(torch.uint8)
    whole = torch.full((4096 + 256,), 0xA5, dtype=torch.uint8)
    whole[128:-128] = payload
    with Session([(func, [1])], target=target) as session:
        a, o = session.buffer(whole.numel()), session.buffer(whole.numel())
        a.write(whole)
        o.write(torch.full_like(whole, 0xA5))
        session.kernels[0](a.at(128, 4096), o.at(128, 4096))
        assert a.read() == whole.numpy().tobytes()
        actual = torch.frombuffer(bytearray(o.read()), dtype=torch.uint8)
        assert torch.equal(actual[128:-128], expected)
        assert torch.all(actual[:128] == 0xA5) and torch.all(actual[-128:] == 0xA5)


def _pipeline_payload():
    import torch

    index = torch.arange(4096, dtype=torch.int32)
    return (index * 73 + (index // 256) * 29 + 11).to(torch.uint8)


def test_pipeline_oracle_distinguishes_slot_generations():
    import torch

    # A 256-byte-periodic input would let the wrong ring slot look correct.
    tiles = _pipeline_payload().reshape(16, 256)
    assert len({row.numpy().tobytes() for row in tiles}) == 16
    for tile in range(3, 16):
        assert not (tiles[tile] == tiles[tile - 3]).all()
    expected = (tiles.to(torch.int32) + 1).to(torch.uint8)
    assert not torch.equal(expected.roll(1, dims=0), expected)
    stale = expected.clone()
    stale[3:] = expected[:-3]
    assert not torch.equal(stale, expected)


def test_consecutive_functions_do_not_inherit_a_rejected_policy():
    func, target = _normal()
    bad = _replace(func, "tl.hexagon_dma_submit", lambda c: I.Call(c.dtype, c.op, [*c.args[:-1], I.const(1)])).with_attr(
        "global_symbol", "a_bad"
    )
    good = func.with_attr("global_symbol", "z_good")
    source = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"a_bad": bad, "z_good": good}), target).inspect_source()
    assert source.count(COMPACT_PREFIX + "> tl_dma_async") == 1
    assert source.count(PREFIX + "> tl_dma_async") == 1


@pytest.mark.parametrize("location", ["inside", "outside", "sibling"])
def test_imported_c_disables_specialization_for_the_entire_module(location):
    func, target = _normal()

    # Imported macros are emitted before all functions, even when the Attr
    # occurs inside a scope or in the last sibling visited by codegen.
    def imported(body):
        return I.AttrStmt(I.const(0), "pragma_import_c", I.StringImm("#define dma_imported_marker 1"), body)

    if location == "inside":
        _assert_policy(_scope(func, imported), target, False)
    elif location == "outside":
        _assert_policy(func.with_body(imported(func.body)), target, False)
    else:
        first = func.with_attr("global_symbol", "a_first")
        last = func.with_body(imported(func.body)).with_attr("global_symbol", "z_import")
        source = tvm.get_global_func("target.build.tilelang_hexagon")(
            tvm.IRModule({"a_first": first, "z_import": last}), target
        ).inspect_source()
        assert source.count(PREFIX + "> tl_dma_async") == 2
        assert "tl_hexagon_dma_compact_" not in source
        assert "#define dma_imported_marker 1" in source


def test_parameter_named_like_the_private_queue_is_renamed_safely():
    @T.prim_func
    def collision(tl_dma_async: T.Tensor((256,), "uint8"), O: T.Tensor((256,), "uint8")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "uint8")
            T.dma_copy(tl_dma_async, s)
            T.dma_wait()
            T.copy(s, O)

    target = determine_target({"kind": "hexagon", "mcpu": "hexagonv79"}, return_object=True)
    with target:
        source = tilelang.lower(collision, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source
    assert COMPACT_PREFIX + "> tl_dma_async" in source
    assert "tl_dma_async_1" in source


@pytest.fixture(scope="module", params=["generic", "load-only", "compact"])
def dma_host(request, tmp_path_factory):
    from pathlib import Path
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    directory = tmp_path_factory.mktemp("dma-load-only")
    headers = directory / "tl_templates/hexagon"
    headers.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    source = (root / "src/tl_templates/hexagon/dma.h").read_text()
    release = 'asm volatile("release(%0):at" : : "r"(next) : "memory");'
    sync = 'asm volatile("syncht" : : : "memory");'
    assert source.count(release) == 2 and source.count(sync) == 3
    (headers / "dma.h").write_text(source.replace(release, "dma_release(next);").replace(sync, "dma_sync();"))
    for stub in (directory / "hexagon_protos.h", directory / "qurt_memory.h", headers / "common.h"):
        stub.write_text("")
    binary = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fno-pie",
            "-no-pie",
            "-D__HEXAGON_ARCH__=79",
            f"-DLOAD_ONLY={int(request.param != 'generic')}",
            f"-DCOMPACT={int(request.param == 'compact')}",
            "-I",
            str(directory),
            str(Path(__file__).with_name("dma_load_only_host.cc")),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=30,
    )
    return binary, request.param != "generic"


@pytest.mark.parametrize(
    "case",
    [
        "cached-load",
        "early-success",
        "busy-then-done",
        "poll-before-done",
        "error-drain",
        "capacity-rings",
        "uncommitted-drain",
        "zero-invalid",
        "old-completed",
    ],
)
def test_actual_header_preserves_load_order_and_cleanup(dma_host, case):
    import subprocess

    subprocess.run([str(dma_host[0]), case], check=True, timeout=10)


def test_actual_header_rejects_invalid_policy_without_mutation(dma_host):
    import subprocess

    binary, load_only = dma_host
    cases = ["reject-direction", "reject-pending"] if load_only else ["mixed-generic", "store-cache-error"]
    for case in cases:
        subprocess.run([str(binary), case], check=True, timeout=10)


def test_noncanonical_endpoint_keeps_generic_policy():
    func, target = _normal()
    # A direct handle expression bypasses the normal address-of BufferLoad.
    func = _replace(func, "tl.hexagon_dma_submit", lambda c: I.Call(c.dtype, c.op, [c.args[0], func.params[0], *c.args[2:]]))
    _assert_policy(func, target, False)


def test_adjacent_program_scopes_do_not_share_completion_policy():
    func, target = _normal()
    source = I.decl_buffer((256,), "uint8", data=func.params[0])
    scopes = []
    for n, direction in enumerate((3, 1, 3)):
        scratch = I.decl_buffer((256,), "uint8", name=f"scratch{n}", scope="shared.dyn")
        src = I.call_intrin("handle", "tirx.address_of", I.BufferLoad(source, [0]))
        dst = I.call_intrin("handle", "tirx.address_of", I.BufferLoad(scratch, [0]))
        endpoints = [dst, src] if direction == 3 else [src, dst]
        body = I.SeqStmt(
            [
                I.Evaluate(I.call_intrin("int32", tvm.ir.Op.get("tl.hexagon_dma_queue"), 1, 1)),
                I.AllocBuffer(scratch),
                I.Evaluate(I.call_intrin("int32", tvm.ir.Op.get("tl.hexagon_dma_submit"), *endpoints, 256, 256, 256, 1, direction)),
                I.Evaluate(I.call_intrin("int32", tvm.ir.Op.get("tl.hexagon_dma_commit"))),
                I.Evaluate(I.call_intrin("int32", tvm.ir.Op.get("tl.hexagon_dma_wait"), 0)),
            ]
        )
        scope = I.AttrStmt(I.const(0), "tl.tile_program_scope", 1, body)
        scopes.append(I.For(I.Var(f"i{n}", "int32"), 0, 1, I.ForKind.SERIAL, scope))
    generated = _emit(func.with_body(I.SeqStmt(scopes)), target)
    assert generated.count(COMPACT_PREFIX + "> tl_dma_async") == 2
    assert generated.count(PREFIX + "> tl_dma_async") == 1


def _session_copy(import_kind=None):
    @T.prim_func
    def copy(A: T.Tensor((256,), "uint8"), O: T.Tensor((256,), "uint8")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "uint8")
            T.dma_copy(A, s)
            T.dma_wait()
            T.copy(s, O)

    if import_kind == "attr":
        return copy.with_body(I.AttrStmt(I.const(0), "pragma_import_c", I.StringImm("#define imported_marker 1"), copy.body))
    if import_kind == "block":

        @T.prim_func
        def imported(A: T.Tensor((256,), "uint8"), O: T.Tensor((256,), "uint8")):
            with T.Kernel(1):
                T.import_source("#define imported_marker 1")
                s = T.alloc_shared((256,), "uint8")
                T.dma_copy(A, s)
                T.dma_wait()
                T.copy(s, O)

        return imported
    return copy


@pytest.mark.parametrize("import_kind", ["attr", "block"])
@pytest.mark.parametrize("reverse", [False, True])
def test_normal_session_import_disables_all_kernel_policies(monkeypatch, import_kind, reverse):
    from tilelang.hexagon.runtime import session

    seen = []

    def stop_before_connection(self, lowered, *args):
        seen.append(lowered)

    monkeypatch.setattr(session.Session, "_open", stop_before_connection)
    normal, imported = _session_copy(), _session_copy(import_kind)
    kernels = [(normal, [1]), (imported, [1])]
    if reverse:
        kernels.reverse()
    target = {"kind": "hexagon", "mcpu": "hexagonv79"}
    session.Session(kernels, target=target)
    assert len(seen) == 1
    for _, source, _, _ in seen[0]:
        assert PREFIX + "> tl_dma_async" in source
        assert "tl_hexagon_dma_compact_" not in source
    # The composition opt-out must not mutate the caller's reusable PrimFunc.
    assert not normal.attrs.get("tl.hexagon_disable_dma_load_only", False)
    session.Session([(normal, [1])], target=target)
    assert COMPACT_PREFIX + "> tl_dma_async" in seen[1][0][1]


def test_session_import_policy_has_a_distinct_lowering_cache_key(tmp_path, monkeypatch):
    from tilelang.env import env
    from tilelang.hexagon.runtime import session

    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)
    calls = []
    lower = tilelang.lower
    monkeypatch.setattr(tilelang, "lower", lambda *a, **k: calls.append(1) or lower(*a, **k))
    enabled = env.is_cache_enabled()
    env.enable_cache()
    try:
        normal, imported = _session_copy(), _session_copy("block")
        target = {"kind": "hexagon", "mcpu": "hexagonv79"}
        (first,) = session.lower_kernels([(normal, [1])], target)
        assert COMPACT_PREFIX + "> tl_dma_async" in first[1]
        combined = session.lower_kernels([(normal, [1]), (imported, [1])], target)
        assert all(PREFIX + "> tl_dma_async" in item[1] for item in combined)
        assert len(calls) == 3
        session.lower_kernels([(imported, [1]), (normal, [1])], target)
        (again,) = session.lower_kernels([(normal, [1])], target)
        assert len(calls) == 3 and again[1] == first[1]
        assert len(list((tmp_path / "hexagon_lowered").glob("*.json"))) == 3
    finally:
        if not enabled:
            env.disable_cache()
