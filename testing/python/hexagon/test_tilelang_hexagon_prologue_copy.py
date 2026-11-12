"""Normal Hexagon lowering of the optional synchronous prologue policy."""

import re
from unittest.mock import patch

import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T

from test_tilelang_hexagon_pipeline import _source

X = tvm.tirx
POLICY = "hexagon.prologue_copy_policy"
CANDIDATE = "tl.hexagon.prologue_copy_candidate"


def _kernel(policy=None, extent=3, minimum=0, mixed=False, contract=None):
    annotations = {} if policy is None else {POLICY: policy}
    second = {"hexagon_dma_src_l2_bypass": 0} if mixed else {}

    @T.prim_func
    def kernel(A: T.Tensor((8, 256), "int32"), B: T.Tensor((8, 256), "int32"), C: T.Tensor((8, 256), "int32")):
        with T.Kernel(1):
            if contract == "dma_only":
                T.annotate_dma_only(A)
            if contract == "write_through":
                T.annotate_write_through(A)
                for j in T.Parallel(256):
                    A[0, j] = A[0, j] + 1
            sa = T.alloc_shared((256,), "int32")
            sb = T.alloc_shared((256,), "int32")
            for i in T.Pipelined(minimum, minimum + extent, num_stages=1, annotations=annotations):
                T.copy(A[i, 0:256], sa)
                T.copy(B[i, 0:256], sb, annotations=second)
                for j in T.Parallel(256):
                    C[i, j] = sa[j] + sb[j]

    return kernel


def _lower(func, arch="v79"):
    observed = {}
    original = tilelang.transform.InjectSoftwarePipeline

    def recorder():
        actual = original()

        def invoke(mod):
            observed["before"] = mod
            result = actual(mod)
            observed["after"] = result
            return result

        return invoke

    with patch.object(tilelang.transform, "InjectSoftwarePipeline", recorder):
        source = _source(func, {"kind": "hexagon", "mcpu": "hexagon" + arch})
    return source, observed


def _calls(mod, name):
    found = []
    for func in mod.functions.values():
        X.stmt_functor.post_order_visit(
            func.body, lambda node: found.append(node) if isinstance(node, X.Call) and node.op.name == name else None
        )
    return found


def _schedule(source):
    return re.findall(r"tl_hexagon_dma_(?:async|compact)_(commit|wait)\(tl_dma_async(?:, (\d+))?\)", source)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("extent", [0, 1, 2, 3])
@pytest.mark.parametrize("minimum", [0, 3])
def test_normal_default_and_preserve_source_are_exact(arch, extent, minimum):
    default, _ = _lower(_kernel(extent=extent, minimum=minimum), arch)
    preserve, _ = _lower(_kernel("preserve", extent, minimum), arch)
    assert default == preserve


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("extent", [1, 2, 3])
def test_normal_sync_changes_only_supported_prologue_payloads(arch, mixed, extent):
    default, _ = _lower(_kernel(extent=extent, minimum=3, mixed=mixed), arch)
    actual, observed = _lower(_kernel("sync_if_supported", extent, 3, mixed), arch)
    candidates = [c for c in _calls(observed["after"], "tl.tileop.copy") if CANDIDATE in c.annotations]
    assert len(candidates) == 2
    assert all(int(c.annotations["no_implicit_async_commit_wait"]) == 1 for c in candidates)
    submit = r"tl_hexagon_dma_(?:async|compact)_submit\("
    assert len(re.findall(submit, actual)) == len(re.findall(submit, default)) - (1 if mixed else 2)
    assert _schedule(actual) == _schedule(default)
    assert "*(int32_t32*)" in actual


def test_dma_only_annotation_survives_until_selector_and_preserves_both_copies():
    default, _ = _lower(_kernel(contract="dma_only"))
    actual, observed = _lower(_kernel("sync_if_supported", contract="dma_only"))
    declarations = []
    for func in observed["before"].functions.values():
        X.stmt_functor.post_order_visit(
            func.body,
            lambda node: declarations.append(node) if isinstance(node, X.SBlock) and "tl.hexagon_dma_only" in node.annotations else None,
        )
    assert declarations
    assert not any(CANDIDATE in call.annotations for call in _calls(observed["after"], "tl.tileop.copy"))
    assert actual == default


def test_write_through_keeps_writeback_before_synchronous_read():
    default, _ = _lower(_kernel(contract="write_through"))
    actual, _ = _lower(_kernel("sync_if_supported", contract="write_through"))

    def writebacks(text):
        return [line.strip() for line in text.splitlines() if "tl_hexagon_cache_writeback(" in line]

    assert writebacks(actual) == writebacks(default) and writebacks(actual)
    assert actual.index("tl_hexagon_cache_writeback(") < actual.index("*(int32_t32*)(((int32_t*)sa)")
    assert _schedule(actual) == _schedule(default)
