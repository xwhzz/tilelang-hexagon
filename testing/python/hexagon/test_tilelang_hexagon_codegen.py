"""Hexagon codegen on CodeGenTileLangCBase: source assembly and line directives."""

import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

N = 128


def _source(func, **config) -> str:
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)
    return str(artifact.kernel_source)


def _copy_kernel(prelude=None, imported=None, dma=False):
    @T.prim_func
    def kernel(A: T.Tensor((N,), "float16"), B: T.Tensor((N,), "float16")):
        with T.Kernel(1, threads=1, prelude=prelude):
            if imported is not None:
                T.import_source(imported)
            A_s = T.alloc_shared((N,), "float16")
            if dma:
                T.dma_copy(A, A_s)
                T.dma_wait()
            else:
                T.copy(A, A_s)
            T.copy(A_s, B)  # line_marker_copy_out

    return kernel


@pytest.mark.parametrize("imports", [{"prelude": "// imported source"}, {"imported": "// imported source"}])
def test_headers_then_imports_then_kernel(imports):
    source = _source(_copy_kernel(dma=True, **imports))
    order = [
        source.index("#include <tl_templates/hexagon/common.h>"),
        source.index("#include <tl_templates/hexagon/dma.h>"),
        source.index("// imported source"),
        source.index("kernel_kernel("),
    ]
    assert order == sorted(order), source


def test_dma_header_only_when_used():
    assert "tl_templates/hexagon/dma.h" not in _source(_copy_kernel())


def test_line_directives_follow_the_pass_config():
    assert "#line" not in _source(_copy_kernel())
    source = _source(_copy_kernel(), **{tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES: True})
    files = set(re.findall(r'^#line \d+ "(.*)"$', source, re.M))
    assert __file__ in files, source
