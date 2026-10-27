import pytest

import tilelang
from examples.hexagon.quantize.q8_0 import Q8_0_TILED_LAYOUT, make_q8_0_dot_prim_func
from tilelang import tvm
from tilelang.backend.target import determine_target


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def test_q8_0_tiled_layout_contract():
    layout = Q8_0_TILED_LAYOUT
    assert layout.source_tile_bytes == 1088
    assert layout.staged_tile_bytes == 1152
    assert layout.staged_weight_bytes(2048) == 64 * 1152
    assert layout.staged_activation_bytes(8192) == 256 * 1152
    with pytest.raises(ValueError, match="K % 32"):
        layout.staged_weight_bytes(2000)


def test_q8_0_dot_atom_hexagon_codegen():
    source = _source(make_q8_0_dot_prim_func(2048, symbol="test_qgemv_q8_0"))
    assert "test_qgemv_q8_0_kernel" in source
    assert "tl_hexagon_q8_0_dot_32x1(2048" in source
    assert "tl_hexagon_hmx" not in source

    nobias = _source(make_q8_0_dot_prim_func(2048, symbol="test_qgemv_q8_0_nobias", with_bias=False))
    assert "tl_hexagon_q8_0_dot_32x1_nobias(2048" in nobias
    assert "float* bias" not in nobias
