import pytest

import tilelang
from tilelang.hexagon.qgemv import Q8_0_TILED_LAYOUT, make_q8_0_dot_prim_func


def test_q8_0_tiled_layout_contract():
    layout = Q8_0_TILED_LAYOUT
    assert layout.source_tile_bytes == 1088
    assert layout.staged_tile_bytes == 1152
    assert layout.staged_weight_bytes(2048) == 64 * 1152
    assert layout.staged_activation_bytes(8192) == 256 * 1152
    with pytest.raises(ValueError, match="K % 32"):
        layout.staged_weight_bytes(2000)


def test_q8_0_dot_atom_hexagon_codegen():
    kernel = tilelang.compile(
        make_q8_0_dot_prim_func(2048, symbol="test_qgemv_q8_0"),
        out_idx=[3],
        target="hexagon",
    )
    source = kernel.get_kernel_source()
    assert "test_qgemv_q8_0_kernel" in source
    assert "tl_hexagon_q8_0_dot_32x1(2048" in source
    assert "tl_hexagon_hmx" not in source

    nobias = tilelang.compile(
        make_q8_0_dot_prim_func(
            2048, symbol="test_qgemv_q8_0_nobias", with_bias=False
        ),
        out_idx=[2],
        target="hexagon",
    ).get_kernel_source()
    assert "tl_hexagon_q8_0_dot_32x1_nobias(2048" in nobias
    assert "float* bias" not in nobias
