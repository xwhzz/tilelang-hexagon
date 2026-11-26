"""Coordinate bindings in IR graphs containing Hexagon and ordinary layouts."""

import json

import pytest

from tilelang import tvm
from tilelang.hexagon.intrinsics.crouton import CroutonLayout
from tilelang.hexagon.intrinsics.panel import PanelLayout
from tilelang.layout import Fragment, Layout


@pytest.mark.parametrize("packed", [CroutonLayout([2, 64, 96], True, True), PanelLayout([2, 64, 96], 8)])
def test_packed_layout_restores_from_parameters_in_mixed_graph(packed):
    captured = tvm.tirx.Var("_i", "int32")
    layout = Layout([8], lambda i: i + captured)
    fragment = Fragment([8], forward_fn=lambda i, rep: (i % 4 + rep * 4, i // 4 + rep + captured), replicate=2)
    point = [1, 37, 63]
    expected = packed.map_forward_index(point)
    graph = json.loads(tvm.ir.save_json([packed, packed.get_forward_vars(), layout, fragment, captured]))
    packed_id, axes_id, *_ = graph["nodes"][graph["root_index"]]["data"]
    fields = graph["nodes"][packed_id]["data"]
    assert graph["nodes"][fields["placeholder_vars"]]["data"] == []
    # Older records carry bindings. Even with a stale cached forward map, the
    # shape, tile order and lane count determine the specialized layout.
    fields["placeholder_vars"] = axes_id
    fields["forward_index"] = fields["input_size"]
    restored, _, ordinary, frag, outer = tvm.ir.load_json(json.dumps(graph))
    tvm.ir.assert_structural_equal(restored.map_forward_index(point), expected)
    assert len(restored.placeholder_vars) == 0
    tvm.ir.assert_structural_equal(ordinary.map_forward_index([3]), [outer + 3])
    tvm.ir.assert_structural_equal(frag.map_forward_index([1, 6]), [outer + 2])
    assert int(frag.map_forward_thread([1, 6])[0]) == 6
