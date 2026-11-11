"""Region traversal keeps the iteration domain and every store on CPU too."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang.layout import Layout


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("guarded", [False, True])
def test_region_physical_order_preserves_other_stores_and_holes(step, guarded):
    @T.prim_func
    def kernel(X: T.Tensor((16, 32), "int32"), Y: T.Tensor((16, 32), "int32"), Z: T.Tensor((16, 32), "int32")):
        with T.Kernel(1):
            tiles = T.alloc_shared((16, 32), "int32")
            T.annotate_layout({tiles: Layout((16, 32), lambda i, j: (j, i))})
            T.copy(X, tiles)
            T.copy(X, Z)
            for i, j in T.Parallel(4, 12):
                if not guarded or (i + j) % 3 != 0:
                    tiles[4 + i, 8 + step * j] = X[4 + i, 8 + step * j] * 2 + i
                    Z[4 + i, 8 + step * j] = X[4 + i, 8 + step * j] - j
            T.copy(tiles, Y)

    compiled = tilelang.compile(kernel, target="c", out_idx=[1, 2])
    x = torch.arange(16 * 32, dtype=torch.int32).reshape(16, 32)
    expected_y, expected_z = x.clone(), x.clone()
    for i in range(4):
        for j in range(12):
            if not guarded or (i + j) % 3 != 0:
                expected_y[4 + i, 8 + step * j] = x[4 + i, 8 + step * j] * 2 + i
                expected_z[4 + i, 8 + step * j] = x[4 + i, 8 + step * j] - j
    y, z = compiled(x)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)


@pytest.mark.parametrize("column_tiles", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("start,guarded", [(4, False), (30, False), (3, True)])
def test_crouton_coordinate_regions_preserve_values_on_cpu(column_tiles, transposed, start, guarded):
    shape = (96, 64) if transposed else (64, 96)

    def position(i, j):
        row, col = (j, i) if transposed else (i, j)
        tile = col // 32 * 2 + row // 32 if column_tiles else row // 32 * 3 + col // 32
        return tile, row % 32 // 2, col % 32 * 2 + row % 2

    @T.prim_func
    def kernel(X: T.Tensor(shape, "int32"), Y: T.Tensor(shape, "int32")):
        with T.Kernel(1):
            tiles = T.alloc_shared(shape, "int32")
            T.annotate_layout({tiles: Layout(shape, position)})
            T.copy(X, tiles)
            for batch in T.serial(2):
                for i, k in T.Parallel(4, 96):
                    row = start + batch * 8 + i
                    if not guarded or (i + k) % 3 != 0:
                        if transposed:
                            tiles[k, row] = X[k, row] * 2 + i - k
                        else:
                            tiles[row, k] = X[row, k] * 2 + i - k
            T.copy(tiles, Y)

    compiled = tilelang.compile(kernel, target="c", out_idx=[1])
    x = torch.arange(64 * 96, dtype=torch.int32).reshape(shape)
    expected = x.clone()
    for batch in range(2):
        for i in range(4):
            for k in range(96):
                if not guarded or (i + k) % 3 != 0:
                    index = (k, start + batch * 8 + i) if transposed else (start + batch * 8 + i, k)
                    expected[index] = x[index] * 2 + i - k
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)
