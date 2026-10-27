"""The Hexagon dialect composes the common surface with Hexagon extensions."""

import pytest

import tilelang.language as T_default
import tilelang.language.common as T_comm
from tilelang.backend import create_backend_context, get_backend
from tilelang.hexagon import language as T

HEXAGON_ONLY_NAMES = {"dma_copy", "dma_wait", "hmx"}

HMX_ATOMS = {
    "alloc_state",
    "alloc_accumulator",
    "alloc_convert_state",
    "alloc_bias_state",
    "acquire",
    "release",
    "load_bias",
    "clear",
    "mma",
    "convert",
    "store",
}


def test_hexagon_language_composes_common_and_hexagon_symbols():
    assert T.__tilelang_dialect__ == "hexagon"
    assert set(T_comm.__all__) <= set(T.__all__)
    assert set(T.__all__) - set(T_comm.__all__) == HEXAGON_ONLY_NAMES
    # T.Kernel is shadowed by the Hexagon launch (num_workers=).
    assert T.Kernel is not T_comm.Kernel
    assert set(T.hmx.__all__) - {"StateKind"} == HMX_ATOMS


def test_hexagon_symbols_stay_out_of_other_dialects():
    assert HEXAGON_ONLY_NAMES.isdisjoint(T_comm.__all__)
    assert HEXAGON_ONLY_NAMES.isdisjoint(T_default.__all__)
    # The definitions live in the dialect, not in the shared language modules.
    for name in ("dma_copy", "dma_wait", "alloc_hmx_accumulator", "hexagon_hmx_mma"):
        assert not hasattr(T_default, name)
        assert not hasattr(T_comm, name)


@pytest.mark.parametrize("num_workers", [0, -1, True, 2.0])
def test_hexagon_kernel_rejects_invalid_num_workers(num_workers):
    with pytest.raises(ValueError, match="num_workers"):
        T.Kernel(4, num_workers=num_workers)


def test_hexagon_backend_context():
    context = create_backend_context("hexagon", "c", "auto")

    assert context.module is get_backend("hexagon")
    assert context.execution_backend.name == "hexagon"
    assert not context.execution_backend.enable_host_codegen
    assert not context.execution_backend.enable_device_compile
