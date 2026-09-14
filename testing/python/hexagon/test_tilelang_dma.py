import re
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.hexagon import _fastrpc
from tilelang.utils.target import determine_target


DMA_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "src/tl_templates/hexagon/dma.h"
)
def _lower_source(prim_func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target):
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return str(artifact.kernel_source)


def _make_worker_dma_probe():
    @T.prim_func
    def worker_dma_probe(
        A: T.Tensor((2, 2048), "float16"),
        B: T.Tensor((2, 2048), "float16"),
    ):
        with T.Kernel(2, threads=1, num_workers=2) as bx:
            T.call_extern("uint32", "tl_hexagon_dma_poll")

    return worker_dma_probe


def _make_dma_raw_poll_probe():
    @T.prim_func
    def dma_raw_poll_probe():
        with T.Kernel(1, threads=1) as _:
            T.call_extern("uint32", "tl_hexagon_dma_poll")

    return dma_raw_poll_probe


def test_hexagon_dma_rejects_worker_pool_until_channels_are_worker_local():
    with pytest.raises(Exception, match="DMA calls inside a"):
        _lower_source(_make_worker_dma_probe())


def test_hexagon_dma_raw_value_call_is_not_treated_as_status_code():
    source = _lower_source(_make_dma_raw_poll_probe())
    call_line = next(
        line.strip() for line in source.splitlines() if "tl_hexagon_dma_poll" in line
    )

    assert call_line == "tl_hexagon_dma_poll();"


def test_hexagon_dma_fastrpc_wrapper_includes_template():
    source = _lower_source(_make_dma_raw_poll_probe())
    wrapper = _fastrpc.gen_dsp("tl_dma_probe", "dma_raw_poll_probe_kernel", source, [])

    assert "#include <tl_templates/hexagon/dma.h>" in wrapper


def test_hexagon_dma_template_exposes_reusable_primitive_layers():
    source = DMA_TEMPLATE.read_text()

    instruction_primitives = [
        "tl_hexagon_dma_pause",
        "tl_hexagon_dma_resume",
        "tl_hexagon_dma_start",
        "tl_hexagon_dma_link",
        "tl_hexagon_dma_poll",
        "tl_hexagon_dma_wait",
        "tl_hexagon_dma_sync_thread",
        "tl_hexagon_dma_tlb_sync",
        "tl_hexagon_dma_config_read",
        "tl_hexagon_dma_config_write",
    ]
    descriptor_primitives = [
        "tl_hexagon_dma_descriptor_init_1d",
        "tl_hexagon_dma_descriptor_init_2d",
        "tl_hexagon_dma_descriptor_done",
        "tl_hexagon_dma_submit_one",
    ]
    queue_primitives = [
        "tl_hexagon_dma_queue_init",
        "tl_hexagon_dma_queue_push_1d",
        "tl_hexagon_dma_queue_push_2d",
        "tl_hexagon_dma_queue_submit_prepared",
        "tl_hexagon_dma_queue_in_flight",
        "tl_hexagon_dma_queue_wait",
        "tl_hexagon_dma_queue_try_pop",
        "tl_hexagon_dma_queue_pop",
        "tl_hexagon_dma_queue_flush",
    ]

    for primitive in (
        instruction_primitives + descriptor_primitives + queue_primitives
    ):
        assert re.search(rf"\b{primitive}\s*\(", source), primitive
    assert "sizeof(tl_hexagon_dma_descriptor_1d) == 16" in source
    assert "sizeof(tl_hexagon_dma_descriptor_2d) == 32" in source
    assert "sizeof(tl_hexagon_dma_descriptor) == 64" in source
    assert "malloc(" not in source
    assert "free(" not in source
