"""Hexagon backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline
from .transform.hmx_requant import normalize_local_hmx_requant

BACKEND = register_backend(
    BackendModule(
        name="hexagon",
        target_kinds=("hexagon",),
        pipelines={"hexagon": PassPipeline("hexagon", pipeline.HexagonPassPipelineBody)},
        device_codegens={
            "hexagon": DeviceCodegen(
                "hexagon",
                build=codegen.build_hexagon,
                build_without_compile=codegen.build_hexagon,
                direct_call=True,
            )
        },
        host_codegens=STANDARD_HOST_CODEGENS,
        normalize_before_validation=normalize_local_hmx_requant,
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
