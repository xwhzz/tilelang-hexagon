import json
from pathlib import Path
import re
import tempfile

import tilelang
from examples.hexagon.example_qmatmul_kstream import (
    make,
    make_staged_dequant,
    make_staged_hmx,
    make_staged_pack,
)
from examples.hexagon.llama_cpp_integration.emit_embeddable import emit
from examples.hexagon.llama_cpp_integration.emit_staged_qmatmul import (
    emit as emit_staged,
)
from tilelang.hexagon.hmx_intrin import (
    make_hmx_activation_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)
from tilelang.utils.target import determine_target


def _offset(layout, *indices):
    return int(layout.map_forward_index(list(indices))[0])


def _lower_source(prim_func):
    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return str(artifact.kernel_source)


def test_hmx_crouton_layout_contract():
    activation = make_hmx_activation_layout((64, 64))
    assert _offset(activation, 0, 0) == 0
    assert _offset(activation, 1, 0) == 1
    assert _offset(activation, 0, 1) == 2
    assert _offset(activation, 0, 32) == 1024
    assert _offset(activation, 32, 0) == 2048

    weight = make_hmx_weight_layout((64, 64))
    assert _offset(weight, 0, 0) == 0
    assert _offset(weight, 1, 0) == 1
    assert _offset(weight, 0, 1) == 2
    assert _offset(weight, 32, 0) == 1024
    assert _offset(weight, 0, 32) == 2048
    for kpair in range(16):
        for n in range(32):
            assert _offset(weight, 2 * kpair, n) == kpair * 64 + 2 * n
            assert _offset(weight, 2 * kpair + 1, n) == kpair * 64 + 2 * n + 1

    output = make_hmx_output_layout((64, 64))
    assert _offset(output, 0, 32) == 1024
    assert _offset(output, 32, 0) == 2048

    staged = make_hmx_weight_layout((2, 32, 32))
    assert _offset(staged, 1, 0, 0) == 1024

    try:
        make_hmx_weight_layout((32, 48))
    except ValueError as err:
        assert "multiples of 32" in str(err)
    else:
        raise AssertionError("invalid HMX weight shape was accepted")


def test_hmx_atom_q4_codegen():
    source = _lower_source(make(32, 64, 64))

    expected_calls = (
        "tl_hexagon_hmx_acc_acquire",
        "tl_hexagon_hmx_load_bias",
        "tl_hexagon_hmx_clear_acc",
        "tl_hexagon_hmx_mma_atom",
        "tl_hexagon_hmx_convert_acc",
        "tl_hexagon_hmx_store_cvt_state",
        "tl_hexagon_hmx_acc_release",
    )
    for call in expected_calls:
        assert call in source

    assert "tl_hexagon_q4_0_dequant_tile_32x32" in source
    assert "tl_hexagon_hmx_gemm" not in source
    assert "tl_hexagon_hmx_pack_a(" not in source
    assert "tl_hexagon_hmx_pack_b(" not in source
    assert "tl_hexagon_hmx_open" not in source
    assert "tl_hexagon_hmx_close" not in source
    assert "tl_hexagon_hmx_dequant_q4_0" not in source
    assert "tl.hexagon_q4_0_dequant" not in source
    assert "tl_hexagon_q4_0_prepare_scale_32" not in source
    assert "tl_hexagon_q4_0_dequant_group_128" not in source
    for old_local in ("q4_scales", "q4_low", "q4_high", "q4_values"):
        assert old_local not in source
    assert re.search(r"\(kt(?:_\d+)? \* 576\)", source)
    assert re.search(r"\(kt(?:_\d+)? \* 1024\)", source)

    offsets = {}
    for name in ("A_hmx", "B_hmx", "C_hmx"):
        match = re.search(
            rf"void\* {name} = .*buf_dyn_shmem \+ (\d+)\)\);", source
        )
        assert match is not None
        offsets[name] = int(match.group(1))
        assert offsets[name] % 2048 == 0
    assert offsets["A_hmx"] + 32 * 64 * 2 <= offsets["B_hmx"]
    assert offsets["B_hmx"] + 2 * 32 * 32 * 2 <= offsets["C_hmx"]


def test_hmx_atom_q4_llama_cpp_abi():
    source = _lower_source(
        make(32, 64, 64, input_dtype="float32", output_dtype="float32")
    )

    assert re.search(
        r"int32_t qmatmul_hmx_atoms_kernel\(float\* A, uint8_t\* W, float\* C\)",
        source,
    )
    assert "tl_hexagon_hmx_pack_a_f32_pair_k32" in source
    assert "tl_hexagon_q4_0_dequant_tile_32x32" in source
    assert "tl_hexagon_hmx_unpack_c_f32_pair_n32" in source
    for old_local in ("q4_scales", "q4_low", "q4_high", "q4_values"):
        assert old_local not in source
    assert "float64" not in source
    assert "tl_hexagon_hmx_convert_acc" in source
    assert "tl_hexagon_hmx_store_cvt_state" in source
    assert "tl_hexagon_hmx_mma_atom" in source
    assert "tl_hexagon_hmx_gemm" not in source


def test_hmx_atom_q4_embeddable_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        source_path, manifest_path = emit(32, 128, 64, Path(tmp))
        source = source_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["implementation"] == "explicit_hmx_atoms"
    assert manifest["dequantization"] == "tilelang_scheduled_hvx_q4_tile_atoms"
    assert manifest["weight_format"]["shape"] == "[N/32][K/32][576] uint8"
    assert manifest["weight_bytes"] == 4 * 2 * 576
    assert manifest["runtime_contract"]["hmx_session"] == "caller-owned"
    assert manifest["runtime_contract"]["dma"].startswith("caller-owned")
    assert manifest["entry"] in source
    assert manifest["kernel_entry"] in source
    assert "tl_bridge_enter(vtcm_base, vtcm_size)" in source
    assert "tl_hexagon_hmx_mma_atom" in source
    assert "tl_hexagon_q4_0_dequant_tile_32x32" in source
    assert "tl_hexagon_hmx_dequant_q4_0" not in source
    assert "q4_scales" not in source
    assert "q4_values" not in source
    for forbidden in (
        "tl_hexagon_hmx_gemm",
        "tl_hexagon_hmx_pack_a(",
        "tl_hexagon_hmx_pack_b(",
    ):
        assert forbidden not in source


def test_hmx_atom_q4_checked_in_dispatch_contract():
    repo_root = Path(__file__).resolve().parents[3]
    integration = repo_root / "examples/hexagon/llama_cpp_integration"
    bridge = (integration / "tl_ggml_matmul.cc").read_text(encoding="utf-8")
    patch = (integration / "ggml-hexagon.patch").read_text(encoding="utf-8")
    assert "o->dma.push_1d" in bridge
    assert "o->dma.push_2d" in bridge
    assert "o->parallel.run" in bridge
    assert "tl_q4_dequant_worker" in bridge
    assert "640u" in bridge
    assert "weight_stage[2]" in bridge
    assert "for (int m0 = 0; m0 < o->m" in bridge
    assert "activation + (size_t)m0 * o->act_stride" in bridge
    assert "o->m == 32 && staged" in bridge
    assert "(o->m % 32) == 0" in bridge
    assert "GGML_HEXAGON_TILELANG_SOURCE_DIR" in patch
    assert "GGML_HEXAGON_TILELANG_Q4_HMX" in patch
    assert "-fno-lto" in patch
    assert "/home/xwh/tilelang-hexagon" not in patch

    for n in (128, 512):
        for k in (2048, 8192):
            stem = f"kernel_qmatmul_hmx_atoms_32x{n}x{k}"
            source = (integration / f"{stem}.cc").read_text(encoding="utf-8")
            manifest = json.loads(
                (integration / f"{stem}.manifest.json").read_text(encoding="utf-8")
            )
            kernel_symbol = f"tl_q4_hmx_m32_n{n}_k{k}_kernel"
            embedded_symbol = f"tl_q4_hmx_m32_n{n}_k{k}_embedded"
            assert manifest["kernel_entry"].startswith(f"int32_t {kernel_symbol}(")
            assert manifest["entry"].startswith(f"int32_t {embedded_symbol}(")
            shape = f"{{{k}, {n}, {manifest['vtcm_bytes']}u, {embedded_symbol}}}"
            assert shape in bridge
            assert f'#include "{stem}.cc"' not in bridge
            assert "tl_bridge_enter(vtcm_base, vtcm_size)" in source
            assert "tl_hexagon_hmx_mma_atom" in source
            assert "tl_hexagon_q4_0_dequant_tile_32x32" in source
            assert "tl_hexagon_hmx_dequant_q4_0" not in source

    for k in (2048, 8192):
        stem = f"kernel_qmatmul_hmx_staged_32x256x{k}"
        source = (integration / f"{stem}.cc").read_text(encoding="utf-8")
        manifest = json.loads(
            (integration / f"{stem}.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["dequantization"] == "parallel_range_of_hvx_q4_tile_atoms"
        assert manifest["padded_weight_format"]["dma"] == "2D 576B -> 640B rows"
        for entry in manifest["entries"].values():
            assert f"int32_t {entry}(" in source
            assert entry in bridge
        assert "int32_t tile_begin, int32_t tile_end" in source
        assert "* (int64_t)640" in source
        assert "tl_hexagon_q4_0_dequant_tile_32x32" in source
        assert "tl_hexagon_hmx_mma_atom" in source


def test_hmx_staged_q4_codegen_contract():
    pack = _lower_source(make_staged_pack(32, 64))
    dequant = _lower_source(make_staged_dequant(256, 64))
    compute = _lower_source(make_staged_hmx(32, 256, 64))

    assert "tl_hexagon_hmx_pack_a_f32_pair_k32" in pack
    assert "uint32_t* bias_vtcm" in pack
    assert "int32_t tile_begin, int32_t tile_end" in dequant
    assert "tl_hexagon_q4_0_dequant_tile_32x32" in dequant
    assert "* (int64_t)640" in dequant
    assert "int32_t dst_stride" in compute
    assert "tl_hexagon_hmx_mma_atom" in compute
    assert "tl_hexagon_hmx_unpack_c_f32_pair_n32" in compute

    with tempfile.TemporaryDirectory() as tmp:
        source_path, manifest_path = emit_staged(2048, Path(tmp))
        source = source_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["vtcm"]["dequant_bytes"] == 256 * 2048 * 2
    assert manifest["vtcm"]["raw_weight_stages"] == 2
    assert all(entry in source for entry in manifest["entries"].values())
