import json
from pathlib import Path
import re
import tempfile

import tilelang
import tilelang.language as T
from tilelang import tvm
from examples.hexagon.example_flash_attention import make_flash
from examples.hexagon.example_qmatmul import make_qmatmul
from examples.hexagon.example_qmatmul_kstream import (
    make,
    make_staged_dequant,
    make_staged_hmx,
    make_staged_pack,
)
from examples.hexagon.example_worker_pool import make_batched_matmul
from examples.hexagon.llama_cpp_integration.emit_embeddable import emit
from examples.hexagon.llama_cpp_integration.emit_staged_qmatmul import (
    emit as emit_staged,
)
from tilelang.hexagon import _fastrpc
from tilelang.hexagon.hmx_intrin import (
    HMXIntrinEmitter,
    make_hmx_activation_layout,
    make_hmx_crouton_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)
from tilelang.utils.target import determine_target


def _offset(layout, *indices):
    mapped = [int(value) for value in layout.map_forward_index(list(indices))]
    output_shape = [int(value) for value in layout.get_output_shape()]
    offset = 0
    for index, extent in zip(mapped, output_shape):
        offset = offset * extent + index
    return offset


def _output_shape(layout):
    return tuple(int(value) for value in layout.get_output_shape())


def _lower_source(prim_func, pass_configs=None):
    target = determine_target("hexagon", return_object=True)
    with tvm.transform.PassContext(config=pass_configs or {}), target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return str(artifact.kernel_source)


def _make_hmx_coordinate_probe():
    emitter = HMXIntrinEmitter(64, 64, 64)

    @T.prim_func
    def hmx_coordinate_probe(
        A_hmx: T.Tensor((64, 64), "float16"),
        B_hmx: T.Tensor((64, 64), "float16"),
        C_hmx: T.Tensor((64, 64), "float16"),
        bias_vtcm: T.Tensor((64,), "uint32"),
    ):
        with T.Kernel(1, threads=1) as _:
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()
            T.annotate_layout(
                {
                    A_hmx: emitter.activation_layout(A_hmx),
                    B_hmx: emitter.weight_layout(B_hmx),
                    C_hmx: emitter.output_layout(C_hmx),
                }
            )
            emitter.acquire(acc)
            emitter.clear(acc)
            emitter.load_bias(bias, bias_vtcm)
            emitter.mma_atom(acc, A_hmx, B_hmx, 1, 1, 1)
            emitter.convert(cvt, acc, bias, bias_vtcm)
            emitter.store(
                cvt,
                acc,
                C_hmx,
                bias,
                bias_vtcm,
                1,
                1,
            )
            emitter.release(acc)

    return hmx_coordinate_probe


def _make_hmx_vtcm_gemm_probe():
    """Compose a complete GEMM from three caller-owned Crouton buffers."""

    emitter = HMXIntrinEmitter(64, 64, 64)

    @T.prim_func
    def hmx_vtcm_gemm_probe(
        A_hmx: T.Tensor((64, 64), "float16"),
        B_hmx: T.Tensor((64, 64), "float16"),
        C_hmx: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(1, threads=1) as _:
            bias_vtcm = T.alloc_shared((64,), "uint32", align=256)
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()
            T.annotate_layout(
                {
                    A_hmx: emitter.activation_layout(A_hmx),
                    B_hmx: emitter.weight_layout(B_hmx),
                    C_hmx: emitter.output_layout(C_hmx),
                }
            )

            for i in T.serial(32):
                bias_vtcm[i] = T.Cast("uint32", 0x3C00)
                bias_vtcm[32 + i] = T.Cast("uint32", 0)

            emitter.acquire(acc)
            for inst_m_idx in T.serial(emitter.num_m_tiles):
                for inst_n_idx in T.serial(emitter.num_n_tiles):
                    emitter.clear(acc)
                    emitter.load_bias(bias, bias_vtcm)
                    emitter.mma_tile(
                        acc,
                        A_hmx,
                        B_hmx,
                        inst_m_idx=inst_m_idx,
                        inst_n_idx=inst_n_idx,
                    )
                    emitter.convert(cvt, acc, bias, bias_vtcm)
                    emitter.store(
                        cvt,
                        acc,
                        C_hmx,
                        bias,
                        bias_vtcm,
                        inst_m_idx=inst_m_idx,
                        inst_n_idx=inst_n_idx,
                    )
            emitter.release(acc)

    return hmx_vtcm_gemm_probe


def _make_hmx_operand_alignment_probe():
    """Place each HMX operand after a smaller live allocation."""

    emitter = HMXIntrinEmitter(32, 32, 32)

    @T.prim_func
    def hmx_operand_alignment_probe():
        with T.Kernel(1, threads=1) as _:
            aa_padding = T.alloc_shared((64,), "float16")  # 128 bytes
            bb_weight = T.alloc_shared((32, 32), "float16")
            cc_config = T.alloc_shared((64,), "uint32")  # 256 bytes
            dd_activation = T.alloc_shared((32, 32), "float16")
            ee_output = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()

            aa_padding[0] = T.float16(0)
            emitter.acquire(acc)
            emitter.clear(acc)
            emitter.load_bias(bias, cc_config)
            emitter.mma_atom(acc, dd_activation, bb_weight)
            emitter.convert(cvt, acc, bias, cc_config)
            emitter.store(
                cvt,
                acc,
                ee_output,
                bias,
                cc_config,
            )
            emitter.release(acc)
            aa_padding[0] = ee_output[0, 0]

    return hmx_operand_alignment_probe


def _make_t_gemm_probe(*, transpose_a=False, transpose_b=False, output_dtype="float16"):
    """Exercise instruction selection, layout inference, and GemmHMX lowering."""

    @T.prim_func
    def t_gemm_probe(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), output_dtype),
    ):
        with T.Kernel(1, threads=1) as _:
            A_hmx = T.alloc_shared((64, 64), "float16")
            B_hmx = T.alloc_shared((64, 64), "float16")
            C_hmx = T.alloc_shared((64, 64), output_dtype)
            T.copy(A, A_hmx)
            T.copy(B, B_hmx)
            T.gemm(
                A_hmx,
                B_hmx,
                C_hmx,
                transpose_A=transpose_a,
                transpose_B=transpose_b,
                clear_accum=True,
            )
            T.copy(C_hmx, C)

    return t_gemm_probe


def _make_fused_copy_probe():
    """Use one logical copy per DDR/native-Crouton matrix boundary."""

    emitter = HMXIntrinEmitter(64, 64, 64)

    @T.prim_func
    def fused_copy_probe(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((64, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(2, 2, threads=1) as (bx, by):
            A_hmx = T.alloc_shared((64, 64), "float16")
            B_hmx = T.alloc_shared((64, 64), "float16")
            C_hmx = T.alloc_shared((64, 64), "float16")
            T.annotate_layout(
                {
                    A_hmx: emitter.activation_layout(A_hmx),
                    B_hmx: emitter.weight_layout(B_hmx),
                    C_hmx: emitter.output_layout(C_hmx),
                }
            )
            T.copy(A[by * 64, 0], A_hmx)
            T.copy(B[0, bx * 64], B_hmx)
            T.copy(C_hmx, C[by * 64, bx * 64])

    return fused_copy_probe


def test_hmx_crouton_layout_contract():
    atom = make_hmx_crouton_layout()
    assert _output_shape(atom) == (1024,)
    assert _offset(atom, 0, 0) == 0
    assert _offset(atom, 1, 0) == 1
    assert _offset(atom, 0, 1) == 2

    activation = make_hmx_activation_layout((64, 64))
    assert _output_shape(activation) == (2, 2, 1024)
    assert _offset(activation, 0, 0) == 0
    assert _offset(activation, 1, 0) == 1
    assert _offset(activation, 0, 1) == 2
    assert _offset(activation, 0, 32) == 1024
    assert _offset(activation, 32, 0) == 2048

    weight = make_hmx_weight_layout((64, 64))
    assert _output_shape(weight) == (2, 2, 1024)
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
    assert _output_shape(output) == (2, 2, 1024)
    assert _offset(output, 0, 32) == 1024
    assert _offset(output, 32, 0) == 2048

    staged = make_hmx_weight_layout((2, 32, 32))
    assert _output_shape(staged) == (2, 1024)
    assert _offset(staged, 1, 0, 0) == 1024

    assert len({_offset(weight, k, n) for k in range(64) for n in range(64)}) == 4096

    activation_rect = make_hmx_activation_layout((64, 128))
    activation_transposed = make_hmx_activation_layout((128, 64), transposed=True)
    for m in range(64):
        for k in range(128):
            assert _offset(activation_rect, m, k) == _offset(activation_transposed, k, m)

    weight_rect = make_hmx_weight_layout((128, 96))
    weight_transposed = make_hmx_weight_layout((96, 128), transposed=True)
    for k in range(128):
        for n in range(96):
            assert _offset(weight_rect, k, n) == _offset(weight_transposed, n, k)

    try:
        make_hmx_weight_layout((32, 48))
    except ValueError as err:
        assert "multiples of 32" in str(err)
    else:
        raise AssertionError("invalid HMX weight shape was accepted")


def test_hmx_emitter_native_interface_contract():
    emitter = HMXIntrinEmitter(64, 96, 128)
    assert emitter.num_m_tiles == 2
    assert emitter.num_n_tiles == 3
    assert emitter.num_k_atoms == 4
    assert _output_shape(emitter.activation_layout((64, 128))) == (
        2,
        4,
        1024,
    )
    assert _output_shape(emitter.weight_layout((128, 96))) == (
        3,
        4,
        1024,
    )
    assert _output_shape(emitter.output_layout((64, 96))) == (
        2,
        3,
        1024,
    )
    assert not hasattr(emitter, "make_mma_load_layout")
    assert not hasattr(emitter, "make_mma_store_layout")
    assert not hasattr(emitter, "mma_num_inst_m")


def test_hmx_atom_uses_standard_tile_coordinates():
    source = _lower_source(_make_hmx_coordinate_probe())
    mma_call = next(
        line for line in source.splitlines() if "tl_hexagon_hmx_mma_atom" in line
    )
    store_call = next(
        line for line in source.splitlines() if "tl_hexagon_hmx_store_cvt_state" in line
    )
    assert "A_hmx[3072]" in mma_call
    assert "B_hmx[3072]" in mma_call
    assert "C_hmx[3072]" in store_call
    # A/W lifetime ends at the multiply packet, so store must not retain either
    # source through dependency-only arguments.
    assert "A_hmx" not in store_call
    assert "B_hmx" not in store_call


def test_hmx_vtcm_gemm_composes_from_three_operand_buffers():
    source = _lower_source(_make_hmx_vtcm_gemm_probe())

    assert re.search(
        r"int32_t hmx_vtcm_gemm_probe_kernel\(half\* A_hmx, half\* B_hmx, half\* C_hmx\)",
        source,
    )
    for call in (
        "tl_hexagon_hmx_acc_acquire",
        "tl_hexagon_hmx_clear_acc",
        "tl_hexagon_hmx_load_bias",
        "tl_hexagon_hmx_mma_atom",
        "tl_hexagon_hmx_convert_acc",
        "tl_hexagon_hmx_store_cvt_state",
        "tl_hexagon_hmx_acc_release",
    ):
        assert call in source
    assert "tl_hexagon_hmx_gemm" not in source
    assert "A_hmx[((inst_m_idx * 2048) + (k_inner * 1024))]" in source
    assert "B_hmx[((inst_n_idx * 2048) + (k_inner * 1024))]" in source
    assert "C_hmx[((inst_m_idx * 2048) + (inst_n_idx * 1024))]" in source
    assert "tl_hexagon_hmx_commit_output_atom" not in source


def test_hmx_operand_alignment_is_propagated_per_role():
    source = _lower_source(
        _make_hmx_operand_alignment_probe(),
        {"tl.disable_shared_memory_reuse": True},
    )

    offsets = {}
    for name in (
        "aa_padding",
        "bb_weight",
        "cc_config",
        "dd_activation",
        "ee_output",
    ):
        match = re.search(
            rf"void\* {name} = .*buf_dyn_shmem \+ (\d+)\)\);", source
        )
        assert match is not None
        offsets[name] = int(match.group(1))

    assert offsets["bb_weight"] % 128 == 0
    assert offsets["bb_weight"] % 2048 != 0
    assert offsets["cc_config"] % 256 == 0
    assert offsets["cc_config"] % 2048 != 0
    assert offsets["dd_activation"] % 2048 == 0
    assert offsets["ee_output"] % 2048 == 0


def test_t_gemm_lowers_to_native_hmx_layout_and_atoms():
    source = _lower_source(_make_t_gemm_probe())

    for call in (
        "tl_hexagon_hmx_acc_acquire",
        "tl_hexagon_hmx_clear_acc",
        "tl_hexagon_hmx_load_bias",
        "tl_hexagon_hmx_mma_atom",
        "tl_hexagon_hmx_convert_acc",
        "tl_hexagon_hmx_store_cvt_state",
        "tl_hexagon_hmx_acc_release",
    ):
        assert call in source
    assert "tl_hexagon_hmx_gemm" not in source
    assert "A_hmx)[((inst_m_idx * 2048) + (k_inner * 1024))]" in source
    assert "B_hmx)[((inst_n_idx * 2048) + (k_inner * 1024))]" in source
    assert "C_hmx)[((inst_m_idx * 2048) + (inst_n_idx * 1024))]" in source
    assert "output_atom" not in source
    assert "tl_hexagon_hmx_commit_output_atom" not in source

    offsets = {}
    for name in ("A_hmx", "B_hmx", "C_hmx"):
        match = re.search(
            rf"void\* {name} = .*buf_dyn_shmem \+ (\d+)\)\);", source
        )
        assert match is not None
        offsets[name] = int(match.group(1))
        assert offsets[name] % 2048 == 0

    # Each logical DDR/native-Crouton boundary becomes one strided helper call.
    assert source.count("tl_hexagon_hmx_pack_crouton") == 2
    assert source.count("tl_hexagon_hmx_unpack_crouton") == 1
    assert "tl_hexagon_hmx_pack_crouton((&(((half*)A_hmx)[0])), (&(A[0])), 64, 64, 64, 1, 0)" in source
    assert "tl_hexagon_hmx_pack_crouton((&(((half*)B_hmx)[0])), (&(B[0])), 64, 64, 64, 1, 1)" in source
    assert "tl_hexagon_hmx_unpack_crouton((&(C[0])), (&(((half*)C_hmx)[0])), 64, 64, 64, 1, 0)" in source


def test_t_gemm_transposed_operands_use_composed_native_layouts():
    transposed_a = _lower_source(_make_t_gemm_probe(transpose_a=True))
    transposed_b = _lower_source(_make_t_gemm_probe(transpose_b=True))

    assert "A_hmx)[((inst_m_idx * 2048) + (k_inner * 1024))]" in transposed_a
    assert "B_hmx)[((inst_n_idx * 2048) + (k_inner * 1024))]" in transposed_b
    assert "(&(((half*)A_hmx)[0])), (&(A[0])), 64, 64, 64, 1, 2)" in transposed_a
    assert "(&(((half*)B_hmx)[0])), (&(B[0])), 64, 64, 64, 1, 3)" in transposed_b
    for source in (transposed_a, transposed_b):
        assert "tl_hexagon_hmx_mma_atom" in source
        assert "tl_hexagon_hmx_gemm" not in source


def test_t_gemm_unsupported_output_dtype_stays_on_scalar_fallback():
    source = _lower_source(_make_t_gemm_probe(output_dtype="float32"))
    assert "tl_hexagon_hmx_" not in source
    assert "tl_hexagon_hmx_gemm" not in source


def test_flash_attention_keeps_row_reduce_outside_crouton_buffers():
    source = _lower_source(make_flash(64, 64, 64, 64))
    rowmax = next(line for line in source.splitlines() if "tl_hvx_rowmax_mat" in line)
    rowsum = next(line for line in source.splitlines() if "tl_hvx_rowsum_mat" in line)

    assert source.count("tl_hexagon_hmx_mma_atom") == 2
    assert "((half*)S)[0]" in rowmax
    assert "((half*)S)[0]" in rowsum
    assert "S_hmx" not in rowmax
    assert "S_hmx" not in rowsum
    assert source.count("tl_hexagon_hmx_pack_crouton") == 4
    assert source.count("tl_hexagon_hmx_unpack_crouton") == 2
    assert "tl_hexagon_hmx_gemm" not in source


def test_hmx_copy_matches_singleton_tile_axes_and_dequant_staging():
    source = _lower_source(make_qmatmul(32, 128, 128))

    assert source.count("tl_hexagon_hmx_pack_crouton") == 2
    assert source.count("tl_hexagon_hmx_unpack_crouton") == 1
    assert "(&(((half*)A_hmx)[0])), (&(A[0])), 32, 128, 128, 1, 0)" in source
    assert "(&(((half*)B_row)[0])), 128, 128, 128, 1, 1)" in source
    assert "tl_hexagon_hmx_unpack_crouton((&(C[0])), (&(((half*)C_hmx)[0])), 32, 128, 128, 1, 0)" in source
    assert "A_row" not in source
    assert "C_row" not in source
    assert not any(
        "B_hmx)[" in line and "=" in line
        for line in source.splitlines()
    )


def test_hmx_copy_fuses_global_matrix_slices_with_one_logical_copy():
    source = _lower_source(_make_fused_copy_probe())

    assert source.count("tl_hexagon_hmx_pack_crouton") == 2
    assert source.count("tl_hexagon_hmx_unpack_crouton") == 1
    assert "tl_hexagon_hmx_pack_crouton((&(((half*)A_hmx)[0])), (&(A[(by * 8192)])), 64, 64, 128, 1, 0)" in source
    assert "tl_hexagon_hmx_pack_crouton((&(((half*)B_hmx)[0])), (&(B[(bx * 64)])), 64, 64, 128, 1, 1)" in source
    assert "tl_hexagon_hmx_unpack_crouton((&(C[((by * 8192) + (bx * 64))])), (&(((half*)C_hmx)[0])), 64, 64, 128, 1, 0)" in source


def test_hmx_pack_unpack_is_worker_local():
    source = _lower_source(make_batched_matmul(2, 64, 64, 64, 2))

    assert "tl_parallel" in source
    assert source.count("tl_hexagon_hmx_pack_crouton") == 2
    assert source.count("tl_hexagon_hmx_unpack_crouton") == 1
    assert source.count("tl_hexagon_hmx_mma_atom") == 1
    assert "A_row" not in source
    assert "B_row" not in source
    assert "C_row" not in source


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

    dsp_wrapper = _fastrpc.gen_dsp(
        "tl_qmatmul_hmx_atoms_kernel",
        "qmatmul_hmx_atoms_kernel",
        source,
        [],
    )
    assert "#include <tl_templates/hexagon/qmatmul.h>" in dsp_wrapper


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
