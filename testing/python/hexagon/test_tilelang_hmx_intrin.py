import pytest
import re

import tilelang
from tilelang.hexagon import language as T
from tilelang import tvm
from examples.hexagon.example_flash_attention import make_flash
from tilelang.hexagon.intrinsics.crouton import (
    make_hmx_activation_layout,
    make_hmx_crouton_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)
from tilelang.hexagon.intrinsics.hmx import HMXIntrinEmitter
from tilelang.backend.target import determine_target


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
        with T.Kernel(1) as _:
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()
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
        with T.Kernel(1) as _:
            bias_vtcm = T.alloc_shared((64,), "uint32", align=256)
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()
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
        with T.Kernel(1) as _:
            aa_padding = T.alloc_shared((64,), "float16")  # 128 bytes
            bb_weight = T.alloc_shared((32, 32), "float16")
            cc_config = T.alloc_shared((64,), "uint32")  # 256 bytes
            dd_activation = T.alloc_shared((32, 32), "float16")
            ee_output = T.alloc_shared((32, 32), "float16")
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()

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


def _make_t_gemm_probe(*, transpose_a=False, transpose_b=False):
    """Exercise instruction selection, layout inference, and GemmHMX lowering."""

    @T.prim_func
    def t_gemm_probe(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(1) as _:
            A_hmx = T.alloc_shared((64, 64), "float16")
            B_hmx = T.alloc_shared((64, 64), "float16")
            C_hmx = T.alloc_shared((64, 64), "float16")
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
        with T.Kernel(2, 2) as (bx, by):
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
    assert _output_shape(atom) == (1, 16, 64)
    assert _offset(atom, 0, 0) == 0
    assert _offset(atom, 1, 0) == 1
    assert _offset(atom, 0, 1) == 2

    activation = make_hmx_activation_layout((64, 64))
    assert _output_shape(activation) == (4, 16, 64)
    assert _offset(activation, 0, 0) == 0
    assert _offset(activation, 1, 0) == 1
    assert _offset(activation, 0, 1) == 2
    assert _offset(activation, 0, 32) == 1024
    assert _offset(activation, 32, 0) == 2048

    weight = make_hmx_weight_layout((64, 64))
    assert _output_shape(weight) == (4, 16, 64)
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
    assert _output_shape(output) == (4, 16, 64)
    assert _offset(output, 0, 32) == 1024
    assert _offset(output, 32, 0) == 2048

    staged = make_hmx_weight_layout((2, 32, 32))
    assert _output_shape(staged) == (2, 16, 64)
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
    assert _output_shape(emitter.activation_layout((64, 128))) == (8, 16, 64)
    assert _output_shape(emitter.weight_layout((128, 96))) == (12, 16, 64)
    assert _output_shape(emitter.output_layout((64, 96))) == (6, 16, 64)
    assert not hasattr(emitter, "make_mma_load_layout")
    assert not hasattr(emitter, "make_mma_store_layout")
    assert not hasattr(emitter, "mma_num_inst_m")


def test_hmx_atom_uses_standard_tile_coordinates():
    source = _lower_source(_make_hmx_coordinate_probe())
    mma_call = next(line for line in source.splitlines() if "tl_hexagon_hmx_mma_atom" in line)
    store_call = next(line for line in source.splitlines() if "tl_hexagon_hmx_store_cvt_state" in line)
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
    assert "A_hmx[(inst_m_idx * 2048)]" in source
    assert "B_hmx[(inst_n_idx * 2048)]" in source
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
        match = re.search(rf"void\* {name} = .*buf_dyn_shmem \+ (\d+)\)\)\)?;", source)
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
    # Output tile t is row tile t / 2 and column tile t % 2 of the 2 x 2.
    assert "A_hmx)[((t >> 1) * 2048)]" in source
    assert "B_hmx)[((t & 1) * 2048)]" in source
    # An fp16 C that the call overwrites takes the HMX tiles directly.
    assert "C_hmx)[(t * 1024)]" in source
    assert "tl_crouton_tile_store" not in source
    assert "output_atom" not in source
    assert "tl_hexagon_hmx_commit_output_atom" not in source

    offsets = {}
    for name in ("A_hmx", "B_hmx", "C_hmx"):
        match = re.search(rf"void\* {name} = .*buf_dyn_shmem \+ (\d+)\)\)\)?;", source)
        assert match is not None
        offsets[name] = int(match.group(1))
        assert offsets[name] % 2048 == 0

    # Each logical DDR/native-Crouton boundary becomes one Crouton copy.
    assert source.count("tl_crouton_copy<") == 3
    assert "tl_crouton_copy<true, false>((&(((half*)A_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(A[0])), 64, 1);" in source
    assert "tl_crouton_copy<true, true>((&(((half*)B_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(B[0])), 64, 1);" in source
    assert "tl_crouton_copy<false, false>((&(((half*)C_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(C[0])), 64, 1);" in source


def test_t_gemm_transposed_operands_use_composed_native_layouts():
    transposed_a = _lower_source(_make_t_gemm_probe(transpose_a=True))
    transposed_b = _lower_source(_make_t_gemm_probe(transpose_b=True))

    assert "A_hmx)[((t >> 1) * 2048)]" in transposed_a
    assert "B_hmx)[((t & 1) * 2048)]" in transposed_b
    # A transposed operand's matrix runs down its storage columns.
    assert "tl_crouton_copy<true, false>((&(((half*)A_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(A[0])), 1, 64);" in transposed_a
    assert "tl_crouton_copy<true, true>((&(((half*)B_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(B[0])), 1, 64);" in transposed_b
    for source in (transposed_a, transposed_b):
        assert "tl_hexagon_hmx_mma_atom" in source
        assert "tl_hexagon_hmx_gemm" not in source


def test_flash_attention_softmax_works_on_crouton_tiles():
    source = _lower_source(make_flash(64, 64, 64, 64))
    lines = source.splitlines()
    # The row reductions read S's Crouton tiles; nothing converts S or temp.
    for kind, acc in (("MAX", "half"), ("SUM", "float")):
        line = next(line for line in lines if f"tl_crouton_reduce<TL_REDUCE_{kind}, {acc}, true, false>(" in line)
        assert "(&(((half*)S)[0]))" in line, line
    assert "tl_hvx_reduce_rows" not in source

    assert source.count("tl_hexagon_hmx_mma_atom") == 2
    # Q, K and V pack into Crouton tiles; only the output unpacks.
    assert source.count("tl_crouton_copy<true,") == 3
    assert source.count("tl_crouton_copy<false,") == 1
    assert "tl_hexagon_hmx_gemm" not in source


def _make_hmx_gemm(M, N, K):
    """C = A @ B on HMX, every operand copied whole between DDR and VTCM."""

    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1) as _:
            A_hmx = T.alloc_shared((M, K), "float16")
            B_hmx = T.alloc_shared((K, N), "float16")
            C_hmx = T.alloc_shared((M, N), "float16")
            T.copy(A, A_hmx)
            T.copy(B, B_hmx)
            T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
            T.copy(C_hmx, C)

    return gemm


def _make_batched_matmul(NB, M, N, K, num_workers):
    """NB independent HMX GEMMs, one program each, on num_workers workers."""

    @T.prim_func
    def batched(A: T.Tensor((NB, M, K), "float16"), B: T.Tensor((NB, K, N), "float16"), C: T.Tensor((NB, M, N), "float16")):
        with T.Kernel(NB, num_workers=num_workers) as bx:
            A_hmx = T.alloc_shared((M, K), "float16")
            B_hmx = T.alloc_shared((K, N), "float16")
            C_hmx = T.alloc_shared((M, N), "float16")
            T.copy(A[bx, :, :], A_hmx)
            T.copy(B[bx, :, :], B_hmx)
            T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
            T.copy(C_hmx, C[bx, :, :])

    return batched


def test_hmx_copy_matches_singleton_tile_axes():
    source = _lower_source(_make_hmx_gemm(32, 128, 128))

    # A and C are one tile high: the tile grid keeps that axis.
    assert "tl_crouton_copy<true, false>((&(((half*)A_hmx)[0])), 1, 4, 0, 0, 32, 128, (&(A[0])), 128, 1);" in source
    assert "tl_crouton_copy<true, true>((&(((half*)B_hmx)[0])), 4, 4, 0, 0, 128, 128, (&(B[0])), 128, 1);" in source
    assert "tl_crouton_copy<false, false>((&(((half*)C_hmx)[0])), 1, 4, 0, 0, 32, 128, (&(C[0])), 128, 1);" in source
    assert not any("B_hmx)[" in line and "=" in line for line in source.splitlines())


def test_hmx_copy_fuses_global_matrix_slices_with_one_logical_copy():
    source = _lower_source(_make_fused_copy_probe())

    assert "tl_crouton_copy<true, false>((&(((half*)A_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(A[(by * 8192)])), 128, 1);" in source
    assert "tl_crouton_copy<true, true>((&(((half*)B_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(B[(bx * 64)])), 128, 1);" in source
    assert "tl_crouton_copy<false, false>((&(((half*)C_hmx)[0])), 2, 2, 0, 0, 64, 64, (&(C[((by * 8192) + (bx * 64))])), 128, 1);" in source


def test_hmx_pack_unpack_is_worker_local():
    source = _lower_source(_make_batched_matmul(2, 64, 64, 64, 2))

    assert "tl_parallel" in source
    assert source.count("tl_crouton_copy<true,") == 2
    assert source.count("tl_crouton_copy<false,") == 1
    assert source.count("tl_hexagon_hmx_mma_atom") == 1
    assert "A_row" not in source
    assert "B_row" not in source
    assert "C_row" not in source


@pytest.mark.parametrize("tiles", [1, 31, 32, 33, 64, 65])
def test_hmx_deep_chunk_bounds_and_native_offsets(tiles):
    emitter = HMXIntrinEmitter(32, 32, tiles * 32, b_transposed=True)

    @T.prim_func
    def probe(A: T.Tensor((32, tiles * 32), "float16"), B: T.Tensor((32, tiles * 32), "float16")):
        with T.Kernel(1):
            acc = T.hmx.alloc_accumulator()
            T.annotate_layout({A: emitter.activation_layout(A), B: emitter.weight_layout(B)})
            emitter.acquire(acc)
            emitter.clear(acc)
            emitter.mma_tile(acc, A, B)
            emitter.release(acc)

    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(probe, target=target, enable_host_codegen=False, enable_device_compile=False)
    source = str(artifact.kernel_source)
    call = next(line for line in source.splitlines() if "tl_hexagon_hmx_mma_atom(" in line)
    if tiles <= 32:
        assert "A[0]" in call and "B[0]" in call
    else:
        assert "k_chunk * 32768" in call
        assert f"k_chunk < {(tiles + 31) // 32}" in source

    calls = []

    def collect(node):
        if isinstance(node, tvm.tirx.Call) and node.op == tvm.ir.Op.get("tl.hexagon_hmx_mma"):
            calls.append(node)

    for func in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    assert len(calls) == 1
    mma = calls[0]
    analyzer = tvm.arith.Analyzer()
    for chunk in range((tiles + 31) // 32):
        expected = min(tiles - chunk * 32, 32)
        for expr, value in [(mma.args[3], expected), (mma.args[1].args[3], expected * 1024), (mma.args[2].args[3], expected * 1024)]:
            variables = []
            tvm.tirx.stmt_functor.post_order_visit(
                expr, lambda node, variables=variables: variables.append(node) if isinstance(node, tvm.tirx.Var) else None
            )
            assert all(var.name == "k_chunk" for var in variables)
            evaluated = tvm.tirx.stmt_functor.substitute(expr, {var: chunk for var in variables})
            assert int(analyzer.simplify(evaluated)) == value


@pytest.mark.parametrize("count", [0, -1, 33])
def test_hmx_deep_rejects_invalid_constant_count(count):
    with pytest.raises(ValueError, match="num_tiles"):
        T.hmx.mma(0, 0, 0, count)
