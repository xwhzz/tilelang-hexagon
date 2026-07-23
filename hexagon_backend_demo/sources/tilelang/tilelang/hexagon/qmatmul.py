"""Register-level data-movement atoms for explicit Q4_0 HMX matmul.

The emitter does not own an operator schedule.  It exposes the HVX atoms needed
around HMX while the caller retains TileLang loops and Crouton layouts.  The
preferred dequant atom consumes one native 576-byte Q4_0 tile so the compiler
can overlap four independent HVX instruction chains; group-granular atoms stay
available for focused lowering tests.
"""

from tilelang import language as T


class Q4HMXIntrinEmitter:
    """Emit v79 HVX atoms used by a Q4_0 HMX kernel."""

    quant_bytes = 512
    group_bytes = 128
    groups_per_tile = 4
    group_output_half = 256

    @staticmethod
    def pack_activation_pair(dst, src, dst_index, row0_index, row1_index):
        """Pack two contiguous 32-float rows into one activation Crouton vector."""

        @T.macro
        def _pack(dst, src):
            T.call_extern(
                "int32",
                "tl_hexagon_hmx_pack_a_f32_pair_k32",
                T.address_of(dst[dst_index]),
                T.address_of(src[row0_index]),
                T.address_of(src[row1_index]),
            )

        return _pack(dst, src)

    @staticmethod
    def store_output_pair_f32(dst, src, row0_index, row1_index, src_index):
        """Convert one HMX output vector into two contiguous FP32 rows."""

        @T.macro
        def _store(dst, src):
            T.call_extern(
                "int32",
                "tl_hexagon_hmx_unpack_c_f32_pair_n32",
                T.address_of(dst[row0_index]),
                T.address_of(dst[row1_index]),
                T.address_of(src[src_index]),
            )

        return _store(dst, src)

    @staticmethod
    def prepare_scale(dst, src, dst_index, scale_index):
        """Safely load and duplicate one tile's 32 FP16 channel scales."""

        @T.macro
        def _prepare(dst, src):
            T.call_extern(
                "int32",
                "tl_hexagon_q4_0_prepare_scale_32",
                T.address_of(dst[dst_index]),
                T.address_of(src[scale_index]),
            )

        return _prepare(dst, src)

    @staticmethod
    def dequant_group(dst, src, scale, dst_index, quant_index, scale_index):
        """Expand one packed 128-byte group into four FP16 Crouton vectors."""

        @T.macro
        def _dequant(dst, src, scale):
            T.call_extern(
                "int32",
                "tl_hexagon_q4_0_dequant_group_128",
                T.address_of(dst[dst_index]),
                T.address_of(src[quant_index]),
                T.address_of(scale[scale_index]),
            )

        return _dequant(dst, src, scale)

    @staticmethod
    def dequant_tile(dst, src, dst_index, tile_index):
        """Expand one native Q4_0 tile into one 32x32 weight Crouton."""

        @T.macro
        def _dequant(dst, src):
            T.call_extern(
                "int32",
                "tl_hexagon_q4_0_dequant_tile_32x32",
                T.address_of(dst[dst_index]),
                T.address_of(src[tile_index]),
            )

        return _dequant(dst, src)
