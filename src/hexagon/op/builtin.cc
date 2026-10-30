/*!
 * \file tl/hexagon/op/builtin.cc
 * \brief Registration of Hexagon-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include "op/builtin_registry.h"

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

TIR_DEFINE_TL_BUILTIN(hexagon_dma_queue)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_dma_submit)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_dma_commit)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_dma_wait)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_cache_writeback)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

#define TL_HEXAGON_MATH_BUILTIN(Name)                                          \
  TIR_DEFINE_TL_BUILTIN(Name)                                                  \
      .set_num_inputs(1)                                                       \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kPure))               \
      .set_attr<TVectorizable>("TVectorizable", true)

TL_HEXAGON_MATH_BUILTIN(hexagon_exp);
TL_HEXAGON_MATH_BUILTIN(hexagon_exp2);
TL_HEXAGON_MATH_BUILTIN(hexagon_log);
TL_HEXAGON_MATH_BUILTIN(hexagon_log2);
TL_HEXAGON_MATH_BUILTIN(hexagon_sqrt);
TL_HEXAGON_MATH_BUILTIN(hexagon_rsqrt);
TL_HEXAGON_MATH_BUILTIN(hexagon_tanh);
TL_HEXAGON_MATH_BUILTIN(hexagon_sigmoid);
TL_HEXAGON_MATH_BUILTIN(hexagon_erf);
TL_HEXAGON_MATH_BUILTIN(hexagon_fabs);
TL_HEXAGON_MATH_BUILTIN(hexagon_exp_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_exp2_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_log_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_log2_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_sqrt_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_rsqrt_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_tanh_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_sigmoid_fast);
TL_HEXAGON_MATH_BUILTIN(hexagon_erf_fast);

#undef TL_HEXAGON_MATH_BUILTIN

TIR_DEFINE_TL_BUILTIN(hexagon_div_fast)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

TIR_DEFINE_TL_BUILTIN(hexagon_reduce_rows)
    .set_num_inputs(10)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_crouton_reduce)
    .set_num_inputs(10)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_load_lanes)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kReadState));

TIR_DEFINE_TL_BUILTIN(hexagon_broadcast_lane)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(hexagon_load_replicate)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kReadState));

TIR_DEFINE_TL_BUILTIN(hexagon_load_repeat)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kReadState));

TIR_DEFINE_TL_BUILTIN(hexagon_load_interleave)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kReadState));

TIR_DEFINE_TL_BUILTIN(hexagon_crouton_copy)
    .set_num_inputs(12)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_crouton_move_tiles)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_crouton_tile_store)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// The HMX GEMMs. HMX reads activation tiles on 2 KiB boundaries and weight
// tiles on 128 B ones, stores whole 2 KiB tiles, and reads its conversion
// config from 256 B: the alignments of the VTCM operands, by argument (the
// flags first).
TIR_DEFINE_TL_BUILTIN(hexagon_hmx_gemm)
    .set_num_inputs(16)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment,
                                       {0, 0, 0, 0, 0, 0, 0, 2048, 0, 2048, 0,
                                        128, 0, 0, 2048, 256});

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_gemm_blockscaled)
    .set_num_inputs(23)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment,
                                       {0, 0, 0,    0, 0,    0,    0,  0,
                                        0, 0, 2048, 0, 0,    0,    0,  0,
                                        0, 0, 0,    0, 2048, 2048, 256});

TIR_DEFINE_TL_BUILTIN(hexagon_hvx_gemm)
    .set_num_inputs(15)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hvx_gemm_weight)
    .set_num_inputs(11)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hvx_gemm_int)
    .set_num_inputs(13)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hvx_gemm_int_scaled)
    .set_num_inputs(19)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_exclusive_pointer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

} // namespace tl
} // namespace tvm
