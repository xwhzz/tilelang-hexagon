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

// Hexagon HMX instruction protocol.  State operands are dependency-only IR
// tokens that make the implicit accumulator/convert/bias dependencies explicit.
// Memory operands are VTCM addresses: activation and output tiles 2 KiB
// aligned, weight tiles 128 B, and the bias/scale block 256 B.
TIR_DEFINE_TL_BUILTIN(hexagon_hmx_acquire)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_release)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_load_bias)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment, {0, 256});

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_clear)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_mma)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment,
                                       {0, 2048, 128, 0});

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_convert)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment,
                                       {0, 0, 0, 256, 0});

TIR_DEFINE_TL_BUILTIN(hexagon_hmx_store)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TSharedOperandAlignment>(kSharedOperandAlignment,
                                       {0, 2048, 0, 0, 256});

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

#undef TL_HEXAGON_MATH_BUILTIN

#define TL_HEXAGON_LANE_BUILTIN(Name)                                          \
  TIR_DEFINE_TL_BUILTIN(Name).set_num_inputs(1).set_attr<TCallEffectKind>(     \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

TL_HEXAGON_LANE_BUILTIN(hexagon_lane_sum);
TL_HEXAGON_LANE_BUILTIN(hexagon_lane_max);
TL_HEXAGON_LANE_BUILTIN(hexagon_lane_min);
TL_HEXAGON_LANE_BUILTIN(hexagon_lane_and);
TL_HEXAGON_LANE_BUILTIN(hexagon_lane_or);
TL_HEXAGON_LANE_BUILTIN(hexagon_lane_xor);

#undef TL_HEXAGON_LANE_BUILTIN

} // namespace tl
} // namespace tvm
