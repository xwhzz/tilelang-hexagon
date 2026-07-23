/*!
 * \file tl/hexagon/op/gemm.cc
 * \brief Hexagon implementation for tl.gemm instruction selection.
 *
 * For now this selects the generic "scalar" instruction, lowered by the Python
 * GemmScalar impl (a triple loop that accumulates in the accum dtype, which
 * hexagon-clang auto-vectorizes onto HVX).  An HMX instruction key + Crouton
 * tile layout is the next step toward using the matrix engine.
 */

#include "op/gemm.h"
#include "op/utils.h" // IsSharedBuffer
#include "support/check.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {
constexpr const char *kHexagonScalar = "cpu.scalar";
constexpr const char *kHexagonHMX = "hexagon.hmx";
} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)block_size;
    (void)target;
    // Route to HMX only for the validated GemmHMX surface: fp16, 32-multiple,
    // 2D, shared-shared (SS), overwrite (clear_accum is a compile-time true).
    // Everything else falls back to the scalar loop hexagon-clang auto-vectorizes
    // onto HVX — crucially the DEFAULT clear_accum=false (the accumulate K-loop
    // pattern) and fragment/sub-rank operands, which GemmHMX can't lower yet.
    bool hmx_ok = op.a_->dtype == DataType::Float(16) &&
                  op.b_->dtype == DataType::Float(16) && op.m_ % 32 == 0 &&
                  op.n_ % 32 == 0 && op.k_ % 32 == 0 && op.a_->shape.size() == 2 &&
                  op.b_->shape.size() == 2 && op.c_->shape.size() == 2 &&
                  IsSharedBuffer(op.a_) && IsSharedBuffer(op.b_) &&
                  IsSharedBuffer(op.c_) && is_one(op.clearAccum_);
    return hmx_ok ? String(kHexagonHMX) : String(kHexagonScalar);
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    return gemm_inst == kHexagonHMX ? String("hmx") : String("scalar");
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonGemmTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonGemm() {
  RegisterGemmImpl(GemmImpl{
      "hexagon.Gemm",
      MatchHexagonGemmTarget,
      hexagon::Gemm::SelectInst,
      hexagon::Gemm::ComputeWarpPartition,
      hexagon::Gemm::ReuseExistingSharedLayout,
      hexagon::Gemm::InstructionKind,
  });
  return true;
}

const bool hexagon_gemm_registered = RegisterHexagonGemm();

} // namespace

} // namespace tl
} // namespace tvm
