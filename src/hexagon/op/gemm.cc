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
#include "support/check.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {
constexpr const char *kHexagonScalar = "cpu.scalar";
} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kHexagonScalar;
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
    (void)gemm_inst;
    return "scalar";
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
