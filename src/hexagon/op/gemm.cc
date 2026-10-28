/*!
 * \file tl/hexagon/op/gemm.cc
 * \brief Hexagon implementation for tl.gemm instruction selection.
 *
 * A GEMM runs on HMX (GemmHMX, tilelang/hexagon/op/gemm/gemm_hmx.py) when its
 * operands fit the matrix unit, and as the scalar loop otherwise, which
 * hexagon-clang vectorizes onto HVX.
 */

#include "hexagon/op/gemm.h"

#include "op/gemm.h"
#include "op/utils.h" // IsSharedBuffer, IsFragmentBuffer
#include "support/check.h"

#include "backend/common/target_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>
#include <string>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {
constexpr const char *kHexagonScalar = "cpu.scalar";
constexpr const char *kHexagonHMX = "hexagon.hmx";
constexpr int kTile = 32;

bool OnTiles(const PrimExpr &value, arith::Analyzer *analyzer) {
  return analyzer->CanProveEqual(floormod(value, kTile), 0);
}

std::string TypeName(DataType dtype) {
  std::ostringstream os;
  os << dtype;
  return os.str();
}

} // namespace

std::string HMXRejection(const GemmNode &op) {
  const DataType fp16 = DataType::Float(16);
  if (op.a_->dtype != fp16 || op.b_->dtype != fp16)
    return "HMX multiplies fp16 operands, not " + TypeName(op.a_->dtype) +
           " x " + TypeName(op.b_->dtype);
  if (op.c_->dtype != fp16 && op.c_->dtype != DataType::Float(32))
    return "HMX writes an fp16 or fp32 C, not " + TypeName(op.c_->dtype);
  if (op.m_ % kTile || op.n_ % kTile || op.k_ % kTile)
    return "HMX needs M, N and K multiples of 32, not " +
           std::to_string(op.m_) + " x " + std::to_string(op.n_) + " x " +
           std::to_string(op.k_);
  arith::Analyzer analyzer;
  const std::pair<const char *, const BufferRegion *> operands[] = {
      {"A", &op.aRegion_}, {"B", &op.bRegion_}, {"C", &op.cRegion_}};
  for (const auto &[name, region] : operands) {
    const Buffer &buffer = (*region)->buffer;
    std::string operand = std::string(name) + " (" + buffer->name + ")";
    if (!IsSharedBuffer(buffer) && !IsFragmentBuffer(buffer))
      return operand + " is not in VTCM: HMX operands are shared buffers or "
                       "fragments";
    const Array<Range> &ranges = (*region)->region;
    size_t rank = ranges.size();
    if (rank < 2)
      return operand + " is not a matrix";
    for (size_t i = 0; i + 2 < rank; ++i) {
      if (!is_one(ranges[i]->extent))
        return operand + " spans more than one matrix of its leading "
                         "dimensions";
    }
    for (size_t i = rank - 2; i < rank; ++i) {
      if (!OnTiles(ranges[i]->min, &analyzer))
        return operand + "'s region does not start on a 32 x 32 tile";
      if (buffer.same_as(op.c_) && buffer->dtype != fp16)
        continue; // An fp32 C stays row-major.
      if (!OnTiles(buffer->shape[i], &analyzer))
        return operand + " is not whole 32 x 32 tiles";
    }
  }
  return "";
}

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)block_size;
    (void)target;
    return HMXRejection(op).empty() ? String(kHexagonHMX)
                                    : String(kHexagonScalar);
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

  // A layout a shared buffer already has stays: C may have any layout, and
  // GemmHMX checks that A and B have the HMX operand layouts.
  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return true;
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
  });
  return true;
}

const bool hexagon_gemm_registered = RegisterHexagonGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // The scalar fallback reports why a GEMM is not on HMX.
  refl::GlobalDef().def("tl.hexagon.GemmHMXRejection", [](ObjectRef gemm) {
    const auto *node = gemm.as<GemmNode>();
    ICHECK(node) << "expected a tl.Gemm, got " << gemm->GetTypeKey();
    return String(hexagon::HMXRejection(*node));
  });
}

} // namespace tl
} // namespace tvm
