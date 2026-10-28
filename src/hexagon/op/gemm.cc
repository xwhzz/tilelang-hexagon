/*!
 * \file tl/hexagon/op/gemm.cc
 * \brief Hexagon implementation for tl.gemm instruction selection.
 *
 * A GEMM runs on HMX (GemmHMX, tilelang/hexagon/op/gemm/gemm_hmx.py) when its
 * operands fit the matrix unit, on HVX (GemmHVX, gemm_hvx.py) when they are
 * row-major matrices of fp16 or fp32, or integers with a panel-layout B, and
 * as the scalar loop otherwise. A block-scaled GEMM of an fp16 A by a 4-bit
 * B runs on HMX (GemmHMXBlockScaled), one of integers on HVX
 * (GemmHVXBlockScaled); any other fails to compile.
 */

#include "hexagon/op/gemm.h"

#include "hexagon/op/utils.h" // GetBufferStrides, IsVTCMBuffer
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
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
constexpr const char *kHexagonHVX = "hexagon.hvx";
constexpr const char *kHexagonHVXBlockScaled = "hexagon.hvx.blockscaled";
constexpr const char *kHexagonHMXBlockScaled = "hexagon.hmx.blockscaled";
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
    if (!IsVTCMBuffer(buffer))
      return operand + " is not in VTCM: HMX operands are shared buffers, "
                       "fragments or global.vtcm buffers";
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

// Why the integer B of `op` is not read in whole panels from VTCM, rows of A
// by rows of B, or "".
std::string PanelRejection(const GemmNode &op) {
  if (op.transA_ || !op.transB_)
    return "HVX multiplies integer rows of A by rows of B: transpose_B, and "
           "A not transposed";
  if (!IsVTCMBuffer(op.b_))
    return "HVX reads an integer B in its panel layout, from VTCM";
  int64_t lane = 32 / op.b_->dtype.bits();
  const Array<Range> &rb = op.bRegion_->region;
  const Range &rows = rb[rb.size() - 2], &cols = rb[rb.size() - 1];
  const auto *nrows = rows->extent.as<IntImmNode>();
  const auto *ncols = cols->extent.as<IntImmNode>();
  arith::Analyzer analyzer;
  auto aligned = [&analyzer](const PrimExpr &min, int64_t to) {
    return analyzer.CanProveEqual(floormod(min, make_const(min.dtype(), to)),
                                  0);
  };
  if (!nrows || !ncols || nrows->value % 32 || ncols->value % lane ||
      !aligned(rows->min, 32) || !aligned(cols->min, lane))
    return "HVX multiplies whole panels of an integer B: its region's rows "
           "start and end on multiples of 32, its columns on multiples of " +
           std::to_string(lane);
  return "";
}

// Why HVX's vrmpy cannot run the integer GEMM `op`, or "".
// C is int32, or fp32 for a block-scaled GEMM.
std::string HVXIntRejection(const GemmNode &op, DataType c_type) {
  DataType a = op.a_->dtype, b = op.b_->dtype, c = op.c_->dtype;
  bool a_ok =
      a == DataType::Int(8) || a == DataType::UInt(8) || a == DataType::Int(16);
  bool b_ok = (b.is_int() || b.is_uint()) && b.lanes() == 1 &&
              (b.bits() == 8 || b.bits() == 4);
  if (!a_ok || !b_ok || c != c_type)
    return "HVX multiplies an int8, uint8 or int16 A by an 8- or 4-bit B "
           "into an " +
           TypeName(c_type) + " C, not " + TypeName(a) + " x " + TypeName(b) +
           " into " + TypeName(c);
  return PanelRejection(op);
}

std::string HVXRejection(const GemmNode &op) {
  const DataType fp16 = DataType::Float(16);
  const DataType fp32 = DataType::Float(32);
  DataType a = op.a_->dtype, b = op.b_->dtype, c = op.c_->dtype;
  bool integer = (a.is_int() || a.is_uint()) && (b.is_int() || b.is_uint());
  if (integer) {
    std::string why = HVXIntRejection(op, DataType::Int(32));
    if (!why.empty())
      return why;
  } else {
    bool operands = a == b && (a == fp16 || a == fp32);
    if (!operands || !(c == fp32 || (c == fp16 && a == fp16)))
      return "HVX multiplies fp16 operands into an fp16 or fp32 C and fp32 "
             "ones into an fp32 C, not " +
             TypeName(a) + " x " + TypeName(b) + " into " + TypeName(c);
  }
  if (op.transA_ && op.transB_)
    return "HVX reads A or B along K, so not both transposed";
  const std::pair<const char *, const BufferRegion *> regions[] = {
      {"A", &op.aRegion_}, {"B", &op.bRegion_}, {"C", &op.cRegion_}};
  for (const auto &[name, region] : regions) {
    const Buffer &buffer = (*region)->buffer;
    std::string operand = std::string(name) + " (" + buffer->name + ")";
    const Array<Range> &ranges = (*region)->region;
    size_t rank = ranges.size();
    if (rank < 2)
      return operand + " is not a matrix";
    for (size_t i = 0; i + 2 < rank; ++i) {
      if (!is_one(ranges[i]->extent))
        return operand + " spans more than one matrix of its leading "
                         "dimensions";
    }
    if (integer && region == &op.bRegion_)
      continue;
    Array<PrimExpr> strides = GetBufferStrides(buffer);
    if (!is_one(strides[rank - 1]) || !strides[rank - 2].as<IntImmNode>())
      return operand + "'s rows are not contiguous, a constant stride apart";
  }
  return "";
}

// Why HVX cannot run the block-scaled GEMM `op`, or "": an integer GEMM into
// an fp32 C, with fp32 scales for A and fp16 ones for B. The group sizes are
// checked as the GEMM lowers (GemmHVXBlockScaled).
std::string HVXBlockScaledRejection(const GemmBlockScaledNode &op) {
  std::string why = HVXIntRejection(op, DataType::Float(32));
  if (!why.empty())
    return why;
  if (op.b_->dtype.is_int() && op.a_->dtype != DataType::Int(16))
    return "HVX weighs the groups of a signed B with the sums of an int16 A, "
           "not " +
           TypeName(op.a_->dtype);
  DataType sfa = op.sfaRegion_->buffer->dtype;
  DataType sfb = op.sfbRegion_->buffer->dtype;
  if (sfa != DataType::Float(32) || sfb != DataType::Float(16))
    return "HVX scales A by fp32 and B by fp16 factors, not " + TypeName(sfa) +
           " and " + TypeName(sfb);
  return "";
}

// Why HMX cannot run the block-scaled GEMM `op`, or "": an fp16 A in VTCM by a
// 4-bit B in the panel layout, its rows along K (transpose_B), into an fp16 or
// fp32 C, with fp32 factors for the rows of A and fp16 ones for B. HVX
// dequantizes B into fp16 weight tiles for HMX; the group sizes are checked as
// the GEMM lowers (GemmHMXBlockScaled).
std::string HMXBlockScaledRejection(const GemmBlockScaledNode &op) {
  const DataType fp16 = DataType::Float(16);
  DataType a = op.a_->dtype, b = op.b_->dtype, c = op.c_->dtype;
  if (a != fp16 || !(b == DataType::Int(4) || b == DataType::UInt(4)) ||
      !(c == fp16 || c == DataType::Float(32)))
    return "HMX multiplies an fp16 A by a 4-bit B into an fp16 or fp32 C, "
           "not " +
           TypeName(a) + " x " + TypeName(b) + " into " + TypeName(c);
  if (op.m_ % kTile || op.n_ % kTile || op.k_ % kTile)
    return "HMX needs M, N and K multiples of 32, not " +
           std::to_string(op.m_) + " x " + std::to_string(op.n_) + " x " +
           std::to_string(op.k_);
  if (!IsVTCMBuffer(op.a_))
    return "HMX reads A (" + op.a_->name + ") from VTCM";
  if (!IsVTCMBuffer(op.c_))
    return "HMX writes C (" + op.c_->name + ") to VTCM";
  std::string why = PanelRejection(op);
  if (!why.empty())
    return why;
  DataType sfa = op.sfaRegion_->buffer->dtype;
  DataType sfb = op.sfbRegion_->buffer->dtype;
  if (sfa != DataType::Float(32) || sfb != fp16)
    return "HMX scales the rows of A by fp32 and B by fp16 factors, not " +
           TypeName(sfa) + " and " + TypeName(sfb);
  return "";
}

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)block_size;
    (void)target;
    if (HMXRejection(op).empty())
      return String(kHexagonHMX);
    return HVXRejection(op).empty() ? String(kHexagonHVX)
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

// A block-scaled GEMM of an fp16 A runs on HMX, any other on HVX. It has no
// scalar fallback: one that unit cannot run fails.
String SelectHexagonBlockScaledInst(const GemmBlockScaled &op, int block_size,
                                    const Target &target) {
  (void)block_size;
  (void)target;
  bool hmx = op->a_->dtype == DataType::Float(16);
  std::string why = hmx ? hexagon::HMXBlockScaledRejection(*op.get())
                        : hexagon::HVXBlockScaledRejection(*op.get());
  ICHECK(why.empty()) << "T.gemm_blockscaled(" << op->a_->name << ", "
                      << op->b_->name << ", " << op->c_->name
                      << ") cannot run on Hexagon: " << why;
  return String(hmx ? hexagon::kHexagonHMXBlockScaled
                    : hexagon::kHexagonHVXBlockScaled);
}

bool RegisterHexagonGemm() {
  RegisterGemmImpl(GemmImpl{
      "hexagon.Gemm",
      MatchHexagonGemmTarget,
      hexagon::Gemm::SelectInst,
      hexagon::Gemm::ComputeWarpPartition,
      hexagon::Gemm::ReuseExistingSharedLayout,
  });
  RegisterGemmBlockScaledImpl(GemmBlockScaledImpl{
      "hexagon.GemmBlockScaled",
      MatchHexagonGemmTarget,
      SelectHexagonBlockScaledInst,
  });
  return true;
}

const bool hexagon_gemm_registered = RegisterHexagonGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // GemmHVX reports why a GEMM is not on HMX, and the scalar fallback why
  // it is on neither.
  refl::GlobalDef()
      .def("tl.hexagon.GemmHMXRejection",
           [](ObjectRef gemm) {
             const auto *node = gemm.as<GemmNode>();
             ICHECK(node) << "expected a tl.Gemm, got " << gemm->GetTypeKey();
             return String(hexagon::HMXRejection(*node));
           })
      .def("tl.hexagon.GemmHVXRejection", [](ObjectRef gemm) {
        const auto *node = gemm.as<GemmNode>();
        ICHECK(node) << "expected a tl.Gemm, got " << gemm->GetTypeKey();
        return String(hexagon::HVXRejection(*node));
      });
}

} // namespace tl
} // namespace tvm
