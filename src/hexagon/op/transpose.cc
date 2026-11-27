/*!
 * \file tl/hexagon/op/transpose.cc
 * \brief Transpose disjoint VTCM tiles through their logical regions.
 */
#include "backend/common/op/transpose.h"
#include "hexagon/target_utils.h"

namespace tvm::tl {
namespace {

Stmt LowerHexagonTranspose(const TransposeNode &op, const LowerArgs &args,
                           arith::Analyzer *analyzer) {
  ICHECK(IsSharedBuffer(op.src) && IsSharedBuffer(op.dst))
      << "Hexagon T.transpose requires shared source and destination buffers";
  ICHECK(!op.src->data.same_as(op.dst->data))
      << "Hexagon T.transpose requires distinct shared buffers";
  ICHECK(op.HasTransposedShape(analyzer))
      << "Hexagon T.transpose requires region shapes (..., M, N) and (..., N, "
         "M)";
  return backend::Transpose::Lower(op, args, analyzer);
}

const bool registered = [] {
  RegisterTransposeImpl(
      {"hexagon.Transpose", TargetIsHexagon, LowerHexagonTranspose});
  return true;
}();

} // namespace
} // namespace tvm::tl
