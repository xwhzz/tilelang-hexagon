/*!
 * \file tl/hexagon/op/reduce.cc
 * \brief Hexagon implementation for tl.reduce.
 *
 * Lowers a row reduction (max/sum over the last axis of a 2D row-major tile) to
 * a single call into the HVX row-reduce primitives in hvx_math.h.  This is the
 * `reduce` half of the backend's {gemm, copy, map, reduce} primitive basis; the
 * map half is the elementwise HVX vectorizer in the codegen.
 *
 * Only the shapes softmax/layernorm need are handled (2D src, reduce dim=1,
 * clear=True, fp16 src, fp32 dst); anything else errors clearly rather than
 * silently falling back to a GPU-shaped lowering Hexagon can't run.  The runtime
 * helper uses HVX for aligned row spans and scalar tails/fallbacks otherwise, so
 * non-64-multiple row widths stay correct.
 */

#include "op/reduce.h"
#include "op/utils.h" // MakeAccessPtrFromRegion

#include "backend/common/target_utils.h" // TargetIsHexagon

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    (void)T;
    Array<PrimExpr> src_extents;
    for (const auto &range : op.srcRegion_->region)
      src_extents.push_back(range->extent);
    int ndim = static_cast<int>(src_extents.size());
    ICHECK(ndim == 2 && op.dim == 1)
        << "Hexagon reduce supports only a 2D row reduction (dim=1); got ndim="
        << ndim << " dim=" << op.dim
        << ". Reshape to [rows, cols] and reduce the last axis.";
    ICHECK(op.clear)
        << "Hexagon reduce supports clear=True only; combine with a running "
           "value in scalar code (e.g. m = max(m, rowmax(S))).";
    ICHECK(op.src->dtype == DataType::Float(16))
        << "Hexagon reduce supports fp16 source only; got " << op.src->dtype;
    ICHECK(op.dst->dtype == DataType::Float(32))
        << "Hexagon reduce writes an fp32 result; declare the destination "
           "float32 (got "
        << op.dst->dtype << ").";
    ICHECK(!op.nan_propagate)
        << "Hexagon reduce does not implement nan_propagate (HVX vmax has fixed "
           "NaN handling); use the default.";
    // The HVX primitive walks contiguous full-width rows (in + i*cols), and
    // MakeAccessPtrFromRegion drops a 2D sub-region offset — so the source region
    // must span the whole buffer.  A sub-tile / column-slice / row-offset reduce
    // would silently read the wrong rows; reject it loudly instead.
    for (int d = 0; d < ndim; ++d) {
      PrimExpr zero = IntImm(op.srcRegion_->region[d]->min.dtype(), 0);
      ICHECK(analyzer->CanProveEqual(op.srcRegion_->region[d]->min, zero) &&
             analyzer->CanProveEqual(src_extents[d], op.src->shape[d]))
          << "Hexagon reduce requires a full-buffer (contiguous-row) source "
             "region; a sub-region reduce is unsupported.";
    }
    if (const auto *cols = src_extents[1].as<IntImmNode>()) {
      ICHECK_GT(cols->value, 0) << "Hexagon reduce requires a positive row width.";
    }

    const char *fn = nullptr;
    if (op.type->isMax())
      fn = "tl_hvx_rowmax_mat";
    else if (op.type->isSum())
      fn = "tl_hvx_rowsum_mat";
    else
      ICHECK(false) << "Hexagon reduce supports max/sum only.";

    PrimExpr src_ptr = MakeAccessPtrFromRegion(op.srcRegion_, 1); // read
    PrimExpr dst_ptr = MakeAccessPtrFromRegion(op.dstRegion_, 2); // write
    // primitive signature: (float* out, const __fp16* in, int rows, int n)
    Array<PrimExpr> args = {StringImm(fn), dst_ptr, src_ptr, src_extents[0],
                            src_extents[1]};
    return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonReduceTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonReduce() {
  RegisterReduceImpl(ReduceImpl{
      "hexagon.Reduce",
      MatchHexagonReduceTarget,
      hexagon::Reduce::Lower,
  });
  return true;
}

const bool hexagon_reduce_registered = RegisterHexagonReduce();

} // namespace

} // namespace tl
} // namespace tvm
