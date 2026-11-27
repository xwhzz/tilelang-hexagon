/*!
 * \file tl/hexagon/op/scan.cc
 * \brief Inclusive scan over a shared tile on Hexagon.
 */
#include "op/scan.h"
#include "hexagon/target_utils.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {
namespace {

Stmt LowerCumSum(const CumSumOpNode &op, const LowerArgs &,
                 arith::Analyzer *analyzer) {
  const auto &src = op.srcRegion_->region;
  const auto &dst = op.dstRegion_->region;
  ICHECK(IsSharedBuffer(op.src) && IsSharedBuffer(op.dst))
      << "Hexagon T.cumsum requires shared source and destination buffers";
  for (const Buffer &buf : {op.src, op.dst}) {
    ICHECK(IsBufferContiguous(buf, analyzer) &&
           analyzer->CanProveEqual(buf->elem_offset, 0))
        << "Hexagon T.cumsum requires compact shared buffers with zero "
           "element offset; pass slices of the allocated buffers directly";
  }
  const DataType dtype = op.src->dtype;
  ICHECK(dtype.is_scalar() && dtype.bits() >= 8 &&
         (dtype.is_float() || dtype.is_int() || dtype.is_uint()))
      << "Hexagon T.cumsum requires a scalar numeric dtype";
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "Hexagon T.cumsum requires equal source and destination dtypes";
  ICHECK_EQ(src.size(), dst.size());
  ICHECK_GE(op.dim, 0);
  bool identical = op.src.same_as(op.dst);
  bool disjoint = false;
  Array<Var> iters;
  Array<PrimExpr> src_index, dst_index;
  for (size_t d = 0; d < src.size(); ++d) {
    ICHECK(analyzer->CanProveEqual(src[d]->extent, dst[d]->extent))
        << "Hexagon T.cumsum requires matching region shapes";
    ICHECK(
        analyzer->CanProve(src[d]->min >= 0) &&
        analyzer->CanProve(dst[d]->min >= 0) &&
        analyzer->CanProve(src[d]->extent >= 0) &&
        analyzer->CanProve(src[d]->min + src[d]->extent <= op.src->shape[d]) &&
        analyzer->CanProve(dst[d]->min + dst[d]->extent <= op.dst->shape[d]))
        << "Hexagon T.cumsum requires provably in-bounds regions";
    identical = identical && analyzer->CanProveEqual(src[d]->min, dst[d]->min);
    if (op.src.same_as(op.dst)) {
      disjoint =
          disjoint ||
          analyzer->CanProve(src[d]->min + src[d]->extent <= dst[d]->min) ||
          analyzer->CanProve(dst[d]->min + dst[d]->extent <= src[d]->min);
    }
    Var index("i", src[d]->extent.dtype());
    iters.push_back(index);
    PrimExpr offset = d == static_cast<size_t>(op.dim) && op.reverse
                          ? src[d]->extent - 1 - index
                          : PrimExpr(index);
    src_index.push_back(src[d]->min + offset);
    dst_index.push_back(dst[d]->min + offset);
  }
  ICHECK(!op.src->data.same_as(op.dst->data) ||
         (op.src.same_as(op.dst) && (identical || disjoint)))
      << "Hexagon T.cumsum requires identical or disjoint regions";

  // A row starts from its first input rather than adding zero. Every later
  // element reads its predecessor after that store has completed.
  Array<PrimExpr> previous = dst_index;
  previous.Set(op.dim, dst_index[op.dim] + (op.reverse ? 1 : -1));
  PrimExpr input = BufferLoad(op.src, src_index);
  Stmt sum =
      BufferStore(op.dst, BufferLoad(op.dst, previous) + input, dst_index);
  const PrimExpr extent = src[op.dim]->extent;
  const PrimExpr zero = make_zero(extent.dtype());
  Stmt body = For(iters[op.dim], make_const(extent.dtype(), 1),
                  max(extent - 1, zero), ForKind::kSerial, sum);
  if (!identical) {
    Stmt head = Substitute(BufferStore(op.dst, input, dst_index),
                           {{iters[op.dim], zero}});
    if (!analyzer->CanProve(extent > 0))
      head = IfThenElse(extent > 0, head);
    body = SeqStmt({head, body});
  }
  for (size_t d = src.size(); d-- > 0;) {
    if (d != static_cast<size_t>(op.dim))
      body = For(iters[d], make_zero(iters[d].dtype()), src[d]->extent,
                 ForKind::kSerial, body);
  }
  return body;
}

const bool registered = [] {
  RegisterCumSumImpl({"hexagon.CumSum", TargetIsHexagon, LowerCumSum});
  return true;
}();
} // namespace
} // namespace tvm::tl
