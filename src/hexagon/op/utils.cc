/*!
 * \file tl/hexagon/op/utils.cc
 * \brief Helpers shared by the Hexagon tile-op lowerings.
 */

#include "utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace hexagon {

using namespace tirx;
using namespace ffi;

Array<PrimExpr> GetBufferStrides(const Buffer &buffer) {
  if (!buffer->strides.empty()) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size())
        << "Explicit buffer strides must match the buffer rank";
    return buffer->strides;
  }

  Array<PrimExpr> strides;
  strides.resize(buffer->shape.size());
  PrimExpr stride = make_const(buffer->shape.back().dtype(), 1);
  for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
    strides.Set(i, stride);
    stride = stride * buffer->shape[i];
  }
  return strides;
}

PrimExpr MakeAccessPtrAtRegion(const Buffer &buffer, const Array<Range> &region,
                               const Array<PrimExpr> &strides, int rw_mask,
                               const PrimExpr &extent, int64_t offset) {
  ICHECK_EQ(region.size(), buffer->shape.size());
  ICHECK_EQ(strides.size(), buffer->shape.size());
  PrimExpr first = buffer->elem_offset;
  for (size_t i = 0; i < region.size(); ++i)
    first = first + region[i]->min * strides[i];
  if (offset != 0)
    first = first + IntImm(first.dtype(), offset);

  PrimExpr ptype = tirx::TypeAnnotation(buffer->dtype);
  Array<PrimExpr> args{ptype, buffer->data, first, extent,
                       IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
}

bool RowMajor(const Buffer &buffer, const LayoutMap &layout_map) {
  Optional<Layout> layout = layout_map.Get(buffer);
  if (!layout)
    return true;
  if (const auto *fragment = layout.value().as<FragmentNode>()) {
    if (!is_one(fragment->ThreadExtent()) ||
        !is_one(fragment->ReplicateExtent()))
      return false;
  }
  PrimExpr row_major = make_zero(DataType::Int(32));
  Array<PrimExpr> shape = layout.value()->InputShape();
  arith::Analyzer analyzer;
  for (size_t i = 0; i < shape.size(); ++i) {
    row_major = row_major * shape[i] + InputPlaceholder(i);
    analyzer.Bind(InputPlaceholder(i), Range(0, shape[i]));
  }
  return analyzer.CanProveEqual(layout.value()->GetLinearizedForwardIndex(),
                                row_major);
}

} // namespace hexagon
} // namespace tl
} // namespace tvm
