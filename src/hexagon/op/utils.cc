/*!
 * \file tl/hexagon/op/utils.cc
 * \brief Region helpers shared by the Hexagon copy and DMA lowerings.
 */

#include "utils.h"

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
                               const PrimExpr &extent) {
  ICHECK_EQ(region.size(), buffer->shape.size());
  ICHECK_EQ(strides.size(), buffer->shape.size());
  PrimExpr offset = buffer->elem_offset;
  for (size_t i = 0; i < region.size(); ++i)
    offset = offset + region[i]->min * strides[i];

  PrimExpr ptype = tirx::TypeAnnotation(buffer->dtype);
  Array<PrimExpr> args{ptype, buffer->data, offset, extent,
                       IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
}

} // namespace hexagon
} // namespace tl
} // namespace tvm
