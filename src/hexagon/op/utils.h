/*!
 * \file tl/hexagon/op/utils.h
 * \brief Region helpers shared by the Hexagon copy and DMA lowerings.
 */

#ifndef TVM_TL_HEXAGON_OP_UTILS_H_
#define TVM_TL_HEXAGON_OP_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>

namespace tvm {
namespace tl {
namespace hexagon {

/*! \brief Explicit buffer strides, or compact row-major strides if none. */
ffi::Array<PrimExpr> GetBufferStrides(const tirx::Buffer &buffer);

/*!
 * \brief A tvm_access_ptr to the first element of `region`, plus `offset`
 * elements, covering `extent` elements.
 *
 * `strides` are the buffer's strides (GetBufferStrides). The offset includes
 * the region minimum of every dimension, so a sub-matrix of a larger tensor
 * starts at its own first element.
 */
PrimExpr MakeAccessPtrAtRegion(const tirx::Buffer &buffer,
                               const ffi::Array<Range> &region,
                               const ffi::Array<PrimExpr> &strides, int rw_mask,
                               const PrimExpr &extent, int64_t offset = 0);

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_UTILS_H_
