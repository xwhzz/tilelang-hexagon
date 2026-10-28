/*!
 * \file tl/hexagon/op/utils.h
 * \brief Helpers shared by the Hexagon tile-op lowerings.
 */

#ifndef TVM_TL_HEXAGON_OP_UTILS_H_
#define TVM_TL_HEXAGON_OP_UTILS_H_

#include "op/operator.h"

#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>

namespace tvm {
namespace tl {
namespace hexagon {

/*!
 * \brief The storage scope of VTCM buffers that every program of a launch
 * shares (T.alloc_global(..., scope="global.vtcm")): device-global memory
 * placed in VTCM, as in TVM's Hexagon backend.
 */
constexpr const char *kGridVTCMScope = "global.vtcm";

/*! \brief Whether `buffer` is in VTCM shared by the programs of a launch. */
bool IsGridVTCMBuffer(const tirx::Buffer &buffer);

/*!
 * \brief Whether `buffer` is in VTCM: a program's shared buffer or fragment,
 * or a buffer the programs of a launch share.
 */
bool IsVTCMBuffer(const tirx::Buffer &buffer);

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

/*!
 * \brief Whether the elements of `buffer` sit at their row-major positions:
 * it has no layout, or one that maps every index there (a fragment one thread
 * holds whole).
 */
bool RowMajor(const tirx::Buffer &buffer, const LayoutMap &layout_map);

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_UTILS_H_
