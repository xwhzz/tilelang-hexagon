/*!
 * \file tl/hexagon/layout/panel.h
 * \brief Panel: the layout of integer GEMM operands that HVX vrmpy reads.
 *
 * vrmpy multiplies the 4 bytes of each 32-bit lane of a register by 4 bytes
 * and adds the products into the lane. A register of a matrix in this layout
 * holds 32 rows, each row's lane `lane` consecutive columns of it: 4 of 8
 * bits, or 8 of 4 bits. A panel is 32 rows by every column, stored one column
 * block of `lane` columns after another, so element (r, c) sits at
 *
 *   (r / 32) * 32 * cols + (c / lane) * 32 * lane + (r % 32) * lane + c % lane.
 *
 * Within a lane the columns keep their order: two 4-bit elements share a
 * byte, the first in its low nibble. Leading dimensions stack whole matrices.
 *
 * The physical shape is [panels, column blocks, 32, lane], panels of all
 * matrices in order. Lowerings recognize the layout by its type. A parameter
 * in this layout arrives packed: the host stores it in the same order.
 */
#ifndef TVM_TL_HEXAGON_LAYOUT_PANEL_H_
#define TVM_TL_HEXAGON_LAYOUT_PANEL_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {

constexpr int64_t kPanelRows = 32;

class PanelLayoutNode : public LayoutNode {
public:
  // Default-constructible, so that reflection can rebuild one (serialization).
  PanelLayoutNode() = default;
  PanelLayoutNode(ffi::Array<PrimExpr> shape, int64_t lane);

  // Columns per 32-bit lane: 32 / element bits.
  int64_t lane = 4;

  // The matrix's rows and columns: the buffer's last two dimensions.
  int64_t Rows() const;
  int64_t Cols() const;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.hexagon.PanelLayout", PanelLayoutNode,
                                    LayoutNode);
};

class PanelLayout : public Layout {
public:
  TVM_DLL PanelLayout(ffi::Array<PrimExpr> shape, int64_t lane);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PanelLayout, Layout,
                                             PanelLayoutNode);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_LAYOUT_PANEL_H_
