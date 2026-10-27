/*!
 * \file tl/hexagon/layout/crouton.h
 * \brief Crouton: the VTCM layout of HMX operands, as a layout object.
 *
 * A Crouton tile is a 32x32 fp16 block stored as interleaved row pairs:
 * element (r, c) of the tile sits at (r / 2) * 64 + c * 2 + r % 2, so one HVX
 * register holds one row pair. A matrix of tiles orders them by rows ([rt, ct],
 * HMX activations and outputs) or by columns ([ct, rt], HMX weights), and
 * leading dimensions stack whole matrices.
 *
 * The matrix is the buffer's last two dimensions, swapped when `transposed`
 * (a transposed GEMM operand keeps its storage shape). The physical shape is
 * [tiles, 16 row pairs, 64 lanes], tiles of all matrices in order, so that a
 * loop in physical order nests by tile and by register. Lowerings recognize
 * the layout by its type and read the tile order from it.
 */
#ifndef TVM_TL_HEXAGON_LAYOUT_CROUTON_H_
#define TVM_TL_HEXAGON_LAYOUT_CROUTON_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {

constexpr int64_t kCroutonTile = 32;
constexpr int64_t kCroutonTileElems = kCroutonTile * kCroutonTile;

class CroutonLayoutNode : public LayoutNode {
public:
  // Default-constructible, so that reflection can rebuild one (serialization).
  CroutonLayoutNode() = default;
  CroutonLayoutNode(ffi::Array<PrimExpr> shape, bool column_tiles,
                    bool transposed);

  // Tiles ordered by the matrix's columns (weights), else by its rows.
  bool column_tiles = false;
  // The matrix is the buffer's last two dimensions swapped.
  bool transposed = false;

  // The matrix's rows and columns, and its tiles along each.
  int64_t Rows() const;
  int64_t Cols() const;
  int64_t RowTiles() const { return Rows() / kCroutonTile; }
  int64_t ColTiles() const { return Cols() / kCroutonTile; }
  // The buffer axes of the matrix's rows and columns.
  int RowAxis() const;
  int ColAxis() const;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.hexagon.CroutonLayout",
                                    CroutonLayoutNode, LayoutNode);
};

class CroutonLayout : public Layout {
public:
  TVM_DLL CroutonLayout(ffi::Array<PrimExpr> shape, bool column_tiles,
                        bool transposed);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CroutonLayout, Layout,
                                             CroutonLayoutNode);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_LAYOUT_CROUTON_H_
