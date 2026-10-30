/*!
 * \file tl/hexagon/op/dma.h
 * \brief Explicit Hexagon DMA copy (T.dma_copy).
 */

#ifndef TVM_TL_HEXAGON_OP_DMA_H_
#define TVM_TL_HEXAGON_OP_DMA_H_

#include "op/copy.h"

#include <string>

namespace tvm {
namespace tl {

class PanelLayoutNode;

namespace hexagon {

//! The panel layout (hexagon/layout/panel.h) of `buffer`, if it has one.
const PanelLayoutNode *PanelOf(const tirx::Buffer &buffer,
                               const LayoutMap &layout_map);

/*!
 * \brief Why `op` cannot be one User-DMA transfer between DDR and VTCM, or
 * an empty string: equal byte-addressable dtypes, row-major buffers, and a
 * static, in-bounds 1D or 2D region on each side. Between two buffers in the
 * panel layout, the region instead covers whole panels of one matrix, which
 * move as rows of their column blocks; between two of one Crouton layout,
 * whole tiles of one matrix, which move as runs of tiles in their order.
 */
std::string DMARejection(const CopyNode &op, const LowerArgs &T,
                         arith::Analyzer *analyzer);

/*!
 * \brief The descriptors of the transfer `op`, and with `commit` the commit
 * that makes them one group of the tile program's queue.
 */
Stmt LowerDMATransfer(const CopyNode &op, const LowerArgs &T,
                      arith::Analyzer *analyzer, bool commit);

/*!
 * \brief Whether DMA moves the copy `op` from DDR faster than HVX loads do:
 * one contiguous block, or rows of at least 512 bytes. On v79 DMA moves
 * 128-byte rows at 3 GB/s, 512-byte ones at 9 GB/s and contiguous blocks at
 * 23 GB/s, while HVX loads 128-byte rows at up to 14 GB/s. A software
 * pipeline's other copies stay HVX copies.
 */
bool DMAOutpacesHVX(const CopyNode &op, arith::Analyzer *analyzer);

/*!
 * \brief The writeback of `region` of the DDR buffer `buffer`
 * (hexagon_cache_writeback): of a Crouton buffer, the whole tiles the region
 * touches, a run of tiles at a time as a Crouton transfer moves them; of a
 * row-major one, the region's rows.
 */
Stmt LowerCacheWriteback(const tirx::Buffer &buffer, const Array<Range> &region,
                         const LowerArgs &T, arith::Analyzer *analyzer);

/*!
 * \brief The data var a pointer argument of hexagon_dma_submit addresses:
 * tvm_access_ptr's buffer, or an address_of's BufferLoad; nullptr otherwise.
 */
const tirx::VarNode *DMAPointedVar(const PrimExpr &ptr);

} // namespace hexagon

/*!
 * \brief One explicit asynchronous DDR <-> VTCM transfer.
 *
 * Parsed from tl.tileop.dma_copy. It shares the copy's regions and access
 * analysis with CopyNode, always lowers to a Hexagon User-DMA submission, and
 * is rejected on every other target.
 */
class HexagonDmaCopyNode : public CopyNode {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HexagonDmaCopy", HexagonDmaCopyNode,
                                    CopyNode);

  static void RegisterReflection() {
    reflection::ObjectDef<HexagonDmaCopyNode>();
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;

protected:
  TileOperator Clone() const override;
};

/*!
 * \brief The writeback of a DDR region's cache lines, which
 * InsertWriteThroughWritebacks puts after each write of a buffer that
 * T.annotate_write_through declares, so that DMA reads what the DSP wrote.
 *
 * Parsed from tl.tileop.cache_writeback; Hexagon only.
 */
class HexagonCacheWritebackNode : public TileOperatorNode {
public:
  tirx::Buffer buffer;
  Array<Range> region;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HexagonCacheWriteback",
                                    HexagonCacheWritebackNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    reflection::ObjectDef<HexagonCacheWritebackNode>();
  }

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const override;

  TileOperator Clone() const override;
};

class HexagonCacheWriteback : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HexagonCacheWriteback,
                                             TileOperator,
                                             HexagonCacheWritebackNode);

  /*! \brief From tl.tileop.cache_writeback's argument: the region. */
  TVM_DLL HexagonCacheWriteback(
      Array<PrimExpr> args,
      Map<String, ObjectRef> annotations = Map<String, ObjectRef>());

  static const Op &Get();
};

class HexagonDmaCopy : public Copy {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HexagonDmaCopy, Copy,
                                             HexagonDmaCopyNode);

  /*!
   * \brief Constructor from tl.tileop.dma_copy call arguments.
   * \param args args[0]/args[1] are the source/destination regions.
   * \param annotations Annotations map from the Call node.
   */
  TVM_DLL
  HexagonDmaCopy(Array<PrimExpr> args,
                 Map<String, ObjectRef> annotations = Map<String, ObjectRef>());

  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_DMA_H_
