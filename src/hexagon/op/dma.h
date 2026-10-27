/*!
 * \file tl/hexagon/op/dma.h
 * \brief Explicit Hexagon DMA copy (T.dma_copy).
 */

#ifndef TVM_TL_HEXAGON_OP_DMA_H_
#define TVM_TL_HEXAGON_OP_DMA_H_

#include "op/copy.h"

namespace tvm {
namespace tl {

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
