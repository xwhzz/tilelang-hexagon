/*!
 * \file tl/hexagon/op/fill.cc
 * \brief Hexagon implementation for tl.fill lowering (e.g. T.clear).
 *
 * Reuses the generic SIMT-loop fill for global, local, VTCM shared buffers and
 * fragments, which one thread holds whole; the loop vectorizes onto HVX.
 */

#include "op/fill.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include "backend/common/target_utils.h"
#include "hexagon/layout/crouton.h"
#include "hexagon/op/utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace hexagon {

// A whole Crouton allocation is a dense permutation, so a uniform value may
// be written directly in storage order. Restrict reads to DDR parameters: a
// read from this VTCM allocation or an alias could observe an earlier
// iteration.
static std::optional<For> PhysicalFill(const FillNode &op, const LowerArgs &T,
                                       arith::Analyzer *analyzer) {
  auto layout = T.layout_map.Get(op.dst);
  if (!layout || !layout.value().as<CroutonLayoutNode>() ||
      !IsVTCMBuffer(op.dst))
    return std::nullopt;
  auto shape = layout.value()->InputShape();
  if (shape.size() != op.region.size())
    return std::nullopt;
  for (size_t i = 0; i < op.region.size(); ++i) {
    if (!analyzer->CanProveEqual(op.region[i]->min, 0) ||
        !analyzer->CanProveEqual(op.region[i]->extent, op.dst->shape[i]) ||
        !analyzer->CanProveEqual(op.region[i]->extent, shape[i]))
      return std::nullopt;
  }
  bool uniform = true;
  PostOrderVisit(op.value, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      auto param = T.parameter_buffers.Get(load->buffer->data);
      uniform &= param && IsGlobalBuffer(param.value());
    } else if (const auto *call = node.as<CallNode>())
      uniform &= SideEffect(GetRef<PrimExpr>(call)) <= CallEffectKind::kPure;
  });
  if (!uniform)
    return std::nullopt;
  Buffer flat = T.buffer_remap[op.dst].GetFlattenedBuffer();
  Var i("i", flat->shape[0].dtype());
  return For(i, 0, flat->shape[0], ForKind::kParallel,
             BufferStore(flat, op.value, {i}));
}

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst) ||
        IsVTCMBuffer(op.dst)) {
      auto physical = PhysicalFill(op, T, analyzer);
      auto init_loop = physical ? *physical : op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    LOG(FATAL) << "Hexagon fill does not support dst scope `" << op.dst.scope()
               << "`.";
    return Stmt();
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonFillTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonFill() {
  RegisterFillImpl(FillImpl{
      "hexagon.Fill",
      MatchHexagonFillTarget,
      hexagon::Fill::Lower,
  });
  return true;
}

const bool hexagon_fill_registered = RegisterHexagonFill();

} // namespace

} // namespace tl
} // namespace tvm
