/*!
 * \file tl/hexagon/op/fill.cc
 * \brief Hexagon implementation for tl.fill lowering (e.g. T.clear).
 *
 * Reuses the generic SIMT-loop fill (same as the CPU backend); hexagon-clang
 * auto-vectorizes the init loop onto HVX.
 */

#include "op/fill.h"
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace hexagon {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    LOG(FATAL) << "Hexagon fill only supports local and global buffers, but got "
               << "dst scope `" << op.dst.scope() << "`.";
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
