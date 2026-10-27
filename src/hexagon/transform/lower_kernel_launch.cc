/*!
 * \file tl/hexagon/transform/lower_kernel_launch.cc
 * \brief Mark the Hexagon launch grid before MaterializeKernelLaunch.
 *
 * MaterializeKernelLaunch turns the launch grid into plain serial loops. This
 * pass marks each grid loop with its axis so MaterializeGrid can find the grid
 * after the loop transformations, and moves the T.Kernel(num_workers=)
 * annotation from the launch block onto the outermost grid loop, which
 * outlives the block.
 */

#include "hexagon/op/builtin.h"
#include "transform/common/attr.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsGridBinding(const ForNode *loop) {
  if (loop->kind != ForKind::kThreadBinding || !loop->thread_binding)
    return false;
  std::string tag = loop->thread_binding.value()->thread_tag;
  return tag.rfind("blockIdx.", 0) == 0;
}

// Takes the worker count off this launch's block, leaving nested launches and
// the kernel body alone.
class NumWorkersExtractor : public StmtMutator {
public:
  Optional<Integer> NumWorkers() const { return num_workers_; }

private:
  Stmt VisitStmt_(const ForNode *op) final { return GetRef<Stmt>(op); }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (!IsDeviceMainBlock(op))
      return GetRef<Stmt>(op);
    auto value = op->annotations.Get(attr::kHexagonNumWorkers);
    if (!value)
      return GetRef<Stmt>(op);
    auto count = value->try_cast<Integer>();
    ICHECK(count && count.value()->value > 0)
        << attr::kHexagonNumWorkers << " must be a positive integer, got "
        << value.value();
    num_workers_ = count.value();
    SBlock block = GetRef<SBlock>(op);
    block.CopyOnWrite()->annotations.erase(attr::kHexagonNumWorkers);
    return block;
  }

  Optional<Integer> num_workers_;
};

class KernelLaunchLowerer : public StmtMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (!IsGridBinding(op))
      return StmtMutator::VisitStmt_(op);
    std::vector<const ForNode *> loops;
    Stmt body = GetRef<Stmt>(op);
    while (const auto *loop = body.as<ForNode>()) {
      if (!IsGridBinding(loop))
        break;
      loops.push_back(loop);
      body = loop->body;
    }
    NumWorkersExtractor extractor;
    body = VisitStmt(extractor(body));
    for (size_t i = loops.size(); i-- > 0;) {
      For loop = GetRef<For>(loops[i]);
      auto *node = loop.CopyOnWrite();
      node->body = body;
      node->annotations.Set(attr::kHexagonGridDim,
                            Integer(static_cast<int>(i)));
      if (i == 0 && extractor.NumWorkers())
        node->annotations.Set(attr::kHexagonNumWorkers,
                              extractor.NumWorkers().value());
      body = loop;
    }
    return body;
  }

  // A launch without grid axes runs one tile program; its worker count has
  // nothing to distribute.
  Stmt VisitStmt_(const SBlockNode *op) final {
    NumWorkersExtractor extractor;
    return extractor(StmtMutator::VisitStmt_(op));
  }
};

} // namespace

namespace transform {

tvm::transform::Pass LowerKernelLaunch() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = KernelLaunchLowerer()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.hexagon.LowerKernelLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.LowerKernelLaunch",
                        LowerKernelLaunch);
}

} // namespace transform
} // namespace tl
} // namespace tvm
