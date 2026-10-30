/*!
 * \file tl/hexagon/transform/mark_dma_only_buffers.cc
 * \brief Transfers of the DDR buffers only DMA moves skip cache maintenance.
 *
 * DMA reads and writes DDR, past the DSP's caches, so the runtime keeps the
 * two in step around each transfer (dma.h): before a load it writes the
 * source's lines back, and around a store it writes the destination's lines
 * back and then drops them, about a cycle every 16 bytes moved. A buffer that
 * T.annotate_dma_only declares is moved by DMA alone, in every kernel that
 * reaches it, so the DSP holds none of its lines: its transfers use
 * TL_HEXAGON_DMA_NO_CACHE. The pass checks this kernel's side of that: a use
 * of the buffer other than as a transfer's DDR endpoint is an error.
 *
 * Runs after LowerTileOp, which lowers T.dma_copy to the submits.
 */

#include "hexagon/op/builtin.h"
#include "hexagon/op/dma.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The block annotation of T.annotate_dma_only: the buffers' data vars.
constexpr const char *kDMAOnly = "tl.hexagon_dma_only";
// dma.h's tl_hexagon_dma_direction.
constexpr int64_t kDDRToVTCM = 0;
constexpr int64_t kVTCMToDDR = 1;
constexpr int64_t kNoCache = 2;

class DMAOnlyCollector : public StmtVisitor {
public:
  std::unordered_set<const VarNode *> vars;

private:
  void VisitStmt_(const SBlockNode *op) final {
    if (auto declared = op->annotations.Get(kDMAOnly))
      for (const Var &var : Downcast<Array<Var>>(declared.value()))
        vars.insert(var.get());
    StmtVisitor::VisitStmt_(op);
  }
};

class DMAOnlyMarker : public StmtExprMutator {
public:
  explicit DMAOnlyMarker(std::unordered_set<const VarNode *> vars)
      : vars_(std::move(vars)) {}

private:
  // `use` is the expression or statement that reaches the buffer.
  [[noreturn]] static void Reject(const VarNode *var, const ObjectRef &use) {
    LOG(FATAL) << "T.annotate_dma_only: `" << var->name_hint
               << "` is read or written other than by T.dma_copy, in " << use
               << "; a buffer declared DMA-only must move by DMA alone";
    __builtin_unreachable();
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    if (!block->annotations.count(kDMAOnly))
      return block;
    auto *n = block.CopyOnWrite();
    n->annotations.erase(kDMAOnly);
    return block;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    const auto *direction = op->op.same_as(hexagon_dma_submit())
                                ? op->args[6].as<IntImmNode>()
                                : nullptr;
    if (direction == nullptr ||
        (direction->value != kDDRToVTCM && direction->value != kVTCMToDDR)) {
      // A pointer to the buffer passed on, such as tvm_access_ptr's.
      for (const PrimExpr &arg : op->args)
        if (const auto *var = arg.as<VarNode>(); var && vars_.count(var))
          Reject(var, GetRef<Call>(op));
      return StmtExprMutator::VisitExpr_(op);
    }
    // (dst, src, dst_stride, src_stride, row_bytes, rows, direction): a
    // load's DDR side is its source, a store's its destination.
    size_t ddr = direction->value == kDDRToVTCM ? 1 : 0;
    if (!vars_.count(hexagon::DMAPointedVar(op->args[ddr])))
      return StmtExprMutator::VisitExpr_(op);
    Array<PrimExpr> args;
    for (size_t i = 0; i < op->args.size(); ++i)
      args.push_back(i == ddr ? op->args[i] : VisitExpr(op->args[i]));
    args.Set(6, IntImm(direction->dtype, kNoCache));
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (vars_.count(op))
      Reject(op, GetRef<Var>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (vars_.count(op->buffer->data.get()))
      Reject(op->buffer->data.get(), GetRef<BufferLoad>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (vars_.count(op->buffer->data.get()))
      Reject(op->buffer->data.get(), GetRef<BufferStore>(op));
    return StmtExprMutator::VisitStmt_(op);
  }

  std::unordered_set<const VarNode *> vars_;
};

} // namespace

tvm::transform::Pass MarkDMAOnlyBuffers() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    DMAOnlyCollector collector;
    collector(f->body);
    if (collector.vars.empty())
      return f;
    std::unordered_set<const VarNode *> params;
    for (const auto &kv : f->buffer_map)
      if (kv.second.scope() == "global")
        params.insert(kv.second->data.get());
    for (const VarNode *var : collector.vars)
      if (!params.count(var))
        LOG(FATAL) << "T.annotate_dma_only takes the kernel's DDR buffer "
                      "parameters; `"
                   << var->name_hint << "` is not one";
    f.CopyOnWrite()->body = DMAOnlyMarker(std::move(collector.vars))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.MarkDMAOnlyBuffers", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.MarkDMAOnlyBuffers",
                        MarkDMAOnlyBuffers);
}

} // namespace tl
} // namespace tvm
