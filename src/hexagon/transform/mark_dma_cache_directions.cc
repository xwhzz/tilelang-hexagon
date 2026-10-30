/*!
 * \file tl/hexagon/transform/mark_dma_cache_directions.cc
 * \brief The cache maintenance a DDR buffer's declarations let its DMA skip.
 *
 * DMA reads and writes DDR, past the DSP's caches, so the runtime keeps the
 * two in step around each transfer (dma.h): before a load it writes the
 * source's lines back, and around a store it writes the destination's lines
 * back and then drops them, about a cycle every 16 bytes moved. Two
 * declarations spare some of that:
 *
 * - A buffer that T.annotate_dma_only declares is moved by DMA alone, in
 *   every kernel that reaches it, so the DSP holds none of its lines: its
 *   loads use TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN and its stores
 *   TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED. A use of the buffer other than as a
 *   transfer's DDR endpoint is an error.
 * - A buffer that T.annotate_write_through declares is written back right
 *   after each of the DSP's writes (InsertWriteThroughWritebacks), so it holds
 *   no dirty line: its loads use TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN. Its
 *   stores keep their maintenance, as the DSP may hold clean lines of it.
 *
 * Both directions bypass the L2, as the other transfers do. Runs after
 * LowerTileOp, which lowers T.dma_copy to the submits.
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

// The block annotations of T.annotate_dma_only and T.annotate_write_through:
// the buffers' data vars.
constexpr const char *kDMAOnly = "tl.hexagon_dma_only";
constexpr const char *kWriteThrough = "tl.hexagon_write_through";
// dma.h's tl_hexagon_dma_direction.
constexpr int64_t kDDRToVTCM = 0;
constexpr int64_t kVTCMToDDR = 1;
constexpr int64_t kDDRToVTCMUnwritten = 3;
constexpr int64_t kVTCMToDDRUncached = 4;

class DeclarationCollector : public StmtVisitor {
public:
  std::unordered_set<const VarNode *> dma_only, write_through;

private:
  void VisitStmt_(const SBlockNode *op) final {
    for (auto [key, vars] : {std::pair{kDMAOnly, &dma_only},
                             std::pair{kWriteThrough, &write_through}})
      if (auto declared = op->annotations.Get(key))
        for (const Var &var : Downcast<Array<Var>>(declared.value()))
          vars->insert(var.get());
    StmtVisitor::VisitStmt_(op);
  }
};

class DirectionMarker : public StmtExprMutator {
public:
  DirectionMarker(std::unordered_set<const VarNode *> dma_only,
                  std::unordered_set<const VarNode *> write_through)
      : dma_only_(std::move(dma_only)),
        write_through_(std::move(write_through)) {}

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
    if (!block->annotations.count(kDMAOnly) &&
        !block->annotations.count(kWriteThrough))
      return block;
    auto *n = block.CopyOnWrite();
    n->annotations.erase(kDMAOnly);
    n->annotations.erase(kWriteThrough);
    return block;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    const auto *direction = op->op.same_as(hexagon_dma_submit())
                                ? op->args[6].as<IntImmNode>()
                                : nullptr;
    if (direction == nullptr ||
        (direction->value != kDDRToVTCM && direction->value != kVTCMToDDR)) {
      // A pointer to a DMA-only buffer passed on, such as tvm_access_ptr's.
      for (const PrimExpr &arg : op->args)
        if (const auto *var = arg.as<VarNode>(); var && dma_only_.count(var))
          Reject(var, GetRef<Call>(op));
      return StmtExprMutator::VisitExpr_(op);
    }
    // (dst, src, dst_stride, src_stride, row_bytes, rows, direction): a
    // load's DDR side is its source, a store's its destination.
    bool load = direction->value == kDDRToVTCM;
    size_t ddr = load ? 1 : 0;
    const VarNode *buffer = hexagon::DMAPointedVar(op->args[ddr]);
    int64_t marked = direction->value;
    if (dma_only_.count(buffer))
      marked = load ? kDDRToVTCMUnwritten : kVTCMToDDRUncached;
    else if (load && write_through_.count(buffer))
      marked = kDDRToVTCMUnwritten;
    if (marked == direction->value)
      return StmtExprMutator::VisitExpr_(op);
    Array<PrimExpr> args;
    for (size_t i = 0; i < op->args.size(); ++i)
      args.push_back(i == ddr ? op->args[i] : VisitExpr(op->args[i]));
    args.Set(6, IntImm(direction->dtype, marked));
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (dma_only_.count(op))
      Reject(op, GetRef<Var>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (dma_only_.count(op->buffer->data.get()))
      Reject(op->buffer->data.get(), GetRef<BufferLoad>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (dma_only_.count(op->buffer->data.get()))
      Reject(op->buffer->data.get(), GetRef<BufferStore>(op));
    return StmtExprMutator::VisitStmt_(op);
  }

  std::unordered_set<const VarNode *> dma_only_, write_through_;
};

} // namespace

tvm::transform::Pass MarkDMACacheDirections() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    DeclarationCollector collector;
    collector(f->body);
    if (collector.dma_only.empty() && collector.write_through.empty())
      return f;
    std::unordered_set<const VarNode *> params;
    for (const auto &kv : f->buffer_map)
      if (kv.second.scope() == "global")
        params.insert(kv.second->data.get());
    for (const VarNode *var : collector.dma_only) {
      if (!params.count(var))
        LOG(FATAL) << "T.annotate_dma_only takes the kernel's DDR buffer "
                      "parameters; `"
                   << var->name_hint << "` is not one";
      if (collector.write_through.count(var))
        LOG(FATAL) << "`" << var->name_hint
                   << "` is declared both DMA-only and write-through; a "
                      "DMA-only buffer has no writes of the DSP's to write "
                      "back";
    }
    f.CopyOnWrite()->body =
        DirectionMarker(std::move(collector.dma_only),
                        std::move(collector.write_through))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.MarkDMACacheDirections",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.MarkDMACacheDirections",
                        MarkDMACacheDirections);
}

} // namespace tl
} // namespace tvm
