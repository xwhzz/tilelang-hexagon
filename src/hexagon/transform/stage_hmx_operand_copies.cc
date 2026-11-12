/*!
 * \file tl/hexagon/transform/stage_hmx_operand_copies.cc
 * \brief Stage a software pipeline's copies into HMX operands through DMA.
 *
 * A software pipeline (T.Pipelined) makes its copies from DDR asynchronous,
 * which on Hexagon are DMA transfers. DMA moves row-major bytes, but a T.gemm
 * on HMX reads its A and B as Crouton tiles. So in a pipelined loop, a copy
 * from DDR into an HMX operand becomes two: one into a row-major staging
 * buffer, which the pipeline multi-buffers and transfers ahead, and one from
 * the staging buffer into the operand, an HVX pack, which the pipeline places
 * with the gemm that reads it. Copies that DMA would move slower than HVX
 * (DMAOutpacesHVX) stay single HVX copies into the operand.
 *
 * Runs before PipelinePlanning.
 */

#include "hexagon/op/dma.h"
#include "hexagon/op/gemm.h"
#include "hexagon/op/utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr int kRead = 1;
constexpr int kWrite = 2;

// The A and B of the gemms in `body` that run on HMX.
std::unordered_set<const BufferNode *> HMXOperands(const Stmt &body) {
  std::unordered_set<const BufferNode *> operands;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr)
      return;
    TileOperator op = ParseOperator(GetRef<Call>(call));
    const auto *gemm = op.defined() ? op.as<GemmNode>() : nullptr;
    if (gemm != nullptr && hexagon::HMXRejection(*gemm).empty()) {
      operands.insert(gemm->a_.get());
      operands.insert(gemm->b_.get());
    }
  });
  return operands;
}

class HMXOperandCopyStager : public StmtExprMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (!op->annotations.count("num_stages"))
      return StmtExprMutator::VisitStmt_(op);
    std::unordered_set<const BufferNode *> outer = operands_;
    for (const BufferNode *buffer : HMXOperands(op->body))
      operands_.insert(buffer);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    operands_ = std::move(outer);
    return stmt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || operands_.empty())
      return StmtExprMutator::VisitStmt_(op);
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    const auto *copy = tile_op.defined() ? tile_op.as<CopyNode>() : nullptr;
    if (copy == nullptr || !IsGlobalBuffer(copy->src) ||
        !operands_.count(copy->dst.get()) ||
        !hexagon::DMAOutpacesHVX(*copy, &analyzer_))
      return StmtExprMutator::VisitStmt_(op);
    // A row-major buffer of the copied region.
    Array<PrimExpr> shape;
    Array<Range> whole;
    for (const Range &r : copy->dst_range) {
      shape.push_back(r->extent);
      whole.push_back(
          Range::FromMinExtent(make_zero(r->extent.dtype()), r->extent));
    }
    Buffer staging = decl_buffer(shape, copy->dst->dtype,
                                 copy->dst->name + "_dma", copy->dst.scope());
    staging_[copy->dst.get()].push_back(staging);
    Call transfer(call->dtype, call->op,
                  {call->args[0], hexagon::RegionArg(staging, whole, kWrite)},
                  call->annotations, call->span);
    Call pack(call->dtype, call->op,
              {hexagon::RegionArg(staging, whole, kRead), call->args[1]}, {},
              call->span);
    return SeqStmt({Evaluate(transfer), Evaluate(pack)});
  }

  // A staging buffer is allocated with its operand.
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    Array<Buffer> allocs = block->alloc_buffers;
    bool added = false;
    for (const Buffer &buffer : op->alloc_buffers) {
      auto it = staging_.find(buffer.get());
      if (it == staging_.end())
        continue;
      for (const Buffer &staging : it->second)
        allocs.push_back(staging);
      staging_.erase(it);
      added = true;
    }
    if (!added)
      return block;
    block.CopyOnWrite()->alloc_buffers = allocs;
    return block;
  }

  arith::Analyzer analyzer_;
  std::unordered_set<const BufferNode *> operands_;
  std::unordered_map<const BufferNode *, std::vector<Buffer>> staging_;
};

} // namespace

tvm::transform::Pass StageHMXOperandCopies() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f.CopyOnWrite()->body = HMXOperandCopyStager()(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.StageHMXOperandCopies",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.StageHMXOperandCopies",
                        StageHMXOperandCopies);
}

} // namespace tl
} // namespace tvm
