/*!
 * \file tl/hexagon/transform/stage_hmx_operand_copies.cc
 * \brief Prepare the operands of the gemms on HMX: pad them to whole tiles
 * (PadHMXOperands), and stage a software pipeline's copies into them through
 * DMA (StageHMXOperandCopies).
 *
 * HMX multiplies 32 x 32 tiles. A gemm whose M, N or K is not a multiple of
 * 32 multiplies the tiles that cover it, zero-padded along K
 * (tl_hexagon_hmx_gemm), so its A and B take whole tiles of VTCM:
 * PadHMXOperands rounds their buffers' matrix dimensions up to multiples of
 * 32. So does it C's, when the gemm writes all of C (kPaddedC): HMX then
 * stores whole tiles, the padding's included, instead of C's rows and
 * columns of its last tiles. Every other access to such a buffer keeps its
 * region, so the padding is never read or written but by HMX. Runs before
 * LayoutInference, which gives the padded buffers their Crouton layouts.
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
#include "layout/layout.h"
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

// The annotation of a gemm on HMX whose C is padded to whole tiles, which it
// may write (GemmHMX).
constexpr const char *kPaddedC = "hexagon.padded_c";

// The operands to pad: the A and B buffers, allocated in `body`, that HMX
// multiplies once they are padded to whole tiles, and the C buffers of such
// gemms that they write whole; by their padded buffers. `writers` gets the
// gemms whose C is padded. A gemm with an operand of a layout of its own
// (T.annotate_layout) keeps its operands: its unit is the one their layouts
// chose.
std::unordered_map<const BufferNode *, Buffer>
PaddedOperands(const Stmt &body,
               std::unordered_set<const CallNode *> *writers) {
  std::unordered_set<const BufferNode *> allocated;
  // The data of the buffers with a layout of their own.
  std::unordered_set<const VarNode *> annotated;
  // The gemms off whole tiles that HMX runs once padded.
  std::vector<std::pair<const CallNode *, Gemm>> gemms;
  arith::Analyzer analyzer;
  auto whole = [&](const Buffer &buffer) {
    size_t n = buffer->shape.size();
    for (size_t i = n - 2; i < n; ++i) {
      if (!analyzer.CanProveEqual(floormod(buffer->shape[i], 32), 0))
        return false;
    }
    return true;
  };
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      for (const Buffer &buffer : block->alloc_buffers)
        allocated.insert(buffer.get());
      if (auto layouts = block->annotations.Get(attr::kLayoutMap)) {
        for (const auto &[key, layout] :
             Downcast<Map<ObjectRef, ObjectRef>>(layouts.value())) {
          if (const auto *var = key.as<VarNode>())
            annotated.insert(var);
          else if (const auto *buffer = key.as<BufferNode>())
            annotated.insert(buffer->data.get());
        }
      }
      return;
    }
    const auto *call = node.as<CallNode>();
    if (call == nullptr)
      return;
    TileOperator op = ParseOperator(GetRef<Call>(call));
    const auto *gemm = op.defined() ? op.as<GemmNode>() : nullptr;
    if (gemm != nullptr && hexagon::HMXRejection(*gemm, /*pad=*/true).empty())
      gemms.emplace_back(call, GetRef<Gemm>(gemm));
  });
  std::unordered_map<const BufferNode *, Buffer> padded;
  auto pad = [&](const BufferNode *buffer) {
    if (padded.count(buffer))
      return;
    Array<PrimExpr> shape = buffer->shape;
    for (size_t i = shape.size() - 2; i < shape.size(); ++i)
      shape.Set(i, analyzer.Simplify(floordiv(shape[i] + 31, 32) * 32));
    padded[buffer] =
        Buffer(buffer->data, buffer->dtype, shape, {}, buffer->elem_offset,
               buffer->name, buffer->data_alignment, buffer->offset_factor,
               buffer->buffer_type);
  };
  auto own = [&](const Buffer &buffer) {
    return allocated.count(buffer.get()) &&
           !annotated.count(buffer->data.get());
  };
  for (const auto &[call, gemm] : gemms) {
    if (!own(gemm->a_) || !own(gemm->b_))
      continue;
    for (const Buffer &buffer : {gemm->a_, gemm->b_}) {
      if (!whole(buffer))
        pad(buffer.get());
    }
    // A C that the gemm writes whole, along its matrix and at one index of
    // its leading dimensions.
    const Buffer &c = gemm->c_;
    const Array<Range> &region = gemm->cRegion_->region;
    bool all = !whole(c) && own(c);
    for (size_t i = 0; i < region.size(); ++i) {
      bool matrix = i + 2 >= region.size();
      all = all &&
            (matrix ? is_zero(region[i]->min) &&
                          analyzer.CanProveEqual(region[i]->extent, c->shape[i])
                    : is_one(region[i]->extent));
    }
    if (all) {
      pad(c.get());
      writers->insert(call);
    }
  }
  return padded;
}

// Replaces each buffer of `padded` by its padded buffer, wherever it appears.
class HMXOperandPadder : public StmtExprMutator {
public:
  HMXOperandPadder(std::unordered_map<const BufferNode *, Buffer> padded,
                   std::unordered_set<const CallNode *> writers)
      : padded_(std::move(padded)), writers_(std::move(writers)) {}

private:
  Buffer Padded(const Buffer &buffer) const {
    auto it = padded_.find(buffer.get());
    return it == padded_.end() ? buffer : it->second;
  }

  BufferRegion Padded(const BufferRegion &region) const {
    Buffer buffer = Padded(region->buffer);
    return buffer.same_as(region->buffer)
               ? region
               : BufferRegion(buffer, region->region);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!writers_.count(op))
      return call;
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set(kPaddedC, Bool(true));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    Buffer buffer = Padded(load->buffer);
    if (!buffer.same_as(load->buffer))
      load.CopyOnWrite()->buffer = buffer;
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer buffer = Padded(store->buffer);
    if (!buffer.same_as(store->buffer))
      store.CopyOnWrite()->buffer = buffer;
    return store;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto *n = block.CopyOnWrite();
    n->alloc_buffers = n->alloc_buffers.Map(
        [this](const Buffer &buffer) { return Padded(buffer); });
    n->reads = n->reads.Map(
        [this](const BufferRegion &region) { return Padded(region); });
    n->writes = n->writes.Map(
        [this](const BufferRegion &region) { return Padded(region); });
    n->match_buffers = n->match_buffers.Map([this](const MatchBufferRegion &m) {
      return MatchBufferRegion(m->buffer, Padded(m->source));
    });
    return block;
  }

  std::unordered_map<const BufferNode *, Buffer> padded_;
  std::unordered_set<const CallNode *> writers_;
};

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

tvm::transform::Pass PadHMXOperands() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    std::unordered_set<const CallNode *> writers;
    auto padded = PaddedOperands(f->body, &writers);
    if (!padded.empty())
      f.CopyOnWrite()->body =
          HMXOperandPadder(std::move(padded), std::move(writers))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.PadHMXOperands", {});
}

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
  refl::GlobalDef()
      .def("tl.hexagon.transform.PadHMXOperands", PadHMXOperands)
      .def("tl.hexagon.transform.StageHMXOperandCopies", StageHMXOperandCopies);
}

} // namespace tl
} // namespace tvm
