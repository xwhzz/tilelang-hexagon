/*!
 * \file tl/hexagon/transform/mark_exclusive_buffers.cc
 * \brief Give the VTCM buffers that share no memory pointers of their own.
 *
 * The buffers of a tile program are byte offsets into one VTCM arena, so to
 * LLVM they are all the same object. The software pipeliner groups memory
 * accesses by underlying object and keeps a loop's iterations apart when a
 * store may reach a later iteration's load of the same object. A map that
 * reads one buffer and writes another then runs one iteration at a time: on
 * the v79 device, exp(x - 1) over a 64x1024 fp16 tile took 101.7 us, and
 * 41.2 us once each buffer was its own object.
 *
 * Runs after MergeSharedMemoryAllocations. A buffer whose bytes no other
 * buffer of its arena overlaps gets its pointer through
 * tl_exclusive (common.h), a call LLVM treats as a fresh allocation; a
 * buffer that reuses memory keeps the plain offset.
 */

#include "arith/ir_visitor_with_analyzer.h"
#include "hexagon/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// A buffer pointer bound to a byte offset into an arena.
struct Slice {
  const VarNode *arena;
  int64_t offset;
  int64_t bytes = -1; // unknown
};

// The bytes of `buffer`, or -1 when its shape is not constant.
int64_t Bytes(const Buffer &buffer) {
  int64_t elements = 1;
  for (const PrimExpr &dim : buffer->shape) {
    const auto *n = dim.as<IntImmNode>();
    if (!n)
      return -1;
    elements *= n->value;
  }
  return elements * buffer->dtype.bytes();
}

// The arena slices of a function, and the bytes each one's accesses reach:
// buffer views (declared, loaded or stored) and tvm_access_ptr ranges, bounded
// under the enclosing loops.
class SliceCollector : public arith::IRVisitorWithAnalyzer {
public:
  std::unordered_map<const VarNode *, Slice> slices;

  void Finish() {
    for (auto &[var, slice] : slices) {
      auto it = reach_.find(var);
      slice.bytes = it == reach_.end() ? -1 : it->second;
    }
  }

private:
  using IRVisitorWithAnalyzer::VisitExpr_;
  using IRVisitorWithAnalyzer::VisitStmt_;

  void VisitStmt_(const BindNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(builtin::handle_add_byte_offset())) {
      const auto *arena = call->args[0].as<VarNode>();
      const auto *offset = call->args[1].as<IntImmNode>();
      if (arena && offset)
        slices[op->var.get()] = {arena, offset->value};
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }
  void VisitStmt_(const DeclBufferNode *op) final {
    View(op->buffer);
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    View(op->buffer);
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    View(op->buffer);
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // (type_annotation, data, offset, extent, rw_mask), in elements.
      const auto *data = op->args[1].as<VarNode>();
      if (data) {
        int64_t end =
            analyzer_.const_int_bound(op->args[2] + op->args[3])->max_value;
        Reach(data, end == arith::ConstIntBound::kPosInf
                        ? -1
                        : end * op->args[0].dtype().bytes());
      }
    }
    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void View(const Buffer &buffer) {
    int64_t bytes = Bytes(buffer);
    if (!buffer->strides.empty() || !is_zero(buffer->elem_offset))
      bytes = -1;
    Reach(buffer->data.get(), bytes);
  }
  // Records that a slice's accesses reach `bytes`; -1 (unknown) sticks.
  void Reach(const VarNode *var, int64_t bytes) {
    auto [it, fresh] = reach_.emplace(var, bytes);
    if (!fresh && it->second >= 0)
      it->second = bytes < 0 ? -1 : std::max(it->second, bytes);
  }

  std::unordered_map<const VarNode *, int64_t> reach_;
};

class ExclusiveMarker : public StmtExprMutator {
public:
  explicit ExclusiveMarker(
      const std::unordered_map<const VarNode *, Slice> &slices) {
    for (const auto &[var, slice] : slices) {
      if (slice.bytes <= 0)
        continue;
      bool alone = true;
      for (const auto &[other, s] : slices) {
        if (other == var || s.arena != slice.arena)
          continue;
        // An unknown extent may overlap anything after its offset.
        int64_t end = s.bytes < 0 ? INT64_MAX : s.offset + s.bytes;
        if (s.offset < slice.offset + slice.bytes && slice.offset < end) {
          alone = false;
          break;
        }
      }
      if (alone)
        exclusive_.push_back(var);
    }
  }

private:
  Stmt VisitStmt_(const BindNode *op) final {
    Bind bind = Downcast<Bind>(StmtExprMutator::VisitStmt_(op));
    if (std::find(exclusive_.begin(), exclusive_.end(), op->var.get()) !=
        exclusive_.end()) {
      bind.CopyOnWrite()->value =
          Call(DataType::Handle(), hexagon_exclusive_pointer(), {bind->value});
    }
    return bind;
  }

  std::vector<const VarNode *> exclusive_;
};

} // namespace

namespace transform {

tvm::transform::Pass MarkExclusiveBuffers() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    SliceCollector collector;
    collector(func->body);
    collector.Finish();
    func.CopyOnWrite()->body = ExclusiveMarker(collector.slices)(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.hexagon.MarkExclusiveBuffers", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.MarkExclusiveBuffers",
                        MarkExclusiveBuffers);
}

} // namespace transform
} // namespace tl
} // namespace tvm
