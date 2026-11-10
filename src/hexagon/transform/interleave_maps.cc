/*!
 * \file tl/hexagon/transform/interleave_maps.cc
 * \brief Scheduling hints for heavy maps and simple v79 half maps.
 *
 * A vectorized map whose elements compute an HVX math function (exp, sigmoid,
 * a division, ...) runs a chain of a few dozen dependent instructions per
 * vector. LLVM software-pipelines a loop only up to an initiation interval it
 * bounds (-pipeliner-max-mii), which such a chain can exceed: SwiGLU's
 * x * sigmoid(x) * y needs 47 cycles, against a bound of 27, and its loop then
 * runs one chain at a time. Unrolled by two, an iteration interleaves two
 * independent chains: on v79 SwiGLU over 128 x 128 fp32 went from 126 to 64
 * cycles a vector, and four chains gained nothing more.
 *
 * Marks such loops, innermost and storing one vector an iteration, for
 * the codegen, which prints them unrolled by two (`#pragma clang loop
 * unroll_count`). Unrolling keeps the iterations in order, so any loop may.
 * A separate, conservative v79 half64 multiply/square pattern disables software
 * pipelining; other arithmetic, aliasing and partial maps retain their
 * schedule.
 */

#include "hexagon/op/builtin.h"
#include "hexagon/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsMathFunction(const CallNode *call) {
  // Built on first use: the Ops are registered by static initializers.
  static const std::vector<Op> functions = {
      hexagon_exp(),       hexagon_exp2(),         hexagon_log(),
      hexagon_log2(),      hexagon_sqrt(),         hexagon_rsqrt(),
      hexagon_tanh(),      hexagon_sigmoid(),      hexagon_erf(),
      hexagon_exp_fast(),  hexagon_exp2_fast(),    hexagon_log_fast(),
      hexagon_log2_fast(), hexagon_sqrt_fast(),    hexagon_rsqrt_fast(),
      hexagon_tanh_fast(), hexagon_sigmoid_fast(), hexagon_erf_fast(),
      hexagon_div_fast()};
  for (const Op &op : functions)
    if (call->op.same_as(op))
      return true;
  return false;
}

// One iteration of a heavy map: no loop, one vector stored, and an HVX math
// function or a float division of vectors on the way.
bool IsHeavyMapIteration(const Stmt &body) {
  int stores = 0;
  bool loop = false, heavy = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (node.as<ForNode>()) {
      loop = true;
    } else if (const auto *store = node.as<BufferStoreNode>()) {
      stores += store->value.dtype().lanes() > 1;
    } else if (const auto *call = node.as<CallNode>()) {
      heavy |= call->dtype.lanes() > 1 && IsMathFunction(call);
    } else if (const auto *div = node.as<DivNode>()) {
      heavy |= div->dtype.lanes() > 1 && div->dtype.is_float();
    }
  });
  return !loop && stores == 1 && heavy;
}

struct ArenaSlice {
  const VarNode *arena;
  int64_t offset;
  bool exclusive;
};
using ArenaSlices = std::unordered_map<const VarNode *, ArenaSlice>;

// Keep MarkExclusiveBuffers' proof, including the arena behind the marker.
// Different buffer variables alone are insufficient: storage reuse gives
// distinct variables the same arena offset.
ArenaSlices CollectArenaSlices(const Stmt &body) {
  ArenaSlices slices;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *bind = node.as<BindNode>();
    if (!bind)
      return;
    const auto *call = bind->value.as<CallNode>();
    bool exclusive = call && call->op.same_as(hexagon_exclusive_pointer()) &&
                     call->args.size() == 1;
    if (exclusive)
      call = call->args[0].as<CallNode>();
    if (!call || !call->op.same_as(builtin::handle_add_byte_offset()) ||
        call->args.size() != 2)
      return;
    const auto *arena = call->args[0].as<VarNode>();
    const auto *offset = call->args[1].as<IntImmNode>();
    if (arena && offset && offset->value >= 0 && offset->value % 128 == 0)
      slices.emplace(bind->var.get(),
                     ArenaSlice{arena, offset->value, exclusive});
  });
  return slices;
}

bool IsInt(const PrimExpr &value, int64_t expected) {
  const auto *n = value.as<IntImmNode>();
  return n && n->value == expected;
}

// A deliberately small scheduling pattern, not algebraic simplification.
// On v79, software-pipelining these half64 maps inhibits a cheaper schedule;
// add/affine/two-input maps need the existing pipeliner. Forward only lexical
// local SSA copies for recognition, leaving the emitted math/casts untouched.
class SimpleHalfMap {
public:
  SimpleHalfMap(const For &loop, const ArenaSlices &slices)
      : loop_(loop), slices_(slices) {}

  bool Matches() {
    const auto *n = loop_->extent.as<IntImmNode>();
    if (!IsInt(loop_->min, 0) ||
        (loop_->step.defined() && !IsInt(loop_->step.value(), 1)) || !n ||
        n->value < 2 || n->value > std::numeric_limits<int64_t>::max() / 128)
      return false;
    elements_ = n->value * 64;
    if (!ReadBody(loop_->body) || !output_ || used_.size() != locals_.size())
      return false;

    PrimExpr value = output_value_;
    bool widened = false;
    if (const auto *cast = value.as<CastNode>()) {
      if (cast->dtype != DataType::Float(16, 64) ||
          cast->value.dtype() != DataType::Float(32, 64))
        return false;
      widened = true;
      value = cast->value;
    }
    const auto *mul = value.as<MulNode>();
    if (!mul)
      return false;
    const BufferLoadNode *input = nullptr;
    if (!widened && IsHalfConstant(mul->b)) {
      input = Input(mul->a, false);
    } else if (!widened && IsHalfConstant(mul->a)) {
      input = Input(mul->b, false);
    } else {
      input = Input(mul->a, widened);
      const auto *other = Input(mul->b, widened);
      if (!input || !other || !input->buffer->data.same_as(other->buffer->data))
        return false;
    }
    if (!input || input->buffer->data.same_as(output_->buffer->data))
      return false;
    auto src = slices_.find(input->buffer->data.get());
    auto dst = slices_.find(output_->buffer->data.get());
    if (src == slices_.end() || dst == slices_.end() ||
        src->second.arena != dst->second.arena ||
        !(src->second.exclusive || dst->second.exclusive))
      return false;
    // Subtract nonnegative offsets instead of adding potentially large ones.
    int64_t distance = src->second.offset >= dst->second.offset
                           ? src->second.offset - dst->second.offset
                           : dst->second.offset - src->second.offset;
    return distance >= elements_ * 2;
  }

private:
  using Dependencies = std::unordered_set<const VarNode *>;
  struct Local {
    Buffer buffer;
    Optional<PrimExpr> value;
    Dependencies dependencies;
  };

  bool FlatBuffer(const Buffer &buffer, int64_t elements) const {
    return buffer->shape.size() == 1 && IsInt(buffer->shape[0], elements) &&
           buffer->strides.empty() && IsInt(buffer->elem_offset, 0);
  }

  bool VectorIndex(const Array<PrimExpr> &indices, bool local) const {
    if (indices.size() != 1)
      return false;
    const auto *ramp = indices[0].as<RampNode>();
    if (!ramp || !IsInt(ramp->lanes, 64) || !IsInt(ramp->stride, 1))
      return false;
    if (local)
      return IsInt(ramp->base, 0);
    const auto *base = ramp->base.as<MulNode>();
    return base && ((base->a.same_as(loop_->loop_var) && IsInt(base->b, 64)) ||
                    (base->b.same_as(loop_->loop_var) && IsInt(base->a, 64)));
  }

  bool SharedAccess(const Buffer &buffer,
                    const Array<PrimExpr> &indices) const {
    return buffer.scope() == "shared.dyn" &&
           buffer->dtype == DataType::Float(16) &&
           FlatBuffer(buffer, elements_) && VectorIndex(indices, false);
  }

  static bool IsHalfConstant(const PrimExpr &value) {
    const auto *broadcast = value.as<BroadcastNode>();
    if (!broadcast || !IsInt(broadcast->lanes, 64))
      return false;
    const auto *constant = broadcast->value.as<FloatImmNode>();
    return constant && constant->dtype == DataType::Float(16) &&
           std::isfinite(constant->value);
  }

  const BufferLoadNode *Input(PrimExpr value, bool widened) const {
    if (widened) {
      const auto *cast = value.as<CastNode>();
      if (!cast || cast->dtype != DataType::Float(32, 64) ||
          cast->value.dtype() != DataType::Float(16, 64))
        return nullptr;
      value = cast->value;
    }
    const auto *load = value.as<BufferLoadNode>();
    return load && !load->predicate && SharedAccess(load->buffer, load->indices)
               ? load
               : nullptr;
  }

  Optional<PrimExpr> Resolve(const PrimExpr &value, Dependencies *used) const {
    if (const auto *load = value.as<BufferLoadNode>()) {
      if (load->predicate)
        return std::nullopt;
      if (auto it = locals_.find(load->buffer->data.get());
          it != locals_.end()) {
        const Local &local = it->second;
        if (!local.buffer.same_as(load->buffer) || !local.value ||
            !VectorIndex(load->indices, true))
          return std::nullopt;
        used->insert(it->first);
        used->insert(local.dependencies.begin(), local.dependencies.end());
        return local.value;
      }
      return SharedAccess(load->buffer, load->indices)
                 ? Optional<PrimExpr>(value)
                 : std::nullopt;
    }
    if (const auto *cast = value.as<CastNode>()) {
      if (!((cast->dtype == DataType::Float(16, 64) &&
             cast->value.dtype() == DataType::Float(32, 64)) ||
            (cast->dtype == DataType::Float(32, 64) &&
             cast->value.dtype() == DataType::Float(16, 64))))
        return std::nullopt;
      if (auto operand = Resolve(cast->value, used))
        return Cast(cast->dtype, operand.value());
    } else if (const auto *mul = value.as<MulNode>()) {
      auto a = Resolve(mul->a, used);
      auto b = Resolve(mul->b, used);
      if (a && b)
        return Mul(a.value(), b.value());
    } else if (IsHalfConstant(value)) {
      return value;
    }
    return std::nullopt;
  }

  bool ReadBody(const Stmt &body) {
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &stmt : seq->seq)
        if (!ReadBody(stmt))
          return false;
      return true;
    }
    if (const auto *attr = body.as<AttrStmtNode>())
      return attr->attr_key == tl::attr::kLexicalAllocScope &&
             IsInt(attr->value, 1) && ReadBody(attr->body);
    if (output_)
      return false;
    if (const auto *alloc = body.as<AllocBufferNode>()) {
      const Buffer &buffer = alloc->buffer;
      return alloc->annotations.empty() && buffer.scope() == "local" &&
             (buffer->dtype == DataType::Float(16) ||
              buffer->dtype == DataType::Float(32)) &&
             FlatBuffer(buffer, 64) &&
             locals_
                 .emplace(buffer->data.get(), Local{buffer, std::nullopt, {}})
                 .second;
    }
    const auto *store = body.as<BufferStoreNode>();
    if (!store || store->predicate)
      return false;
    Dependencies used;
    auto value = Resolve(store->value, &used);
    if (!value)
      return false;
    if (auto it = locals_.find(store->buffer->data.get());
        it != locals_.end()) {
      Local &local = it->second;
      if (!local.buffer.same_as(store->buffer) || local.value ||
          !VectorIndex(store->indices, true) ||
          value.value().dtype() != local.buffer->dtype.with_lanes(64))
        return false;
      local.value = value;
      local.dependencies = std::move(used);
      return true;
    }
    if (!SharedAccess(store->buffer, store->indices) ||
        value.value().dtype() != DataType::Float(16, 64))
      return false;
    output_ = store;
    output_value_ = value.value();
    used_ = std::move(used);
    return true;
  }

  For loop_;
  const ArenaSlices &slices_;
  int64_t elements_ = 0;
  std::unordered_map<const VarNode *, Local> locals_;
  const BufferStoreNode *output_ = nullptr;
  PrimExpr output_value_;
  Dependencies used_;
};

class MapInterleaver : public StmtExprMutator {
public:
  explicit MapInterleaver(const PrimFunc &func) {
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      simple_maps_ = TargetIsHexagon(target.value()) &&
                     HexagonArchVersion(target.value()) == 79 &&
                     target.value()
                             ->GetAttr<Integer>("min_vector_bits", Integer(0))
                             .value()
                             ->value == 1024 &&
                     target.value()
                             ->GetAttr<Integer>("max_vector_bits", Integer(0))
                             .value()
                             ->value == 1024;
    }
    if (simple_maps_)
      slices_ = CollectArenaSlices(func->body);
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    // The loops over the vectors of a map are serial, or unrolled kind (which
    // the codegen prints as plain loops).
    if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled)
      return loop;
    if (const auto *n = loop->extent.as<IntImmNode>(); n && n->value < 2)
      return loop;
    if (IsHeavyMapIteration(loop->body)) {
      loop.CopyOnWrite()->annotations.Set(attr::kHexagonUnrollCount,
                                          Integer(2));
    } else if (simple_maps_ && loop->annotations.empty() &&
               SimpleHalfMap(loop, slices_).Matches()) {
      loop.CopyOnWrite()->annotations.Set(
          attr::kHexagonDisableSoftwarePipelining, Integer(1));
    }
    return loop;
  }

  bool simple_maps_ = false;
  ArenaSlices slices_;
};

} // namespace

namespace transform {

tvm::transform::Pass InterleaveMaps() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = MapInterleaver(func)(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.InterleaveMaps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.InterleaveMaps", InterleaveMaps);
}

} // namespace transform
} // namespace tl
} // namespace tvm
