/*!
 * \file tl/hexagon/transform/vectorize_map_tail.cc
 * \brief Vectorize the full-register prefix of a pointwise parallel map.
 */
#include "vectorize_map_tail.h"
#include "hexagon/op/builtin.h"
#include "hexagon/target_utils.h"
#include "transform/loop_partition.h"

#include <tvm/arith/pattern.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::hexagon {

using namespace tirx;

ffi::Optional<Stmt> VectorizeMapTail(const For &loop,
                                     const arith::Analyzer *analyzer,
                                     const LayoutMap &layouts,
                                     const Target &target) {
  const auto *store = loop->body.as<BufferStoreNode>();
  const auto *extent = as_const_int(loop->extent);
  if (!store || store->predicate || !extent || !loop->HasTrivialStep() ||
      !loop->annotations.empty() || loop->thread_binding.defined())
    return std::nullopt;

  DataType type = store->buffer->dtype;
  if (!type.is_scalar() || type.bits() < 8 || type.bits() > 32 ||
      !(type.is_int() || type.is_uint() || type.is_float()))
    return std::nullopt;
  int lanes = MinVectorBits(target) / type.bits();
  const auto *div = store->value.as<DivNode>();
  bool group_div = type == DataType::Float(32) && div &&
                   div->a.as<BufferLoadNode>() && div->b.as<BufferLoadNode>() &&
                   lanes == 32 && *extent >= 128 && *extent % 128 == 0 &&
                   MaxVectorLoadBits(target, true) == 1024;
  if (lanes < 2 || *extent <= lanes || (!group_div && *extent % lanes == 0) ||
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>("tirx.disable_vectorize", Bool(false))
          .value())
    return std::nullopt;
  // Partitioning erases user scheduling annotations. Carry eligibility from
  // the unconstrained parallel-map path while preserving its vector width.
  if (group_div) {
    For ordinary = PragmaUnrollLoop(VectorizeLoop(loop, analyzer, layouts));
    ordinary.CopyOnWrite()->annotations.Set(attr::kHexagonGroupDiv, Integer(1));
    return ordinary;
  }

  // Require the same physical element offset for every access. This also
  // admits in-place maps and excludes shifted reads of the destination.
  auto offset =
      [&](const Buffer &buffer,
          const ffi::Array<PrimExpr> &indices) -> ffi::Optional<PrimExpr> {
    if (buffer->dtype != type || buffer.scope() != "global" ||
        layouts.count(buffer))
      return std::nullopt;
    auto physical = buffer.OffsetOf(indices);
    if (physical.size() != 1 || SideEffect(physical[0]) > CallEffectKind::kPure)
      return std::nullopt;
    auto coeff = arith::DetectLinearEquation(physical[0], {loop->loop_var});
    if (coeff.size() != 2 || !is_one(coeff[0]))
      return std::nullopt;
    return coeff[1];
  };
  auto dst = offset(store->buffer, store->indices);
  if (!dst)
    return std::nullopt;
  static const auto effects =
      Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  static const auto vectorizable =
      Op::GetAttrMap<TVectorizable>("TVectorizable");
  bool legal = true;
  bool uses_vector_intrinsic = false;
  PreOrderVisit(store->value, [&](const ffi::ObjectRef &node) {
    if (!legal)
      return false;
    if (const auto *load = node.as<BufferLoadNode>()) {
      auto src = offset(load->buffer, load->indices);
      legal =
          !load->predicate && src && ExprDeepEqual()(dst.value(), src.value());
      return false; // The address was checked separately.
    }
    auto expr = Downcast<PrimExpr>(node);
    if (auto *call = node.as<CallNode>()) {
      // Generic TIR math can be marked vectorizable and still only have a
      // scalar lowering on Hexagon. Its HVX builtins register vector support.
      auto op = call->op.as<Op>();
      legal =
          op && op.value()->name.find("tl.hexagon_") == 0 &&
          vectorizable.get(op.value(), false) && effects.count(op.value()) &&
          effects[op.value()]->value == static_cast<int>(CallEffectKind::kPure);
      uses_vector_intrinsic = true;
    } else {
      legal = node.as<FloatImmNode>() || node.as<IntImmNode>() ||
              (node.as<VarNode>() && !node.same_as(loop->loop_var)) ||
              node.as<AddNode>() || node.as<SubNode>() || node.as<MulNode>();
    }
    legal = legal && expr.dtype() == type;
    return legal;
  });
  if (!legal)
    return std::nullopt;

  // Short arithmetic maps can run faster as scalar loops on v73. Vector
  // math intrinsics benefit from splitting even a single full register.
  constexpr int kMinV73ArithmeticElements = 256;
  if (!uses_vector_intrinsic && *extent < kMinV73ArithmeticElements &&
      HexagonArchVersion(target) == 73)
    return std::nullopt;

  For prefix = loop;
  int64_t remainder = *extent % lanes;
  PrimExpr length = make_const(loop->extent.dtype(), *extent - remainder);
  prefix.CopyOnWrite()->extent = length;
  int width = GetVectorizeSize(prefix, analyzer, layouts);
  if (width < lanes)
    return std::nullopt;
  prefix = PragmaUnrollLoop(VectorizeLoop(prefix, analyzer, layouts, width));
  Var tail = loop->loop_var.copy_with_suffix("_tail");
  Stmt body =
      Substitute(loop->body, {{loop->loop_var, loop->min + length + tail}});
  return SeqStmt(
      {prefix, For(tail, make_zero(tail.dtype()),
                   make_const(tail.dtype(), remainder), ForKind::kSerial, body,
                   std::nullopt, {}, std::nullopt, loop->span)});
}

} // namespace tvm::tl::hexagon
