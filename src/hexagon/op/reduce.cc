/*!
 * \file tl/hexagon/op/reduce.cc
 * \brief Hexagon implementation of tl.reduce.
 *
 * One HVX thread reduces in place: the source and destination are shared
 * buffers or fragments, which are both VTCM, or locals. Float sums accumulate
 * in fp32 and integers in the destination type; the result rounds once. Max
 * and min are exact, so an fp16 source keeps them in fp16 unless an fp32
 * destination value joins in.
 *
 * Reducing the innermost axis runs each output's row through a vector
 * accumulator of whole HVX registers, which a lane tree (tl_hvx_lane_*)
 * finishes, and a scalar loop takes the tail. Along any other axis the outputs
 * are contiguous instead: a block of them accumulates in registers while the
 * loop walks the reduced axis.
 */

#include "op/reduce.h"

#include "backend/common/op/reduce.h"
#include "backend/common/target_utils.h"
#include "hexagon/op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

// One HVX register.
constexpr int kVectorBits = 1024;

bool InPlace(const Buffer &buffer) {
  return IsSharedBuffer(buffer) || IsFragmentBuffer(buffer) ||
         IsLocalBuffer(buffer, /*allow_var=*/true);
}

bool Supported(DataType t) {
  return (t.is_float() && (t.bits() == 16 || t.bits() == 32)) ||
         ((t.is_int() || t.is_uint()) && t.bits() >= 8 && t.bits() <= 32);
}

PrimExpr As(DataType t, PrimExpr e) {
  return e.dtype() == t ? e : Cast(t, std::move(e));
}

DataType AccType(const ReduceOpNode &op) {
  DataType dst = op.dst->dtype;
  if (!dst.is_float())
    return dst;
  bool exact = !op.type->IsSum() && !op.type->IsAbsSum();
  if (exact && op.src->dtype == DataType::Float(16) &&
      (op.clear || dst == DataType::Float(16)))
    return DataType::Float(16);
  return DataType::Float(32);
}

Stmt Loop(const Var &var, PrimExpr extent, ForKind kind, Stmt body) {
  return For(var, make_zero(DataType::Int(32)), cast(DataType::Int(32), extent),
             kind, std::move(body));
}

class ReduceLowering {
public:
  explicit ReduceLowering(const ReduceOpNode &op)
      : op_(op), ndim_(static_cast<int>(op.srcRegion_->region.size())),
        src_t_(op.src->dtype), dst_t_(op.dst->dtype), acc_t_(AccType(op)) {
    // The vectorized loops fill whole registers with their narrowest type.
    int bits = std::min(src_t_.bits(), acc_t_.bits());
    if (!Innermost())
      bits = std::min(bits, dst_t_.bits());
    lanes_ = kVectorBits / bits;
    acc_ = decl_buffer({lanes_}, acc_t_, "acc", "local");
    res_ = decl_buffer({1}, acc_t_, "res", "local");
  }

  Stmt Lower() {
    for (int d = 0; d < ndim_; ++d)
      vars_.emplace_back("i" + std::to_string(d), DataType::Int(32));
    Stmt body = Innermost() ? RowReduce() : ColumnReduce();
    // Loops over the outputs: every axis but the reduced one, and in a column
    // reduction the innermost axis too, which ColumnReduce walks itself.
    for (int d = ndim_ - 1; d >= 0; --d) {
      if (d == op_.dim || (!Innermost() && d == ndim_ - 1))
        continue;
      body = Loop(vars_[d], Extent(d), ForKind::kSerial, body);
    }
    return SeqStmt({AllocBuffer(acc_), AllocBuffer(res_), body});
  }

private:
  bool Innermost() const { return op_.dim == ndim_ - 1; }
  PrimExpr Extent(int d) const { return op_.srcRegion_->region[d]->extent; }

  // Reduces the innermost axis of one output's row.
  Stmt RowReduce() {
    PrimExpr extent = Extent(op_.dim);
    PrimExpr blocks = floordiv(extent, lanes_);
    PrimExpr full = blocks * lanes_;
    Var k0("k0", DataType::Int(32)), k("k", DataType::Int(32));
    Array<Stmt> seq;
    if (is_zero(blocks)) {
      // Shorter than a register: the scalar loop below does it all.
      seq.push_back(BufferStore(res_, Init(), {0}));
    } else {
      seq.push_back(Lanes([&](const Var &) { return Init(); }, acc_));
      seq.push_back(Loop(k0, blocks, ForKind::kSerial,
                         Lanes(
                             [&](const Var &v) {
                               return Accumulate(BufferLoad(acc_, {v}),
                                                 Source(k0 * lanes_ + v));
                             },
                             acc_)));
      seq.push_back(BufferStore(
          res_,
          Call(acc_t_, LaneOp(), {BufferLoad(acc_, {Ramp(0, 1, lanes_)})}),
          {0}));
    }
    seq.push_back(Loop(
        k, extent - full, ForKind::kSerial,
        BufferStore(res_, Accumulate(BufferLoad(res_, {0}), Source(full + k)),
                    {0})));
    seq.push_back(Store({}, BufferLoad(res_, {0})));
    return SeqStmt::Flatten(seq);
  }

  // Reduces an outer axis for one row of outputs along the innermost axis.
  Stmt ColumnReduce() {
    int last = ndim_ - 1;
    PrimExpr reduced = Extent(op_.dim);
    PrimExpr extent = Extent(last);
    PrimExpr blocks = floordiv(extent, lanes_);
    PrimExpr full = blocks * lanes_;

    // Blocks of whole registers.
    Var j0("j0", DataType::Int(32)), k("k", DataType::Int(32));
    auto in_block = [&](const Var &v) { return j0 * lanes_ + v; };
    Array<Stmt> block;
    block.push_back(Lanes(
        [&](const Var &v) { return Start({{last, in_block(v)}}); }, acc_));
    block.push_back(Loop(k, reduced, ForKind::kSerial,
                         Lanes(
                             [&](const Var &v) {
                               return Accumulate(
                                   BufferLoad(acc_, {v}),
                                   Source(k, {{last, in_block(v)}}));
                             },
                             acc_)));
    Var v("v", DataType::Int(32));
    block.push_back(Loop(v, lanes_, ForKind::kVectorized,
                         Store({{last, in_block(v)}}, BufferLoad(acc_, {v}))));

    // The tail, one output at a time.
    Var j("j", DataType::Int(32)), kt("k", DataType::Int(32));
    std::vector<std::pair<int, PrimExpr>> at{{last, full + j}};
    Array<Stmt> tail;
    tail.push_back(BufferStore(res_, Start(at), {0}));
    tail.push_back(Loop(
        kt, reduced, ForKind::kSerial,
        BufferStore(res_, Accumulate(BufferLoad(res_, {0}), Source(kt, at)),
                    {0})));
    tail.push_back(Store(at, BufferLoad(res_, {0})));

    return SeqStmt(
        {Loop(j0, blocks, ForKind::kSerial, SeqStmt::Flatten(block)),
         Loop(j, extent - full, ForKind::kSerial, SeqStmt::Flatten(tail))});
  }

  // `acc[v] = value(v)` for every lane v, as a vectorized loop.
  template <typename F> Stmt Lanes(F value, const Buffer &acc) const {
    Var v("v", DataType::Int(32));
    return Loop(v, lanes_, ForKind::kVectorized,
                BufferStore(acc, value(v), {v}));
  }

  // The source element at `reduced` on the reduced axis, and at the loop
  // variables or `at` elsewhere.
  PrimExpr Source(PrimExpr reduced,
                  const std::vector<std::pair<int, PrimExpr>> &at = {}) const {
    Array<PrimExpr> indices;
    for (int d = 0; d < ndim_; ++d) {
      PrimExpr i = d == op_.dim ? reduced : Index(d, at);
      indices.push_back(op_.srcRegion_->region[d]->min + i);
    }
    return BufferLoad(op_.src, indices);
  }

  // The destination indices of the output at the loop variables, or at `at`.
  Array<PrimExpr>
  Output(const std::vector<std::pair<int, PrimExpr>> &at) const {
    bool keepdim = static_cast<int>(op_.dstRegion_->region.size()) == ndim_;
    Array<PrimExpr> indices;
    for (int d = 0; d < ndim_; ++d) {
      if (d == op_.dim && !keepdim)
        continue;
      indices.push_back(d == op_.dim ? make_zero(DataType::Int(32))
                                     : Index(d, at));
    }
    for (size_t d = 0; d < indices.size(); ++d)
      indices.Set(d, op_.dstRegion_->region[d]->min + indices[d]);
    return indices;
  }

  PrimExpr Index(int d, const std::vector<std::pair<int, PrimExpr>> &at) const {
    for (const auto &[axis, value] : at) {
      if (axis == d)
        return value;
    }
    return vars_[d];
  }

  // Where a column accumulation starts: the identity, or the destination.
  PrimExpr Start(const std::vector<std::pair<int, PrimExpr>> &at) const {
    if (op_.clear)
      return Init();
    return As(acc_t_, BufferLoad(op_.dst, Output(at)));
  }

  // Stores a result; a row result combines with the destination unless
  // cleared, while a column accumulation started from it.
  Stmt Store(const std::vector<std::pair<int, PrimExpr>> &at,
             PrimExpr value) const {
    Array<PrimExpr> indices = Output(at);
    if (!op_.clear && Innermost())
      value = Combine(As(acc_t_, BufferLoad(op_.dst, indices)), value);
    return BufferStore(op_.dst, As(dst_t_, value), indices);
  }

  PrimExpr Init() const {
    return As(acc_t_, backend::reduce::MakeInitValue(op_));
  }

  PrimExpr Abs(PrimExpr x) const {
    if (x.dtype().is_float())
      return Call(x.dtype(), hexagon_fabs(), {x});
    if (x.dtype().is_uint())
      return x;
    return Max(x, make_zero(x.dtype()) - x);
  }

  // An accumulator with one source element.
  PrimExpr Accumulate(PrimExpr acc, PrimExpr x) const {
    x = As(acc_t_, std::move(x));
    if (op_.type->IsAbsSum() || op_.type->IsAbsMax())
      x = Abs(x);
    return Combine(std::move(acc), std::move(x));
  }

  // Two partial results.
  PrimExpr Combine(PrimExpr a, PrimExpr b) const {
    const ReduceType &type = op_.type;
    if (type->IsSum() || type->IsAbsSum())
      return a + b;
    if (type->IsMax() || type->IsAbsMax())
      return Max(a, b);
    if (type->IsMin())
      return Min(a, b);
    if (type->IsBitAnd())
      return a & b;
    if (type->IsBitOr())
      return a | b;
    ICHECK(type->IsBitXor());
    return a ^ b;
  }

  // The lane tree that finishes a vector accumulator.
  const Op &LaneOp() const {
    const ReduceType &type = op_.type;
    if (type->IsSum() || type->IsAbsSum())
      return hexagon_lane_sum();
    if (type->IsMax() || type->IsAbsMax())
      return hexagon_lane_max();
    if (type->IsMin())
      return hexagon_lane_min();
    if (type->IsBitAnd())
      return hexagon_lane_and();
    if (type->IsBitOr())
      return hexagon_lane_or();
    return hexagon_lane_xor();
  }

  const ReduceOpNode &op_;
  int ndim_;
  DataType src_t_, dst_t_, acc_t_;
  int lanes_ = 1;
  Buffer acc_, res_;
  std::vector<Var> vars_;
};

struct Reduce {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)lower_args;
    (void)analyzer;
    if (!InPlace(op.src) || !InPlace(op.dst)) {
      LOG(FATAL) << "Hexagon reduce reads and writes shared, fragment or "
                    "local buffers, got `"
                 << op.src.scope() << "` and `" << op.dst.scope() << "`";
    }
    if (!Supported(op.src->dtype) || !Supported(op.dst->dtype)) {
      LOG(FATAL) << "Hexagon reduce supports float16, float32 and 8 to 32-bit "
                    "integers, got "
                 << op.src->dtype << " to " << op.dst->dtype;
    }
    if (op.nan_propagate) {
      LOG(FATAL) << "Hexagon reduce does not propagate NaN: HVX max and min "
                    "return the other operand";
    }
    return ReduceLowering(op).Lower();
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonReduceTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonReduce() {
  RegisterReduceImpl(ReduceImpl{
      "hexagon.Reduce",
      MatchHexagonReduceTarget,
      hexagon::Reduce::Lower,
  });
  return true;
}

const bool hexagon_reduce_registered = RegisterHexagonReduce();

} // namespace

} // namespace tl
} // namespace tvm
