/*!
 * \file tl/hexagon/op/dma.cc
 * \brief Explicit Hexagon DMA copy (T.dma_copy).
 *
 * Each T.dma_copy lowers to the descriptors of one asynchronous User-DMA
 * transfer and one commit, so it is one group of the tile program's queue;
 * T.dma_wait(n) waits until at most n groups are pending. An explicit async
 * request either becomes DMA or fails: it never silently falls back to a
 * synchronous copy.
 */

#include "dma.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>

#include <algorithm>
#include <optional>
#include <string>

#include "hexagon/op/builtin.h"
#include "hexagon/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

// A region of a buffer as a matrix of `rows` x `cols` elements whose rows are
// `stride` elements apart: the last dimension, or the last two.
struct Matrix {
  PrimExpr rows, cols, stride;
  Array<PrimExpr> strides;
};

std::optional<Matrix> MatrixOf(const Buffer &buf, const Array<Range> &region,
                               arith::Analyzer *analyzer, std::string *why) {
  size_t rank = buf->shape.size();
  ICHECK(rank > 0 && region.size() == rank);
  Array<PrimExpr> strides = GetBufferStrides(buf);
  if (!analyzer->CanProveEqual(strides.back(), 1)) {
    *why = "Hexagon T.dma_copy requires unit innermost stride";
    return std::nullopt;
  }
  for (size_t i = 0; i < rank; ++i) {
    if (i + 2 < rank && !analyzer->CanProveEqual(region[i]->extent, 1)) {
      *why = "Hexagon T.dma_copy supports only 1D/2D regions";
      return std::nullopt;
    }
    if (!analyzer->CanProve(region[i]->min >= 0) ||
        !analyzer->CanProve(region[i]->min + region[i]->extent <=
                            buf->shape[i])) {
      *why = "Hexagon T.dma_copy requires provably in-bounds regions";
      return std::nullopt;
    }
  }
  PrimExpr cols = region.back()->extent;
  PrimExpr rows = rank == 1 ? PrimExpr(1) : region[rank - 2]->extent;
  PrimExpr stride = rank == 1 ? cols : strides[rank - 2];
  return Matrix{rows, cols, stride, strides};
}

int64_t PositiveConstant(const PrimExpr &e, arith::Analyzer *analyzer) {
  const auto *v = analyzer->Simplify(e).as<IntImmNode>();
  return v && v->value > 0 ? v->value : 0;
}

// Descriptor fields are 24 bits wide, and a 2D descriptor has at most 65535
// rows; larger transfers take several descriptors of one group.
constexpr int64_t kField = 0x00ffffff;
constexpr int64_t kRows = 0xffff;
// 1D chunks stay aligned for every element size.
constexpr int64_t kChunk = int64_t(1) << 23;

} // namespace

std::string DMARejection(const CopyNode &op, const LowerArgs &T,
                         arith::Analyzer *analyzer) {
  const bool load = IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst);
  const bool store = IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst);
  if (!(load || store) || op.src->dtype != op.dst->dtype ||
      op.src->dtype.lanes() != 1 || op.src->dtype.bits() % 8 != 0 ||
      T.layout_map.count(op.src) || T.layout_map.count(op.dst))
    return "Hexagon T.dma_copy requires equal scalar byte-addressable dtypes "
           "and row-major global <-> shared buffers (no Crouton layouts)";
  if (op.annotations.count("parallel_loop_layout") ||
      op.annotations.count("coalesced_width"))
    return "Hexagon T.dma_copy does not support SIMT copy hints";
  std::string why;
  auto src = MatrixOf(op.src, op.src_range, analyzer, &why);
  auto dst = src ? MatrixOf(op.dst, op.dst_range, analyzer, &why) : src;
  if (!src || !dst)
    return why;
  if (!analyzer->CanProveEqual(src->rows, dst->rows) ||
      !analyzer->CanProveEqual(src->cols, dst->cols))
    return "Hexagon T.dma_copy requires matching region shapes";
  int64_t rows = PositiveConstant(src->rows, analyzer);
  int64_t cols = PositiveConstant(src->cols, analyzer);
  int64_t ss = PositiveConstant(src->stride, analyzer);
  int64_t ds = PositiveConstant(dst->stride, analyzer);
  if (!rows || !cols || !ss || !ds)
    return "Hexagon T.dma_copy requires positive static extents and strides";
  if (ss < cols || ds < cols)
    return "Hexagon T.dma_copy requires rows that do not overlap";
  int64_t bytes = op.src->dtype.bytes();
  if (rows == 1 || (ss == cols && ds == cols))
    return "";
  if (cols * bytes > kField || ss * bytes > kField || ds * bytes > kField)
    return "Hexagon T.dma_copy rows and strides must fit the 24-bit DMA "
           "descriptor fields (16 MiB)";
  // The runtime builds the v75+ type-9 2D descriptor; older architectures
  // would need another descriptor format.
  const int arch = HexagonArchVersion(T.target);
  if (arch != 0 && arch < 75)
    return "2D T.dma_copy needs Hexagon v75 or newer; the target is hexagonv" +
           std::to_string(arch) +
           ". Copy contiguous rows with 1D transfers instead.";
  return "";
}

bool DMAOutpacesHVX(const CopyNode &op, arith::Analyzer *analyzer) {
  constexpr int64_t kMinRowBytes = 512;
  // The row length, and whether the rows of a buffer's region follow each
  // other in memory.
  const Array<Range> &region = op.src_range;
  int64_t cols = PositiveConstant(region.back()->extent, analyzer);
  int64_t rows =
      region.size() == 1
          ? 1
          : PositiveConstant(region[region.size() - 2]->extent, analyzer);
  if (!rows || !cols)
    return false;
  auto dense = [&](const Buffer &buffer) {
    Array<PrimExpr> strides = GetBufferStrides(buffer);
    return strides.size() < 2 ||
           analyzer->CanProveEqual(strides[strides.size() - 2],
                                   IntImm(DataType::Int(64), cols));
  };
  bool contiguous = rows == 1 || (dense(op.src) && dense(op.dst));
  return contiguous || cols * op.src->dtype.bytes() >= kMinRowBytes;
}

Stmt LowerDMATransfer(const CopyNode &op, const LowerArgs &T,
                      arith::Analyzer *analyzer, bool commit) {
  std::string why = DMARejection(op, T, analyzer);
  ICHECK(why.empty()) << why;
  const bool load = IsGlobalBuffer(op.src);
  std::string unused;
  Matrix src = *MatrixOf(op.src, op.src_range, analyzer, &unused);
  Matrix dst = *MatrixOf(op.dst, op.dst_range, analyzer, &unused);
  int64_t rows = PositiveConstant(src.rows, analyzer);
  int64_t cols = PositiveConstant(src.cols, analyzer);
  int64_t ss = PositiveConstant(src.stride, analyzer);
  int64_t ds = PositiveConstant(dst.stride, analyzer);
  int64_t bytes = op.src->dtype.bytes();
  auto u32 = [](int64_t n) { return IntImm(DataType::UInt(32), n); };
  PrimExpr direction = IntImm(DataType::Int(32), load ? 0 : 1);
  // One descriptor of `count` rows starting `src_offset`/`dst_offset` elements
  // into the regions. Strides and row sizes are in bytes.
  auto submit = [&](int64_t src_offset, int64_t dst_offset, int64_t count,
                    int64_t row_bytes, int64_t src_stride, int64_t dst_stride) {
    int64_t src_span = (count - 1) * src_stride + row_bytes;
    int64_t dst_span = (count - 1) * dst_stride + row_bytes;
    PrimExpr sp = MakeAccessPtrAtRegion(
        op.src, op.src_range, src.strides, 1,
        IntImm(DataType::Int(32), src_span / bytes), src_offset);
    PrimExpr dp = MakeAccessPtrAtRegion(
        op.dst, op.dst_range, dst.strides, 2,
        IntImm(DataType::Int(32), dst_span / bytes), dst_offset);
    return Evaluate(Call(DataType::Int(32), hexagon_dma_submit(),
                         {dp, sp, u32(dst_stride), u32(src_stride),
                          u32(row_bytes), u32(count), direction}));
  };
  Array<Stmt> seq;
  if (rows == 1 || (ss == cols && ds == cols)) {
    int64_t total = rows * cols * bytes;
    for (int64_t done = 0; done < total; done += kChunk) {
      int64_t n = std::min(kChunk, total - done);
      seq.push_back(submit(done / bytes, done / bytes, 1, n, n, n));
    }
  } else {
    for (int64_t row = 0; row < rows; row += kRows) {
      seq.push_back(submit(row * ss, row * ds, std::min(kRows, rows - row),
                           cols * bytes, ss * bytes, ds * bytes));
    }
  }
  if (commit)
    seq.push_back(Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {})));
  return SeqStmt::Flatten(seq);
}

namespace {

void CheckHexagonTarget(const Target &target) {
  ICHECK(TargetIsHexagon(target))
      << "T.dma_copy is supported only on Hexagon, got target "
      << target->str();
}

} // namespace

} // namespace hexagon

HexagonDmaCopy::HexagonDmaCopy(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  ObjectPtr<HexagonDmaCopyNode> node = make_object<HexagonDmaCopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  node->annotations = annotations;
  data_ = std::move(node);
}

TileOperator HexagonDmaCopyNode::Clone() const {
  auto op = make_object<HexagonDmaCopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return HexagonDmaCopy(op);
}

LayoutMap HexagonDmaCopyNode::InferLayout(const LayoutInferArgs &layout_args,
                                          InferLevel level) const {
  (void)level;
  hexagon::CheckHexagonTarget(layout_args.target);
  // DMA moves row-major bytes; it neither requires nor proposes a layout.
  return {};
}

Stmt HexagonDmaCopyNode::Lower(const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) const {
  hexagon::CheckHexagonTarget(lower_args.target);
  return hexagon::LowerDMATransfer(*this, lower_args, analyzer,
                                   /*commit=*/true);
}

TIR_REGISTER_TL_TILE_OP(HexagonDmaCopy, dma_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  HexagonDmaCopyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  // The commit and wait of a software pipeline's async copies (the target's
  // "async_copy_commit" and "async_copy_wait"): DMA groups of the queue.
  refl::GlobalDef()
      .def("tl.hexagon.AsyncCopyCommit",
           []() -> Stmt {
             return Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {}));
           })
      .def("tl.hexagon.AsyncCopyWait", [](PrimExpr pending) -> Stmt {
        return Evaluate(Call(DataType::Int(32), hexagon_dma_wait(), {pending}));
      });
}

} // namespace tl
} // namespace tvm
