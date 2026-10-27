/*
 * \file codegen_hexagon.cc
 */
#include "codegen_hexagon.h"
#include "hexagon/op/builtin.h"
#include "support/check.h"

#include <tvm/ir/op.h> // Op::Get (exp intrinsic match)
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h> // UsesVar (HVX elementwise vectorizer)
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using namespace ffi;

namespace {

// A Hexagon builtin printed as a call to its runtime template function, with
// the call's arguments in order.
struct HexagonBuiltin {
  Op op;
  const char *function;
  bool uses_hmx;
};

const HexagonBuiltin *FindHexagonBuiltin(const CallNode *call) {
  // Built on first use: the Ops are registered by static initializers in
  // other translation units.
  static const std::vector<HexagonBuiltin> builtins = {
      {tl::hexagon_hmx_acquire(), "tl_hexagon_hmx_acc_acquire", true},
      {tl::hexagon_hmx_release(), "tl_hexagon_hmx_acc_release", true},
      {tl::hexagon_hmx_load_bias(), "tl_hexagon_hmx_load_bias", true},
      {tl::hexagon_hmx_clear(), "tl_hexagon_hmx_clear_acc", true},
      {tl::hexagon_hmx_mma(), "tl_hexagon_hmx_mma_atom", true},
      {tl::hexagon_hmx_convert(), "tl_hexagon_hmx_convert_acc", true},
      {tl::hexagon_hmx_store(), "tl_hexagon_hmx_store_cvt_state", true},
  };
  for (const auto &builtin : builtins) {
    if (call->op.same_as(builtin.op))
      return &builtin;
  }
  return nullptr;
}

bool IsHexagonHMXIntrinsic(const CallNode *call) {
  const HexagonBuiltin *builtin = FindHexagonBuiltin(call);
  return builtin != nullptr && builtin->uses_hmx;
}

bool IsHexagonDMAExtern(const CallNode *call) {
  if (!call->op.same_as(tirx::builtin::call_extern()) || call->args.empty())
    return false;
  const auto *name = call->args[0].as<StringImmNode>();
  return name && name->value.find("tl_hexagon_dma_") == 0;
}

bool IsHexagonAsyncCopy(const CallNode *call) {
  if (!IsHexagonDMAExtern(call))
    return false;
  const auto *name = call->args[0].as<StringImmNode>();
  return name->value == "tl_hexagon_dma_async_copy_1d" ||
         name->value == "tl_hexagon_dma_async_copy_2d";
}

bool IsDMAWait(const CallNode *call) {
  return call->op.same_as(Op::Get("tl.dma_wait"));
}

// The VTCM region's first 2 KiB hold the HMX output scales; kernel arenas
// start above them (TL_VTCM_ARENA_OFFSET in tl_templates/hexagon/vtcm.h).
constexpr size_t kVtcmArenaOffset = 2048;

// A parallel loop that MaterializeGrid made from a T.Kernel(num_workers=) grid.
bool IsWorkerLoop(const ForNode *loop) {
  return loop->kind == ForKind::kParallel &&
         loop->annotations.count(tl::attr::kHexagonNumWorkers);
}

bool IsSharedScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn" || scope == "shared.tmem";
}

size_t ConstantElements(const tirx::Buffer &buffer) {
  size_t elems = 1;
  for (const auto &dim : buffer->shape) {
    const auto *imm = dim.as<IntImmNode>();
    ICHECK(imm) << "Can only handle constant size allocation for now";
    elems *= static_cast<size_t>(imm->value);
  }
  ICHECK_GT(elems, 0) << "Can only handle constant size allocation";
  return elems;
}

// The bytes a shared buffer takes in a VTCM arena: whole 2 KiB tiles, one HMX
// Crouton tile each, which keeps every buffer HVX (128 B) aligned too.
size_t VtcmBytes(const tirx::Buffer &buffer) {
  size_t bytes = ConstantElements(buffer) * buffer->dtype.bytes();
  return (bytes + 2047) & ~size_t(2047);
}

// The VTCM arena of one tile program in `body`.
size_t VtcmArenaBytes(const Stmt &body) {
  size_t bytes = 0;
  tirx::PostOrderVisit(body, [&bytes](const ffi::ObjectRef &node) {
    const auto *alloc = node.as<AllocBufferNode>();
    if (alloc && IsSharedScope(GetPtrStorageScope(alloc->buffer->data)))
      bytes += VtcmBytes(alloc->buffer);
  });
  return bytes;
}

bool UsesHMX(const Stmt &body) {
  bool found = false;
  tirx::PostOrderVisit(body, [&found](const ffi::ObjectRef &node) {
    if (const auto *call = node.as<CallNode>())
      found |= IsHexagonHMXIntrinsic(call);
  });
  return found;
}

// Whether `body` uses the managed DMA queue outside worker loops, which own
// their queues.
bool UsesManagedDMA(const Stmt &body) {
  struct Finder : public tirx::StmtExprVisitor {
    bool found = false;
    void VisitStmt_(const ForNode *op) final {
      if (!IsWorkerLoop(op))
        tirx::StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      found |= IsHexagonAsyncCopy(op) || IsDMAWait(op);
      tirx::StmtExprVisitor::VisitExpr_(op);
    }
  } finder;
  finder(body);
  return finder.found;
}

} // namespace

CodeGenTileLangHexagon::CodeGenTileLangHexagon() {}

void CodeGenTileLangHexagon::Init(bool output_ssa) {
  preamble_stream_ << "// tilelang Hexagon (cDSP) kernel\n";
  preamble_stream_ << "#include <tl_templates/hexagon/common.h>\n";
  preamble_stream_ << "\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenTileLangHexagon::EmitPreamble_(std::ostream &os) const {
  if (uses_dma_async_)
    os << "#include <tl_templates/hexagon/dma.h>\n";
}

void CodeGenTileLangHexagon::PrintFuncPrefix(std::ostream &os) { // NOLINT(*)
  // Generated kernels are compiled as C++ by hexagon-clang++; keep the entry
  // symbol C-linkage so the (C) FastRPC skel can call it without mangling.
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n";
}

void CodeGenTileLangHexagon::PrintType(DataType t,
                                       std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      os << "half"; // backed by native __fp16 in tl_templates/hexagon/common.h
      break;
    case 32:
      os << "float";
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 128)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "int8_t";
      break;
    case 16:
      os << "int16_t";
      break;
    case 32:
      os << "int32_t";
      break;
    case 64:
      os << "int64_t";
      break;
    case 1:
      os << "int32_t";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 128)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Hexagon C type";
}

void CodeGenTileLangHexagon::AddFunction(const PrimFunc &f) {
  // Clear previous generated state and reserve keywords.
  this->InitFuncState(f);
  ReserveKeywordsAsUnique();
  vtcm_offset_ = 0;
  vtcm_capacity_ = 0;
  if (auto target = f->GetAttr<Target>(tvm::attr::kTarget)) {
    if (auto capacity = target.value()->GetAttr<Integer>("vtcm-capacity"))
      vtcm_capacity_ = static_cast<size_t>(capacity.value()->value);
  }

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenTileLangHexagon: Expect PrimFunc to have the global_symbol "
         "attribute";
  function_names_.push_back(global_symbol.value());

  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  dma_async_ = false;
  bool raw_dma = false;
  tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (IsHexagonAsyncCopy(call) || IsDMAWait(call))
        dma_async_ = true;
      else if (IsHexagonDMAExtern(call))
        raw_dma = true;
    }
  });
  dma_capacity_ = 16;
  if (auto capacity = f->GetAttr<Integer>("hexagon.dma_queue_capacity")) {
    ICHECK(capacity.value()->value > 0 && capacity.value()->value <= 256)
        << "hexagon.dma_queue_capacity must be in [1, 256]";
    dma_capacity_ = static_cast<int>(capacity.value()->value);
  }
  if (dma_async_) {
    ICHECK(!raw_dma)
        << "Hexagon async DMA cannot share a kernel with raw DMA calls";
    uses_dma_async_ = true;
  }
  // Names the runtime calls bind; reserve them before the parameters so no
  // parameter can take them.
  for (const char *name : {"tl_dma_async", "tl_status", "tl_arena"})
    name_supply_->ReserveName(name);

  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    AllocVarID(v.get());
    if (v.dtype().is_handle()) {
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    }
  }

  // The kernel entry returns an int32 status (TL_OK = 0); the skel maps a
  // nonzero status to AEE_EFAILED. LowerDeviceKernelLaunch leaves the device
  // function void, so the codegen declares the status type and emits the
  // returns itself, after the DMA drain.
  this->PrintFuncPrefix(stream);
  stream << "int32_t";
  this->PrintExtraAttrs(f, stream);
  stream << " " << static_cast<std::string>(global_symbol.value()) << "(";
  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end())
        PrintStorageScope(it->second, stream);
      CodeGenC::PrintType(GetType(v), stream);
      if (no_alias)
        PrintRestrict(v, stream);
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << GetVarID(v.get());
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  dma_in_scope_ = dma_async_ && UsesManagedDMA(f->body);
  if (dma_in_scope_)
    EmitDMAContext();
  this->PrintStmt(f->body);
  if (dma_in_scope_)
    EmitDMADrain();
  this->PrintIndent();
  this->stream << "return TL_OK;\n";
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangHexagon::EmitDMAContext() {
  // The descriptors and FIFO belong to this scope: the kernel, or one tile
  // program of a worker. Never share a queue between threads.
  PrintIndent();
  stream << "tl_hexagon_dma_async_context<" << dma_capacity_
         << "> tl_dma_async;\n";
  PrintIndent();
  stream << "if (tl_dma_async.status != TL_OK) return TL_ERR_DMA;\n";
}

void CodeGenTileLangHexagon::EmitDMADrain() {
  PrintIndent();
  stream << "if (tl_dma_async.wait(0) != TL_OK) return TL_ERR_DMA;\n";
}

void CodeGenTileLangHexagon::PrintWorkerLoop(const ForNode *op) {
  ICHECK(!in_worker_) << "Hexagon worker loops cannot nest";
  ICHECK(is_zero(op->min)) << "Hexagon worker loop `" << op->loop_var
                           << "` must start at 0";
  int64_t num_workers =
      op->annotations.Get(tl::attr::kHexagonNumWorkers)->cast<Integer>()->value;
  size_t arena_bytes = VtcmArenaBytes(op->body);
  bool outer_dma = dma_in_scope_;
  dma_in_scope_ = dma_async_ && UsesManagedDMA(op->body);

  // tl_parallel_for runs the body once per tile program, on the worker pool.
  // Each worker gets a private VTCM arena; a worker that issues HMX enables
  // it for its thread (tl_hmx_worker_scope).
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  stream << "if (int32_t tl_status = tl_parallel_for";
  if (UsesHMX(op->body))
    stream << "<tl_hmx_worker_scope>";
  stream << "(" << extent << ", " << num_workers << ", ";
  if (arena_bytes > 0)
    stream << "tl_vtcm_worker_arenas(" << arena_bytes << "u), ";
  stream << "[&](";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << AllocVarID(op->loop_var.get());
  if (arena_bytes > 0)
    stream << ", uint8_t *tl_arena";
  stream << ") -> int32_t {\n";
  int scope = BeginScope();
  in_worker_ = true;
  size_t outer_offset = vtcm_offset_;
  vtcm_offset_ = 0;
  if (dma_in_scope_)
    EmitDMAContext();
  PrintStmt(op->body);
  if (dma_in_scope_)
    EmitDMADrain();
  PrintIndent();
  stream << "return TL_OK;\n";
  vtcm_offset_ = outer_offset;
  in_worker_ = false;
  dma_in_scope_ = outer_dma;
  EndScope(scope);
  PrintIndent();
  stream << "})) return tl_status;\n";
}

void CodeGenTileLangHexagon::VisitStmt_(const AllocBufferNode *op) {
  std::string vid = AllocVarID(op->buffer->data.get());
  size_t constant_size = ConstantElements(op->buffer);

  this->PrintIndent();
  if (IsSharedScope(GetPtrStorageScope(op->buffer->data))) {
    // VTCM-backed: the HMX matrix engine reads operands from VTCM (mxmem) and
    // HVX wants its scratch there, so a `shared` tile cannot live on the stack.
    // It gets a compile-time byte offset into the tile program's VTCM arena:
    // the kernel arena (tl_vtcm_arena()) or, in a worker, the worker's own
    // arena (tl_arena). The offsets repeat for every program.
    size_t offset = vtcm_offset_;
    vtcm_offset_ += VtcmBytes(op->buffer);
    ICHECK(vtcm_capacity_ == 0 ||
           kVtcmArenaOffset + vtcm_offset_ <= vtcm_capacity_)
        << "alloc_shared total (" << kVtcmArenaOffset + vtcm_offset_
        << " bytes) exceeds the target's VTCM capacity (" << vtcm_capacity_
        << " bytes)";
    PrintType(op->buffer->dtype, stream);
    stream << "* " << vid << " = (";
    PrintType(op->buffer->dtype, stream);
    stream << "*)(" << (in_worker_ ? "tl_arena" : "tl_vtcm_arena()") << " + "
           << offset << ");\n";
    // Publish static VTCM usage, from the region base, for embedding/export
    // capacity checks. Worker arenas are sized by tl_parallel_for instead.
    if (!in_worker_) {
      this->PrintIndent();
      stream << "tl_vtcm_shared_high_water = "
             << kVtcmArenaOffset + vtcm_offset_ << "u;\n";
    }
  } else {
    // Stack-local (fragments / small scratch).
    PrintType(op->buffer->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
}

void CodeGenTileLangHexagon::VisitExpr_(const BroadcastNode *op,
                                        std::ostream &os) { // NOLINT(*)
  // Emit ((float4)(v)) — the vec_type broadcast ctor in common.h fills all
  // lanes from one value.  (Repeating the value would form a comma expression,
  // which hexagon-clang rejects under -Werror=unused-value.)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->dtype, os);
  os << ")(" << v << "))";
}

void CodeGenTileLangHexagon::VisitExpr_(const CallNode *op,
                                        std::ostream &os) { // NOLINT(*)
  if (const HexagonBuiltin *builtin = FindHexagonBuiltin(op)) {
    os << builtin->function << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i != 0)
        os << ", ";
      os << PrintExpr(op->args[i]);
    }
    os << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangHexagon::VisitStmt_(const EvaluateNode *op) { // NOLINT(*)
  const auto *call = op->value.as<CallNode>();
  if (dma_in_scope_ && call &&
      (call->op.same_as(tl::hexagon_hmx_acquire()) ||
       call->op.same_as(tl::hexagon_hmx_release()))) {
    CodeGenC::VisitStmt_(op);
    PrintIndent();
    stream << "tl_dma_async.release_hmx = "
           << (call->op.same_as(tl::hexagon_hmx_acquire())
                   ? "tl_hmx_unit_release"
                   : "nullptr")
           << ";\n";
    return;
  }
  if (call && (IsHexagonAsyncCopy(call) || IsDMAWait(call))) {
    ICHECK(dma_in_scope_);
    PrintIndent();
    stream << "if (tl_dma_async.";
    if (IsDMAWait(call)) {
      const auto *pending = call->args[0].as<IntImmNode>();
      ICHECK(pending && pending->value >= 0 && pending->value <= dma_capacity_)
          << "T.dma_wait pending must be static and within queue capacity";
      stream << "wait(" << pending->value;
    } else {
      const auto *name = call->args[0].as<StringImmNode>();
      stream << (name->value == "tl_hexagon_dma_async_copy_1d" ? "copy_1d("
                                                               : "copy_2d(");
      for (size_t i = 1; i < call->args.size(); ++i) {
        if (i != 1)
          stream << ", ";
        stream << PrintExpr(call->args[i]);
      }
    }
    stream << ") != TL_OK) return TL_ERR_DMA;\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHexagon::VisitExpr_(const MinNode *op,
                                        std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenTileLangHexagon::VisitExpr_(const MaxNode *op,
                                        std::ostream &os) { // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void
CodeGenTileLangHexagon::PrintTernaryCondExpr(const T *op, const char *compare,
                                             std::ostream &os) { // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());
  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

// ---------------------------------------------------------------------------
// HVX elementwise ("map") vectorizer.
//
// The `map` half of the {gemm, copy, map, reduce} basis.  Hexagon's scalar unit
// has no fp16, so an elementwise loop emitted as scalar C compiles to
// per-element libcalls (__extendhfsf2 et al.).  Here we recognise the clean
// innermost loop a `T.serial` elementwise nest lowers to —
//     for (j = 0; j < N; ++j)  dst[base + j] = f(src[base + j], ..., scalar[i])
// (inner `j` contiguous, N a multiple of the 64-lane HVX width, per-row values
// j-independent) — and emit one HVX vector per 64 columns via the validated
// hvx_math.h lane helpers: widen fp16->fp32, compute in fp32 (lo/hi pair = 2x32
// lanes), narrow back.  fp16<->fp32 casts are transparent (everything is fp32
// internally); j-independent subexpressions are evaluated once and splatted.
// Anything that doesn't fit falls back to CodeGenC's scalar loop.
// ---------------------------------------------------------------------------
namespace {
constexpr int kHvxF16Lanes = 64; // 1024-bit HVX register / 16-bit lane

// Does expression `e` depend on the (inner) loop var `j`?
bool UsesLoopVar(const PrimExpr &e, const tirx::Var &j) {
  return tirx::UsesVar(e, [&](const VarNode *v) { return v == j.get(); });
}

// Match an exp call in either form: the op-level intrinsic (tl.__exp and
// tirx.exp are BOTH base-e e^x — the CUDA backend lowers tl.__exp to `exp`,
// __exp10 to `exp10`; the DSL docstring claiming 2**x is wrong) or — after the
// pipeline's intrinsic lowering — the call_extern to a libm symbol (exp2f
// base-2 / expf base-e).  Sets *arg to the operand and *base_e true for base-e
// (which the emitter rescales by log2e before exp2).
bool MatchExpCall(const CallNode *call, PrimExpr *arg, bool *base_e) {
  if (call->args.size() == 1) {
    if (call->op.same_as(Op::Get("tl.__exp")) ||
        call->op.same_as(Op::Get("tirx.exp"))) {
      *arg = call->args[0];
      *base_e = true; // e^x
      return true;
    }
  }
  if (call->args.size() == 2) {
    if (const auto *s = call->args[0].as<StringImmNode>()) {
      if (s->value == "exp2f") {
        *arg = call->args[1];
        *base_e = false;
        return true;
      }
      if (s->value == "expf") {
        *arg = call->args[1];
        *base_e = true;
        return true;
      }
    }
  }
  return false;
}

// Match a sqrt call: the op-level tl.ieee_fsqrt or — after intrinsic lowering —
// the call_extern to sqrtf/sqrt.  Sets *arg to the operand.
bool MatchSqrtCall(const CallNode *call, PrimExpr *arg) {
  if (call->args.size() == 1 && call->op.same_as(Op::Get("tl.ieee_fsqrt"))) {
    *arg = call->args[0];
    return true;
  }
  if (call->args.size() == 2) {
    if (const auto *s = call->args[0].as<StringImmNode>())
      if (s->value == "sqrtf" || s->value == "sqrt") {
        *arg = call->args[1];
        return true;
      }
  }
  return false;
}

// Match the rsqrt idiom 1.0 / sqrt(arg) (what T.rsqrt lowers to: 1.0f /
// sqrtf(x)).  Emitting tl_hvx_rsqrt_vsf(arg) directly is one HVX op and more
// accurate than recip(sqrt(arg)).
bool MatchRsqrt(const DivNode *d, PrimExpr *arg) {
  const auto *num = d->a.as<FloatImmNode>();
  if (!num || num->value != 1.0)
    return false;
  const auto *call = d->b.as<CallNode>();
  return call && MatchSqrtCall(call, arg);
}
} // namespace

void CodeGenTileLangHexagon::VisitStmt_(const ForNode *op) {
  if (IsWorkerLoop(op)) {
    PrintWorkerLoop(op);
    return;
  }
  if (TryEmitHvxElementwise(op))
    return;
  CodeGenC::VisitStmt_(op);
}

bool CodeGenTileLangHexagon::AccessAligned128(const BufferNode *buf,
                                              const PrimExpr &idx,
                                              const tirx::Var &j,
                                              arith::Analyzer *ana) {
  // Only shared (VTCM) tiles are guaranteed 128-byte aligned (the merge pass
  // aligns them to 128 for Hexagon); local/global buffers are not.
  std::string scope = GetPtrStorageScope(buf->data);
  if (scope != "shared" && scope != "shared.dyn" && scope != "shared.tmem")
    return false;
  // idx = base + j with j stepping 64 (j % 64 == 0 at each chunk start), so the
  // byte address (2 bytes/elem) is 128-aligned iff base % 64 == 0.
  arith::ModularSet ms = ana->modular_set(ana->Simplify(idx - j));
  return ms->coeff % 64 == 0 && ms->base % 64 == 0;
}

bool CodeGenTileLangHexagon::TryEmitHvxElementwise(const ForNode *op) {
  // Innermost loop over a static [0, N) extent that is a multiple of the HVX
  // fp16 width.
  const auto *mn = op->min.as<IntImmNode>();
  const auto *ext = op->extent.as<IntImmNode>();
  if (!mn || mn->value != 0 || !ext)
    return false;
  int64_t N = ext->value;
  if (N <= 0 || N % kHvxF16Lanes != 0)
    return false;
  // Body must be a single elementwise BufferStore.
  Stmt body = op->body;
  if (const auto *seq = body.as<SeqStmtNode>()) {
    if (seq->seq.size() != 1)
      return false;
    body = seq->seq[0];
  }
  const auto *store = body.as<BufferStoreNode>();
  if (!store || store->indices.size() != 1)
    return false;
  // Scalar fp16 store only: this skips the half8 vector copies (lanes>1 Ramp
  // store) and fp32 elementwise (left scalar).
  DataType vt = store->value.dtype();
  if (vt.lanes() != 1 || !vt.is_float() || vt.bits() != 16)
    return false;
  PrimExpr sidx = store->indices[0];
  if (sidx.dtype().lanes() != 1)
    return false;
  tirx::Var j = op->loop_var;
  arith::Analyzer ana;
  ana.Bind(j, Range::FromMinExtent(IntImm(j.dtype(), 0), op->extent));
  // Store must be unit-stride in j (idx - j independent of j), so the original
  // index doubles as the chunk base address with j stepping by 64.
  if (UsesLoopVar(ana.Simplify(sidx - j), j))
    return false;
  // Supportability probe: a dry run of the emitter (emit=false) — side-effect
  // free (no stream / name / var allocation), so it's safe before AllocVarID(j)
  // and it shares the single op-set with the real emit below.
  if (!EmitHvxExpr(store->value, j, &ana, /*emit=*/false))
    return false;

  // ---- emit ----
  std::string jid = AllocVarID(j.get());
  PrintIndent();
  stream << "for (int " << jid << " = 0; " << jid << " < " << N << "; " << jid
         << " += " << kHvxF16Lanes << ") {\n";
  int scope = this->BeginScope();
  std::optional<HvxLanes> rr =
      EmitHvxExpr(store->value, j, &ana, /*emit=*/true);
  ICHECK(rr) << "HVX emit failed after the probe passed (op-set drift?)";
  HvxLanes r = *rr;
  std::string sref = GetBufferRef(vt, store->buffer.get(), sidx);
  PrintIndent();
  if (AccessAligned128(store->buffer.get(), sidx, j, &ana))
    stream << "*(HVX_Vector*)(&" << sref << ") = tl_hvx_narrow_hf(" << r.lo
           << ", " << r.hi << ");\n";
  else
    stream << "tl_hvx_storeu(&" << sref << ", tl_hvx_narrow_hf(" << r.lo << ", "
           << r.hi << "));\n";
  this->EndScope(scope);
  PrintIndent();
  stream << "}\n";
  return true;
}

CodeGenTileLangHexagon::HvxLanes
CodeGenTileLangHexagon::EmitHvxOp(const HvxLanes &a, const HvxLanes &b,
                                  const char *fn, bool emit) {
  if (!emit)
    return {"", "", a.bcast && b.bcast};
  std::string lo = name_supply_->FreshName("tl_v");
  PrintIndent();
  stream << "HVX_Vector " << lo << " = " << fn << "(" << a.lo << ", " << b.lo
         << ");\n";
  if (a.bcast && b.bcast)
    return {lo, lo, true};
  std::string hi = name_supply_->FreshName("tl_v");
  PrintIndent();
  stream << "HVX_Vector " << hi << " = " << fn << "(" << a.hi << ", " << b.hi
         << ");\n";
  return {lo, hi, false};
}

CodeGenTileLangHexagon::HvxLanes
CodeGenTileLangHexagon::EmitHvxUnary(const HvxLanes &a, const char *fn,
                                     bool emit) {
  if (!emit)
    return {"", "", a.bcast};
  std::string lo = name_supply_->FreshName("tl_v");
  PrintIndent();
  stream << "HVX_Vector " << lo << " = " << fn << "(" << a.lo << ");\n";
  if (a.bcast)
    return {lo, lo, true};
  std::string hi = name_supply_->FreshName("tl_v");
  PrintIndent();
  stream << "HVX_Vector " << hi << " = " << fn << "(" << a.hi << ");\n";
  return {lo, hi, false};
}

std::optional<CodeGenTileLangHexagon::HvxLanes>
CodeGenTileLangHexagon::EmitHvxExpr(const PrimExpr &e, const tirx::Var &j,
                                    arith::Analyzer *ana, bool emit) {
  // j-independent subexpression: evaluate once (scalar C) and splat to fp32.
  if (!UsesLoopVar(e, j)) {
    if (!emit)
      return HvxLanes{"", "", true};
    // Render the scalar FIRST: a j-independent subexpr that lowers to
    // statements (a select/if_then_else/Let) emits them to `stream` here —
    // before the splat line — instead of splitting it mid-token.
    std::string ev = PrintExpr(e);
    std::string s = name_supply_->FreshName("tl_b");
    PrintIndent();
    stream << "HVX_Vector " << s << " = tl_hvx_splat_f((float)(" << ev
           << "));\n";
    return HvxLanes{s, s, true};
  }
  // j-contiguous fp16 load: widen to an fp32 (lo, hi) pair.
  if (const auto *load = e.as<BufferLoadNode>()) {
    if (load->indices.size() != 1 || load->indices[0].dtype().lanes() != 1)
      return std::nullopt;
    if (load->dtype.lanes() != 1 || !load->dtype.is_float() ||
        load->dtype.bits() != 16)
      return std::nullopt; // contiguous loads must be scalar fp16 (widened)
    if (UsesLoopVar(ana->Simplify(load->indices[0] - j), j))
      return std::nullopt; // not unit-stride in j
    if (!emit)
      return HvxLanes{"", "", false};
    std::string lo = name_supply_->FreshName("tl_lo");
    std::string hi = name_supply_->FreshName("tl_hi");
    PrintIndent();
    stream << "HVX_Vector " << lo << ", " << hi << ";\n";
    std::string lref =
        GetBufferRef(load->dtype, load->buffer.get(), load->indices[0]);
    PrintIndent();
    if (AccessAligned128(load->buffer.get(), load->indices[0], j, ana))
      stream << "tl_hvx_widen_hf(*(const HVX_Vector*)(&" << lref << "), &" << lo
             << ", &" << hi << ");\n";
    else
      stream << "tl_hvx_widen_hf(tl_hvx_loadu(&" << lref << "), &" << lo
             << ", &" << hi << ");\n";
    return HvxLanes{lo, hi, false};
  }
  // fp16<->fp32 casts are transparent (internal compute is fp32).
  if (const auto *c = e.as<CastNode>())
    return EmitHvxExpr(c->value, j, ana, emit);
  // Binary arithmetic (Add/Sub/Mul) — one body for all three.
  auto binop = [&](const PrimExpr &x, const PrimExpr &y,
                   const char *fn) -> std::optional<HvxLanes> {
    auto a = EmitHvxExpr(x, j, ana, emit);
    if (!a)
      return std::nullopt;
    auto b = EmitHvxExpr(y, j, ana, emit);
    if (!b)
      return std::nullopt;
    return EmitHvxOp(*a, *b, fn, emit);
  };
  if (const auto *a = e.as<AddNode>())
    return binop(a->a, a->b, "tl_hvx_add_sf");
  if (const auto *s = e.as<SubNode>())
    return binop(s->a, s->b, "tl_hvx_sub_sf");
  if (const auto *m = e.as<MulNode>())
    return binop(m->a, m->b, "tl_hvx_mul_sf");
  if (const auto *d = e.as<DivNode>()) {
    PrimExpr sa;
    if (MatchRsqrt(d, &sa)) { // 1/sqrt(x) -> rsqrt(x) (one op, accurate)
      auto a = EmitHvxExpr(sa, j, ana, emit);
      if (!a)
        return std::nullopt;
      return EmitHvxUnary(*a, "tl_hvx_rsqrt_vsf", emit);
    }
    auto a = EmitHvxExpr(d->a, j, ana, emit);
    if (!a)
      return std::nullopt;
    auto b = EmitHvxExpr(d->b, j, ana, emit);
    if (!b)
      return std::nullopt;
    return EmitHvxOp(*a, EmitHvxUnary(*b, "tl_hvx_recip_vsf", emit),
                     "tl_hvx_mul_sf", emit);
  }
  if (const auto *call = e.as<CallNode>()) {
    PrimExpr arg_expr;
    bool base_e;
    if (!MatchExpCall(call, &arg_expr, &base_e))
      return std::nullopt;
    auto a = EmitHvxExpr(arg_expr, j, ana, emit);
    if (!a)
      return std::nullopt;
    HvxLanes arg = *a;
    if (base_e) {
      // base-e: exp(x) = exp2(x * log2e); base-2 needs no rescale.
      HvxLanes log2e{"", "", true};
      if (emit) {
        std::string s = name_supply_->FreshName("tl_b");
        PrintIndent();
        stream << "HVX_Vector " << s
               << " = tl_hvx_splat_f(1.4426950408889634f);\n";
        log2e = {s, s, true};
      }
      arg = EmitHvxOp(arg, log2e, "tl_hvx_mul_sf", emit);
    }
    return EmitHvxUnary(arg, "tl_hvx_exp2_vsf", emit);
  }
  return std::nullopt; // unsupported elementwise node
}

} // namespace codegen
} // namespace tvm
