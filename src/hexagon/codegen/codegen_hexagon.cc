/*
 * \file codegen_hexagon.cc
 */
#include "codegen_hexagon.h"
#include "hexagon/op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/storage_size.h"

#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
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
  // Takes the tile program's DMA queue (tl_dma_async) as the first argument.
  bool on_dma_queue = false;
  // Returns a kernel status, which the statement returns on failure.
  bool returns_status = false;
  // Prints only the first `printed_args` arguments (-1: all). The others are
  // dependency-only operands for compiler analyses.
  int printed_args = -1;
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
      {tl::hexagon_dma_submit(), "tl_hexagon_dma_async_submit", false, true,
       true},
      {tl::hexagon_dma_commit(), "tl_hexagon_dma_async_commit", false, true,
       true},
      {tl::hexagon_dma_wait(), "tl_hexagon_dma_async_wait", false, true, true,
       1},
      {tl::hexagon_exp(), "tl_hvx_exp", false},
      {tl::hexagon_exp2(), "tl_hvx_exp2", false},
      {tl::hexagon_log(), "tl_hvx_log", false},
      {tl::hexagon_log2(), "tl_hvx_log2", false},
      {tl::hexagon_sqrt(), "tl_hvx_sqrt", false},
      {tl::hexagon_rsqrt(), "tl_hvx_rsqrt", false},
      {tl::hexagon_tanh(), "tl_hvx_tanh", false},
      {tl::hexagon_sigmoid(), "tl_hvx_sigmoid", false},
      {tl::hexagon_erf(), "tl_hvx_erf", false},
      {tl::hexagon_fabs(), "tl_hvx_fabs", false},
      {tl::hexagon_lane_sum(), "tl_hvx_lane_sum", false},
      {tl::hexagon_lane_max(), "tl_hvx_lane_max", false},
      {tl::hexagon_lane_min(), "tl_hvx_lane_min", false},
      {tl::hexagon_lane_and(), "tl_hvx_lane_and", false},
      {tl::hexagon_lane_or(), "tl_hvx_lane_or", false},
      {tl::hexagon_lane_xor(), "tl_hvx_lane_xor", false},
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

int64_t ConstantBytes(const tirx::Buffer &buffer) {
  return tl::GetBufferStorageSizeBytes(
      static_cast<int64_t>(ConstantElements(buffer)), buffer->dtype);
}

// The bytes of the VTCM arena of the tile program in `body`: its one shared
// allocation (VerifyVTCMArena), or 0.
int64_t VtcmArenaBytes(const Stmt &body) {
  int64_t bytes = 0;
  tirx::PostOrderVisit(body, [&bytes](const ffi::ObjectRef &node) {
    const auto *alloc = node.as<AllocBufferNode>();
    if (alloc && IsSharedScope(GetPtrStorageScope(alloc->buffer->data)))
      bytes += ConstantBytes(alloc->buffer);
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
  arena_bound_ = false;

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenTileLangHexagon: Expect PrimFunc to have the global_symbol "
         "attribute";
  function_names_.push_back(global_symbol.value());

  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  dma_in_scope_ = false;
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
  // returns itself.
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
  this->PrintStmt(f->body);
  this->PrintIndent();
  this->stream << "return TL_OK;\n";
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangHexagon::PrintWorkerLoop(const ForNode *op) {
  ICHECK(!in_worker_) << "Hexagon worker loops cannot nest";
  ICHECK(is_zero(op->min)) << "Hexagon worker loop `" << op->loop_var
                           << "` must start at 0";
  int64_t num_workers =
      op->annotations.Get(tl::attr::kHexagonNumWorkers)->cast<Integer>()->value;
  int64_t arena_bytes = VtcmArenaBytes(op->body);

  // tl_parallel_for runs the body once per tile program, on the worker pool.
  // Each worker gets a private VTCM arena; a worker that issues HMX enables
  // it for its thread (tl_hmx_worker_scope). Each program declares its own
  // DMA queue.
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
  PrintStmt(op->body);
  PrintIndent();
  stream << "return TL_OK;\n";
  in_worker_ = false;
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
    // MergeSharedMemoryAllocations laid out the tile program's shared buffers
    // in this one arena, which starts at the kernel's VTCM base or, in a
    // worker, at the worker's slice (tl_arena).
    ICHECK(!arena_bound_) << "A Hexagon tile program has one VTCM arena";
    arena_bound_ = true;
    PrintType(op->buffer->dtype, stream);
    stream << "* " << vid << " = (";
    PrintType(op->buffer->dtype, stream);
    stream << "*)" << (in_worker_ ? "tl_arena" : "tl_vtcm_arena()") << ";\n";
    // Publish the kernel's VTCM usage for embedding/export capacity checks.
    // Worker arenas are sized by tl_parallel_for instead.
    if (!in_worker_) {
      this->PrintIndent();
      stream << "tl_vtcm_shared_high_water = " << ConstantBytes(op->buffer)
             << "u;\n";
    }
  } else {
    // Stack-local scratch. Whole-register vector accesses to it are aligned
    // vmem, which drops the low address bits, so align it like a register.
    PrintType(op->buffer->dtype, stream);
    stream << ' ' << vid << '[' << constant_size
           << "] __attribute__((aligned(128)));\n";
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
    ICHECK(!builtin->on_dma_queue || dma_in_scope_)
        << builtin->function << " needs the tile program's DMA queue";
    os << builtin->function << "(";
    const char *separator = "";
    if (builtin->on_dma_queue) {
      os << "tl_dma_async";
      separator = ", ";
    }
    size_t printed = builtin->printed_args < 0
                         ? op->args.size()
                         : static_cast<size_t>(builtin->printed_args);
    for (size_t i = 0; i < printed; ++i) {
      os << separator << PrintExpr(op->args[i]);
      separator = ", ";
    }
    os << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangHexagon::VisitStmt_(const EvaluateNode *op) { // NOLINT(*)
  const auto *call = op->value.as<CallNode>();
  if (call && call->op.same_as(tl::hexagon_dma_queue())) {
    // The queue lives as long as the tile program; its destructor drains it on
    // early returns too.
    const auto *descriptors = call->args[0].as<IntImmNode>();
    const auto *groups = call->args[1].as<IntImmNode>();
    ICHECK(descriptors && groups) << "hexagon_dma_queue needs constant sizes";
    PrintIndent();
    stream << "tl_hexagon_dma_async_context<" << descriptors->value << ", "
           << groups->value << "> tl_dma_async;\n";
    PrintIndent();
    stream << "if (tl_dma_async.status != TL_OK) return TL_ERR_DMA;\n";
    dma_in_scope_ = true;
    uses_dma_async_ = true;
    return;
  }
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
  const HexagonBuiltin *builtin = call ? FindHexagonBuiltin(call) : nullptr;
  if (builtin && builtin->returns_status) {
    std::string value = PrintExpr(op->value);
    PrintIndent();
    stream << "if (int32_t tl_status = " << value << ") return tl_status;\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHexagon::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tl::attr::kTileProgramScope) {
    // A tile program declares its DMA queue and binds its VTCM arena in this
    // scope, which is the C block of its loop body, lambda or function.
    bool outer_dma = dma_in_scope_;
    bool outer_arena = arena_bound_;
    arena_bound_ = false;
    CodeGenC::VisitStmt_(op);
    dma_in_scope_ = outer_dma;
    arena_bound_ = outer_arena;
    return;
  }
  CodeGenC::VisitStmt_(op);
}

// vec_type has no comparisons, and a vector min or max is one HVX instruction,
// so vectors print as the common.h functions.
void CodeGenTileLangHexagon::VisitExpr_(const MinNode *op,
                                        std::ostream &os) { // NOLINT(*)
  if (op->dtype.lanes() > 1) {
    os << "tl_vec_min(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
    return;
  }
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenTileLangHexagon::VisitExpr_(const MaxNode *op,
                                        std::ostream &os) { // NOLINT(*)
  if (op->dtype.lanes() > 1) {
    os << "tl_vec_max(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
    return;
  }
  PrintTernaryCondExpr(op, ">", os);
}

void CodeGenTileLangHexagon::VisitExpr_(const DivNode *op,
                                        std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_float() && op->dtype.lanes() > 1) {
    os << "tl_hvx_div(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
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

void CodeGenTileLangHexagon::VisitStmt_(const ForNode *op) {
  if (IsWorkerLoop(op)) {
    PrintWorkerLoop(op);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

} // namespace codegen
} // namespace tvm
