/*
 * \file codegen_hexagon.cc
 */
#include "codegen_hexagon.h"
#include "support/check.h"

#include <tvm/runtime/logging.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using namespace ffi;

CodeGenTileLangHexagon::CodeGenTileLangHexagon() {}

void CodeGenTileLangHexagon::Init(bool output_ssa) {
  decl_stream << "// tilelang Hexagon (cDSP) kernel\n";
  decl_stream << "#include <tl_templates/hexagon/common.h>\n";
  decl_stream << "\n";
  CodeGenC::Init(output_ssa);
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
    if (!fail && (lanes >= 2 && lanes <= 16)) {
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
    if (!fail && (lanes >= 2 && lanes <= 16)) {
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
  // Reserve the first 2KB VTCM tile (offset 0) for the HMX output scales that
  // tl_hexagon_hmx_gemm writes at tl_vtcm_base()+0; alloc_shared buffers start
  // above it, so operands and scales can never overlap regardless of the runtime
  // VTCM size.  (Non-HMX shared kernels just leave the first tile unused.)
  vtcm_offset_ = 2048;

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenTileLangHexagon: Expect PrimFunc to have the global_symbol "
         "attribute";
  function_names_.push_back(global_symbol.value());

  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  // Worker-pool mode: `hexagon.num_workers` > 0 fans the grid's outermost block
  // loop across HW threads via tl_parallel instead of a serial for-loop.  Reset the
  // per-emit flags too (defensive: an exception mid-emit on a prior function must
  // not leak state into this one).
  wp_num_workers_ = 0;
  wp_emit_ = false;
  wp_outermost_pending_ = false;
  if (auto nw = f->GetAttr<Integer>("hexagon.num_workers")) {
    wp_num_workers_ = static_cast<int>(nw.value()->value);
  }
  // Reserve the worker callback's fixed identifiers BEFORE assigning param ids, so a
  // kernel parameter named e.g. "tl_nw" can't collide with them (mirrors how the
  // CUDA backend reserves its runtime helper names).
  if (wp_num_workers_ > 0) {
    for (const char *nm : {"tl_p", "tl_a", "tl_args", "tl_wid", "tl_nw"})
      name_supply_->ReserveName(nm);
  }

  // Pre-assign param ids and register handle element types (needed before the body
  // is emitted, and reused by the worker-pool args struct / unpack / signature).
  std::vector<std::string> pvids;
  pvids.reserve(f->params.size());
  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    pvids.push_back(AllocVarID(v.get()));
    if (v.dtype().is_handle()) {
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    }
  }

  // Emit the parameter signature list (storage scope + type + restrict).
  auto emit_signature = [&]() {
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
      stream << ' ' << pvids[i];
    }
  };

  std::string name = static_cast<std::string>(global_symbol.value());

  if (wp_num_workers_ > 0) {
    // (1) args struct: one plain field per kernel parameter.
    stream << "typedef struct {\n";
    for (size_t i = 0; i < f->params.size(); ++i) {
      stream << "  ";
      CodeGenC::PrintType(GetType(f->params[i]), stream);
      stream << " " << pvids[i] << ";\n";
    }
    stream << "} " << name << "_args_t;\n";
    // (2) worker callback: unpack params into same-named locals (so the body emits
    //     unchanged), then run the grid with the OUTERMOST block loop strided across
    //     workers (wp_outermost_pending_ tells the AttrStmt handler).
    stream << "static void " << name
           << "_worker(void* tl_p, int tl_wid, int tl_nw) {\n";
    stream << "  " << name << "_args_t* tl_a = (" << name << "_args_t*)tl_p;\n";
    for (size_t i = 0; i < f->params.size(); ++i) {
      stream << "  ";
      CodeGenC::PrintType(GetType(f->params[i]), stream);
      stream << " " << pvids[i] << " = tl_a->" << pvids[i] << ";\n";
    }
    this->PreFunctionBody(f);
    wp_emit_ = true;
    wp_outermost_pending_ = true;
    int worker_scope = this->BeginScope();
    this->PrintStmt(f->body);
    this->EndScope(worker_scope);
    wp_emit_ = false;
    // The body must have contained a blockIdx.x loop for the stride to attach to;
    // otherwise tl_parallel would run the WHOLE grid on every worker (duplicated
    // work + a write race on the outputs).  Reject that loudly.
    ICHECK(!wp_outermost_pending_)
        << "CodeGenTileLangHexagon: a hexagon.num_workers kernel must have a "
           "blockIdx.x grid loop to distribute across workers, but none was found.";
    wp_outermost_pending_ = false;
    stream << "}\n";
    // (3) entry: build the args struct and dispatch (workers capped by HW threads).
    this->PrintFuncPrefix(stream);
    CodeGenC::PrintType(f->ret_type, stream);
    this->PrintExtraAttrs(f, stream);
    stream << " " << name << "(";
    emit_signature();
    stream << ") {\n";
    stream << "  " << name << "_args_t tl_args = {";
    for (size_t i = 0; i < f->params.size(); ++i)
      stream << (i ? ", " : " ") << pvids[i];
    stream << " };\n";
    stream << "  int tl_nw = tl_num_workers();\n";
    stream << "  if (tl_nw > " << wp_num_workers_ << ") tl_nw = " << wp_num_workers_
           << ";\n";
    stream << "  tl_parallel(" << name << "_worker, &tl_args, tl_nw);\n";
    stream << "}\n\n";
    return;
  }

  // Serial path (single HW thread): the grid lowers to nested for-loops.
  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  stream << " " << name << "(";
  emit_signature();
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangHexagon::VisitStmt_(const AllocBufferNode *op) {
  std::string vid = AllocVarID(op->buffer->data.get());
  const auto &shape = op->buffer->shape;
  size_t constant_size = 1;
  for (const auto &dim : shape) {
    const IntImmNode *dim_imm = dim.as<IntImmNode>();
    ICHECK(dim_imm) << "Can only handle constant size allocation for now";
    constant_size *= dim_imm->value;
  }
  ICHECK_GT(constant_size, 0) << "Can only handle constant size allocation";

  std::string scope = GetPtrStorageScope(op->buffer->data);
  bool is_shared = scope == "shared" || scope == "shared.dyn" || scope == "shared.tmem";

  if (is_shared && wp_emit_) {
    LOG(FATAL) << "CodeGenTileLangHexagon: alloc_shared inside a hexagon.num_workers "
                  "(worker-pool) kernel is not supported yet — concurrent workers each "
                  "need a private VTCM partition (the next increment). Use the "
                  "worker-pool only for HVX kernels without shared memory for now, or "
                  "drop the hexagon.num_workers attr.";
  }
  this->PrintIndent();
  if (is_shared) {
    // VTCM-backed: the HMX matrix engine reads operands from VTCM (mxmem) and HVX
    // wants its scratch there, so a `shared` tile cannot live on the stack.
    // Assign a compile-time byte offset into the one session-acquired VTCM arena
    // (tl_vtcm_base()).  Shared buffers are declared inside the serialized block
    // loop, so a *fixed* offset (reused across block iterations) is correct —
    // not a runtime bump that would overrun.
    size_t nbytes = constant_size * op->buffer->dtype.bytes();
    size_t offset = vtcm_offset_;
    // 2048-byte align == one HMX Crouton tile, also HVX(128B)-aligned.
    vtcm_offset_ += (nbytes + size_t(2047)) & ~size_t(2047);
    ICHECK_LE(vtcm_offset_, size_t(8) << 20)
        << "alloc_shared total exceeds 8MB VTCM (" << vtcm_offset_ << " bytes)";
    PrintType(op->buffer->dtype, stream);
    stream << "* " << vid << " = (";
    PrintType(op->buffer->dtype, stream);
    stream << "*)((char*)tl_vtcm_base() + " << offset << ");\n";
    // Publish the running bottom high-water so a top-down VTCM consumer (the HMX
    // gemm scratch) won't overlap this (and prior) live shared tiles.  Monotonic
    // and emitted before the compute, so the value seen at the gemm call is the
    // end of all currently-live shared buffers.
    this->PrintIndent();
    stream << "tl_vtcm_shared_high_water = " << vtcm_offset_ << "u;\n";
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

void CodeGenTileLangHexagon::VisitStmt_(const AttrStmtNode *op) {
  // The lowered kernel keeps the GPU launch as nested `thread_extent` attrs
  // (blockIdx/threadIdx).  A Hexagon kernel runs on a single (for M1) thread,
  // so we serialize the whole grid into nested loops, binding each thread var
  // to its loop index.  (Mapping blocks onto a worker pool is an M3 concern.)
  if (op->attr_key == "thread_extent") {
    const IterVarNode *iv = op->node.as<IterVarNode>();
    ICHECK(iv != nullptr) << "thread_extent attr expects an IterVar node";
    // Mirror CodeGenC's contract: only the first binding of a real (tagged)
    // thread var introduces an induction variable.  A tag-less extent, or a var
    // re-bound in a sibling/nested scope (e.g. across a reduction), must NOT
    // re-emit a loop or re-AllocVarID (which would fatally fail the SSA check);
    // it just descends into the body under the already-bound variable.
    if (iv->thread_tag.length() != 0 && !var_idmap_.count(iv->var.get())) {
      std::string vid = AllocVarID(iv->var.get());
      std::string extent = PrintExpr(op->value);
      // Declare the induction var in the *enclosing* C scope, not the for-header,
      // so a later sibling-scope binding of the same thread var (the else branch,
      // which descends without re-emitting a loop) still sees it in scope instead
      // of referencing an identifier whose for-block has already closed.
      PrintIndent();
      stream << "int " << vid << " = 0;\n";
      PrintIndent();
      if (wp_emit_ && wp_outermost_pending_ && iv->thread_tag == "blockIdx.x") {
        // Worker-pool: distribute the blockIdx.x grid loop across HW threads (each
        // worker takes a strided slice); inner thread loops stay serial.  Gating on
        // the tag (not just "first tagged") keeps the stride on the block dim.
        wp_outermost_pending_ = false;
        stream << "for (" << vid << " = tl_wid; " << vid << " < " << extent << "; "
               << vid << " += tl_nw) {\n";
      } else {
        stream << "for (" << vid << " = 0; " << vid << " < " << extent << "; ++"
               << vid << ") {\n";
      }
      int scope = this->BeginScope();
      this->PrintStmt(op->body);
      this->EndScope(scope);
      this->PrintIndent();
      stream << "}\n";
    } else {
      this->PrintStmt(op->body);
    }
  } else {
    CodeGenC::VisitStmt_(op);
  }
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

} // namespace codegen
} // namespace tvm
