/*
 * \file codegen_hexagon.cc
 */
#include "codegen_hexagon.h"
#include "support/check.h"

#include <tvm/ir/op.h>             // Op::Get (exp intrinsic match)
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>     // UsesVar (HVX elementwise vectorizer)
#include <tvm/tirx/stmt_functor.h> // PostOrderVisit (per-worker VTCM pre-pass)

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
  wp_stride_ = 0;
  if (auto nw = f->GetAttr<Integer>("hexagon.num_workers")) {
    wp_num_workers_ = static_cast<int>(nw.value()->value);
  }
  // Reserve the worker callback's fixed identifiers BEFORE assigning param ids, so a
  // kernel parameter named e.g. "tl_nw" can't collide with them (mirrors how the
  // CUDA backend reserves its runtime helper names).
  if (wp_num_workers_ > 0) {
    for (const char *nm : {"tl_p", "tl_a", "tl_args", "tl_wid", "tl_nw", "tl_en"})
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
    // Pre-pass: size each worker's private VTCM region.  It holds the alloc_shared
    // operands at the bottom (wp_operand_bytes_) and the HMX gemm's Crouton scratch
    // at the top (wp_gemm_scratch_, max over gemms — scratch is reused per gemm), so
    // wp_stride_ = operands + scratch and worker w owns [2048+w*stride, 2048+(w+1)*stride).
    wp_operand_bytes_ = 0;
    wp_gemm_scratch_ = 0;
    wp_uses_hmx_ = false;
    tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &node) {
      if (const auto *a = node.as<AllocBufferNode>()) {
        std::string sc = GetPtrStorageScope(a->buffer->data);
        if (sc == "shared" || sc == "shared.dyn" || sc == "shared.tmem") {
          size_t n = 1;
          for (const auto &d : a->buffer->shape) {
            const auto *imm = d.as<IntImmNode>();
            // Match the emission's static-shape requirement (the ICHECK in
            // VisitStmt_(AllocBufferNode)) so this pre-pass sum can never diverge
            // from the per-alloc vtcm_offset_ bump (a divergence would overlap or
            // waste worker regions).
            ICHECK(imm) << "worker-pool alloc_shared requires a static shape";
            n *= static_cast<size_t>(imm->value);
          }
          wp_operand_bytes_ +=
              (n * a->buffer->dtype.bytes() + size_t(2047)) & ~size_t(2047);
        }
      } else if (const auto *call = node.as<CallNode>()) {
        // T.gemm->HMX lowers to tl_hexagon_hmx_gemm(C,A,B,M,N,K,ta,tb); it needs
        // Crouton scratch (a_sz+b_sz+c_sz tiles) in this worker's region — size it
        // from M=args[4]/N=args[5]/K=args[6] and keep the max.
        if (!call->args.empty()) {
          if (const auto *s = call->args[0].as<StringImmNode>()) {
            if (s->value == "tl_hexagon_hmx_gemm") {
              wp_uses_hmx_ = true;
              auto cdim = [&](size_t i) -> size_t {
                const auto *imm = call->args[i].as<IntImmNode>();
                ICHECK(imm) << "worker-pool T.gemm requires static M/N/K";
                return static_cast<size_t>(imm->value);
              };
              size_t M = cdim(4), N = cdim(5), K = cdim(6);
              size_t tiles = (M / 32) * (K / 32) + (N / 32) * (K / 32) +
                             (M / 32) * (N / 32);
              size_t sc_bytes = tiles * 1024u * 2u; // tile elems * sizeof(fp16)
              if (sc_bytes > wp_gemm_scratch_)
                wp_gemm_scratch_ = sc_bytes;
            }
          }
        }
      }
    });
    wp_stride_ = wp_operand_bytes_ + wp_gemm_scratch_;
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
    if (wp_uses_hmx_) {
      // Each worker enables HMX for its OWN thread (per-thread SHARED lock), except
      // the session thread (worker 0 / inline fallback) which session_init already
      // enabled — keyed on tid so it's exactly once per thread.
      stream << "  int tl_en = (qurt_thread_get_id() != tl_hmx_session_tid);\n";
      // Refuse (skip this worker's blocks) rather than run MACs with HMX unlocked if
      // the per-thread enable fails — mirrors the template-path worker.
      stream << "  if (tl_en && tl_hmx_enable() != 0) return;\n";
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
    if (wp_uses_hmx_)
      stream << "  if (tl_en) tl_hmx_disable();\n";
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
    if (wp_stride_ > 0) {
      // Cap workers so nw private VTCM regions (wp_stride_ bytes each) fit alongside
      // the reserved scale tile.  The compile-time guard above is against an 8MB
      // constant; here we check the ACTUAL runtime grant.  If not even one region
      // fits (smaller SKU / partial VTCM grant), skip the dispatch rather than let a
      // worker write past the arena.
      stream << "  tl_vtcm_acquire();\n";
      stream << "  if ((size_t)tl_vtcm_total >= 2048u + " << wp_stride_ << "u) {\n";
      stream << "    unsigned tl_cap = (unsigned)((tl_vtcm_total - 2048u) / "
             << wp_stride_ << "u);\n";
      stream << "    if ((unsigned)tl_nw > tl_cap) tl_nw = (int)tl_cap;\n";
      stream << "    tl_parallel(" << name << "_worker, &tl_args, tl_nw);\n";
      stream << "  }\n";
    } else {
      stream << "  tl_parallel(" << name << "_worker, &tl_args, tl_nw);\n";
    }
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
    stream << "*)((char*)tl_vtcm_base() + " << offset;
    if (wp_emit_) {
      // Worker-pool: each worker (tl_wid) gets a disjoint VTCM slice of wp_stride_
      // bytes, so concurrent blocks never share a buffer.
      stream << " + (size_t)tl_wid * " << wp_stride_ << "u";
    }
    stream << ");\n";
    // Publish the running bottom high-water so a top-down VTCM consumer (the HMX
    // gemm scratch) won't overlap this (and prior) live shared tiles.  Skipped in
    // worker-pool mode: the per-worker gemm (tl_hexagon_hmx_gemm_mt) uses the
    // codegen-passed op_floor for its own slice, not this global high-water, so the
    // value is unused there and would be a racy cross-worker write.
    if (!wp_emit_) {
      this->PrintIndent();
      stream << "tl_vtcm_shared_high_water = " << vtcm_offset_ << "u;\n";
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
  // In a worker-pool kernel a T.gemm (tl_hexagon_hmx_gemm) is REWRITTEN to the
  // region-aware tl_hexagon_hmx_gemm_mt so its Crouton scratch lives in THIS
  // worker's VTCM slice (the plain entry carves from the global VTCM top, which
  // would collide across workers).  Any OTHER tl_hexagon_hmx* call has no per-worker
  // variant (global scratch) and is rejected loudly.
  if (wp_emit_ && !op->args.empty()) {
    if (const auto *s = op->args[0].as<StringImmNode>()) {
      if (s->value == "tl_hexagon_hmx_gemm") {
        // Rewrite T.gemm to the region-aware worker variant: its Crouton scratch
        // lives in THIS worker's VTCM slice, not the global top, so concurrent
        // workers' gemms don't collide.  Append the worker's region end and operand
        // floor (region = [2048+wid*stride, 2048+(wid+1)*stride), operands fill the
        // bottom wp_operand_bytes_, scratch the rest).
        os << "tl_hexagon_hmx_gemm_mt(";
        for (size_t i = 1; i < op->args.size(); ++i) { // C,A,B,M,N,K,ta,tb
          if (i > 1)
            os << ", ";
          os << PrintExpr(op->args[i]);
        }
        os << ", (char*)tl_vtcm_base() + 2048u + (size_t)(tl_wid + 1) * "
           << wp_stride_ << "u"; // region_end (top of this worker's slice)
        os << ", (char*)tl_vtcm_base() + 2048u + (size_t)tl_wid * " << wp_stride_
           << "u + " << wp_operand_bytes_ << "u)"; // op_floor (operand high-water)
        return;
      }
      if (s->value.find("tl_hexagon_hmx") == 0) {
        LOG(FATAL) << "CodeGenTileLangHexagon: this HMX call inside a "
                      "hexagon.num_workers kernel is unsupported — only T.gemm "
                      "(tl_hexagon_hmx_gemm) has a per-worker scratch variant.";
      }
    }
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangHexagon::VisitStmt_(const AttrStmtNode *op) {
  // The lowered kernel keeps the GPU launch as nested `thread_extent` attrs
  // (blockIdx/threadIdx); we serialize the grid into nested C loops, binding each
  // thread var to its loop index.  In worker-pool mode (hexagon.num_workers > 0,
  // set up in AddFunction) the outermost blockIdx.x loop is instead STRIDED across
  // HW threads (`bx = tl_wid; bx += tl_nw`); inner thread loops stay serial.
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

// ---------------------------------------------------------------------------
// HVX elementwise ("map") vectorizer.
//
// The `map` half of the {gemm, copy, map, reduce} basis.  Hexagon's scalar unit
// has no fp16, so an elementwise loop emitted as scalar C compiles to per-element
// libcalls (__extendhfsf2 et al.).  Here we recognise the clean innermost loop a
// `T.serial` elementwise nest lowers to —
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

// Match an exp call in either form: the op-level intrinsic (tl.__exp base-2 /
// tirx.exp base-e) or — after the pipeline's intrinsic lowering — the
// call_extern to the libm symbol (exp2f / expf).  Sets *arg to the operand and
// *base_e true for base-e (which the emitter rescales by log2e before exp2).
bool MatchExpCall(const CallNode *call, PrimExpr *arg, bool *base_e) {
  if (call->args.size() == 1) {
    if (call->op.same_as(Op::Get("tl.__exp"))) {
      *arg = call->args[0];
      *base_e = false;
      return true;
    }
    if (call->op.same_as(Op::Get("tirx.exp"))) {
      *arg = call->args[0];
      *base_e = true;
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
} // namespace

void CodeGenTileLangHexagon::VisitStmt_(const ForNode *op) {
  if (TryEmitHvxElementwise(op))
    return;
  CodeGenC::VisitStmt_(op);
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
  auto j_free = [&](const PrimExpr &x) {
    return !tirx::UsesVar(x, [&](const VarNode *v) { return v == j.get(); });
  };
  // Store must be unit-stride in j (idx - j independent of j), so the original
  // index doubles as the chunk base address with j stepping by 64.
  if (!j_free(ana.Simplify(sidx - j)))
    return false;
  if (!HvxExprSupported(store->value, j, &ana))
    return false;

  // ---- emit ----
  std::string jid = AllocVarID(j.get());
  PrintIndent();
  stream << "for (int " << jid << " = 0; " << jid << " < " << N << "; " << jid
         << " += " << kHvxF16Lanes << ") {\n";
  int scope = this->BeginScope();
  HvxLanes r = EmitHvxExpr(store->value, j, &ana);
  PrintIndent();
  stream << "tl_hvx_storeu(&" << GetBufferRef(vt, store->buffer.get(), sidx)
         << ", tl_hvx_narrow_hf(" << r.lo << ", " << r.hi << "));\n";
  this->EndScope(scope);
  PrintIndent();
  stream << "}\n";
  return true;
}

bool CodeGenTileLangHexagon::HvxExprSupported(const PrimExpr &e,
                                              const tirx::Var &j,
                                              arith::Analyzer *ana) {
  auto uses_j = [&](const PrimExpr &x) {
    return tirx::UsesVar(x, [&](const VarNode *v) { return v == j.get(); });
  };
  if (!uses_j(e))
    return true; // j-independent -> broadcast splat
  if (const auto *load = e.as<BufferLoadNode>()) {
    if (load->indices.size() != 1 || load->indices[0].dtype().lanes() != 1)
      return false;
    if (load->dtype.lanes() != 1 || !load->dtype.is_float() ||
        load->dtype.bits() != 16)
      return false; // j-contiguous loads must be scalar fp16 (widened to fp32)
    return !uses_j(ana->Simplify(load->indices[0] - j)); // unit stride
  }
  if (const auto *c = e.as<CastNode>())
    return HvxExprSupported(c->value, j, ana);
  if (const auto *a = e.as<AddNode>())
    return HvxExprSupported(a->a, j, ana) && HvxExprSupported(a->b, j, ana);
  if (const auto *s = e.as<SubNode>())
    return HvxExprSupported(s->a, j, ana) && HvxExprSupported(s->b, j, ana);
  if (const auto *m = e.as<MulNode>())
    return HvxExprSupported(m->a, j, ana) && HvxExprSupported(m->b, j, ana);
  if (const auto *d = e.as<DivNode>())
    return HvxExprSupported(d->a, j, ana) && HvxExprSupported(d->b, j, ana);
  if (const auto *call = e.as<CallNode>()) {
    PrimExpr arg;
    bool base_e;
    if (MatchExpCall(call, &arg, &base_e))
      return HvxExprSupported(arg, j, ana);
    return false;
  }
  return false;
}

CodeGenTileLangHexagon::HvxLanes
CodeGenTileLangHexagon::EmitHvxOp(const HvxLanes &a, const HvxLanes &b,
                                  const char *fn) {
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
CodeGenTileLangHexagon::EmitHvxUnary(const HvxLanes &a, const char *fn) {
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

CodeGenTileLangHexagon::HvxLanes
CodeGenTileLangHexagon::EmitHvxExpr(const PrimExpr &e, const tirx::Var &j,
                                    arith::Analyzer *ana) {
  auto uses_j = [&](const PrimExpr &x) {
    return tirx::UsesVar(x, [&](const VarNode *v) { return v == j.get(); });
  };
  // j-independent subexpression: evaluate once (scalar C) and splat to fp32.
  if (!uses_j(e)) {
    std::string s = name_supply_->FreshName("tl_b");
    PrintIndent();
    stream << "HVX_Vector " << s << " = tl_hvx_splat_f((float)(" << PrintExpr(e)
           << "));\n";
    return {s, s, true};
  }
  // j-contiguous fp16 load: widen to an fp32 (lo, hi) pair.
  if (const auto *load = e.as<BufferLoadNode>()) {
    std::string lo = name_supply_->FreshName("tl_lo");
    std::string hi = name_supply_->FreshName("tl_hi");
    PrintIndent();
    stream << "HVX_Vector " << lo << ", " << hi << ";\n";
    PrintIndent();
    stream << "tl_hvx_widen_hf(tl_hvx_loadu(&"
           << GetBufferRef(load->dtype, load->buffer.get(), load->indices[0])
           << "), &" << lo << ", &" << hi << ");\n";
    return {lo, hi, false};
  }
  // fp16<->fp32 casts are transparent (internal compute is fp32).
  if (const auto *c = e.as<CastNode>())
    return EmitHvxExpr(c->value, j, ana);
  if (const auto *a = e.as<AddNode>()) {
    HvxLanes la = EmitHvxExpr(a->a, j, ana), lb = EmitHvxExpr(a->b, j, ana);
    return EmitHvxOp(la, lb, "tl_hvx_add_sf");
  }
  if (const auto *s = e.as<SubNode>()) {
    HvxLanes la = EmitHvxExpr(s->a, j, ana), lb = EmitHvxExpr(s->b, j, ana);
    return EmitHvxOp(la, lb, "tl_hvx_sub_sf");
  }
  if (const auto *m = e.as<MulNode>()) {
    HvxLanes la = EmitHvxExpr(m->a, j, ana), lb = EmitHvxExpr(m->b, j, ana);
    return EmitHvxOp(la, lb, "tl_hvx_mul_sf");
  }
  if (const auto *d = e.as<DivNode>()) {
    HvxLanes la = EmitHvxExpr(d->a, j, ana), lb = EmitHvxExpr(d->b, j, ana);
    return EmitHvxOp(la, EmitHvxUnary(lb, "tl_hvx_recip_vsf"), "tl_hvx_mul_sf");
  }
  if (const auto *call = e.as<CallNode>()) {
    PrimExpr arg_expr;
    bool base_e;
    ICHECK(MatchExpCall(call, &arg_expr, &base_e)); // detection guaranteed a match
    HvxLanes arg = EmitHvxExpr(arg_expr, j, ana);
    if (base_e) {
      // base-e: exp(x) = exp2(x * log2e); base-2 (exp2f/tl.__exp) needs no rescale.
      std::string s = name_supply_->FreshName("tl_b");
      PrintIndent();
      stream << "HVX_Vector " << s
             << " = tl_hvx_splat_f(1.4426950408889634f);\n";
      arg = EmitHvxOp(arg, {s, s, true}, "tl_hvx_mul_sf");
    }
    return EmitHvxUnary(arg, "tl_hvx_exp2_vsf");
  }
  LOG(FATAL) << "EmitHvxExpr: unsupported expr (detection should have caught it)";
  return {"", "", false};
}

} // namespace codegen
} // namespace tvm
