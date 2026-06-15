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

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenTileLangHexagon: Expect PrimFunc to have the global_symbol "
         "attribute";
  function_names_.push_back(global_symbol.value());

  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }
      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
      if (no_alias) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangHexagon::VisitStmt_(const AllocBufferNode *op) {
  // Stack-local allocation (e.g. fragments / small scratch).  VTCM-scoped
  // allocations are lowered to an explicit arena by a later pass + the runtime
  // prologue; until then this handles constant-size local buffers.
  std::string vid = AllocVarID(op->buffer->data.get());
  this->PrintIndent();

  PrintType(op->buffer->dtype, stream);
  const auto &shape = op->buffer->shape;
  size_t constant_size = 1;
  for (const auto &dim : shape) {
    const IntImmNode *dim_imm = dim.as<IntImmNode>();
    ICHECK(dim_imm) << "Can only handle constant size stack allocation for now";
    constant_size *= dim_imm->value;
  }
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  stream << ' ' << vid << '[' << constant_size << "];\n";
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
      stream << "for (" << vid << " = 0; " << vid << " < " << extent
             << "; ++" << vid << ") {\n";
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
