/*!
 * \file annotate_read_only_params.cc
 * \brief Annotate PrimFunc parameters that are read-only (never written).
 */

#include "support/check.h"
#include <string>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

/*!
 * \brief A simple visitor that marks handle parameters as written when they
 *        appear on the LHS of a BufferStore or in a tvm_access_ptr with write
 * flag.
 */
class ReadWriteMarker : public StmtExprVisitor {
public:
  explicit ReadWriteMarker(
      const std::unordered_set<const VarNode *> &param_or_data_vars)
      : param_or_data_vars_(param_or_data_vars) {}

  const std::unordered_set<const VarNode *> &written() const {
    return written_;
  }

private:
  void MarkData(const Var &var) {
    if (param_or_data_vars_.count(var.get())) {
      written_.insert(var.get());
    } else if (auto it = bindings_.find(var.get()); it != bindings_.end()) {
      if (expanded_.insert(var.get()).second)
        MarkPointer(it->second);
    }
  }

  // Follow possible pointer bases. An explicit access_ptr mask describes
  // how a callee uses that address. Numeric loads in an offset or predicate
  // are not destinations.
  void MarkPointer(const PrimExpr &ptr) {
    if (const auto *var = ptr.as<VarNode>()) {
      MarkData(GetRef<Var>(var));
      return;
    }
    PreOrderVisit(ptr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        MarkData(GetRef<Var>(var));
      } else if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::tvm_access_ptr())) {
          MarkAccess(call);
          return false;
        }
        if (call->op.same_as(builtin::handle_add_byte_offset())) {
          MarkPointer(call->args[0]);
          return false;
        }
        if (call->op.same_as(builtin::address_of())) {
          if (const auto *load = call->args[0].as<BufferLoadNode>())
            MarkData(load->buffer->data);
          return false;
        }
        if (call->op.same_as(builtin::if_then_else())) {
          MarkPointer(call->args[1]);
          MarkPointer(call->args[2]);
          return false;
        }
      } else if (const auto *select = node.as<SelectNode>()) {
        MarkPointer(select->true_value);
        MarkPointer(select->false_value);
        return false;
      }
      return !node.as<BufferLoadNode>();
    });
  }

  void MarkAccess(const CallNode *call) {
    const auto *mask = call->args[4].as<IntImmNode>();
    if (!mask || (mask->value & 2))
      MarkPointer(call->args[1]);
  }

  void VisitStmt_(const BindNode *op) final {
    bindings_[op->var.get()] = op->value;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    bindings_[op->var.get()] = op->value;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    MarkData(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      MarkAccess(op);
    } else {
      const bool opaque = op->op.same_as(builtin::call_extern());
      for (const PrimExpr &arg : op->args) {
        const auto *call = arg.as<CallNode>();
        if ((opaque && arg.dtype().is_handle()) ||
            (call && call->op.same_as(builtin::address_of()))) {
          MarkPointer(arg);
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const VarNode *, PrimExpr> bindings_;
  std::unordered_set<const VarNode *> expanded_;
  std::unordered_set<const VarNode *> param_or_data_vars_;
  std::unordered_set<const VarNode *> written_;
};

/*!
 * \brief Annotate PrimFunc with indices of read-only handle parameters.
 *
 * Adds an Array<Integer> attribute "tl.readonly_param_indices" that lists
 * parameter indices which correspond to handle parameters that are never
 * written inside the function body. This can be used by codegen to emit
 * `const` qualifiers to enable read-only caching (e.g., __ldg on CUDA).
 */
static tirx::PrimFunc MarkReadOnlyParams(tirx::PrimFunc f) {
  // Gather handle params and their corresponding buffer data vars (aliases).
  std::unordered_set<const VarNode *> param_or_data_vars;

  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var &p = f->params[i];
    if (!p->dtype.is_handle())
      continue;
    param_or_data_vars.insert(p.get());
    // If there is a buffer_map entry for this param, include its data var too.
    if (auto opt = f->buffer_map.Get(p)) {
      const VarNode *data = opt.value()->data.get();
      param_or_data_vars.insert(data);
    }
  }
  if (param_or_data_vars.empty())
    return f;

  ReadWriteMarker marker(param_or_data_vars);
  marker(f->body);

  // Determine read-only parameter indices among all params (handle only)
  Array<Integer> readonly_indices;
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var &v = f->params[i];
    if (!v->dtype.is_handle())
      continue;

    bool is_written = false;
    // Direct param var written?
    if (marker.written().count(v.get())) {
      is_written = true;
    } else {
      // Or any aliased data var written?
      if (auto opt = f->buffer_map.Get(v)) {
        if (marker.written().count(opt.value()->data.get())) {
          is_written = true;
        }
      }
    }

    if (!is_written) {
      readonly_indices.push_back(Integer(static_cast<int>(i)));
    }
  }

  return WithAttr(std::move(f), "tl.readonly_param_indices", readonly_indices);
}

namespace transform {
using namespace tirx::transform;

Pass AnnotateReadOnlyParams() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return MarkReadOnlyParams(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateReadOnlyParams", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.AnnotateReadOnlyParams",
                        AnnotateReadOnlyParams);
}

} // namespace transform
} // namespace tl
} // namespace tvm
