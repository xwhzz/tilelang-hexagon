/*!
 * \file tl/hexagon/transform/keep_kernel_params.cc
 * \brief A device kernel takes every parameter of its PrimFunc, in order.
 *
 * SplitHostDevice gives a device kernel the variables its body uses. Nothing
 * compiles the host function on Hexagon: the FastRPC skel calls the kernel
 * itself, with every parameter of the PrimFunc in order
 * (tilelang/hexagon/runtime/fastrpc.py), so a parameter the body never reads
 * would shift the ones after it. Each kernel takes the parameters of the host
 * function that calls it instead, in the host's order, a buffer by its data
 * pointer, and the host's call passes all of them. The kernel's other
 * arguments, such as a buffer's symbolic shape, follow them.
 *
 * Runs right after SplitHostDevice, before anything indexes the kernel's
 * parameters (AnnotateReadOnlyParams).
 */

#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The host function's arguments, in order: a buffer's data pointer, or the
// scalar itself.
Array<Var> HostArguments(const PrimFunc &host) {
  Array<Var> args;
  for (const Var &param : host->params) {
    if (auto buffer = host->buffer_map.Get(param))
      args.push_back(buffer.value()->data);
    else
      args.push_back(param);
  }
  return args;
}

// Passes every host argument at the host's calls of device kernels, and gives
// each kernel the parameters that receive them: its own for the arguments it
// used, new ones for the rest. Arguments that are not the host's keep their
// parameters, after these.
class KernelCallRewriter : public StmtExprMutator {
public:
  KernelCallRewriter(const IRModule &mod, Array<Var> args)
      : mod_(mod), args_(std::move(args)) {}

  std::unordered_map<GlobalVar, PrimFunc, ObjectPtrHash, ObjectPtrEqual>
      kernels;

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    auto gvar = op->op.as<GlobalVar>();
    auto kernel = gvar ? mod_->functions.Get(gvar.value()) : std::nullopt;
    if (!kernel || !kernel.value().as<PrimFuncNode>())
      return StmtExprMutator::VisitExpr_(op);
    PrimFunc f = Downcast<PrimFunc>(kernel.value());
    ICHECK_EQ(op->args.size(), f->params.size());
    std::unordered_map<const VarNode *, Var> used;
    Array<Var> extra_params;
    Array<PrimExpr> extra_args;
    for (size_t i = 0; i < op->args.size(); ++i) {
      const auto *arg = op->args[i].as<VarNode>();
      if (arg != nullptr && HostArgument(arg)) {
        used[arg] = f->params[i];
      } else {
        extra_params.push_back(f->params[i]);
        extra_args.push_back(op->args[i]);
      }
    }
    Array<Var> params;
    Array<PrimExpr> args;
    for (const Var &arg : args_) {
      auto it = used.find(arg.get());
      params.push_back(it != used.end()
                           ? it->second
                           : Var(arg->name_hint, arg->type_annotation));
      args.push_back(arg);
    }
    params.insert(params.end(), extra_params.begin(), extra_params.end());
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    auto known = kernels.find(gvar.value());
    ICHECK(known == kernels.end() ||
           known->second->params.size() == params.size())
        << "Hexagon: kernel " << gvar.value()->name_hint
        << " has calls with different arguments";
    f.CopyOnWrite()->params = params;
    kernels[gvar.value()] = f;
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

  bool HostArgument(const VarNode *var) const {
    for (const Var &arg : args_)
      if (arg.get() == var)
        return true;
    return false;
  }

  const IRModule &mod_;
  Array<Var> args_;
};

} // namespace

tvm::transform::Pass KeepKernelParams() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto &[gvar, base_func] : mod->functions) {
      auto host = base_func.as<PrimFunc>();
      if (!host)
        continue;
      KernelCallRewriter rewriter(mod, HostArguments(host.value()));
      Stmt body = rewriter(host.value()->body);
      if (rewriter.kernels.empty())
        continue;
      PrimFunc f = host.value();
      f.CopyOnWrite()->body = body;
      updates->Add(gvar, f);
      for (const auto &[kernel, func] : rewriter.kernels)
        updates->Add(kernel, func);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.hexagon.KeepKernelParams", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.KeepKernelParams",
                        KeepKernelParams);
}

} // namespace tl
} // namespace tvm
