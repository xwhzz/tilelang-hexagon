/*
 * \file rt_mod_hexagon.cc
 * \brief Register the Hexagon device-source build function with TVM:
 *        "target.build.tilelang_hexagon" -> emit C/C++ kernel source.
 *
 * Mirrors the CPU C backend (rt_mod_c.cc): generate source and wrap it in a
 * C-source module.  Actual compilation to a Hexagon cDSP skel (+ FastRPC glue)
 * is performed by the Python adapter using the Hexagon SDK, so this stage only
 * produces the kernel source string.
 */
#include "codegen_hexagon.h"
#include "support/check.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ir/cast.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using namespace ffi;

Module BuildTileLangHexagon(IRModule mod, Target target) {
  CodeGenTileLangHexagon cg;
  cg.Init(/*output_ssa=*/false);

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>())
        << "BuildTileLangHexagon: Can only take PrimFunc";
    funcs.push_back({gvar, Downcast<PrimFunc>(base_func)});
  }

  // Deterministic emission order for stable output.
  std::sort(funcs.begin(), funcs.end(),
            [](const auto &a, const auto &b) {
              return a.first->name_hint < b.first->name_hint;
            });

  for (const auto &[gvar, prim_func] : funcs) {
    cg.AddFunction(prim_func);
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("target.build.tilelang_hexagon", BuildTileLangHexagon);
}

} // namespace codegen
} // namespace tvm
