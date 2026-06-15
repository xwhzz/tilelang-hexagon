/*
 * \file codegen_hexagon.h
 * \brief Generate C/C++ device source for the Qualcomm Hexagon NPU (cDSP).
 *
 * Emits a plain `extern "C"` kernel function per device PrimFunc, to be compiled
 * by hexagon-clang++ (HVX/HMX) and invoked from a FastRPC skel.  Hardware
 * intimacy (HMX/HVX/VTCM) is deliberately kept out of the codegen and lives in
 * tl_templates/hexagon headers; this class is a thin source emitter built on
 * TVM's CodeGenC (the same pattern as the CPU/Metal backends).
 */
#ifndef TVM_TL_CODEGEN_HEXAGON_H_
#define TVM_TL_CODEGEN_HEXAGON_H_

#include "target/source/codegen_c.h"
#include "tvm/target/codegen.h"
#include <string>
#include <tvm/tirx/expr.h>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace codegen {

class CodeGenTileLangHexagon : public CodeGenC {
public:
  CodeGenTileLangHexagon();

  /*! \brief Emit the preamble (template include) and initialize CodeGenC. */
  void Init(bool output_ssa);

  /*! \brief Emit one device kernel function from a PrimFunc. */
  void AddFunction(const PrimFunc &f);

  using CodeGenC::PrintType;
  void PrintType(DataType t, std::ostream &os) final;   // NOLINT(*)
  void PrintFuncPrefix(std::ostream &os) final;         // NOLINT(*)

  void VisitStmt_(const AllocBufferNode *op) final;      // NOLINT(*)
  // Vectorized copy/fill emit broadcasts; mirror the CPU C backend's emission.
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  // Serialize the GPU-style grid: bind each thread_extent IterVar to a loop.
  void VisitStmt_(const AttrStmtNode *op) final;         // NOLINT(*)
  // Use the ternary operator for min/max so we don't depend on a host stdlib.
  void VisitExpr_(const MinNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream &os) final; // NOLINT(*)

  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

private:
  /*! \brief names of the kernel functions emitted in this module */
  ffi::Array<ffi::String> function_names_;

  template <typename T>
  inline void PrintTernaryCondExpr(const T *op, const char *compare,
                                   std::ostream &os); // NOLINT(*)
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_CODEGEN_HEXAGON_H_
