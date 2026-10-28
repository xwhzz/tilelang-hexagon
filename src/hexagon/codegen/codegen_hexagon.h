/*
 * \file codegen_hexagon.h
 * \brief Generate C/C++ device source for the Qualcomm Hexagon NPU (cDSP).
 *
 * Emits a plain `extern "C"` kernel function per device PrimFunc, to be
 * compiled by hexagon-clang++ (HVX/HMX) and invoked from a FastRPC skel.
 * Hardware intimacy (HMX/HVX/VTCM) is deliberately kept out of the codegen and
 * lives in tl_templates/hexagon headers; this class is a thin source emitter
 * built on CodeGenTileLangCBase, like the CPU and Metal backends.
 */
#ifndef TVM_TL_CODEGEN_HEXAGON_H_
#define TVM_TL_CODEGEN_HEXAGON_H_

#include "backend/common/codegen/codegen_c_base.h"
#include "tvm/target/codegen.h"
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/expr.h>
#include <vector>

namespace tvm {
namespace codegen {

class CodeGenTileLangHexagon : public CodeGenTileLangCBase {
public:
  CodeGenTileLangHexagon();

  /*! \brief Emit the preamble (template include) and initialize CodeGenC. */
  void Init(bool output_ssa);

  /*! \brief Emit one device kernel function from a PrimFunc. */
  void AddFunction(const PrimFunc &f);

  using CodeGenC::PrintType;
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintFuncPrefix(std::ostream &os) final;       // NOLINT(*)

  void VisitStmt_(const AllocBufferNode *op) final; // NOLINT(*)
  // Vectorized copy/fill emit broadcasts; mirror the CPU C backend's emission.
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  // Declare the DMA queue, and return the status of failing runtime calls.
  void VisitStmt_(const EvaluateNode *op) final; // NOLINT(*)
  // Scope the DMA queue to its tile program.
  void VisitStmt_(const AttrStmtNode *op) final; // NOLINT(*)
  // Use the ternary operator for min/max so we don't depend on a host stdlib.
  void VisitExpr_(const MinNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream &os) final; // NOLINT(*)
  // HVX has no divide: print float vector division as tl_hvx_div (hvx_math.h).
  void VisitExpr_(const DivNode *op, std::ostream &os) final; // NOLINT(*)
  // A lane-wise select prints as tl_vec_select, a ramp as tl_vec_ramp.
  void VisitExpr_(const SelectNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const RampNode *op, std::ostream &os) final;   // NOLINT(*)
  // Print Hexagon builtins as calls to their runtime template functions.
  void VisitExpr_(const CallNode *op, std::ostream &os) final; // NOLINT(*)
  // Print a worker loop as a tl_parallel_for call.
  void VisitStmt_(const ForNode *op) final; // NOLINT(*)
  // A contiguous vector load is one vector access wherever it starts (C's
  // printer gathers the lanes of one it cannot prove aligned). A vector
  // access that may not start on its alignment prints as tl_hvx_vec_loadu /
  // tl_hvx_vec_storeu (hvx_lanes.h), vmemu.
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) final; // NOLINT(*)
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode *buffer, DataType t, PrimExpr base,
                     const std::string &value) final;

  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

protected:
  void EmitPreamble_(std::ostream &os) const final;

private:
  /*! \brief names of the kernel functions emitted in this module */
  ffi::Array<ffi::String> function_names_;

  /*! \brief whether the current tile program's VTCM arena is bound */
  bool arena_bound_ = false;
  /*! \brief whether a tile program's DMA queue (tl_dma_async) is in scope */
  bool dma_in_scope_ = false;
  /*! \brief whether any emitted kernel uses the managed DMA queue */
  bool uses_dma_async_ = false;
  /*! \brief whether the emitter is inside a worker loop body */
  bool in_worker_ = false;
  /*! \brief the program index of the worker loop being emitted */
  ffi::Optional<tirx::Var> worker_program_;
  // Print a T.Kernel(num_workers=) grid loop as a tl_parallel_for call whose
  // lambda runs one tile program.
  void PrintWorkerLoop(const ForNode *op);
  // Whether a vector of `t` at element `base` provably starts on a multiple of
  // its natural alignment, the smaller of its size and a register.
  bool VectorAligned(DataType t, const PrimExpr &base);
  /*! \brief modular arithmetic of vector access offsets */
  arith::Analyzer analyzer_;

  template <typename T>
  inline void PrintTernaryCondExpr(const T *op, const char *compare,
                                   std::ostream &os); // NOLINT(*)
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_CODEGEN_HEXAGON_H_
