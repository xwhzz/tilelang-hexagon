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
#include <tvm/arith/analyzer.h>
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
  // Reject HMX gemm/matmul calls inside a worker-pool kernel (scratch is global).
  void VisitExpr_(const CallNode *op, std::ostream &os) final; // NOLINT(*)
  // HVX-vectorize an innermost elementwise loop (the `map` half of the backend's
  // primitive basis); fall back to CodeGenC's scalar loop otherwise.
  void VisitStmt_(const ForNode *op) final; // NOLINT(*)

  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

private:
  /*! \brief names of the kernel functions emitted in this module */
  ffi::Array<ffi::String> function_names_;

  /*! \brief compile-time bump offset (bytes) for placing alloc_shared buffers
   *  in the VTCM arena; reset per function in AddFunction. */
  size_t vtcm_offset_ = 0;

  /*! \brief worker-pool (multithreaded grid) state, set per function.  When the
   *  device func carries `hexagon.num_workers` > 0, the outermost block loop is
   *  distributed across HW threads via tl_parallel rather than a serial loop. */
  int wp_num_workers_ = 0;            // requested worker count (0 = serial / off)
  bool wp_emit_ = false;             // currently emitting inside the worker callback
  bool wp_outermost_pending_ = false; // next thread_extent is the (strided) block loop
  size_t wp_stride_ = 0;       // per-worker VTCM region = wp_operand_bytes_ + gemm scratch
  size_t wp_operand_bytes_ = 0; // per-worker alloc_shared bytes (operand high-water)
  size_t wp_gemm_scratch_ = 0;  // per-worker HMX gemm Crouton scratch bytes (max over gemms)
  bool wp_uses_hmx_ = false;    // a T.gemm->HMX appears in the body (workers enable HMX)

  template <typename T>
  inline void PrintTernaryCondExpr(const T *op, const char *compare,
                                   std::ostream &os); // NOLINT(*)

  // ---- HVX elementwise ("map") vectorizer (see codegen_hexagon.cc) ----
  // A subexpression's value as a pair of 32-lane fp32 HVX vectors backing 64
  // fp16 lanes; `bcast` means lo == hi (a splatted j-independent scalar).
  struct HvxLanes {
    std::string lo, hi;
    bool bcast;
  };
  // Try to emit `op` (an innermost loop) as a full-width HVX elementwise loop;
  // returns false (emitting nothing) if the loop isn't a vectorizable map.
  bool TryEmitHvxElementwise(const ForNode *op);
  // Whether `e` is a supported elementwise expression over loop var `j`
  // (j-contiguous fp16 loads / j-independent broadcasts / + - * / exp).
  bool HvxExprSupported(const PrimExpr &e, const tirx::Var &j,
                        arith::Analyzer *ana);
  // Recursively emit the HVX statements computing `e`; returns the result lanes.
  HvxLanes EmitHvxExpr(const PrimExpr &e, const tirx::Var &j,
                       arith::Analyzer *ana);
  HvxLanes EmitHvxOp(const HvxLanes &a, const HvxLanes &b, const char *fn);
  HvxLanes EmitHvxUnary(const HvxLanes &a, const char *fn);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_CODEGEN_HEXAGON_H_
