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
#include <optional>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/expr.h>
#include <unordered_set>
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
  // Print Hexagon builtins as calls to their runtime template functions.
  void VisitExpr_(const CallNode *op, std::ostream &os) final; // NOLINT(*)
  // Print a worker loop as a tl_parallel_for call. HVX-vectorize an innermost
  // elementwise loop (the `map` half of the backend's primitive basis); fall
  // back to CodeGenC's scalar loop otherwise.
  void VisitStmt_(const ForNode *op) final; // NOLINT(*)

  ffi::Array<ffi::String> GetFunctionNames() { return function_names_; }

protected:
  void EmitPreamble_(std::ostream &os) const final;

private:
  /*! \brief names of the kernel functions emitted in this module */
  ffi::Array<ffi::String> function_names_;

  /*! \brief compile-time bump offset (bytes) for placing alloc_shared buffers
   *  in the current VTCM arena: the kernel's, or a worker's. */
  size_t vtcm_offset_ = 0;
  /*! \brief the target's vtcm-capacity in bytes; 0 when the target has none
   *  and only the runtime checks the grant. */
  size_t vtcm_capacity_ = 0;
  /*! \brief whether a tile program's DMA queue (tl_dma_async) is in scope */
  bool dma_in_scope_ = false;
  /*! \brief whether any emitted kernel uses the managed DMA queue */
  bool uses_dma_async_ = false;
  /*! \brief whether the emitter is inside a worker loop body */
  bool in_worker_ = false;
  // Print a T.Kernel(num_workers=) grid loop as a tl_parallel_for call whose
  // lambda runs one tile program.
  void PrintWorkerLoop(const ForNode *op);

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
  // One recursive walk for both phases: emit=false is the side-effect-free
  // supportability probe (returns nullopt if `e` isn't a supported elementwise
  // expr); emit=true writes the HVX statements and returns the result lanes.
  // Single op-set — no separate detection function to drift out of sync.
  std::optional<HvxLanes> EmitHvxExpr(const PrimExpr &e, const tirx::Var &j,
                                      arith::Analyzer *ana, bool emit);
  HvxLanes EmitHvxOp(const HvxLanes &a, const HvxLanes &b, const char *fn,
                     bool emit);
  HvxLanes EmitHvxUnary(const HvxLanes &a, const char *fn, bool emit);
  // Whether a vectorized access buf[idx] (idx = base + j, j stepping 64 fp16)
  // is 128-byte (HVX vector) aligned: shared VTCM tiles are merge-aligned to
  // 128, so it reduces to base % 64 == 0 (provable via the analyzer).  Drives
  // the aligned-vmem vs unaligned-vmemu choice.
  bool AccessAligned128(const BufferNode *buf, const PrimExpr &idx,
                        const tirx::Var &j, arith::Analyzer *ana);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_CODEGEN_HEXAGON_H_
