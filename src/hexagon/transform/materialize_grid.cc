/*!
 * \file tl/hexagon/transform/materialize_grid.cc
 * \brief Give the Hexagon launch grid its final loops.
 *
 * Runs after the loop transformations, on the grid loops LowerKernelLaunch
 * marked. The body of the innermost grid loop is the tile program.
 *
 *  - StorageRewrite hoists allocations to function scope. Those used only by
 *    the tile program are sunk back into it, so every program, and so every
 *    worker, owns its buffers; global and global.vtcm buffers stay, one for
 *    all the programs.
 *  - The program body becomes a tile program scope, where
 *    MergeSharedMemoryAllocations places the program's VTCM arena.
 *  - With num_workers > 1 the grid becomes one parallel loop over the
 *    linearized grid index that carries the worker count. Only the perfectly
 *    nested grid loops are linearized; a grid axis below an intervening
 *    statement stays serial inside the workers.
 *  - Programs that meet at T.sync_grid must all run at once, so their grid
 *    may have no more programs than workers, and no serial grid axis.
 *  - Hexagon kernels are called directly, so the whole function body is
 *    device code.
 */

#include "hexagon/op/builtin.h"
#include "op/builtin.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsGridLoop(const ForNode *loop) {
  return loop->annotations.count(attr::kHexagonGridDim) > 0;
}

bool ContainsGridLoop(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>())
      found |= IsGridLoop(loop);
  });
  return found;
}

// The innermost grid loop: its body is the tile program.
bool IsProgramLoop(const ForNode *loop) {
  return IsGridLoop(loop) && !ContainsGridLoop(loop->body);
}

bool IsSharedScope(const Var &data) {
  std::string scope = GetPtrStorageScope(data);
  return scope == "shared" || scope == "shared.dyn";
}

// Records, for every buffer data var, whether its uses all lie in one tile
// program body. Declarations (AllocBuffer, DeclBuffer) are not uses.
class ProgramUseAnalysis : public StmtExprVisitor {
public:
  //! The program loop whose body holds every use of `data`, if one does.
  const ForNode *Owner(const VarNode *data) const {
    auto it = owner_.find(data);
    return it == owner_.end() || shared_.count(data) ? nullptr : it->second;
  }

  //! True if anything uses `data`.
  bool Used(const VarNode *data) const { return owner_.count(data) > 0; }

  //! True if a tile program uses `data`.
  bool UsedByProgram(const VarNode *data) const {
    return used_by_program_.count(data) > 0;
  }

  std::vector<const ForNode *> programs;

private:
  void VisitStmt_(const ForNode *op) final {
    if (!IsProgramLoop(op)) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    programs.push_back(op);
    VisitExpr(op->min);
    VisitExpr(op->extent);
    const ForNode *outer = program_;
    program_ = op;
    VisitStmt(op->body);
    program_ = outer;
  }

  void VisitStmt_(const DeclBufferNode *op) final {}

  void VisitStmt_(const BufferStoreNode *op) final {
    Use(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Use(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final { Use(op); }

  void Use(const VarNode *data) {
    if (program_)
      used_by_program_.insert(data);
    auto [it, inserted] = owner_.emplace(data, program_);
    if (!inserted && it->second != program_)
      shared_.insert(data);
  }

  const ForNode *program_ = nullptr;
  std::unordered_map<const VarNode *, const ForNode *> owner_;
  std::unordered_set<const VarNode *> shared_;
  std::unordered_set<const VarNode *> used_by_program_;
};

bool ContainsGridSync(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>())
      found |= call->op.same_as(tl::sync_grid());
  });
  return found;
}

// Rejects runtime calls whose state is not per worker.
void CheckWorkerBody(const Stmt &body) {
  PostOrderVisit(body, [](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (!call || !call->op.same_as(builtin::call_extern()) ||
        call->args.empty())
      return;
    const auto *name = call->args[0].as<StringImmNode>();
    if (!name)
      return;
    const std::string &callee = name->value;
    ICHECK(callee.rfind("tl_hexagon_hmx", 0) != 0)
        << "`" << callee
        << "` is not worker-safe: inside T.Kernel(num_workers=...) only the "
           "T.hmx atoms may use HMX";
  });
}

class GridMaterializer : public StmtExprMutator {
public:
  explicit GridMaterializer(const ProgramUseAnalysis &uses) : uses_(uses) {}

private:
  // Allocations outside the program are moved to the start of the program
  // body, along with the declarations of the buffers they back.
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    if (program_)
      return StmtExprMutator::VisitStmt_(op);
    Array<Stmt> kept;
    for (const Stmt &stmt : op->seq) {
      if (const ForNode *program = SinkTarget(stmt)) {
        sunk_[program].push_back(stmt);
        continue;
      }
      kept.push_back(VisitStmt(stmt));
    }
    if (kept.empty())
      return Evaluate(0);
    if (kept.size() == 1)
      return kept[0];
    return SeqStmt(kept);
  }

  const ForNode *SinkTarget(const Stmt &stmt) {
    if (const auto *alloc = stmt.as<AllocBufferNode>()) {
      const Var &data = alloc->buffer->data;
      // DDR workspaces, and VTCM the programs share (global.vtcm), stay
      // outside the programs: one buffer for all of them.
      std::string scope = GetPtrStorageScope(data);
      if (scope == "global" || scope == "global.vtcm")
        return nullptr;
      const ForNode *program = uses_.Owner(data.get());
      if (!program) {
        // A buffer the program shares with code outside it cannot be private
        // to each program, and a shared buffer must live in the arena.
        ICHECK(!uses_.UsedByProgram(data.get()) &&
               !(IsSharedScope(data) && uses_.Used(data.get())))
            << "Buffer `" << alloc->buffer->name
            << "` is used outside the Hexagon tile program body";
        return nullptr;
      }
      sunk_data_[data.get()] = program;
      return program;
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      auto it = sunk_data_.find(decl->buffer->data.get());
      return it == sunk_data_.end() ? nullptr : it->second;
    }
    return nullptr;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!IsGridLoop(op))
      return StmtExprMutator::VisitStmt_(op);

    std::vector<const ForNode *> chain{op};
    while (const auto *next = chain.back()->body.as<ForNode>()) {
      if (!IsGridLoop(next))
        break;
      chain.push_back(next);
    }
    for (const ForNode *loop : chain) {
      ICHECK(!loop->step.defined() || is_one(loop->step.value()))
          << "Hexagon grid loop `" << loop->loop_var->name_hint
          << "` must have a unit step";
    }
    const ForNode *last = chain.back();
    Stmt body = IsProgramLoop(last) ? MakeProgram(last) : VisitStmt(last->body);

    int64_t num_workers = 1;
    if (auto value = op->annotations.Get(attr::kHexagonNumWorkers))
      num_workers = value->cast<Integer>()->value;
    if (ContainsGridSync(body))
      CheckCoresident(chain, num_workers);
    // The programs of a num_workers grid follow the worker rules even when the
    // grid has only one, so legality does not depend on the grid size.
    if (num_workers > 1)
      CheckWorkerBody(body);
    if (num_workers > 1 && !HasSingleIteration(chain))
      return MakeParallel(chain, body, num_workers);

    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
      body = MakeLoop(*it, ForKind::kSerial, body, {});
    return body;
  }

  Stmt MakeProgram(const ForNode *program) {
    program_ = program;
    Stmt body = VisitStmt(program->body);
    program_ = nullptr;
    auto it = sunk_.find(program);
    if (it != sunk_.end()) {
      Array<Stmt> seq = it->second;
      seq.push_back(body);
      body = SeqStmt::Flatten(seq);
    }
    return AttrStmt(Integer(0), attr::kTileProgramScope, Integer(1), body);
  }

  // T.sync_grid makes the programs of the launch wait for each other, so each
  // needs its own worker.
  static void CheckCoresident(const std::vector<const ForNode *> &chain,
                              int64_t num_workers) {
    ICHECK(IsProgramLoop(chain.back()))
        << "T.sync_grid() needs every tile program of the launch running at "
           "once, but the grid axes below `"
        << chain.back()->loop_var->name_hint
        << "` run in the workers one program after another: a statement "
           "separates them from it";
    int64_t programs = 1;
    for (const ForNode *loop : chain) {
      const auto *extent = loop->extent.as<IntImmNode>();
      ICHECK(extent) << "T.sync_grid() needs a grid of constant size, but "
                     << "grid axis `" << loop->loop_var->name_hint
                     << "` has extent " << loop->extent;
      programs *= extent->value;
    }
    ICHECK(programs <= num_workers)
        << "T.sync_grid() needs every tile program of the launch running at "
           "once, one per worker, but the grid has "
        << programs << " programs and num_workers=" << num_workers;
  }

  static bool HasSingleIteration(const std::vector<const ForNode *> &chain) {
    for (const ForNode *loop : chain) {
      const auto *extent = loop->extent.as<IntImmNode>();
      if (!extent || extent->value > 1)
        return false;
    }
    return true;
  }

  // One parallel loop over the linearized index of `chain`, which rebinds
  // each grid variable from it.
  Stmt MakeParallel(const std::vector<const ForNode *> &chain, Stmt body,
                    int64_t num_workers) {
    Map<String, Any> annotations;
    annotations.Set(attr::kHexagonNumWorkers,
                    Integer(static_cast<int>(num_workers)));
    const ForNode *head = chain.front();
    if (chain.size() == 1)
      return MakeLoop(head, ForKind::kParallel, body, annotations);

    DataType dtype = head->loop_var.dtype();
    std::string name;
    arith::Analyzer analyzer;
    PrimExpr extent = make_const(dtype, 1);
    for (const ForNode *loop : chain) {
      name += loop->loop_var->name_hint + "_";
      extent = analyzer.Simplify(extent * cast(dtype, loop->extent));
    }
    Var fused(name + "fused", dtype);
    analyzer.Bind(fused, Range::FromMinExtent(make_zero(dtype), extent));
    // Row-major: the innermost grid axis varies fastest.
    PrimExpr stride = make_const(dtype, 1);
    std::vector<Stmt> binds;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      const ForNode *loop = *it;
      PrimExpr index = floordiv(fused, stride);
      if (loop != head)
        index = floormod(index, cast(dtype, loop->extent));
      binds.push_back(Bind(loop->loop_var,
                           analyzer.Simplify(cast(dtype, loop->min) + index)));
      stride = analyzer.Simplify(stride * cast(dtype, loop->extent));
    }
    Array<Stmt> seq(binds.rbegin(), binds.rend());
    seq.push_back(body);
    return For(fused, make_zero(dtype), extent, ForKind::kParallel,
               SeqStmt::Flatten(seq), std::nullopt, annotations, std::nullopt,
               head->span);
  }

  // `loop` rebuilt with `kind`, `body`, and its non-grid annotations added to
  // `annotations`.
  static Stmt MakeLoop(const ForNode *loop, ForKind kind, Stmt body,
                       Map<String, Any> annotations) {
    for (const auto &[key, value] : loop->annotations) {
      if (key != attr::kHexagonGridDim && key != attr::kHexagonNumWorkers)
        annotations.Set(key, value);
    }
    return For(loop->loop_var, loop->min, loop->extent, kind, std::move(body),
               std::nullopt, annotations, std::nullopt, loop->span);
  }

  const ProgramUseAnalysis &uses_;
  const ForNode *program_ = nullptr;
  std::unordered_map<const ForNode *, Array<Stmt>> sunk_;
  std::unordered_map<const VarNode *, const ForNode *> sunk_data_;
};

} // namespace

namespace transform {

tvm::transform::Pass MaterializeGrid() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    ProgramUseAnalysis uses;
    uses(func->body);
    ICHECK_LE(uses.programs.size(), 1U)
        << "Hexagon supports one T.Kernel per prim_func, but `"
        << func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("")
        << "` launches " << uses.programs.size();
    // A launch without grid axes is one tile program: the whole body.
    Stmt body = uses.programs.empty()
                    ? AttrStmt(Integer(0), attr::kTileProgramScope, Integer(1),
                               func->body)
                    : GridMaterializer(uses)(func->body);
    func.CopyOnWrite()->body =
        AttrStmt(Integer(0), tirx::attr::device_scope, Integer(0), body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.MaterializeGrid", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.MaterializeGrid",
                        MaterializeGrid);
}

} // namespace transform
} // namespace tl
} // namespace tvm
