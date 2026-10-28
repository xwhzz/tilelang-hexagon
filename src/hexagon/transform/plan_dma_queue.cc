/*!
 * \file tl/hexagon/transform/plan_dma_queue.cc
 * \brief Size and place the DMA queue of every Hexagon tile program.
 *
 * T.dma_copy lowers to hexagon_dma_submit calls and one hexagon_dma_commit,
 * and T.dma_wait(n) to hexagon_dma_wait(n). Each tile program owns one queue.
 * This pass bounds the descriptors and the committed groups in flight at any
 * point of the program, declares the queue with those capacities at the start
 * of the tile program scope, and drains it at the end unless the program
 * already waits for everything.
 *
 * A transfer keeps its VTCM buffers in use until the wait that completes it,
 * so every wait lists the VTCM buffers whose transfers it may complete, after
 * its count. Shared memory planning sees them as accesses there.
 */

#include "hexagon/op/builtin.h"
#include "transform/common/attr.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The runtime queue holds at most this many descriptors and groups.
constexpr int64_t kMaxInFlight = 256;
// Loops with at most this many iterations are simulated exactly.
constexpr int64_t kMaxSimulatedTrips = 256;

bool IsDMACall(const CallNode *call) {
  return call->op.same_as(hexagon_dma_submit()) ||
         call->op.same_as(hexagon_dma_commit()) ||
         call->op.same_as(hexagon_dma_wait());
}

bool IsRawDMACall(const CallNode *call) {
  if (!call->op.same_as(builtin::call_extern()) || call->args.empty())
    return false;
  const auto *name = call->args[0].as<StringImmNode>();
  return name && std::string(name->value).rfind("tl_hexagon_dma_", 0) == 0;
}

bool UsesDMA(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>())
      found |= IsDMACall(call);
  });
  return found;
}

// A buffer of the tile program's VTCM arena, whose space
// MergeSharedMemoryAllocations reuses: a DMA buffer must stay live until the
// wait that completes it.
bool IsArenaBuffer(const VarNode *var) {
  const auto *ptr = var->type_annotation.as<PointerTypeNode>();
  return ptr &&
         (ptr->storage_scope == "shared" || ptr->storage_scope == "shared.dyn");
}

// Descriptors in flight, and the VTCM buffers they access, numbered in order
// of their first submission.
struct Group {
  int64_t descriptors = 0;
  std::set<int> buffers;

  bool operator==(const Group &other) const {
    return descriptors == other.descriptors && buffers == other.buffers;
  }

  static Group Join(const Group &a, const Group &b) {
    Group out{std::max(a.descriptors, b.descriptors), a.buffers};
    out.buffers.insert(b.buffers.begin(), b.buffers.end());
    return out;
  }
};

// What may be in flight: each committed, pending group (oldest first), and the
// open group.
struct Queue {
  std::deque<Group> groups;
  Group open;

  bool operator==(const Queue &other) const {
    return groups == other.groups && open == other.open;
  }

  // A queue that bounds both: waits keep the newest groups, so align them.
  static Queue Join(const Queue &a, const Queue &b) {
    Queue out;
    out.open = Group::Join(a.open, b.open);
    size_t n = std::max(a.groups.size(), b.groups.size());
    for (size_t i = 0; i < n; ++i) {
      Group x =
          i < a.groups.size() ? a.groups[a.groups.size() - 1 - i] : Group();
      Group y =
          i < b.groups.size() ? b.groups[b.groups.size() - 1 - i] : Group();
      out.groups.push_front(Group::Join(x, y));
    }
    return out;
  }
};

class QueueBound : public StmtVisitor {
public:
  int64_t descriptors = 0;
  int64_t groups = 0;
  // The VTCM buffers each submission accesses, in order of first submission.
  std::vector<Var> buffers;
  // The buffers whose transfers each wait may complete.
  std::unordered_map<const CallNode *, std::set<int>> completes;

  void Run(const Stmt &body) {
    VisitStmt(body);
    Check();
  }

  //! True if nothing can be in flight at the end of the body.
  bool Drained() const {
    return queue_.groups.empty() && queue_.open.descriptors == 0;
  }

  //! The buffers of the transfers that may be in flight at the end.
  std::set<int> InFlightBuffers() const {
    std::set<int> out = queue_.open.buffers;
    for (const Group &group : queue_.groups)
      out.insert(group.buffers.begin(), group.buffers.end());
    return out;
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !IsDMACall(call))
      return;
    if (call->op.same_as(hexagon_dma_submit())) {
      ++queue_.open.descriptors;
      for (const PrimExpr &arg : call->args) {
        PostOrderVisit(arg, [this](const ObjectRef &node) {
          if (const auto *var = node.as<VarNode>(); var && IsArenaBuffer(var))
            queue_.open.buffers.insert(BufferIndex(var));
        });
      }
      descriptors = std::max(descriptors, InFlight());
    } else if (call->op.same_as(hexagon_dma_commit())) {
      queue_.groups.push_back(std::move(queue_.open));
      queue_.open = Group();
      groups = std::max(groups, static_cast<int64_t>(queue_.groups.size()));
    } else {
      const auto *pending = call->args[0].as<IntImmNode>();
      ICHECK(pending && pending->value >= 0)
          << "T.dma_wait needs a constant, nonnegative count";
      std::set<int> &done = completes[call];
      while (static_cast<int64_t>(queue_.groups.size()) > pending->value) {
        const std::set<int> &retired = queue_.groups.front().buffers;
        done.insert(retired.begin(), retired.end());
        queue_.groups.pop_front();
      }
    }
    Check();
  }

  int BufferIndex(const VarNode *var) {
    auto [it, inserted] = index_.emplace(var, static_cast<int>(buffers.size()));
    if (inserted)
      buffers.push_back(GetRef<Var>(var));
    return it->second;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    Queue before = queue_;
    VisitStmt(op->then_case);
    Queue then_queue = queue_;
    queue_ = before;
    if (op->else_case)
      VisitStmt(op->else_case.value());
    queue_ = Queue::Join(then_queue, queue_);
  }

  void VisitStmt_(const ForNode *op) final {
    if (!UsesDMA(op->body))
      return;
    const auto *extent = op->extent.as<IntImmNode>();
    if (extent && extent->value <= kMaxSimulatedTrips) {
      for (int64_t i = 0; i < extent->value; ++i)
        VisitStmt(op->body);
      return;
    }
    Iterate(op->body, /*known_trips=*/extent != nullptr);
  }

  void VisitStmt_(const WhileNode *op) final {
    if (UsesDMA(op->body))
      Iterate(op->body, /*known_trips=*/false);
  }

  // The queue after any number of iterations, zero included: iterate until it
  // stops growing. A loop that grows it on every iteration either has more
  // than kMaxInFlight known iterations or an unknown count.
  void Iterate(const Stmt &body, bool known_trips) {
    unknown_trips_ += !known_trips;
    for (int64_t i = 0; i <= kMaxInFlight; ++i) {
      Queue before = queue_;
      VisitStmt(body);
      queue_ = Queue::Join(before, queue_);
      if (queue_ == before) {
        unknown_trips_ -= !known_trips;
        return;
      }
    }
    descriptors = std::max(descriptors, kMaxInFlight + 1);
    Check();
  }

  int64_t InFlight() const {
    int64_t n = queue_.open.descriptors;
    for (const Group &group : queue_.groups)
      n += group.descriptors;
    return n;
  }

  void Check() const {
    if (descriptors <= kMaxInFlight && groups <= kMaxInFlight)
      return;
    if (unknown_trips_ > 0) {
      LOG(FATAL) << "A loop with an unknown trip count submits DMA copies "
                    "that it does not wait for, so the DMA queue has no bound";
    }
    LOG(FATAL) << "A Hexagon tile program keeps more than " << kMaxInFlight
               << " DMA transfers in flight; wait for older T.dma_copy calls "
                  "first";
  }

  Queue queue_;
  int unknown_trips_ = 0;
  std::unordered_map<const VarNode *, int> index_;
};

// Appends to every wait the VTCM buffers whose transfers it may complete.
class WaitCompletions : public StmtExprMutator {
public:
  explicit WaitCompletions(const QueueBound &bound) : bound_(bound) {}

  static Call Wait(PrimExpr pending, const std::set<int> &buffers,
                   const std::vector<Var> &vars) {
    Array<PrimExpr> args{std::move(pending)};
    for (int i : buffers)
      args.push_back(vars[i]);
    return Call(DataType::Int(32), hexagon_dma_wait(), args);
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(hexagon_dma_wait()))
      return StmtExprMutator::VisitExpr_(op);
    ICHECK_EQ(op->args.size(), 1U) << "hexagon_dma_wait is planned once";
    auto it = bound_.completes.find(op);
    if (it == bound_.completes.end() || it->second.empty())
      return GetRef<PrimExpr>(op);
    return Wait(op->args[0], it->second, bound_.buffers);
  }

  const QueueBound &bound_;
};

// Declares the queue at the start of every tile program that uses DMA and
// drains it at the end.
class DMAQueuePlanner : public StmtMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kTileProgramScope || !UsesDMA(op->body))
      return StmtMutator::VisitStmt_(op);
    QueueBound bound;
    bound.Run(op->body);
    auto capacity = [](int64_t n) {
      return IntImm(DataType::Int(32), std::max<int64_t>(n, 1));
    };
    Stmt queue =
        Evaluate(Call(DataType::Int(32), hexagon_dma_queue(),
                      {capacity(bound.descriptors), capacity(bound.groups)}));
    Array<Stmt> body{queue, WaitCompletions(bound)(op->body)};
    if (!bound.Drained()) {
      body.push_back(Evaluate(
          WaitCompletions::Wait(IntImm(DataType::Int(32), 0),
                                bound.InFlightBuffers(), bound.buffers)));
    }
    return AttrStmt(op->node, op->attr_key, op->value, SeqStmt::Flatten(body),
                    op->span);
  }
};

// Rejects DMA outside a tile program, and managed and raw DMA in one kernel.
class DMAOwnershipChecker : public StmtExprVisitor {
public:
  static void Check(const Stmt &body) {
    DMAOwnershipChecker checker;
    checker(body);
    ICHECK(!(checker.managed_ && checker.raw_))
        << "Hexagon async DMA cannot share a kernel with raw DMA calls";
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    bool scope = op->attr_key == attr::kTileProgramScope;
    programs_ += scope;
    StmtExprVisitor::VisitStmt_(op);
    programs_ -= scope;
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsDMACall(op)) {
      managed_ = true;
      ICHECK(programs_ > 0) << "Hexagon T.dma_copy must run inside a tile "
                               "program";
    }
    raw_ |= IsRawDMACall(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  int programs_ = 0;
  bool managed_ = false;
  bool raw_ = false;
};

} // namespace

namespace transform {

tvm::transform::Pass PlanDMAQueue() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    DMAOwnershipChecker::Check(func->body);
    func.CopyOnWrite()->body = DMAQueuePlanner()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.PlanDMAQueue", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.PlanDMAQueue", PlanDMAQueue);
}

} // namespace transform
} // namespace tl
} // namespace tvm
