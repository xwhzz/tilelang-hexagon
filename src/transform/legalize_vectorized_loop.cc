/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/utils.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <queue>
#include <unordered_set>
#include <vector>

#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using arith::IRMutatorWithAnalyzer;

// Class to legalize vectorized loops by transforming them appropriately
class LoopVectorizedLegalizer : IRMutatorWithAnalyzer {
public:
  // Static method to substitute and transform the given PrimFunc
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    // Create an instance of the legalizer with the analyzer
    LoopVectorizedLegalizer substituter(&analyzer);
    // Get a mutable copy of the function node
    PrimFuncNode *fptr = f.CopyOnWrite();
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  // Constructor initializing the base class with the analyzer
  LoopVectorizedLegalizer(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  static bool IsNonTrivialVectorizedLoop(const For &loop,
                                         arith::Analyzer *analyzer) {
    PrimExpr extent = analyzer->Simplify(loop->extent);
    auto extent_as_int = as_const_int(extent);
    return !extent_as_int || *extent_as_int > 1;
  }

  // A fact of the scope a statement is in: a loop variable's range, the
  // value a Bind statement before it gave a variable, or the condition of a
  // branch or an assumption it is under.
  struct Fact {
    Var var;
    PrimExpr value;
    Range range;
    PrimExpr condition;
  };

  // The constraints entered into a LoopAnalyzer, kept until the loop is
  // vectorized and then exited innermost first.
  using Constraints =
      std::vector<std::unique_ptr<With<arith::ConstraintContext>>>;

  // An analyzer that knows the facts of the current scope `loop` depends on:
  // the enclosing conditions, and the ranges and values of the variables the
  // loop reads, transitively. The mutator's own analyzer keeps every binding
  // of the function, its siblings' included, and vectorizing clones it per
  // loop: in a megakernel with thousands of bindings that made this pass
  // quadratic.
  void LoopAnalyzer(const For &loop, arith::Analyzer *analyzer,
                    Constraints *constraints) const {
    std::unordered_set<const VarNode *> needed;
    auto depend = [&needed](const Array<Var> &vars) {
      for (const Var &v : vars)
        needed.insert(v.get());
    };
    depend(UndefinedVars(loop, {}));
    // Innermost first: a fact the loop needs needs the facts of its own
    // variables.
    std::vector<const Fact *> used;
    for (auto it = scope_.rbegin(); it != scope_.rend(); ++it) {
      if (it->condition.defined()) {
        used.push_back(&*it);
        depend(UndefinedVars(it->condition));
      } else if (needed.count(it->var.get())) {
        used.push_back(&*it);
        depend(UndefinedVars(it->value.defined()
                                 ? it->value
                                 : it->range->min + it->range->extent));
      }
    }
    for (auto it = used.rbegin(); it != used.rend(); ++it) {
      const Fact &f = **it;
      if (f.condition.defined()) {
        constraints->push_back(std::make_unique<With<arith::ConstraintContext>>(
            analyzer, f.condition));
      } else if (f.value.defined()) {
        analyzer->Bind(f.var, f.value);
      } else {
        analyzer->Bind(f.var, f.range);
        constraints->push_back(std::make_unique<With<arith::ConstraintContext>>(
            analyzer, f.range->extent > 0));
      }
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // IRMutatorWithAnalyzer's, with each branch's condition a fact of its
    // scope.
    return constraint_scope_.WithNewScope([&]() -> Stmt {
      PrimExpr condition = this->VisitExpr(op->condition);
      PrimExpr real_condition = condition;
      static auto op_likely = Op::Get("tirx.likely");
      if (auto call = condition.as<CallNode>()) {
        if (call->op.same_as(op_likely))
          real_condition = call->args[0];
      }
      Stmt then_case;
      ffi::Optional<Stmt> else_case;
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_, real_condition);
        scope_.push_back({Var(), PrimExpr(), Range(), real_condition});
        WithRecordIterPredicate(real_condition, [&] {
          then_case = this->VisitStmt(op->then_case);
        });
        scope_.pop_back();
      });
      if (op->else_case) {
        PrimExpr neg_condition =
            analyzer_->rewrite_simplify(Not(real_condition));
        constraint_scope_.WithNewScope([&]() {
          constraint_scope_.Current().Emplace(analyzer_, neg_condition);
          scope_.push_back({Var(), PrimExpr(), Range(), neg_condition});
          else_case = this->VisitStmt(op->else_case.value());
          scope_.pop_back();
        });
      }
      if (is_one(real_condition))
        return then_case;
      if (is_zero(real_condition))
        return else_case.value_or(Evaluate(0));
      if (condition.same_as(op->condition) &&
          then_case.same_as(op->then_case) && else_case.same_as(op->else_case))
        return ffi::GetRef<Stmt>(op);
      auto n = this->CopyOnWrite(op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      return Stmt(n);
    });
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::tilelang_assume)
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    scope_.push_back(
        {Var(), PrimExpr(), Range(), Downcast<PrimExpr>(op->node)});
    Stmt stmt = IRMutatorWithAnalyzer::VisitStmt_(op);
    scope_.pop_back();
    return stmt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // A Bind's variable is in scope until the end of its sequence.
    size_t depth = scope_.size();
    Stmt stmt = IRMutatorWithAnalyzer::VisitStmt_(op);
    scope_.resize(depth);
    return stmt;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    Stmt stmt = IRMutatorWithAnalyzer::VisitStmt_(op);
    const auto *bind = stmt.as<BindNode>();
    if (SideEffect(bind->value) <= CallEffectKind::kPure)
      scope_.push_back({bind->var, bind->value, Range(), PrimExpr()});
    return stmt;
  }

  // Override the VisitStmt_ method to handle ForNode (loop statements)
  Stmt VisitStmt_(const ForNode *op) final {
    // Visit and potentially modify the loop node
    scope_.push_back({op->loop_var, PrimExpr(),
                      Range::FromMinExtent(op->min, op->extent), PrimExpr()});
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    scope_.pop_back();
    // If the loop is not vectorized, proceed with the default behavior
    if (for_node->kind != ForKind::kVectorized) {
      return for_node;
    }
    // Change the loop kind from vectorized to serial
    for_node.CopyOnWrite()->kind = ForKind::kSerial;
    // The analyzer outlives the constraints entered into it.
    auto analyzer = std::make_unique<arith::Analyzer>();
    Constraints constraints;
    LoopAnalyzer(for_node, analyzer.get(), &constraints);
    int vectorize_size = GetVectorizeSize(for_node, analyzer.get());
    if (vectorize_size <= 1 &&
        IsNonTrivialVectorizedLoop(for_node, analyzer.get())) {
      LOG(WARNING) << "T.vectorized loop over `" << for_node->loop_var
                   << "` with extent " << for_node->extent
                   << " is lowered as a serial loop because TileLang could "
                   << "not find a valid vectorization plan. Scalar accumulator "
                   << "updates inside the loop are a common cause; move "
                   << "reductions to T.unroll or T.serial if this is intended.";
    }
    // Apply vectorization transformation to the loop
    Stmt vectorized =
        VectorizeLoop(for_node, analyzer.get(), {}, vectorize_size);
    // Constraints exit in reverse order of entry: a vector's elements are
    // destroyed in an order the standard leaves open.
    while (!constraints.empty())
      constraints.pop_back();
    return vectorized;
  }

  std::vector<Fact> scope_;
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass LegalizeVectorizedLoop() {
  using namespace tirx::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LoopVectorizedLegalizer::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeVectorizedLoop", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeVectorizedLoop",
                        LegalizeVectorizedLoop);
}

} // namespace tl
} // namespace tvm
