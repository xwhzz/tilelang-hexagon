/*!
 * \file tl/hexagon/transform/verify_vtcm_arena.cc
 * \brief Check that the VTCM arena of every Hexagon tile program fits.
 *
 * MergeSharedMemoryAllocations lays out the shared buffers of a tile program in
 * one allocation, its VTCM arena. A kernel's arena starts at the VTCM base. A
 * T.Kernel(num_workers=N) grid gives each of N workers its own slice, which the
 * runtime rounds up to whole HMX tiles so that every slice keeps the arena's
 * alignment. Buffers all the programs share (global.vtcm) come first, whole
 * HMX tiles each, and the arenas after them. All of it must fit the target's
 * vtcm-capacity; the runtime still checks the region it is granted.
 */

#include "hexagon/op/builtin.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"
#include "transform/common/storage_size.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <string>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// A worker's slice is a whole number of HMX tiles (TL_VTCM_HMX_ALIGNMENT).
constexpr int64_t kSliceAlignment = 2048;

class ArenaVerifier : public StmtVisitor {
public:
  ArenaVerifier(int64_t capacity, int64_t grid)
      : capacity_(capacity), grid_(grid) {}

private:
  void VisitStmt_(const ForNode *op) final {
    auto workers = op->annotations.Get(attr::kHexagonNumWorkers);
    if (!workers) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    int64_t outer = workers_;
    workers_ = workers->cast<Integer>()->value;
    StmtVisitor::VisitStmt_(op);
    workers_ = outer;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kTileProgramScope) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    bool outer = has_arena_;
    has_arena_ = false;
    StmtVisitor::VisitStmt_(op);
    has_arena_ = outer;
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    const Buffer &buffer = op->buffer;
    std::string scope = GetPtrStorageScope(buffer->data);
    if (scope == "shared") {
      LOG(FATAL) << "Hexagon keeps shared buffers in one VTCM arena per tile "
                 << "program; allocate `" << buffer->data->name_hint
                 << "` with T.alloc_shared's default scope instead of "
                 << "\"shared\"";
    }
    if (scope == "shared.dyn") {
      ICHECK(!has_arena_) << "MergeSharedMemoryAllocations leaves one VTCM "
                             "arena per tile program";
      has_arena_ = true;
      auto elements = GetRef<AllocBuffer>(op).ConstantAllocationSize();
      if (!elements) {
        LOG(FATAL) << "Hexagon VTCM buffers need constant shapes; `"
                   << buffer->data->name_hint << "` does not have one";
      }
      Check(GetBufferStorageSizeBytes(elements.value(), buffer->dtype));
    }
    StmtVisitor::VisitStmt_(op);
  }

  void Check(int64_t bytes) const {
    if (capacity_ <= 0)
      return;
    // What the global.vtcm buffers leave, and how to say so.
    int64_t capacity = capacity_ - grid_;
    std::string room =
        grid_ ? "the " + std::to_string(capacity) + " bytes the target's " +
                    std::to_string(capacity_) + " leave after its " +
                    std::to_string(grid_) + " bytes of global.vtcm buffers"
              : "the target's " + std::to_string(capacity_);
    if (workers_ == 0) {
      if (bytes > capacity) {
        LOG(FATAL) << "The tile program's shared buffers take " << bytes
                   << " bytes of VTCM, more than " << room
                   << "; use smaller tiles";
      }
      return;
    }
    int64_t slice =
        (bytes + kSliceAlignment - 1) / kSliceAlignment * kSliceAlignment;
    if (workers_ * slice <= capacity)
      return;
    int64_t fit = capacity / slice;
    if (fit == 0) {
      LOG(FATAL) << "A worker's VTCM arena takes " << slice
                 << " bytes, more than " << room << "; use smaller tiles";
    }
    LOG(FATAL) << "T.Kernel(num_workers=" << workers_
               << ") needs a VTCM arena of " << slice << " bytes per worker, "
               << workers_ * slice << " bytes in all, more than " << room
               << "; use num_workers=" << fit << " or fewer, or smaller tiles";
  }

  int64_t capacity_;
  int64_t grid_;
  int64_t workers_ = 0;
  bool has_arena_ = false;
};

} // namespace

namespace transform {

tvm::transform::Pass VerifyVTCMArena() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    int64_t capacity = 0;
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      if (auto vtcm = target.value()->GetAttr<Integer>("vtcm-capacity"))
        capacity = vtcm.value()->value;
    }
    // The global.vtcm buffers, whole HMX tiles each.
    int64_t grid = 0;
    PostOrderVisit(func->body, [&grid](const ObjectRef &node) {
      const auto *alloc = node.as<AllocBufferNode>();
      if (!alloc || GetPtrStorageScope(alloc->buffer->data) != "global.vtcm")
        return;
      auto elements = GetRef<AllocBuffer>(alloc).ConstantAllocationSize();
      if (!elements) {
        LOG(FATAL) << "Hexagon VTCM buffers need constant shapes; `"
                   << alloc->buffer->data->name_hint << "` does not have one";
      }
      int64_t bytes =
          GetBufferStorageSizeBytes(elements.value(), alloc->buffer->dtype);
      grid += (bytes + kSliceAlignment - 1) / kSliceAlignment * kSliceAlignment;
    });
    if (capacity > 0 && grid > capacity) {
      LOG(FATAL) << "The kernel's global.vtcm buffers take " << grid
                 << " bytes of VTCM, more than the target's " << capacity;
    }
    ArenaVerifier verifier(capacity, grid);
    verifier(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.VerifyVTCMArena", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.VerifyVTCMArena",
                        VerifyVTCMArena);
}

} // namespace transform
} // namespace tl
} // namespace tvm
