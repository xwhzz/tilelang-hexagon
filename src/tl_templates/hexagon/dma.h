#pragma once
// The User-DMA queue of a tile program (T.dma_copy, T.dma_wait): descriptors,
// cache maintenance of DDR endpoints, and the queue the codegen declares.
//
// Queues are thread-confined: submit, poll and drain from the same QuRT thread.
// The managed codegen gives every tile program its own queue, and every worker
// its own VTCM arena (validated on v79). This is not a claim of one physical
// DMA engine per worker. This header never resets DMA during initialization or
// borrows another runtime's queue.

#include <hexagon_protos.h>
#include <qurt_memory.h>
#include <tl_templates/hexagon/common.h>

#include <cstddef>
#include <cstdint>

#ifndef TL_DMA_INLINE
#define TL_DMA_INLINE static inline
#endif
// The submit, commit and wait path of every transfer, inlined: as calls it
// cost several times its instructions (#78).
#define TL_DMA_HOT TL_DMA_INLINE __attribute__((always_inline))

enum tl_hexagon_dma_direction {
  TL_HEXAGON_DMA_DDR_TO_VTCM = 0,
  TL_HEXAGON_DMA_VTCM_TO_DDR = 1,
  // An empty slot, which needs no cache maintenance.
  TL_HEXAGON_DMA_NO_CACHE = 2,
  // DDR to VTCM from memory of which no dirty cache line can hold newer bytes
  // than DDR, so the source needs no writeback: memory the DSP has not
  // written since it was handed the memory, such as a kernel's read-only
  // argument, or a buffer only DMA moves (T.annotate_dma_only).
  TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN = 3,
  // VTCM to DDR into memory the DSP holds no cache lines of, a buffer only
  // DMA moves (T.annotate_dma_only): no writeback before, no invalidate after.
  TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED = 4,
};

enum tl_hexagon_dma_result {
  TL_HEXAGON_DMA_OK = 0,
  TL_HEXAGON_DMA_ERR_ARGUMENT = -1,
  TL_HEXAGON_DMA_ERR_RANGE = -2,
  TL_HEXAGON_DMA_ERR_CACHE = -3,
  TL_HEXAGON_DMA_ERR_ENGINE = -4,
  TL_HEXAGON_DMA_ERR_QUEUE_FULL = -5,
  TL_HEXAGON_DMA_ERR_QUEUE_EMPTY = -6,
  TL_HEXAGON_DMA_ERR_BUSY = -7,
};

enum tl_hexagon_dma_engine_state {
  TL_HEXAGON_DMA_STATUS_IDLE = 0,
  TL_HEXAGON_DMA_STATUS_RUN = 1,
  TL_HEXAGON_DMA_STATUS_ERROR = 2,
};

constexpr uint32_t TL_HEXAGON_DMA_STATUS_MASK = 0x3u;
constexpr uint32_t TL_HEXAGON_DMA_FIELD24_MAX = 0x00ffffffu;
constexpr uint32_t TL_HEXAGON_DMA_ROWS_MAX = 0x0000ffffu;
constexpr uint32_t TL_HEXAGON_DMA_DONE_MASK = 0x80000000u;
constexpr uint32_t TL_HEXAGON_DMA_DST_BYPASS_MASK = 0x10000000u;
constexpr uint32_t TL_HEXAGON_DMA_SRC_BYPASS_MASK = 0x20000000u;
constexpr uint32_t TL_HEXAGON_DMA_ORDERED_MASK = 0x40000000u;

// Hardware descriptor payload sizes are 16 bytes for 1D and 32 bytes for the
// v75+ type-9 2D form.  Queue slots occupy a full cache line so adjacent live
// descriptors cannot share one.
struct alignas(16) tl_hexagon_dma_descriptor_1d {
  uint32_t word[4];
};

struct alignas(16) tl_hexagon_dma_descriptor_2d {
  uint32_t word[8];
};

struct alignas(64) tl_hexagon_dma_descriptor {
  uint32_t word[8];
  uint8_t padding[32];
};

static_assert(sizeof(tl_hexagon_dma_descriptor_1d) == 16,
              "Hexagon 1D DMA descriptor must be 16 bytes");
static_assert(sizeof(tl_hexagon_dma_descriptor_2d) == 32,
              "Hexagon 2D DMA descriptor must be 32 bytes");
static_assert(sizeof(tl_hexagon_dma_descriptor) == 64,
              "Hexagon DMA queue slot must occupy one cache line");

struct tl_hexagon_dma_options {
  bool src_l2_bypass;
  bool dst_l2_bypass;
  bool ordered;
};

struct tl_hexagon_dma_transfer {
  void *dst;
  const void *src;
  uint32_t dst_stride;
  uint32_t src_stride;
  uint32_t row_bytes;
  uint32_t rows;
  int direction;
  // Field by field: hexagon-clang copies the struct with a memcpy call, on
  // every submit and completion.
  tl_hexagon_dma_transfer &operator=(const tl_hexagon_dma_transfer &o) {
    dst = o.dst;
    src = o.src;
    dst_stride = o.dst_stride;
    src_stride = o.src_stride;
    row_bytes = o.row_bytes;
    rows = o.rows;
    direction = o.direction;
    return *this;
  }
};

struct tl_hexagon_dma_queue {
  tl_hexagon_dma_descriptor *descriptors;
  tl_hexagon_dma_transfer *transfers;
  tl_hexagon_dma_descriptor *tail;
  uint32_t head;
  uint32_t count;
  uint32_t capacity;
};

// ---------------------------------------------------------------------------
// User-DMA instructions
// ---------------------------------------------------------------------------

TL_DMA_INLINE uint32_t tl_hexagon_dma_pause() {
  return static_cast<uint32_t>(Q6_R_dmpause());
}

TL_DMA_INLINE void tl_hexagon_dma_start(void *next) {
  asm volatile("release(%0):at" : : "r"(next) : "memory");
  Q6_dmstart_A(next);
}

TL_DMA_INLINE void tl_hexagon_dma_link(void *tail, void *next) {
  asm volatile("release(%0):at" : : "r"(next) : "memory");
  Q6_dmlink_AA(tail, next);
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_poll() {
  return static_cast<uint32_t>(Q6_R_dmpoll());
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_engine_status() {
  return tl_hexagon_dma_poll() & TL_HEXAGON_DMA_STATUS_MASK;
}

TL_DMA_INLINE bool tl_hexagon_dma_engine_idle() {
  return tl_hexagon_dma_engine_status() == TL_HEXAGON_DMA_STATUS_IDLE;
}

// dmpause resets DMA0.  Call this only after acquiring exclusive ownership of
// the engine; queue initialization intentionally does not call it.
TL_DMA_INLINE int tl_hexagon_dma_engine_reset() {
  uint32_t status = tl_hexagon_dma_pause() & TL_HEXAGON_DMA_STATUS_MASK;
  return status == TL_HEXAGON_DMA_STATUS_IDLE ? TL_HEXAGON_DMA_OK
                                              : TL_HEXAGON_DMA_ERR_ENGINE;
}

// ---------------------------------------------------------------------------
// Descriptor construction and submission
// ---------------------------------------------------------------------------

// A transfer's DDR side bypasses the L2, which spares the L2 the lines it
// moves: through the L2, 4 workers' DMA loads read DDR at 30 GB/s on v79
// rather than 57.
TL_DMA_HOT tl_hexagon_dma_options
tl_hexagon_dma_options_for_direction(int direction) {
  return {
      direction == TL_HEXAGON_DMA_DDR_TO_VTCM ||
          direction == TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN,
      direction == TL_HEXAGON_DMA_VTCM_TO_DDR ||
          direction == TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED,
      false,
  };
}

TL_DMA_HOT uint32_t
tl_hexagon_dma_option_bits(const tl_hexagon_dma_options &options) {
  return (options.src_l2_bypass ? TL_HEXAGON_DMA_SRC_BYPASS_MASK : 0u) |
         (options.dst_l2_bypass ? TL_HEXAGON_DMA_DST_BYPASS_MASK : 0u) |
         (options.ordered ? TL_HEXAGON_DMA_ORDERED_MASK : 0u);
}

TL_DMA_HOT void
tl_hexagon_dma_descriptor_clear(tl_hexagon_dma_descriptor *descriptor) {
  if (descriptor == nullptr)
    return;
  // The engine reads the words only; the padding keeps descriptors on lines
  // of their own. Four doubleword stores, where loops became memset calls.
  uint64_t *words = reinterpret_cast<uint64_t *>(descriptor->word);
  words[0] = 0;
  words[1] = 0;
  words[2] = 0;
  words[3] = 0;
}

TL_DMA_HOT bool tl_hexagon_dma_address32(const void *ptr, uint64_t bytes,
                                         uint32_t *address) {
  if (ptr == nullptr || address == nullptr)
    return false;
  uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
  if (begin > UINT32_MAX)
    return false;
  if (bytes != 0) {
    uint64_t end = static_cast<uint64_t>(begin) + bytes - 1;
    if (end > UINT32_MAX)
      return false;
  }
  *address = static_cast<uint32_t>(begin);
  return true;
}

TL_DMA_HOT uint64_t tl_hexagon_dma_2d_span(uint32_t stride, uint32_t row_bytes,
                                           uint32_t rows) {
  if (rows == 0 || row_bytes == 0)
    return 0;
  return static_cast<uint64_t>(rows - 1) * stride + row_bytes;
}

TL_DMA_INLINE void
tl_hexagon_dma_descriptor_mark_done(tl_hexagon_dma_descriptor *descriptor) {
  descriptor->word[1] |= TL_HEXAGON_DMA_DONE_MASK;
}

TL_DMA_HOT bool
tl_hexagon_dma_descriptor_done(const tl_hexagon_dma_descriptor *descriptor) {
  if (descriptor == nullptr)
    return false;
  const volatile uint32_t *state = &descriptor->word[1];
  return (*state & TL_HEXAGON_DMA_DONE_MASK) != 0;
}

TL_DMA_HOT int
tl_hexagon_dma_descriptor_init_1d(tl_hexagon_dma_descriptor *descriptor,
                                  void *dst, const void *src, uint32_t bytes,
                                  tl_hexagon_dma_options options) {
  if (descriptor == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (bytes > TL_HEXAGON_DMA_FIELD24_MAX)
    return TL_HEXAGON_DMA_ERR_RANGE;

  tl_hexagon_dma_descriptor_clear(descriptor);
  if (bytes == 0) {
    tl_hexagon_dma_descriptor_mark_done(descriptor);
    return TL_HEXAGON_DMA_OK;
  }

  uint32_t src_address = 0;
  uint32_t dst_address = 0;
  if (src == nullptr || dst == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (!tl_hexagon_dma_address32(src, bytes, &src_address) ||
      !tl_hexagon_dma_address32(dst, bytes, &dst_address))
    return TL_HEXAGON_DMA_ERR_RANGE;

  descriptor->word[0] = 0; // null next pointer, ready state
  descriptor->word[1] = bytes | tl_hexagon_dma_option_bits(options);
  descriptor->word[2] = src_address;
  descriptor->word[3] = dst_address;
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_HOT int tl_hexagon_dma_descriptor_init_2d(
    tl_hexagon_dma_descriptor *descriptor, void *dst, const void *src,
    uint32_t dst_stride, uint32_t src_stride, uint32_t row_bytes, uint32_t rows,
    tl_hexagon_dma_options options) {
  if (descriptor == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (row_bytes > TL_HEXAGON_DMA_FIELD24_MAX ||
      src_stride > TL_HEXAGON_DMA_FIELD24_MAX ||
      dst_stride > TL_HEXAGON_DMA_FIELD24_MAX || rows > TL_HEXAGON_DMA_ROWS_MAX)
    return TL_HEXAGON_DMA_ERR_RANGE;

  tl_hexagon_dma_descriptor_clear(descriptor);
  if (rows == 0 || row_bytes == 0) {
    tl_hexagon_dma_descriptor_mark_done(descriptor);
    return TL_HEXAGON_DMA_OK;
  }

  uint32_t src_address = 0;
  uint32_t dst_address = 0;
  if (src == nullptr || dst == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  uint64_t src_span = tl_hexagon_dma_2d_span(src_stride, row_bytes, rows);
  uint64_t dst_span = tl_hexagon_dma_2d_span(dst_stride, row_bytes, rows);
  if (!tl_hexagon_dma_address32(src, src_span, &src_address) ||
      !tl_hexagon_dma_address32(dst, dst_span, &dst_address))
    return TL_HEXAGON_DMA_ERR_RANGE;

  descriptor->word[0] = 0;
  descriptor->word[1] = (dst_stride & TL_HEXAGON_DMA_FIELD24_MAX) | (1u << 24) |
                        tl_hexagon_dma_option_bits(options);
  descriptor->word[2] = src_address;
  descriptor->word[3] = dst_address;
  descriptor->word[4] = 9u; // v75+ 2D descriptor with 24-bit fields
  descriptor->word[5] =
      (row_bytes & TL_HEXAGON_DMA_FIELD24_MAX) | ((rows & 0xffu) << 24);
  descriptor->word[6] =
      ((rows >> 8) & 0xffu) | ((src_stride & TL_HEXAGON_DMA_FIELD24_MAX) << 8);
  descriptor->word[7] = 0; // zero source/destination width offset
  return TL_HEXAGON_DMA_OK;
}

// ---------------------------------------------------------------------------
// Cache maintenance for DDR endpoints
// ---------------------------------------------------------------------------

// Writes back (QURT_MEM_CACHE_FLUSH) or invalidates the cache lines of
// [ptr, ptr + bytes). A small writeback runs inline and, with `sync`, ends with
// the syncht that completes it; a transfer of several rows syncs once.
TL_DMA_INLINE int tl_hexagon_dma_cache_clean(const void *ptr, uint32_t bytes,
                                             qurt_mem_cache_op_t operation,
                                             bool sync = true) {
  constexpr uint64_t kCacheLineBytes = 64;
  constexpr uint64_t kCleanBatchBytes = 4 * kCacheLineBytes;
  constexpr uint32_t kInlineCleanLimit = 128u * 1024u;

  if (bytes == 0)
    return TL_HEXAGON_DMA_OK;
  if (ptr == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;

  if (operation == QURT_MEM_CACHE_FLUSH) {
    if (bytes <= kInlineCleanLimit) {
      // Avoid a QuRT call for small ranges. On v79, dccleana writes back L1D
      // and L2 without invalidating; syncht completes the cache operations.
      uint32_t address;
      if (!tl_hexagon_dma_address32(ptr, bytes, &address))
        return TL_HEXAGON_DMA_ERR_RANGE;
      uint64_t line = static_cast<uint64_t>(address) & ~(kCacheLineBytes - 1);
      const uint64_t end = static_cast<uint64_t>(address) + bytes;
      asm volatile("" : : : "memory");
      for (; line + kCleanBatchBytes <= end; line += kCleanBatchBytes) {
        Q6_dccleana_A(reinterpret_cast<void *>(static_cast<uintptr_t>(line)));
        Q6_dccleana_A(reinterpret_cast<void *>(
            static_cast<uintptr_t>(line + kCacheLineBytes)));
        Q6_dccleana_A(reinterpret_cast<void *>(
            static_cast<uintptr_t>(line + 2 * kCacheLineBytes)));
        Q6_dccleana_A(reinterpret_cast<void *>(
            static_cast<uintptr_t>(line + 3 * kCacheLineBytes)));
      }
      for (; line < end; line += kCacheLineBytes)
        Q6_dccleana_A(reinterpret_cast<void *>(static_cast<uintptr_t>(line)));
      if (sync)
        asm volatile("syncht" : : : "memory");
      return TL_HEXAGON_DMA_OK;
    }

    // Bound large writebacks by cache size, preserving valid cache contents.
    // Destination invalidation remains range-based in the fallback below.
    const int status =
        qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_ALL, QURT_MEM_DCACHE);
    return status == 0 ? TL_HEXAGON_DMA_OK : TL_HEXAGON_DMA_ERR_CACHE;
  }

  const int status = qurt_mem_cache_clean(
      static_cast<qurt_addr_t>(reinterpret_cast<uintptr_t>(ptr)),
      static_cast<qurt_size_t>(bytes), operation, QURT_MEM_DCACHE);
  return status == 0 ? TL_HEXAGON_DMA_OK : TL_HEXAGON_DMA_ERR_CACHE;
}

TL_DMA_INLINE int tl_hexagon_dma_prepare_ddr(const void *dst, const void *src,
                                             uint32_t bytes, int direction,
                                             bool sync = true) {
  if (direction == TL_HEXAGON_DMA_DDR_TO_VTCM)
    return tl_hexagon_dma_cache_clean(src, bytes, QURT_MEM_CACHE_FLUSH, sync);
  if (direction == TL_HEXAGON_DMA_VTCM_TO_DDR)
    // Preserve dirty bytes sharing the destination's boundary cache lines.
    return tl_hexagon_dma_cache_clean(dst, bytes, QURT_MEM_CACHE_FLUSH, sync);
  if (direction == TL_HEXAGON_DMA_NO_CACHE ||
      direction == TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN ||
      direction == TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED)
    return TL_HEXAGON_DMA_OK;
  return TL_HEXAGON_DMA_ERR_ARGUMENT;
}

TL_DMA_INLINE int tl_hexagon_dma_finish_ddr(const void *dst, uint32_t bytes,
                                            int direction) {
  if (direction == TL_HEXAGON_DMA_DDR_TO_VTCM ||
      direction == TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN ||
      direction == TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED ||
      direction == TL_HEXAGON_DMA_NO_CACHE)
    return TL_HEXAGON_DMA_OK;
  if (direction == TL_HEXAGON_DMA_VTCM_TO_DDR)
    return tl_hexagon_dma_cache_clean(dst, bytes, QURT_MEM_CACHE_INVALIDATE);
  return TL_HEXAGON_DMA_ERR_ARGUMENT;
}

TL_DMA_HOT bool tl_hexagon_dma_valid_direction(int direction) {
  return direction == TL_HEXAGON_DMA_DDR_TO_VTCM ||
         direction == TL_HEXAGON_DMA_VTCM_TO_DDR ||
         direction == TL_HEXAGON_DMA_NO_CACHE ||
         direction == TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN ||
         direction == TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED;
}

// Writes back the DDR rows of a transfer: every row's lines, then one syncht.
TL_DMA_HOT int
tl_hexagon_dma_prepare_transfer(const tl_hexagon_dma_transfer &transfer) {
  if (transfer.direction != TL_HEXAGON_DMA_DDR_TO_VTCM &&
      transfer.direction != TL_HEXAGON_DMA_VTCM_TO_DDR)
    return tl_hexagon_dma_valid_direction(transfer.direction)
               ? TL_HEXAGON_DMA_OK
               : TL_HEXAGON_DMA_ERR_ARGUMENT;
  for (uint32_t row = 0; row < transfer.rows; ++row) {
    const uint8_t *src_row = static_cast<const uint8_t *>(transfer.src) +
                             static_cast<uint64_t>(row) * transfer.src_stride;
    uint8_t *dst_row = static_cast<uint8_t *>(transfer.dst) +
                       static_cast<uint64_t>(row) * transfer.dst_stride;
    int status =
        tl_hexagon_dma_prepare_ddr(dst_row, src_row, transfer.row_bytes,
                                   transfer.direction, /*sync=*/false);
    if (status != TL_HEXAGON_DMA_OK)
      return status;
  }
  if (transfer.rows != 0 && (transfer.direction == TL_HEXAGON_DMA_DDR_TO_VTCM ||
                             transfer.direction == TL_HEXAGON_DMA_VTCM_TO_DDR))
    asm volatile("syncht" : : : "memory");
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_HOT int
tl_hexagon_dma_finish_transfer(const tl_hexagon_dma_transfer &transfer) {
  // Only a store to DDR has cache lines to invalidate, row by row.
  if (transfer.direction != TL_HEXAGON_DMA_VTCM_TO_DDR)
    return TL_HEXAGON_DMA_OK;
  for (uint32_t row = 0; row < transfer.rows; ++row) {
    uint8_t *dst_row = static_cast<uint8_t *>(transfer.dst) +
                       static_cast<uint64_t>(row) * transfer.dst_stride;
    int status = tl_hexagon_dma_finish_ddr(dst_row, transfer.row_bytes,
                                           transfer.direction);
    if (status != TL_HEXAGON_DMA_OK)
      return status;
  }
  return TL_HEXAGON_DMA_OK;
}

// ---------------------------------------------------------------------------
// Descriptor queue
// ---------------------------------------------------------------------------

// The slot `offset` (at most the capacity) past the head. head + offset is
// under twice the capacity, so one compare wraps it: a modulo by the runtime
// capacity is a division libcall on Hexagon, on every submit and pop.
TL_DMA_HOT uint32_t
tl_hexagon_dma_queue_index(const tl_hexagon_dma_queue *queue, uint32_t offset) {
  uint32_t index = queue->head + offset;
  return index >= queue->capacity ? index - queue->capacity : index;
}

TL_DMA_INLINE int tl_hexagon_dma_queue_init(
    tl_hexagon_dma_queue *queue, tl_hexagon_dma_descriptor *descriptors,
    tl_hexagon_dma_transfer *transfers, uint32_t capacity) {
  if (queue == nullptr || descriptors == nullptr || transfers == nullptr ||
      capacity == 0)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if ((reinterpret_cast<uintptr_t>(descriptors) & 63u) != 0)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;

  uint32_t ignored = 0;
  uint64_t descriptor_bytes =
      static_cast<uint64_t>(capacity) * sizeof(tl_hexagon_dma_descriptor);
  if (!tl_hexagon_dma_address32(descriptors, descriptor_bytes, &ignored))
    return TL_HEXAGON_DMA_ERR_RANGE;

  queue->descriptors = descriptors;
  queue->transfers = transfers;
  queue->tail = nullptr;
  queue->head = 0;
  queue->count = 0;
  queue->capacity = capacity;
  for (uint32_t i = 0; i < capacity; ++i) {
    tl_hexagon_dma_descriptor_clear(&descriptors[i]);
    transfers[i] = {nullptr, nullptr, 0, 0, 0, 0, TL_HEXAGON_DMA_NO_CACHE};
  }
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_HOT bool tl_hexagon_dma_queue_full(const tl_hexagon_dma_queue *queue) {
  return queue != nullptr && queue->count == queue->capacity;
}

TL_DMA_HOT tl_hexagon_dma_descriptor *
tl_hexagon_dma_queue_next_descriptor(tl_hexagon_dma_queue *queue) {
  if (queue == nullptr || queue->capacity == 0 ||
      tl_hexagon_dma_queue_full(queue))
    return nullptr;
  return &queue->descriptors[tl_hexagon_dma_queue_index(queue, queue->count)];
}

// Submits the descriptor in the next queue slot, which the caller initialized
// and whose cache maintenance it performed, with its completion metadata.
TL_DMA_HOT int
tl_hexagon_dma_queue_submit_prepared(tl_hexagon_dma_queue *queue,
                                     const tl_hexagon_dma_transfer &transfer) {
  tl_hexagon_dma_descriptor *descriptor =
      tl_hexagon_dma_queue_next_descriptor(queue);
  if (descriptor == nullptr)
    return queue == nullptr ? TL_HEXAGON_DMA_ERR_ARGUMENT
                            : TL_HEXAGON_DMA_ERR_QUEUE_FULL;

  if (!tl_hexagon_dma_descriptor_done(descriptor)) {
    uint32_t engine_status = tl_hexagon_dma_engine_status();
    if (engine_status == TL_HEXAGON_DMA_STATUS_IDLE) {
      tl_hexagon_dma_start(descriptor);
    } else if (engine_status == TL_HEXAGON_DMA_STATUS_RUN &&
               queue->tail != nullptr) {
      tl_hexagon_dma_link(queue->tail, descriptor);
    } else {
      return engine_status == TL_HEXAGON_DMA_STATUS_RUN
                 ? TL_HEXAGON_DMA_ERR_BUSY
                 : TL_HEXAGON_DMA_ERR_ENGINE;
    }
    queue->tail = descriptor;
  }

  uint32_t index = tl_hexagon_dma_queue_index(queue, queue->count);
  queue->transfers[index] = transfer;
  ++queue->count;
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_HOT int tl_hexagon_dma_queue_push_1d(tl_hexagon_dma_queue *queue,
                                            void *dst, const void *src,
                                            uint32_t bytes, int direction) {
  if (!tl_hexagon_dma_valid_direction(direction))
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  tl_hexagon_dma_descriptor *descriptor =
      tl_hexagon_dma_queue_next_descriptor(queue);
  if (descriptor == nullptr)
    return queue == nullptr ? TL_HEXAGON_DMA_ERR_ARGUMENT
                            : TL_HEXAGON_DMA_ERR_QUEUE_FULL;

  int status = tl_hexagon_dma_descriptor_init_1d(
      descriptor, dst, src, bytes,
      tl_hexagon_dma_options_for_direction(direction));
  if (status != TL_HEXAGON_DMA_OK)
    return status;

  tl_hexagon_dma_transfer transfer = {
      dst, src, bytes, bytes, bytes, bytes == 0 ? 0u : 1u, direction};
  status = tl_hexagon_dma_prepare_transfer(transfer);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  return tl_hexagon_dma_queue_submit_prepared(queue, transfer);
}

TL_DMA_HOT int tl_hexagon_dma_queue_push_2d(tl_hexagon_dma_queue *queue,
                                            void *dst, const void *src,
                                            uint32_t dst_stride,
                                            uint32_t src_stride,
                                            uint32_t row_bytes, uint32_t rows,
                                            int direction) {
  if (!tl_hexagon_dma_valid_direction(direction))
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  tl_hexagon_dma_descriptor *descriptor =
      tl_hexagon_dma_queue_next_descriptor(queue);
  if (descriptor == nullptr)
    return queue == nullptr ? TL_HEXAGON_DMA_ERR_ARGUMENT
                            : TL_HEXAGON_DMA_ERR_QUEUE_FULL;

  int status = tl_hexagon_dma_descriptor_init_2d(
      descriptor, dst, src, dst_stride, src_stride, row_bytes, rows,
      tl_hexagon_dma_options_for_direction(direction));
  if (status != TL_HEXAGON_DMA_OK)
    return status;

  tl_hexagon_dma_transfer transfer = {dst,       src,  dst_stride, src_stride,
                                      row_bytes, rows, direction};
  status = tl_hexagon_dma_prepare_transfer(transfer);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  return tl_hexagon_dma_queue_submit_prepared(queue, transfer);
}

TL_DMA_HOT int tl_hexagon_dma_queue_push(tl_hexagon_dma_queue *queue, void *dst,
                                         const void *src, uint32_t dst_stride,
                                         uint32_t src_stride,
                                         uint32_t row_bytes, uint32_t rows,
                                         int direction) {
  if (rows == 1 && dst_stride == row_bytes && src_stride == row_bytes)
    return tl_hexagon_dma_queue_push_1d(queue, dst, src, row_bytes, direction);
  return tl_hexagon_dma_queue_push_2d(queue, dst, src, dst_stride, src_stride,
                                      row_bytes, rows, direction);
}

TL_DMA_HOT int
tl_hexagon_dma_queue_try_pop(tl_hexagon_dma_queue *queue,
                             tl_hexagon_dma_transfer *completed) {
  if (queue == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (queue->count == 0)
    return TL_HEXAGON_DMA_ERR_QUEUE_EMPTY;

  tl_hexagon_dma_poll();
  tl_hexagon_dma_descriptor *descriptor = &queue->descriptors[queue->head];
  if (!tl_hexagon_dma_descriptor_done(descriptor)) {
    uint32_t status = tl_hexagon_dma_engine_status();
    return status == TL_HEXAGON_DMA_STATUS_ERROR ? TL_HEXAGON_DMA_ERR_ENGINE
                                                 : TL_HEXAGON_DMA_ERR_BUSY;
  }

  tl_hexagon_dma_transfer transfer = queue->transfers[queue->head];
  int status = tl_hexagon_dma_finish_transfer(transfer);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  if (completed != nullptr)
    *completed = transfer;

  tl_hexagon_dma_descriptor_clear(descriptor);
  queue->transfers[queue->head] = {
      nullptr, nullptr, 0, 0, 0, 0, TL_HEXAGON_DMA_NO_CACHE};
  queue->head = tl_hexagon_dma_queue_index(queue, 1);
  --queue->count;
  if (queue->count == 0)
    queue->tail = nullptr;
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_HOT int
tl_hexagon_dma_queue_pop(tl_hexagon_dma_queue *queue,
                         tl_hexagon_dma_transfer *completed = nullptr) {
  int status = TL_HEXAGON_DMA_ERR_BUSY;
  while (status == TL_HEXAGON_DMA_ERR_BUSY)
    status = tl_hexagon_dma_queue_try_pop(queue, completed);
  return status;
}

TL_DMA_INLINE int tl_hexagon_dma_queue_flush(tl_hexagon_dma_queue *queue) {
  if (queue == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  while (queue->count != 0) {
    int status = tl_hexagon_dma_queue_pop(queue);
    if (status != TL_HEXAGON_DMA_OK)
      return status;
  }
  return TL_HEXAGON_DMA_OK;
}

// The queue of one tile program, used by explicit T.dma_copy/T.dma_wait. The
// calling thread owns the DMA instruction context for its lifetime.
// Initialization never resets another queue's outstanding work on this thread.
//
// Descriptors complete in commit groups: T.dma_copy submits the descriptors of
// one transfer and commits them as a group, and T.dma_wait(n) completes the
// oldest groups until at most n are pending. The compiler sizes both rings for
// the most descriptors and groups the program keeps in flight.
//
// Compile managed async kernels with -fno-exceptions (the DSP has no C++
// unwinder). Stack storage survives until the program drains its queue,
// including returns caused by another helper's error. No dynamic allocation or
// global queue state.
template <uint32_t Descriptors, uint32_t Groups>
struct tl_hexagon_dma_async_context {
  static_assert(Descriptors > 0 && Descriptors <= 256,
                "DMA queue capacity out of range");
  static_assert(Groups > 0 && Groups <= 256, "DMA group capacity out of range");
  tl_hexagon_dma_descriptor descriptors[Descriptors];
  tl_hexagon_dma_transfer transfers[Descriptors];
  tl_hexagon_dma_queue queue{};
  // The end of every committed, pending group as a count of submitted
  // descriptors, oldest first, in a ring.
  uint32_t group_end[Groups];
  uint32_t group_head = 0;
  uint32_t group_count = 0;
  uint32_t submitted = 0;
  uint32_t retired = 0;
  int status;
  tl_hexagon_dma_async_context() {
    status = tl_hexagon_dma_engine_idle()
                 ? tl_hexagon_dma_queue_init(&queue, descriptors, transfers,
                                             Descriptors)
                 : TL_HEXAGON_DMA_ERR_BUSY;
  }
  ~tl_hexagon_dma_async_context() {
    // Only reset after an error while draining our own outstanding descriptors.
    // This stops DMA before descriptor storage and the VTCM lease can go away.
    if (queue.count && tl_hexagon_dma_queue_flush(&queue) != TL_HEXAGON_DMA_OK)
      tl_hexagon_dma_engine_reset();
  }
  tl_hexagon_dma_async_context(const tl_hexagon_dma_async_context &) = delete;
  tl_hexagon_dma_async_context &
  operator=(const tl_hexagon_dma_async_context &) = delete;

  __attribute__((always_inline)) int
  submit(void *dst, const void *src, uint32_t dst_stride, uint32_t src_stride,
         uint32_t row_bytes, uint32_t rows, int direction) {
    int result = tl_hexagon_dma_queue_push(
        &queue, dst, src, dst_stride, src_stride, row_bytes, rows, direction);
    if (result == TL_HEXAGON_DMA_OK)
      ++submitted;
    return result;
  }
  __attribute__((always_inline)) int commit() {
    if (group_count == Groups)
      return TL_HEXAGON_DMA_ERR_QUEUE_FULL;
    group_end[(group_head + group_count) % Groups] = submitted;
    ++group_count;
    return TL_HEXAGON_DMA_OK;
  }
  __attribute__((always_inline)) int wait(uint32_t groups) {
    while (group_count > groups) {
      // Unsigned differences stay correct when the counters wrap.
      while (group_end[group_head] - retired != 0) {
        int result = tl_hexagon_dma_queue_pop(&queue);
        if (result != TL_HEXAGON_DMA_OK)
          return result;
        ++retired;
      }
      group_head = (group_head + 1) % Groups;
      --group_count;
    }
    // Prevent the compiler from hoisting consumers before DMA completion.
    asm volatile("" : : : "memory");
    return TL_HEXAGON_DMA_OK;
  }
};

// The kernel-facing calls: a failure becomes the kernel status TL_ERR_DMA.
template <uint32_t Descriptors, uint32_t Groups>
TL_DMA_INLINE int tl_hexagon_dma_async_submit(
    tl_hexagon_dma_async_context<Descriptors, Groups> &context, void *dst,
    const void *src, uint32_t dst_stride, uint32_t src_stride,
    uint32_t row_bytes, uint32_t rows, int direction) {
  return context.submit(dst, src, dst_stride, src_stride, row_bytes, rows,
                        direction) == TL_HEXAGON_DMA_OK
             ? TL_OK
             : TL_ERR_DMA;
}

template <uint32_t Descriptors, uint32_t Groups>
TL_DMA_INLINE int tl_hexagon_dma_async_commit(
    tl_hexagon_dma_async_context<Descriptors, Groups> &context) {
  return context.commit() == TL_HEXAGON_DMA_OK ? TL_OK : TL_ERR_DMA;
}

template <uint32_t Descriptors, uint32_t Groups>
TL_DMA_INLINE int tl_hexagon_dma_async_wait(
    tl_hexagon_dma_async_context<Descriptors, Groups> &context,
    uint32_t groups) {
  return context.wait(groups) == TL_HEXAGON_DMA_OK ? TL_OK : TL_ERR_DMA;
}
