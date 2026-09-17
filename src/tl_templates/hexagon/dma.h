#pragma once
// Hexagon User-DMA instruction, descriptor, queue, and managed async-copy helpers.
//
// DMA0 is a shared hardware resource.  This header deliberately does not
// allocate memory, reset the engine during queue initialization, or provide
// locking.  The caller owns the engine for the lifetime of a queue and supplies
// descriptor/metadata storage.  This makes the primitives usable by a future
// TileLang DMA lowering as well as callers managing their own queues.

#include <hexagon_protos.h>
#include <qurt_memory.h>

#include <cstddef>
#include <cstdint>

#ifndef TL_DMA_INLINE
#define TL_DMA_INLINE static inline
#endif

enum tl_hexagon_dma_direction {
  TL_HEXAGON_DMA_DDR_TO_VTCM = 0,
  TL_HEXAGON_DMA_VTCM_TO_DDR = 1,
  // Raw queues may use this when the caller handles cache maintenance.
  TL_HEXAGON_DMA_NO_CACHE = 2,
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
// Raw User-DMA instructions
// ---------------------------------------------------------------------------

TL_DMA_INLINE uint32_t tl_hexagon_dma_pause() {
  return static_cast<uint32_t>(Q6_R_dmpause());
}

TL_DMA_INLINE void tl_hexagon_dma_resume(uint32_t state) {
  asm volatile("dmresume(%0)" : : "r"(state) : "memory");
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

TL_DMA_INLINE uint32_t tl_hexagon_dma_wait() {
  return static_cast<uint32_t>(Q6_R_dmwait());
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_sync_thread() {
  return static_cast<uint32_t>(Q6_R_dmsyncht());
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_tlb_sync() {
  return static_cast<uint32_t>(Q6_R_dmtlbsynch());
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_config_read(uint32_t index) {
  uint32_t value = 0;
  asm volatile("%0 = dmcfgrd(%1)" : "=r"(value) : "r"(index) : "memory");
  return value;
}

TL_DMA_INLINE void tl_hexagon_dma_config_write(uint32_t index,
                                                uint32_t value) {
  asm volatile("dmcfgwr(%0, %1)" : : "r"(index), "r"(value) : "memory");
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

TL_DMA_INLINE tl_hexagon_dma_options tl_hexagon_dma_make_options(
    bool src_l2_bypass, bool dst_l2_bypass, bool ordered) {
  return {src_l2_bypass, dst_l2_bypass, ordered};
}

TL_DMA_INLINE tl_hexagon_dma_options tl_hexagon_dma_options_for_direction(
    int direction) {
  return {
      direction == TL_HEXAGON_DMA_DDR_TO_VTCM,
      direction == TL_HEXAGON_DMA_VTCM_TO_DDR,
      false,
  };
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_option_bits(
    const tl_hexagon_dma_options &options) {
  return (options.src_l2_bypass ? TL_HEXAGON_DMA_SRC_BYPASS_MASK : 0u) |
         (options.dst_l2_bypass ? TL_HEXAGON_DMA_DST_BYPASS_MASK : 0u) |
         (options.ordered ? TL_HEXAGON_DMA_ORDERED_MASK : 0u);
}

TL_DMA_INLINE void tl_hexagon_dma_descriptor_clear(
    tl_hexagon_dma_descriptor *descriptor) {
  if (descriptor == nullptr)
    return;
  for (uint32_t i = 0; i < 8; ++i)
    descriptor->word[i] = 0;
  for (uint32_t i = 0; i < sizeof(descriptor->padding); ++i)
    descriptor->padding[i] = 0;
}

TL_DMA_INLINE bool tl_hexagon_dma_address32(const void *ptr, uint64_t bytes,
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

TL_DMA_INLINE uint64_t tl_hexagon_dma_2d_span(uint32_t stride,
                                               uint32_t row_bytes,
                                               uint32_t rows) {
  if (rows == 0 || row_bytes == 0)
    return 0;
  return static_cast<uint64_t>(rows - 1) * stride + row_bytes;
}

TL_DMA_INLINE void tl_hexagon_dma_descriptor_mark_done(
    tl_hexagon_dma_descriptor *descriptor) {
  descriptor->word[1] |= TL_HEXAGON_DMA_DONE_MASK;
}

TL_DMA_INLINE bool tl_hexagon_dma_descriptor_done(
    const tl_hexagon_dma_descriptor *descriptor) {
  if (descriptor == nullptr)
    return false;
  const volatile uint32_t *state = &descriptor->word[1];
  return (*state & TL_HEXAGON_DMA_DONE_MASK) != 0;
}

TL_DMA_INLINE int tl_hexagon_dma_descriptor_init_1d(
    tl_hexagon_dma_descriptor *descriptor, void *dst, const void *src,
    uint32_t bytes, tl_hexagon_dma_options options) {
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

TL_DMA_INLINE int tl_hexagon_dma_descriptor_init_2d(
    tl_hexagon_dma_descriptor *descriptor, void *dst, const void *src,
    uint32_t dst_stride, uint32_t src_stride, uint32_t row_bytes,
    uint32_t rows, tl_hexagon_dma_options options) {
  if (descriptor == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (row_bytes > TL_HEXAGON_DMA_FIELD24_MAX ||
      src_stride > TL_HEXAGON_DMA_FIELD24_MAX ||
      dst_stride > TL_HEXAGON_DMA_FIELD24_MAX ||
      rows > TL_HEXAGON_DMA_ROWS_MAX)
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
  descriptor->word[1] = (dst_stride & TL_HEXAGON_DMA_FIELD24_MAX) |
                        (1u << 24) | tl_hexagon_dma_option_bits(options);
  descriptor->word[2] = src_address;
  descriptor->word[3] = dst_address;
  descriptor->word[4] = 9u; // v75+ 2D descriptor with 24-bit fields
  descriptor->word[5] = (row_bytes & TL_HEXAGON_DMA_FIELD24_MAX) |
                        ((rows & 0xffu) << 24);
  descriptor->word[6] = ((rows >> 8) & 0xffu) |
                        ((src_stride & TL_HEXAGON_DMA_FIELD24_MAX) << 8);
  descriptor->word[7] = 0; // zero source/destination width offset
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_INLINE int tl_hexagon_dma_submit_one(
    tl_hexagon_dma_descriptor *descriptor) {
  if (descriptor == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (tl_hexagon_dma_descriptor_done(descriptor))
    return TL_HEXAGON_DMA_OK;
  if (!tl_hexagon_dma_engine_idle())
    return TL_HEXAGON_DMA_ERR_BUSY;
  tl_hexagon_dma_start(descriptor);
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_INLINE int tl_hexagon_dma_start_and_wait(
    tl_hexagon_dma_descriptor *descriptor) {
  int status = tl_hexagon_dma_submit_one(descriptor);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  if (tl_hexagon_dma_descriptor_done(descriptor))
    return TL_HEXAGON_DMA_OK;
  uint32_t engine_status = tl_hexagon_dma_wait() & TL_HEXAGON_DMA_STATUS_MASK;
  asm volatile("" : : : "memory");
  return engine_status == TL_HEXAGON_DMA_STATUS_IDLE
             ? TL_HEXAGON_DMA_OK
             : TL_HEXAGON_DMA_ERR_ENGINE;
}

// ---------------------------------------------------------------------------
// Cache maintenance for DDR endpoints
// ---------------------------------------------------------------------------

TL_DMA_INLINE int tl_hexagon_dma_cache_clean(const void *ptr, uint32_t bytes,
                                              qurt_mem_cache_op_t operation) {
  if (bytes == 0)
    return TL_HEXAGON_DMA_OK;
  if (ptr == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  int status = qurt_mem_cache_clean(
      static_cast<qurt_addr_t>(reinterpret_cast<uintptr_t>(ptr)),
      static_cast<qurt_size_t>(bytes), operation, QURT_MEM_DCACHE);
  return status == 0 ? TL_HEXAGON_DMA_OK : TL_HEXAGON_DMA_ERR_CACHE;
}

TL_DMA_INLINE int tl_hexagon_dma_prepare_ddr(const void *dst, const void *src,
                                              uint32_t bytes, int direction) {
  if (direction == TL_HEXAGON_DMA_DDR_TO_VTCM)
    return tl_hexagon_dma_cache_clean(src, bytes, QURT_MEM_CACHE_FLUSH);
  if (direction == TL_HEXAGON_DMA_VTCM_TO_DDR)
    // Preserve dirty bytes sharing the destination's boundary cache lines.
    return tl_hexagon_dma_cache_clean(dst, bytes, QURT_MEM_CACHE_FLUSH);
  if (direction == TL_HEXAGON_DMA_NO_CACHE)
    return TL_HEXAGON_DMA_OK;
  return TL_HEXAGON_DMA_ERR_ARGUMENT;
}

TL_DMA_INLINE int tl_hexagon_dma_finish_ddr(const void *dst, uint32_t bytes,
                                             int direction) {
  if (direction == TL_HEXAGON_DMA_DDR_TO_VTCM ||
      direction == TL_HEXAGON_DMA_NO_CACHE)
    return TL_HEXAGON_DMA_OK;
  if (direction == TL_HEXAGON_DMA_VTCM_TO_DDR)
    return tl_hexagon_dma_cache_clean(dst, bytes,
                                      QURT_MEM_CACHE_INVALIDATE);
  return TL_HEXAGON_DMA_ERR_ARGUMENT;
}

TL_DMA_INLINE int tl_hexagon_dma_prepare_transfer(
    const tl_hexagon_dma_transfer &transfer) {
  for (uint32_t row = 0; row < transfer.rows; ++row) {
    const uint8_t *src_row = static_cast<const uint8_t *>(transfer.src) +
                             static_cast<uint64_t>(row) * transfer.src_stride;
    uint8_t *dst_row = static_cast<uint8_t *>(transfer.dst) +
                       static_cast<uint64_t>(row) * transfer.dst_stride;
    int status = tl_hexagon_dma_prepare_ddr(dst_row, src_row,
                                            transfer.row_bytes,
                                            transfer.direction);
    if (status != TL_HEXAGON_DMA_OK)
      return status;
  }
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_INLINE int tl_hexagon_dma_finish_transfer(
    const tl_hexagon_dma_transfer &transfer) {
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
// Caller-owned descriptor queue
// ---------------------------------------------------------------------------

TL_DMA_INLINE uint32_t tl_hexagon_dma_queue_index(
    const tl_hexagon_dma_queue *queue, uint32_t offset) {
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(queue->head) + offset) % queue->capacity);
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
    transfers[i] = {nullptr, nullptr, 0, 0, 0, 0,
                    TL_HEXAGON_DMA_NO_CACHE};
  }
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_INLINE bool tl_hexagon_dma_queue_empty(
    const tl_hexagon_dma_queue *queue) {
  return queue == nullptr || queue->count == 0;
}

TL_DMA_INLINE bool tl_hexagon_dma_queue_full(
    const tl_hexagon_dma_queue *queue) {
  return queue != nullptr && queue->count == queue->capacity;
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_queue_depth(
    const tl_hexagon_dma_queue *queue) {
  return queue == nullptr ? 0 : queue->count;
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_queue_capacity(
    const tl_hexagon_dma_queue *queue) {
  return queue == nullptr ? 0 : queue->capacity;
}

TL_DMA_INLINE tl_hexagon_dma_descriptor *
tl_hexagon_dma_queue_next_descriptor(tl_hexagon_dma_queue *queue) {
  if (queue == nullptr || queue->capacity == 0 ||
      tl_hexagon_dma_queue_full(queue))
    return nullptr;
  return &queue->descriptors[tl_hexagon_dma_queue_index(queue, queue->count)];
}

TL_DMA_INLINE const tl_hexagon_dma_transfer *tl_hexagon_dma_queue_front(
    const tl_hexagon_dma_queue *queue) {
  if (tl_hexagon_dma_queue_empty(queue))
    return nullptr;
  return &queue->transfers[queue->head];
}

// Submit the descriptor in the next queue slot.  This is the raw extension
// point for custom descriptor options: the caller initializes the descriptor,
// performs any required cache maintenance, then supplies completion metadata.
TL_DMA_INLINE int tl_hexagon_dma_queue_submit_prepared(
    tl_hexagon_dma_queue *queue,
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

TL_DMA_INLINE int tl_hexagon_dma_queue_push_1d(
    tl_hexagon_dma_queue *queue, void *dst, const void *src, uint32_t bytes,
    int direction) {
  if (direction != TL_HEXAGON_DMA_DDR_TO_VTCM &&
      direction != TL_HEXAGON_DMA_VTCM_TO_DDR &&
      direction != TL_HEXAGON_DMA_NO_CACHE)
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

TL_DMA_INLINE int tl_hexagon_dma_queue_push_2d(
    tl_hexagon_dma_queue *queue, void *dst, const void *src,
    uint32_t dst_stride, uint32_t src_stride, uint32_t row_bytes,
    uint32_t rows, int direction) {
  if (direction != TL_HEXAGON_DMA_DDR_TO_VTCM &&
      direction != TL_HEXAGON_DMA_VTCM_TO_DDR &&
      direction != TL_HEXAGON_DMA_NO_CACHE)
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

  tl_hexagon_dma_transfer transfer = {dst,       src,       dst_stride,
                                      src_stride, row_bytes, rows,
                                      direction};
  status = tl_hexagon_dma_prepare_transfer(transfer);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  return tl_hexagon_dma_queue_submit_prepared(queue, transfer);
}

TL_DMA_INLINE int tl_hexagon_dma_queue_push(
    tl_hexagon_dma_queue *queue, void *dst, const void *src,
    uint32_t dst_stride, uint32_t src_stride, uint32_t row_bytes,
    uint32_t rows, int direction) {
  if (rows == 1 && dst_stride == row_bytes && src_stride == row_bytes)
    return tl_hexagon_dma_queue_push_1d(queue, dst, src, row_bytes, direction);
  return tl_hexagon_dma_queue_push_2d(queue, dst, src, dst_stride, src_stride,
                                      row_bytes, rows, direction);
}

TL_DMA_INLINE int tl_hexagon_dma_queue_push_ddr_to_vtcm(
    tl_hexagon_dma_queue *queue, void *vtcm_dst, const void *ddr_src,
    uint32_t vtcm_stride, uint32_t ddr_stride, uint32_t row_bytes,
    uint32_t rows) {
  return tl_hexagon_dma_queue_push(queue, vtcm_dst, ddr_src, vtcm_stride,
                                   ddr_stride, row_bytes, rows,
                                   TL_HEXAGON_DMA_DDR_TO_VTCM);
}

TL_DMA_INLINE int tl_hexagon_dma_queue_push_vtcm_to_ddr(
    tl_hexagon_dma_queue *queue, void *ddr_dst, const void *vtcm_src,
    uint32_t ddr_stride, uint32_t vtcm_stride, uint32_t row_bytes,
    uint32_t rows) {
  return tl_hexagon_dma_queue_push(queue, ddr_dst, vtcm_src, ddr_stride,
                                   vtcm_stride, row_bytes, rows,
                                   TL_HEXAGON_DMA_VTCM_TO_DDR);
}

TL_DMA_INLINE uint32_t tl_hexagon_dma_queue_in_flight(
    tl_hexagon_dma_queue *queue) {
  if (queue == nullptr || queue->count == 0)
    return 0;
  tl_hexagon_dma_poll();
  uint32_t in_flight = 0;
  for (uint32_t offset = 0; offset < queue->count; ++offset) {
    uint32_t index = tl_hexagon_dma_queue_index(queue, offset);
    if (!tl_hexagon_dma_descriptor_done(&queue->descriptors[index]))
      ++in_flight;
  }
  return in_flight;
}

TL_DMA_INLINE int tl_hexagon_dma_queue_wait(tl_hexagon_dma_queue *queue,
                                             uint32_t max_in_flight) {
  if (queue == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  while (true) {
    uint32_t engine_status = tl_hexagon_dma_engine_status();
    if (engine_status == TL_HEXAGON_DMA_STATUS_ERROR)
      return TL_HEXAGON_DMA_ERR_ENGINE;

    uint32_t in_flight = 0;
    for (uint32_t offset = 0; offset < queue->count; ++offset) {
      uint32_t index = tl_hexagon_dma_queue_index(queue, offset);
      if (!tl_hexagon_dma_descriptor_done(&queue->descriptors[index]))
        ++in_flight;
    }
    if (in_flight <= max_in_flight)
      return TL_HEXAGON_DMA_OK;
  }
}

TL_DMA_INLINE int tl_hexagon_dma_queue_try_pop(
    tl_hexagon_dma_queue *queue, tl_hexagon_dma_transfer *completed) {
  if (queue == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (queue->count == 0)
    return TL_HEXAGON_DMA_ERR_QUEUE_EMPTY;

  tl_hexagon_dma_poll();
  tl_hexagon_dma_descriptor *descriptor = &queue->descriptors[queue->head];
  if (!tl_hexagon_dma_descriptor_done(descriptor)) {
    uint32_t status = tl_hexagon_dma_engine_status();
    return status == TL_HEXAGON_DMA_STATUS_ERROR
               ? TL_HEXAGON_DMA_ERR_ENGINE
               : TL_HEXAGON_DMA_ERR_BUSY;
  }

  tl_hexagon_dma_transfer transfer = queue->transfers[queue->head];
  int status = tl_hexagon_dma_finish_transfer(transfer);
  if (status != TL_HEXAGON_DMA_OK)
    return status;
  if (completed != nullptr)
    *completed = transfer;

  tl_hexagon_dma_descriptor_clear(descriptor);
  queue->transfers[queue->head] = {nullptr, nullptr, 0, 0, 0, 0,
                                   TL_HEXAGON_DMA_NO_CACHE};
  queue->head = (queue->head + 1) % queue->capacity;
  --queue->count;
  if (queue->count == 0)
    queue->tail = nullptr;
  return TL_HEXAGON_DMA_OK;
}

TL_DMA_INLINE int tl_hexagon_dma_queue_pop(
    tl_hexagon_dma_queue *queue, tl_hexagon_dma_transfer *completed = nullptr) {
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

TL_DMA_INLINE int tl_hexagon_dma_queue_reset(tl_hexagon_dma_queue *queue) {
  if (queue == nullptr)
    return TL_HEXAGON_DMA_ERR_ARGUMENT;
  if (queue->count != 0 || !tl_hexagon_dma_engine_idle())
    return TL_HEXAGON_DMA_ERR_BUSY;
  return tl_hexagon_dma_queue_init(queue, queue->descriptors, queue->transfers,
                                   queue->capacity);
}

// Per-kernel queue used by explicit T.dma_copy/T.dma_wait. The caller must
// own DMA0 exclusively. Initialization never resets someone else's engine.
// Compile managed async kernels with -fno-exceptions (the DSP has no C++ unwinder).
// Stack storage survives until the kernel drains its queue, including returns
// caused by another helper's error. No dynamic allocation or global queue state.
template <uint32_t Capacity> struct tl_hexagon_dma_async_context {
  static_assert(Capacity > 0 && Capacity <= 256, "DMA queue capacity out of range");
  tl_hexagon_dma_descriptor descriptors[Capacity];
  tl_hexagon_dma_transfer transfers[Capacity];
  tl_hexagon_dma_queue queue{};
  int status;
  void (*release_hmx)() = nullptr;

  tl_hexagon_dma_async_context() {
    status = tl_hexagon_dma_engine_idle()
                 ? tl_hexagon_dma_queue_init(&queue, descriptors, transfers, Capacity)
                 : TL_HEXAGON_DMA_ERR_BUSY;
  }
  ~tl_hexagon_dma_async_context() {
    // Only reset after an error while draining our own outstanding descriptors.
    // This stops DMA before descriptor storage and the VTCM lease can go away.
    if (queue.count && tl_hexagon_dma_queue_flush(&queue) != TL_HEXAGON_DMA_OK)
      tl_hexagon_dma_engine_reset();
    if (release_hmx)
      release_hmx();
  }
  tl_hexagon_dma_async_context(const tl_hexagon_dma_async_context &) = delete;
  tl_hexagon_dma_async_context &operator=(const tl_hexagon_dma_async_context &) = delete;

  int copy_1d(void *dst, const void *src, uint32_t bytes, int direction) {
    return tl_hexagon_dma_queue_push_1d(&queue, dst, src, bytes, direction);
  }
  int copy_2d(void *dst, const void *src, uint32_t dst_stride,
              uint32_t src_stride, uint32_t row_bytes, uint32_t rows, int direction) {
    return tl_hexagon_dma_queue_push_2d(&queue, dst, src, dst_stride, src_stride,
                                      row_bytes, rows, direction);
  }
  int wait(uint32_t pending) {
    // Count FIFO submissions, not hardware in-flight descriptors: later copies
    // can complete early, but the newest pending calls still retain their slots.
    if (pending > Capacity)
      return TL_HEXAGON_DMA_ERR_ARGUMENT;
    while (queue.count > pending) {
      int result = tl_hexagon_dma_queue_pop(&queue);
      if (result != TL_HEXAGON_DMA_OK)
        return result;
    }
    // Prevent the compiler from hoisting consumers before DMA completion.
    asm volatile("" : : : "memory");
    return TL_HEXAGON_DMA_OK;
  }
};
