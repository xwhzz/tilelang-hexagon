#pragma once
// ---------------------------------------------------------------------------
// tilelang runtime bridge — bind the tilelang HMX/VTCM runtime to a host skel's
// already-acquired VTCM region + HMX lock, so an embedded kernel does NOT open
// its own session.  Include this from the kernel .cc (it pulls <hmx.h>); the
// host's stock op function includes only <tl_embed.h> (C-safe, no HMX).
//
// Co-compile the bridge and the kernel that reads the runtime state in ONE TU:
// hmx.h's session state is `static`, so tl_bridge_enter() and the gemm must
// share a translation unit.  (Multiple ops in separate TUs are fine — each is
// self-contained.)  See docs/llama_cpp_integration.md.
// ---------------------------------------------------------------------------
#include <tl_templates/hexagon/hmx.h>

// Bind the tilelang VTCM arena to the caller's region and prime the HMX scales.
// Precondition: the caller already holds the HMX lock on this thread and owns
// `vtcm_base` for the kernel's duration.
static inline int tl_bridge_enter(void *vtcm_base, unsigned int vtcm_size) {
  if (tl_vtcm_bind_region(vtcm_base, vtcm_size) != 0)
    return -1;
  tl_hmx_fill_unit_scales((uint32_t *)tl_vtcm_base_ptr);  // unit scales @ base+0
  tl_hmx_inited    = 1;                        // ride the caller's HMX lock
  return 0;
}

// Unbind (do NOT release VTCM/HMX — the host owns them).
static inline void tl_bridge_exit(void) {
  tl_vtcm_base_ptr = 0;
  tl_vtcm_total    = 0;
  tl_hmx_inited    = 0;
}
