#pragma once

// Each DSP skel owns one copy of its process resources. Kernels compiled
// separately from the dispatcher share those objects within the same DSO.
#ifdef TL_HEXAGON_SEPARATE_KERNELS
#define TL_HEXAGON_SESSION_STATE inline __attribute__((visibility("hidden")))
#else
#define TL_HEXAGON_SESSION_STATE static
#endif
