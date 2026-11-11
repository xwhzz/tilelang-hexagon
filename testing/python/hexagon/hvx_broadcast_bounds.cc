// Exercise the production broadcast implementation. The host build substitutes
// only the HVX types/intrinsics; ASan checks the real helper's local output.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TL_LANES_HOST_TEST
struct alignas(128) HVX_Vector {
  uint8_t bytes[128];
};
struct HVX_VectorPair {
  HVX_Vector lo, hi;
};
struct HVX_VectorPred {};
template <typename T, int N> struct vec_type {
  T values[N];
};

static HVX_Vector Q6_V_vror_VR(HVX_Vector v, int offset) {
  HVX_Vector out;
  for (int i = 0; i < 128; ++i)
    out.bytes[i] = v.bytes[(i + offset) & 127];
  return out;
}

static HVX_Vector Q6_V_vdelta_VV(HVX_Vector v, HVX_Vector control) {
  for (int bit = 1; bit < 128; bit <<= 1) {
    HVX_Vector next;
    for (int i = 0; i < 128; ++i)
      next.bytes[i] = v.bytes[(control.bytes[i] & bit) ? i ^ bit : i];
    v = next;
  }
  return v;
}

// Unused helpers in the same header need declarations, not mocked behavior.
HVX_Vector Q6_V_valign_VVR(HVX_Vector, HVX_Vector, int);
HVX_VectorPred Q6_Q_vsetq2_R(int);
HVX_VectorPred Q6_Q_vsetq_R(int);
HVX_VectorPred Q6_Q_and_QQn(HVX_VectorPred, HVX_VectorPred);
void Q6_vmem_QRIV(HVX_VectorPred, HVX_Vector *, HVX_Vector);
HVX_Vector Q6_V_vzero();
HVX_VectorPair Q6_W_vshuff_VVR(HVX_Vector, HVX_Vector, int);
HVX_Vector Q6_V_lo_W(HVX_VectorPair);
#endif

#include <tl_templates/hexagon/hvx_lanes.h>

template <typename T, int N, int G> static int check() {
  constexpr int M = 128 / sizeof(T);
  uint32_t state = 0x73a79b04;
  for (int trial = 0; trial < 8; ++trial) {
    alignas(128) T input[M];
    for (auto &value : input) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      value = static_cast<T>(state);
    }
    vec_type<T, M> x;
    memcpy(&x, input, sizeof(input));
    for (int group = 0; group < M / G; ++group) {
      struct alignas(256) Guarded {
        uint8_t before[256];
        vec_type<T, N> value;
        uint8_t after[128];
      } guarded;
      memset(&guarded, 0xa5, sizeof(guarded));
      guarded.value = tl_hvx_broadcast_lane<N, G>(x, group);
      T output[N];
      memcpy(output, &guarded.value, sizeof(output));
      for (int lane = 0; lane < N; ++lane)
        if (output[lane] != input[group * G + lane % G])
          return 1;
      for (uint8_t byte : guarded.before)
        if (byte != 0xa5)
          return 2;
      for (uint8_t byte : guarded.after)
        if (byte != 0xa5)
          return 3;
    }
  }
  return 0;
}

template <int G> static int widths() {
  // FP16/FP32-sized bit patterns, without floating-point arithmetic: partial
  // registers, one register, and two registers. The input is one full register.
  return check<uint16_t, 8, G>() || check<uint16_t, 16, G>() ||
         check<uint16_t, 32, G>() || check<uint16_t, 64, G>() ||
         check<uint16_t, 128, G>() || check<uint32_t, 4, G>() ||
         check<uint32_t, 8, G>() || check<uint32_t, 16, G>() ||
         check<uint32_t, 32, G>() || check<uint32_t, 64, G>();
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 4;
  switch (atoi(argv[1])) {
  case 1:
    return widths<1>();
  case 2:
    return widths<2>();
  case 4:
    return widths<4>();
  default:
    return 5;
  }
}
