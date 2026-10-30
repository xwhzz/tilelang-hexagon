#pragma once
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>

// v73/v75 have qfloat conversions, but they do not implement IEEE casts:
// multiplying a half by 1.0 changes finite values and treats Inf/NaN as finite.
// Integer vector operations preserve finite values, signed zeros and infinities
// and narrow with round-to-nearest, ties-to-even. NaN payload/quieting is not a
// cross-architecture contract. v79 retains native vcvt behavior (including
// narrowing exact -0.0f to +0.0h). Both paths keep the even/odd lane
// convention.

#if __HVX_ARCH__ < 79
typedef uint32_t tl_hvx_u32 __attribute__((ext_vector_type(32)));
TL_DEVICE tl_hvx_u32 tl_hvx_as_u32(HVX_Vector v) {
  tl_hvx_u32 r;
  __builtin_memcpy(&r, &v, 128);
  return r;
}
TL_DEVICE HVX_Vector tl_hvx_as_vector(tl_hvx_u32 v) {
  HVX_Vector r;
  __builtin_memcpy(&r, &v, 128);
  return r;
}
TL_DEVICE tl_hvx_u32 tl_hvx_widen_half_bits(tl_hvx_u32 h) {
  tl_hvx_u32 sign = (h & 0x8000u) << 16;
  tl_hvx_u32 mag = h & 0x7fffu;
  tl_hvx_u32 m = h & 0x3ffu;
  tl_hvx_u32 lz = tl_hvx_as_u32(Q6_Vuw_vcl0_Vuw(tl_hvx_as_vector(m)));
  tl_hvx_u32 sub = ((134u - lz) << 23) | ((m << (lz - 8u)) & 0x7fffffu);
  tl_hvx_u32 normal = (mag << 13) + 0x38000000u;
  tl_hvx_u32 out = mag < 0x400u ? sub : normal;
  out = mag == 0 ? tl_hvx_u32{} : out;
  out = mag >= 0x7c00u ? ((m << 13) | 0x7f800000u) : out;
  return out | sign;
}
TL_DEVICE tl_hvx_u32 tl_hvx_narrow_float_bits(tl_hvx_u32 bits) {
  tl_hvx_u32 sign = (bits >> 16) & 0x8000u;
  tl_hvx_u32 mag = bits & 0x7fffffffu;
  tl_hvx_u32 m = (mag & 0x7fffffu) | 0x800000u;
  tl_hvx_u32 shift = 126u - (mag >> 23);
  shift = shift < 14u ? tl_hvx_u32(14u) : shift;
  shift = shift > 24u ? tl_hvx_u32(24u) : shift;
  tl_hvx_u32 bias = (tl_hvx_u32(1u) << (shift - 1u)) - 1u + ((m >> shift) & 1u);
  tl_hvx_u32 sub = (m + bias) >> shift;
  tl_hvx_u32 norm = (mag + 0xfffu + ((mag >> 13) & 1u) - 0x38000000u) >> 13;
  tl_hvx_u32 out = mag < 0x38800000u ? sub : norm;
  out = mag <= 0x33000000u ? tl_hvx_u32{} : out;
  out = mag >= 0x477ff000u ? tl_hvx_u32(0x7c00u) : out;
  out = mag > 0x7f800000u ? ((mag >> 13) & 0x3ffu) | 0x7e00u : out;
  return out | sign;
}
#endif

TL_DEVICE HVX_VectorPair tl_hvx_convert_hf_to_sf(HVX_Vector h) {
#if __HVX_ARCH__ >= 79
  return Q6_Wsf_vcvt_Vhf(h);
#else
  tl_hvx_u32 bits = tl_hvx_as_u32(h);
  return Q6_W_vcombine_VV(
      tl_hvx_as_vector(tl_hvx_widen_half_bits(bits >> 16)),
      tl_hvx_as_vector(tl_hvx_widen_half_bits(bits & 0xffffu)));
#endif
}

TL_DEVICE HVX_Vector tl_hvx_convert_sf_to_hf(HVX_Vector even, HVX_Vector odd) {
#if __HVX_ARCH__ >= 79
  return Q6_Vhf_vcvt_VsfVsf(even, odd);
#else
  return tl_hvx_as_vector(tl_hvx_narrow_float_bits(tl_hvx_as_u32(even)) |
                          (tl_hvx_narrow_float_bits(tl_hvx_as_u32(odd)) << 16));
#endif
}
