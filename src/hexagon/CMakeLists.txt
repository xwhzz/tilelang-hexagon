# Hexagon backend: source files that are always safe to compile.
#
# Hexagon NPU codegen is pure C++ source generation (no Hexagon SDK needed at
# tilelang build time); always compile it so that "target.build.tilelang_hexagon"
# is available. Compilation of the generated source to a cDSP skel is done by
# the Python adapter using the Hexagon SDK.

file(GLOB TILE_LANG_HEXAGON_SRCS
  src/hexagon/codegen/codegen_hexagon.cc
  src/hexagon/codegen/rt_mod_hexagon.cc
  src/hexagon/layout/*.cc
  src/hexagon/op/*.cc
  src/hexagon/target.cc
  src/hexagon/target_utils.cc
  src/hexagon/transform/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HEXAGON_SRCS})
