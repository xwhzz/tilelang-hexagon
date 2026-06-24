cmake_minimum_required(VERSION 3.14.3)
project(tl_matmul_kernel C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src
)

if(${OS_TYPE} MATCHES "HLOS")
    add_executable(tl_matmul_kernel_test
        ${CMAKE_CURRENT_BINARY_DIR}/tl_matmul_kernel_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tl_matmul_kernel_host.c
    )
    build_idl(tl_matmul_kernel.idl tl_matmul_kernel_test)
    set_common_compile_and_link_options(tl_matmul_kernel_test)
    target_compile_definitions(tl_matmul_kernel_test PUBLIC VERIFY_PRINT_ERROR)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(tl_matmul_kernel_test PUBLIC -llog -ldl)
    endif()
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(tl_matmul_kernel_test ${dsprpc})
    copy_binaries(tl_matmul_kernel_test)

    # Persistent agent: same FastRPC stub, served over a TCP socket (reuses the
    # IDL artifacts from tl_matmul_kernel_test, ordered via add_dependencies).
    add_executable(tl_matmul_kernel_agent
        ${CMAKE_CURRENT_BINARY_DIR}/tl_matmul_kernel_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tl_matmul_kernel_agent.c
    )
    add_dependencies(tl_matmul_kernel_agent tl_matmul_kernel_test)
    set_common_compile_and_link_options(tl_matmul_kernel_agent)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(tl_matmul_kernel_agent PUBLIC -llog -ldl)
    endif()
    link_custom_library(tl_matmul_kernel_agent ${dsprpc})
    copy_binaries(tl_matmul_kernel_agent)
else()
    add_library(tl_matmul_kernel_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/tl_matmul_kernel_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tl_matmul_kernel_dsp.cc
    )
    build_idl(tl_matmul_kernel.idl tl_matmul_kernel_skel)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mhmx -mhvx -Wno-error")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mhmx -mhvx -Wno-error")
    copy_binaries(tl_matmul_kernel_skel)
endif()
