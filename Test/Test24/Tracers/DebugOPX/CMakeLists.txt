set(TEST_TEST24_DEBUG_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24DebugOPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24DebugOPXConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT 101)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_DEBUG_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_DebugOPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DebugOPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24DebugOPXConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24DebugOPXConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/DebugOPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu
)
add_custom_target (Test24_Tracer_DebugOPX_BUILD_KERNEL)
add_custom_command(TARGET Test24_Tracer_DebugOPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST24_DEBUG_OPX_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test24_Tracer_DebugOPX_BUILD_KERNEL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu    ${TEST_TEST24_DEBUG_OPX_CUDA_PATH}/RayDebug.cu)

rtlib_compile_ptx(Test24_Tracer_DebugOPX_BUILD_KERNEL ${TEST_TEST24_DEBUG_OPX_CUDA_PATH}/RayDebug.cu   ${TEST_TEST24_DEBUG_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_DebugOPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include   ${CMAKE_CURRENT_BINARY_DIR}  ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_DebugOPX PUBLIC  TestLib Test24_Share)
add_dependencies(Test24_Tracer_DebugOPX Test24_Tracer_DebugOPX_BUILD_KERNEL)
set_target_properties(Test24_Tracer_DebugOPX  Test24_Tracer_DebugOPX_BUILD_KERNEL   PROPERTIES  FOLDER Test24/Tracer/DebugOPX)