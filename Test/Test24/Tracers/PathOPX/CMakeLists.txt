set(TEST_TEST24_PATH_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24PathOPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24PathOPXConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT 101)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_PATH_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_PathOPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/PathOPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test24PathOPXConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/PathOPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

add_custom_command(TARGET Test24_Tracer_PathOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST24_PATH_OPX_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test24_Tracer_PathOPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu    ${TEST_TEST24_PATH_OPX_CUDA_PATH}/RayTrace.cu)

rtlib_compile_ptx(Test24_Tracer_PathOPX ${TEST_TEST24_PATH_OPX_CUDA_PATH}/RayTrace.cu   ${TEST_TEST24_PATH_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_PathOPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_PathOPX PUBLIC  TestLib)