set(TEST_TEST24_WRS_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24WRSOPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24WRSOPXConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT      101)
set(RAY_TRACE_ENABLE_TEXTURE_SAMPLE 1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_WRS_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_WRSOPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WRSOPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24WRSOPXConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24WRSOPXConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/WRSOPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)
add_custom_target( Test24_Tracer_WRSOPX_Build_Kernel)
add_custom_command(TARGET Test24_Tracer_WRSOPX_Build_Kernel PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST24_WRS_OPX_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test24_Tracer_WRSOPX_Build_Kernel POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu    ${TEST_TEST24_WRS_OPX_CUDA_PATH}/RayTrace.cu)
rtlib_compile_ptx(Test24_Tracer_WRSOPX_Build_Kernel ${TEST_TEST24_WRS_OPX_CUDA_PATH}/RayTrace.cu   ${TEST_TEST24_WRS_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_WRSOPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include   ${CMAKE_CURRENT_BINARY_DIR}  ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_WRSOPX PUBLIC  TestLib Test24_Share)
add_dependencies(Test24_Tracer_WRSOPX Test24_Tracer_WRSOPX_Build_Kernel)
set_target_properties(Test24_Tracer_WRSOPX Test24_Tracer_WRSOPX_Build_Kernel PROPERTIES  FOLDER Test24/Tracer/WRSOPX)