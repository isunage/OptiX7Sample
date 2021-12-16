set(TEST_TEST24_RESTIR_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24RESTIROPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24RESTIROPXConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT 101)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_ReSTIROPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ReSTIROPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test24ReSTIROPXConfig.h
    ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RayTraceConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/ReSTIROPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayFirst.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RaySecond.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ReservoirReuse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

add_custom_command(TARGET Test24_Tracer_ReSTIROPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RayTrace.h  )
add_custom_command(TARGET Test24_Tracer_ReSTIROPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayFirst.cu    ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RayFirst.cu )
add_custom_command(TARGET Test24_Tracer_ReSTIROPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RaySecond.cu   ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RaySecond.cu)
add_custom_command(TARGET Test24_Tracer_ReSTIROPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ReservoirReuse.cu    ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/ReservoirReuse.cu )

rtlib_compile_ptx(Test24_Tracer_ReSTIROPX ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RayFirst.cu         ${TEST_TEST24_RESTIR_OPX_CUDA_PATH} )
rtlib_compile_ptx(Test24_Tracer_ReSTIROPX ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/RaySecond.cu        ${TEST_TEST24_RESTIR_OPX_CUDA_PATH} )
rtlib_compile_ptx(Test24_Tracer_ReSTIROPX ${TEST_TEST24_RESTIR_OPX_CUDA_PATH}/ReservoirReuse.cu   ${TEST_TEST24_RESTIR_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_ReSTIROPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_ReSTIROPX PUBLIC  TestLib Test24_Share)