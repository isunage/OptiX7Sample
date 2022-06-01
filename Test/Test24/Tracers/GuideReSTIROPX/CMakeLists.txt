set(TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24GuideReSTIROPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideReSTIROPXConfig.h
)

set(RAY_TRACE_MAX_VERTEX_COUNT     101)
set(RAY_TRACE_ENABLE_TEXTURE_SAMPLE  1)
set(RAY_TRACE_ENABLE_JITTER_SAMPLE   1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_GuideReSTIROPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GuideReSTIROPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24GuideReSTIROPXConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideReSTIROPXConfig.h
    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayTraceConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/GuideReSTIROPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayFirst.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RaySecond.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ReservoirReuse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

target_include_directories(Test24_Tracer_GuideReSTIROPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include   ${CMAKE_CURRENT_BINARY_DIR}  ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_GuideReSTIROPX PUBLIC  TestLib Test24_Share)

add_custom_target(Test24_Tracer_GuideReSTIROPX_Build_Kernel )

add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_Build_Kernel PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h           ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayTrace.h  )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_Build_Kernel POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayFirst.cu          ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayFirst.cu )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_Build_Kernel POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RaySecond.cu         ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RaySecond.cu)
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_Build_Kernel POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ReservoirReuse.cu    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/ReservoirReuse.cu )

rtlib_compile_ptx(Test24_Tracer_GuideReSTIROPX_Build_Kernel ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayFirst.cu         ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH} )
rtlib_compile_ptx(Test24_Tracer_GuideReSTIROPX_Build_Kernel ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RaySecond.cu        ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH} )
rtlib_compile_ptx(Test24_Tracer_GuideReSTIROPX_Build_Kernel ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/ReservoirReuse.cu   ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH} )

add_dependencies(Test24_Tracer_GuideReSTIROPX Test24_Tracer_GuideReSTIROPX_Build_Kernel)
set_target_properties(Test24_Tracer_GuideReSTIROPX  Test24_Tracer_GuideReSTIROPX_Build_Kernel PROPERTIES FOLDER Test24/Tracer/GuideReSTIROPX)