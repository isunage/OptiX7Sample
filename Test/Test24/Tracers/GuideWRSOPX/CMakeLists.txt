set(TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24GuideWRSOPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideWRSOPXConfig.h
)

set(PATH_GUIDING_MAX_DEPTH       1024)
set(RAY_TRACE_S_FILTER           SpatialFilterNearest)
set(RAY_TRACE_D_FILTER           DirectionalFilterNearest)
# Specular�ʂɑ΂���WRS���s�����ADefault�ł͍s��Ȃ��i��ق�shinness�������Ȃ���΂ނ��땪�U�͑�������X���j
set(RAY_TRACE_NEE_SPECULAR 0)
# float�̎w����23->2^-23,2^-23,2^-23
set(PATH_GUIDING_STREE_MAX_DEPTH 69)
# float�̎w����23
set(PATH_GUIDING_DTREE_MAX_DEPTH 24)
set(PATH_GUIDING_LI_COSINE       1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuidingConfig.h.in 
    ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/PathGuidingConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT 101)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_GuideWRSOPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GuideWRSOPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTPathGuidingUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideWRSOPXConfig.h
    ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/PathGuidingConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/GuideWRSOPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuide.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

add_custom_target (Test24_Tracer_GuideWRSOPX_BUILD_KERNEL)
add_custom_command(TARGET Test24_Tracer_GuideWRSOPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test24_Tracer_GuideWRSOPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h  ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/PathGuiding.h )
add_custom_command(TARGET Test24_Tracer_GuideWRSOPX_BUILD_KERNEL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuide.cu    ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/RayGuide.cu)

rtlib_compile_ptx(Test24_Tracer_GuideWRSOPX_BUILD_KERNEL ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH}/RayGuide.cu   ${TEST_TEST24_GUIDE_WRS_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_GuideWRSOPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_GuideWRSOPX PUBLIC  TestLib Test24_Share)

add_dependencies(Test24_Tracer_GuideWRSOPX Test24_Tracer_GuideWRSOPX_BUILD_KERNEL)
set_target_properties(Test24_Tracer_GuideWRSOPX  Test24_Tracer_GuideWRSOPX_BUILD_KERNEL PROPERTIES  FOLDER Test24/Tracers/GuideWRSOPX )