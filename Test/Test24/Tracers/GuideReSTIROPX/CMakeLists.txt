set(TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24GuideReSTIROPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideReSTIROPXConfig.h
)

set(PATH_GUIDING_MAX_DEPTH       1024)
set(RAY_TRACE_S_FILTER           SpatialFilterNearest)
set(RAY_TRACE_D_FILTER           DirectionalFilterNearest)
# Specular�ʂɑ΂���ReSTIR���s�����ADefault�ł͍s��Ȃ��i��ق�shinness�������Ȃ���΂ނ��땪�U�͑�������X���j
set(RAY_TRACE_NEE_SPECULAR 0)
# float�̎w����23->2^-23,2^-23,2^-23
set(PATH_GUIDING_STREE_MAX_DEPTH 69)
# float�̎w����23
set(PATH_GUIDING_DTREE_MAX_DEPTH 24)
set(PATH_GUIDING_LI_COSINE       1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuidingConfig.h.in 
    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/PathGuidingConfig.h
)


set(RAY_TRACE_MAX_VERTEX_COUNT 101)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayTraceConfig.h
)

add_library(Test24_Tracer_GuideReSTIROPX STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GuideReSTIROPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/RTPathGuidingUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Reservoir.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/LightUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SurfaceParameters.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/MaterialParameters.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24GuideReSTIROPXConfig.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24GuideReSTIROPXConfig.h
    ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/PathGuidingConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/GuideReSTIROPXTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuide.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

add_custom_target (Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL)
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h           ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayTrace.h            )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.h           ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/Callable.h            )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h        ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/PathGuiding.h         )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Reservoir.h          ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/Reservoir.h           )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/SurfaceParameters.h  ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/SurfaceParameters.h   )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/MaterialParameters.h ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/MaterialParameters.h  )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/LightUtils.h         ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/LightUtils.h          )
add_custom_command(TARGET Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuide.cu          ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayGuide.cu           )

rtlib_compile_ptx(Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH}/RayGuide.cu   ${TEST_TEST24_GUIDE_RESTIR_OPX_CUDA_PATH} )

target_include_directories(Test24_Tracer_GuideReSTIROPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ext ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test24_Tracer_GuideReSTIROPX PUBLIC  TestLib Test24_Share)

add_dependencies(Test24_Tracer_GuideReSTIROPX Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL)
set_target_properties(Test24_Tracer_GuideReSTIROPX  Test24_Tracer_GuideReSTIROPX_BUILD_KERNEL PROPERTIES  FOLDER Test24/Tracer/GuideReSTIROPX )