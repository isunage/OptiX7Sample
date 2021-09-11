set(TEST_TEST_PG_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST_PG_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPGConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestPGConfig.h
)

set(PATH_GUIDING_MAX_DEPTH       1024)
set(RAY_TRACE_S_FILTER SpatialFilterNearest)
set(RAY_TRACE_D_FILTER DirectionalFilterNearest)
set(PATH_GUIDING_STREE_MAX_DEPTH 32)
set(PATH_GUIDING_DTREE_MAX_DEPTH 16)
set(PATH_GUIDING_LI_COSINE       1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuidingConfig.h.in 
    ${TEST_TEST_PG_CUDA_PATH}/PathGuidingConfig.h
)

set(RAY_TRACE_MAX_VERTEX_COUNT 32)
set(RAY_TRACE_ENABLE_SAMPLE    1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceConfig.h.in 
    ${TEST_TEST_PG_CUDA_PATH}/RayTraceConfig.h
)

add_executable(TestPathGuiding2
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTPathGuidingUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTSubPass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTRecordBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/SceneBuilder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h
    ${CMAKE_CURRENT_BINARY_DIR}/TestPGConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuiding.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu
)

add_custom_command(TARGET TestPathGuiding2 PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST_PG_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET TestPathGuiding2 PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h  ${TEST_TEST_PG_CUDA_PATH}/PathGuiding.h )
add_custom_command(TARGET TestPathGuiding2 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu    ${TEST_TEST_PG_CUDA_PATH}/RayDebug.cu)
add_custom_command(TARGET TestPathGuiding2 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuiding.cu  ${TEST_TEST_PG_CUDA_PATH}/RayGuiding.cu)

rtlib_compile_ptx(TestPathGuiding2 ${TEST_TEST_PG_CUDA_PATH}/RayGuiding.cu ${TEST_TEST_PG_CUDA_PATH} )
rtlib_compile_ptx(TestPathGuiding2 ${TEST_TEST_PG_CUDA_PATH}/RayDebug.cu   ${TEST_TEST_PG_CUDA_PATH} )

target_include_directories(TestPathGuiding2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(TestPathGuiding2 RTLib)