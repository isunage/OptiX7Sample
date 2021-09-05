set(TEST_TEST_PG_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST_PG_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPGConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestPGConfig.h
)

add_executable(TestPathGuiding
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

add_custom_command(TARGET TestPathGuiding PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST_PG_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET TestPathGuiding PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h  ${TEST_TEST_PG_CUDA_PATH}/PathGuiding.h )
add_custom_command(TARGET TestPathGuiding POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu    ${TEST_TEST_PG_CUDA_PATH}/RayDebug.cu)
add_custom_command(TARGET TestPathGuiding POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayGuiding.cu  ${TEST_TEST_PG_CUDA_PATH}/RayGuiding.cu)
rtlib_compile_ptx(TestPathGuiding ${TEST_TEST_PG_CUDA_PATH}/RayGuiding.cu ${TEST_TEST_PG_CUDA_PATH}/RayTrace.h ${TEST_TEST_PG_CUDA_PATH}/PathGuiding.h )
rtlib_compile_ptx(TestPathGuiding ${TEST_TEST_PG_CUDA_PATH}/RayDebug.cu   ${TEST_TEST_PG_CUDA_PATH}/RayTrace.h)
target_include_directories(TestPathGuiding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries(TestPathGuiding RTLib)