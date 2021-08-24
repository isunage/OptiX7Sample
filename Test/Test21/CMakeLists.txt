set(TEST_TEST21_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST21_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test21Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test21Config.h 
)

add_executable(Test21
${CMAKE_CURRENT_SOURCE_DIR}/include/RTApplication.h
${CMAKE_CURRENT_SOURCE_DIR}/include/RTPipeline.h
${CMAKE_CURRENT_SOURCE_DIR}/include/RTSubPass.h
${CMAKE_CURRENT_SOURCE_DIR}/include/RTRecordBuffer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/RTTracer.h
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
${CMAKE_CURRENT_SOURCE_DIR}/include/SceneBuilder.h

${CMAKE_CURRENT_SOURCE_DIR}/src/RTApplication.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/RTPipeline.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/RTTracer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu)

add_custom_command(TARGET Test21 PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${TEST_TEST21_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test21 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu ${TEST_TEST21_CUDA_PATH}/RayTrace.cu)
add_custom_command(TARGET Test21 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu ${TEST_TEST21_CUDA_PATH}/RayDebug.cu)
rtlib_compile_ptx(Test21 ${TEST_TEST21_CUDA_PATH}/RayTrace.cu ${TEST_TEST21_CUDA_PATH}/RayTrace.h)
rtlib_compile_ptx(Test21 ${TEST_TEST21_CUDA_PATH}/RayDebug.cu ${TEST_TEST21_CUDA_PATH}/RayTrace.h)
target_include_directories(Test21 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test21 RTLib)