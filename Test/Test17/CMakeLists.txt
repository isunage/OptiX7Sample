set(TEST_TEST17_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST17_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test17Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test17Config.h 
)
add_executable(Test17 
${CMAKE_CURRENT_SOURCE_DIR}/include/Application.h
${CMAKE_CURRENT_SOURCE_DIR}/include/SceneBuilder.h
${CMAKE_CURRENT_SOURCE_DIR}/include/PathTracer.h
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h

${CMAKE_CURRENT_SOURCE_DIR}/src/Application.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/SceneBuilder.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu)
add_custom_command(TARGET Test17 PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${TEST_TEST17_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test17 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu ${TEST_TEST17_CUDA_PATH}/RayTrace.cu)
add_custom_command(TARGET Test17 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu ${TEST_TEST17_CUDA_PATH}/RayDebug.cu)
rtlib_compile_ptx(Test17 ${TEST_TEST17_CUDA_PATH}/RayTrace.cu ${TEST_TEST17_CUDA_PATH}/RayTrace.h)
rtlib_compile_ptx(Test17 ${TEST_TEST17_CUDA_PATH}/RayDebug.cu ${TEST_TEST17_CUDA_PATH}/RayTrace.h)
target_include_directories(Test17 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test17 RTLib)