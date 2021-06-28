set(TEST_TEST15_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST15_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test15Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test15Config.h 
)
add_executable(Test15 
${CMAKE_CURRENT_SOURCE_DIR}/include/Application.h
${CMAKE_CURRENT_SOURCE_DIR}/include/SceneBuilder.h
${CMAKE_CURRENT_SOURCE_DIR}/include/PathTracer.h
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h

${CMAKE_CURRENT_SOURCE_DIR}/src/Application.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/SceneBuilder.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu)
add_custom_command(TARGET Test15 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${TEST_TEST15_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test15 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu ${TEST_TEST15_CUDA_PATH}/RayTrace.cu)
rtlib_compile_ptx(Test15 ${TEST_TEST15_CUDA_PATH}/RayTrace.cu ${TEST_TEST15_CUDA_PATH}/RayTrace.h)
target_include_directories(Test15 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test15 RTLib)