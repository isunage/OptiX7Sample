file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda)
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
set(TEST_TEST13_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST13_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test13Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test13Config.h 
)
add_executable(Test13 main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu)
add_custom_command(TARGET Test13 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${CMAKE_CURRENT_BINARY_DIR}/cuda/RayTrace.h )
add_custom_command(TARGET Test13 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu ${CMAKE_CURRENT_BINARY_DIR}/cuda/RayTrace.cu)
rtlib_compile_ptx(Test13 ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h)
target_include_directories(Test13 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test13 PUBLIC RTLib)