file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
set(TEST_TEST9_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test9Config.h 
    ${CMAKE_CURRENT_BINARY_DIR}/Test9Config.h 
)
add_executable(Test9 main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu)
target_include_directories(Test9 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test9 PUBLIC RTLib)
