file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/cuda)
set(TEST_TEST6_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test6Config.h 
    ${CMAKE_CURRENT_BINARY_DIR}/Test6Config.h 
)
add_executable(Test6 main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu)
target_include_directories(Test6 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test6 PUBLIC RTLib)
