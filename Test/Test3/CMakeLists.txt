##cuda driver api test
set(TEST_TEST3_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST3_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test3Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test3Config.h 
)
add_executable(Test3 main.cpp)
add_custom_command(TARGET Test3 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/simpleKernel.cu  ${CMAKE_CURRENT_BINARY_DIR}/cuda/simpleKernel.cu )
target_include_directories(Test3 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test3 PUBLIC RTLib)