add_executable(Test1 main.cpp)
set(TEST_TEST1_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST1_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test1Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test1Config.h 
)
target_include_directories(Test1 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Test1 PUBLIC RTLib)