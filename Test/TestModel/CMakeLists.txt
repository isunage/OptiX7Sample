set(TEST_TEST_MODEL_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST_MODEL_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestModelConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestModelConfig.h
)
add_executable(TestModel ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_include_directories(TestModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries(TestModel RTLib)