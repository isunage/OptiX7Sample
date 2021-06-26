set(TEST_TEST16_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test16Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test16Config.h 
)
add_executable(Test16 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(Test16 RTLib)