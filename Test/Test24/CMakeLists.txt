set(TEST_TEST24_DEF_WIDTH  1024)
set(TEST_TEST24_DEF_HEIGHT 1024)
set(TEST_TEST24_DEF_TITLE  "Test24Application")
set(TEST_TEST24_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test24Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h)

add_executable(Test24 
# HEADER
${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Application.h
${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
# SOURCE
${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Application.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_include_directories(Test24 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(Test24 PUBLIC TestLib)