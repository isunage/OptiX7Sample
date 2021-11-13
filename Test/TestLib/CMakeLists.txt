add_library(TestLib 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTFramebuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTGui.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTApplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTFramebuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTGui.cpp
)

target_include_directories(
    TestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(
    TestLib PUBLIC RTLib imgui-filedialog
)

set(TEST_TESTLIB_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
string(REPLACE "/" "\\\\" TEST_TESTLIB_DATA_PATH ${TEST_TESTLIB_DATA_PATH})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/TestLibConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
)

add_executable(TestLib_Test 
    ${CMAKE_CURRENT_BINARY_DIR}/TestLibConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(
    TestLib_Test PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
    TestLib_Test PUBLIC TestLib
)

