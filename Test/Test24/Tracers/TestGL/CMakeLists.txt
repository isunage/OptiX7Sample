add_library(
    Test24_Tracer_TestGL STATIC EXCLUDE_FROM_ALL 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestGLTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TestGLTracer.cpp
)

target_include_directories(Test24_Tracer_TestGL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(Test24_Tracer_TestGL PUBLIC TestLib Test24_Share)