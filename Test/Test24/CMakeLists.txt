# VARIABLE
set(TEST_TEST24_DEF_WIDTH  1024)
set(TEST_TEST24_DEF_HEIGHT 1024)
set(TEST_TEST24_DEF_TITLE "Test24Application")
set(TEST_TEST24_GLSL_PATH "${CMAKE_CURRENT_BINARY_DIR}/glsl")
set(TEST_TEST24_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")
# CONFIGURE
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Test24Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h)
# EXECUTIVE
add_executable(Test24 
# HEADER
${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Application.h
${CMAKE_CURRENT_SOURCE_DIR}/include/Tracers/Test24SimpleViewGLTracer.h
${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
# SOURCE
${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Application.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Test24SimpleViewGLTracer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
# GLSL
${CMAKE_CURRENT_SOURCE_DIR}/glsl/SimpleView.Vert.glsl
${CMAKE_CURRENT_SOURCE_DIR}/glsl/SimpleView.Frag.glsl
)
# COPY GLSL
add_custom_command(TARGET Test24 PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/glsl/SimpleView.Vert.glsl ${TEST_TEST24_GLSL_PATH}/SimpleView.Vert.glsl )
add_custom_command(TARGET Test24 PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/glsl/SimpleView.Frag.glsl ${TEST_TEST24_GLSL_PATH}/SimpleView.Frag.glsl )
# SET INCLUDE_DIR
target_include_directories(Test24 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
# LINK LIBRARIES
target_link_libraries(Test24 PUBLIC TestLib)