set(TEST_TEST24_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST24_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
)

set(TEST24_DEBUG_TRACER_ENABLE_SAMPLE 1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerConfig.h.in
    ${TEST_TEST24_CUDA_PATH}/Test24DebugTracerConfig.h
)

add_executable(Test24 
    # include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Renderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test24Application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Tracers/Test24DebugTracer.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test24Config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayType.h
    # src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24Application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test24DebugTracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp 
    # tracers: Debug
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerConfig.h.in
    ${TEST_TEST24_CUDA_PATH}/Test24DebugTracerConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerDeclare.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerKernel.cu
)

add_custom_command(TARGET Test24  PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayType.h      ${TEST_TEST24_CUDA_PATH}/RayType.h )
add_custom_command(TARGET Test24  PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerDeclare.h  ${TEST_TEST24_CUDA_PATH}/Test24DebugTracerDeclare.h )
add_custom_command(TARGET Test24 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Test24DebugTracerKernel.cu  ${TEST_TEST24_CUDA_PATH}/Test24DebugTracerKernel.cu)

rtlib_compile_ptx(Test24 ${TEST_TEST24_CUDA_PATH}/Test24DebugTracerKernel.cu ${TEST_TEST24_CUDA_PATH} )

target_include_directories(Test24 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${TEST_TEST24_CUDA_PATH})

target_link_libraries(Test24 PUBLIC TestLib imgui-filedialog)

