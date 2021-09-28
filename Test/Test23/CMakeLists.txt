set(TEST_TEST23_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST23_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test23Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test23Config.h
)

add_executable(Test23
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTFrameBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTAssets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test23Application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test23Config.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test23Application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTAssets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu
)

add_custom_command(TARGET Test23  PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST23_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test23 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu    ${TEST_TEST23_CUDA_PATH}/RayDebug.cu)
add_custom_command(TARGET Test23 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu    ${TEST_TEST23_CUDA_PATH}/RayTrace.cu)

rtlib_compile_ptx(Test23 ${TEST_TEST23_CUDA_PATH}/RayTrace.cu  ${TEST_TEST23_CUDA_PATH} )
rtlib_compile_ptx(Test23 ${TEST_TEST23_CUDA_PATH}/RayDebug.cu  ${TEST_TEST23_CUDA_PATH} )

target_include_directories(Test23 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test23 RTLib)