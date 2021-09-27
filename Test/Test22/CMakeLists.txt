set(TEST_TEST22_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(TEST_TEST22_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Data")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test22Config.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test22Config.h
)

add_executable(Test22
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTPipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTFrameBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RTAssets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Test22Application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_BINARY_DIR}/Test22Config.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Test22Application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTAssets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu
)

add_custom_command(TARGET Test22  PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h     ${TEST_TEST22_CUDA_PATH}/RayTrace.h )
add_custom_command(TARGET Test22 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayDebug.cu    ${TEST_TEST22_CUDA_PATH}/RayDebug.cu)
add_custom_command(TARGET Test22 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu    ${TEST_TEST22_CUDA_PATH}/RayTrace.cu)

rtlib_compile_ptx(Test22 ${TEST_TEST22_CUDA_PATH}/RayTrace.cu  ${TEST_TEST22_CUDA_PATH} )
rtlib_compile_ptx(Test22 ${TEST_TEST22_CUDA_PATH}/RayDebug.cu  ${TEST_TEST22_CUDA_PATH} )

target_include_directories(Test22 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(Test22 RTLib)