set(TEST_TEST24_CALLABLE_OPX_CUDA_PATH "${CMAKE_CURRENT_BINARY_DIR}/cuda")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Test24CallableOPXConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Test24CallableOPXConfig.h
)

add_library(Test24_Tracer_CallableOPX STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CallableOPXTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Payload.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/HitRecord.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/MaterialParameters.h

    ${CMAKE_CURRENT_BINARY_DIR}/Test24CallableOPXConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CallableOPXTracer.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu
)

add_custom_command(TARGET Test24_Tracer_CallableOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.h             ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/RayTrace.h            )
add_custom_command(TARGET Test24_Tracer_CallableOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTraceUtils.h        ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/RayTraceUtils.h       )
add_custom_command(TARGET Test24_Tracer_CallableOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Payload.h              ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/Payload.h             )
add_custom_command(TARGET Test24_Tracer_CallableOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/HitRecord.h            ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/HitRecord.h           )
add_custom_command(TARGET Test24_Tracer_CallableOPX PRE_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/MaterialParameters.h   ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/MaterialParameters.h  )
add_custom_command(TARGET Test24_Tracer_CallableOPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/RayTrace.cu            ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/RayTrace.cu           )
add_custom_command(TARGET Test24_Tracer_CallableOPX POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Callable.cu            ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH}/Callable.cu           )

target_include_directories(Test24_Tracer_CallableOPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include   ${CMAKE_CURRENT_BINARY_DIR}  PRIVATE ${TEST_TEST24_CALLABLE_OPX_CUDA_PATH})
target_link_libraries(Test24_Tracer_CallableOPX PUBLIC TestLib Test24_Share)