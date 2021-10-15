set(PATH_GUIDING_MAX_DEPTH       1024)
set(RAY_TRACE_S_FILTER           SpatialFilterNearest)
set(RAY_TRACE_D_FILTER           DirectionalFilterNearest)
# Specular面に対してNEEを行うか、Defaultでは行わない（よほどshinnessが高くなければむしろ分散は増加する傾向）
set(RAY_TRACE_NEE_SPECULAR 0)
# floatの指数部23->2^-23,2^-23,2^-23
set(PATH_GUIDING_STREE_MAX_DEPTH 69)
# floatの指数部23
set(PATH_GUIDING_DTREE_MAX_DEPTH 24)
set(PATH_GUIDING_LI_COSINE       1)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuidingConfig.h.in 
    ${CMAKE_CURRENT_BINARY_DIR}/cuda/PathGuidingConfig.h
)

add_library(TestLib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTPathGuidingUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTApplication.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTFrameBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTTracer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/RTAssets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/Assets/ImgAssets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestLib/Assets/ObjAssets.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/PathGuiding.h
    ${CMAKE_CURRENT_BINARY_DIR}/cuda/PathGuidingConfig.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTPathGuidingUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RTAssets.cpp
)

target_include_directories(TestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cuda)
target_link_libraries(TestLib PUBLIC RTLib)