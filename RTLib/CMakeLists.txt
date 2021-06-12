set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OptiX 7.0.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(thirdparty)
add_library(RTLib STATIC 
${CMAKE_CURRENT_SOURCE_DIR}/src/Core.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/Exceptions.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Camera.cpp)
##config setup
set(RTLIB_OPTIX_INCLUDE_DIR ${OptiX_INCLUDE_DIR})
set(RTLIB_CUDA_INCLUDE_DIRS )
foreach(dir ${CUDAToolkit_INCLUDE_DIRS})
    set(RTLIB_CUDA_INCLUDE_DIRS "${RTLIB_CUDA_INCLUDE_DIRS} \\\n  \"${dir}\",")
endforeach()
set(RTLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WIN32)
    string(REPLACE "/" "\\\\" RTLIB_OPTIX_INCLUDE_DIR ${RTLIB_OPTIX_INCLUDE_DIR})
    string(REPLACE "/" "\\\\" RTLIB_CUDA_INCLUDE_DIRS ${RTLIB_CUDA_INCLUDE_DIRS})
    string(REPLACE "/" "\\\\" RTLIB_INCLUDE_DIR       ${RTLIB_INCLUDE_DIR})
endif()
set(CUDA_NVRTC_FLAGS -arch compute_60 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 --std=c++11)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(RTLIB_NVRTC_OPTIONS "${RTLIB_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/RTLib/Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/RTLib/Config.h @ONLY)
##include directories
target_include_directories(RTLib PUBLIC ${RTLIB_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
##link libraries
target_link_libraries(RTLib PUBLIC CUDA::cudart CUDA::nvrtc glad glm::glm stb glfw tinyobjloader OptiX::OptiX)
