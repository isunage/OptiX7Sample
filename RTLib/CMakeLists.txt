set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OptiX 7.0.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(thirdparty)
##config setup
set(RTLIB_OPTIX_INCLUDE_DIR ${OptiX_INCLUDE_DIR})
set(RTLIB_CUDA_INCLUDE_DIRS )
set(RTLIB_CUDA_SM_VERSION 75)
foreach(dir ${CUDAToolkit_INCLUDE_DIRS})
    set(RTLIB_CUDA_INCLUDE_DIRS "${RTLIB_CUDA_INCLUDE_DIRS} \\\n  \"${dir}\",")
endforeach()
set(RTLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WIN32)
    string(REPLACE "/" "\\\\" RTLIB_OPTIX_INCLUDE_DIR ${RTLIB_OPTIX_INCLUDE_DIR})
    string(REPLACE "/" "\\\\" RTLIB_CUDA_INCLUDE_DIRS ${RTLIB_CUDA_INCLUDE_DIRS})
    string(REPLACE "/" "\\\\" RTLIB_INCLUDE_DIR       ${RTLIB_INCLUDE_DIR})
endif()
set(CUDA_NVRTC_OPTIONS -arch compute_${RTLIB_CUDA_SM_VERSION} -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 --std=c++11)
foreach(option ${CUDA_NVRTC_OPTIONS})
  set(RTLIB_NVRTC_OPTIONS "${RTLIB_NVRTC_OPTIONS} \\\n  \"${option}\",")
endforeach()
set(RTLIB_NVCC_OPTIONS -arch compute_${RTLIB_CUDA_SM_VERSION} -use_fast_math -lineinfo -rdc true -m64 --std c++17)
##configure
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/RTLib/Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/RTLib/Config.h @ONLY)
##add library
add_library(RTLib STATIC 
${CMAKE_CURRENT_SOURCE_DIR}/src/Optix.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/Exceptions.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Camera.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/Mesh.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/TraversalHandle.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/RectRenderer.cpp
)
##include directories
target_include_directories(RTLib PUBLIC ${RTLIB_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
##link libraries
target_link_libraries(RTLib PUBLIC CUDA::cudart CUDA::nvrtc glad glm::glm stb glfw tinyobjloader OptiX::OptiX imgui imgui-glfw range-v3)

set(RTLIB_OPTIX_INCLUDE_DIR ${RTLIB_OPTIX_INCLUDE_DIR} PARENT_SCOPE)
set(RTLIB_INCLUDE_DIR       ${RTLIB_INCLUDE_DIR}       PARENT_SCOPE)
set(RTLIB_NVCC_OPTIONS      ${RTLIB_NVCC_OPTIONS}      PARENT_SCOPE)
function(rtlib_compile_ptx target source include_dirs)
  set(rtlib_compile_options "-ptx")
  set(RTLIB_OPTIX_INCLUDE_DIR ${RTLIB_OPTIX_INCLUDE_DIR} PARENT_SCOPE)
  set(RTLIB_INCLUDE_DIR       ${RTLIB_INCLUDE_DIR}       PARENT_SCOPE)
  set(RTLIB_NVCC_OPTIONS      ${RTLIB_NVCC_OPTIONS}      PARENT_SCOPE)
  set(rtlib_compile_options "${rtlib_compile_options} -I\"${RTLIB_OPTIX_INCLUDE_DIR}\"")
  set(rtlib_compile_options "${rtlib_compile_options} -I\"${RTLIB_INCLUDE_DIR}\"")
  foreach(include_dir ${include_dirs})
    set(rtlib_compile_options "${rtlib_compile_options} -I\"${include_dir}\"")
  endforeach(include_dir )
  foreach(option ${RTLIB_NVCC_OPTIONS})
    set(rtlib_compile_options "${rtlib_compile_options} ${option}")
  endforeach(option )
  get_filename_component(source_name_we ${source} NAME_WE)
  add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${source} ${rtlib_compile_options} -o ${CMAKE_CURRENT_BINARY_DIR}/cuda/${source_name_we}.ptx)
endfunction()