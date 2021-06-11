#ifndef RTLIB_CONFIG_H
#define RTLIB_CONFIG_H
#define RTLIB_OPTIX_INCLUDE_DIR "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.2.0\\include"
#define RTLIB_CUDA_INCLUDE_DIRS  \
  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include",
#define RTLIB_INCLUDE_DIR "D:\\Users\\shums\\Documents\\CMake\\OptiX7Sample\\RTLib\\include"
#define RTLIB_NVRTC_OPTIONS  \
  "-arch", \
  "compute_60", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64", \
  "--std=c++11",
#endif
