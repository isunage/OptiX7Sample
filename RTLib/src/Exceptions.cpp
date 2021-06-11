#include "../include/RTLib/Exceptions.h"
#include <sstream>
namespace rtlib{
    std::string OptixException::getMessage(OptixResult        result, 
                                           const char*        call, 
                                           const char*        file, 
                                           size_t             line,
                                           const std::string& log) noexcept{
        const char* resultStr = optixGetErrorString(result);
        std::stringstream ss;
        ss << "OptiX Throw " << resultStr << " Exception!\n";
        ss << "Call: " << call << "\n";
        ss << "File: " << file << "\n";
        ss << "Line: " << line << "\n";
        ss << " Log: " << log;
        return ss.str();
    }
    std::string CUDAException::getMessage( cudaError_t        result, 
                                           const char*        call, 
                                           const char*        file, 
                                           size_t             line,
                                           const std::string& log) noexcept{
        const char* resultStr = cudaGetErrorString(result);
        std::stringstream ss;
        ss << "CUDA Runtime Throw " << resultStr << " Exception!\n";
        ss << "Call: " << call << " ";
        ss << "File: " << file << " ";
        ss << "line: " << line << " ";
        ss << " Log: " << log;
        return ss.str();
    }
    std::string CUException::getMessage(   CUresult        result, 
                                           const char*        call, 
                                           const char*        file, 
                                           size_t             line,
                                           const std::string& log) noexcept{
        const char* resultStr = nullptr;
        CUresult isSuccess = cuGetErrorString(result,&resultStr);
        std::stringstream ss;
        if(isSuccess==CUDA_SUCCESS){
            ss << "CUDA Driver Throw " << resultStr << " Exception!\n";
        }else{
            ss << "CUDA Driver Throw Unknown Exception!\n";
        }
        ss << "Call: " << call << " ";
        ss << "File: " << file << " ";
        ss << "line: " << line << " ";
        ss << " Log: " << log;
        return ss.str();
    }
    std::string NVRTCException::getMessage(nvrtcResult        result, 
                                           const char*        call, 
                                           const char*        file, 
                                           size_t             line,
                                           const std::string& log) noexcept{
        const char* resultStr = nvrtcGetErrorString(result);
        std::stringstream ss;
        ss << "NVRTC Throw " << resultStr << " Exception!\n";
        ss << "Call: " << call << " ";
        ss << "File: " << file << " ";
        ss << "line: " << line << " ";
        ss << " Log: " << log;
        return ss.str();
    }
}