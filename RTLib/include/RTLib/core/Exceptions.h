#ifndef RTLIB_EXCEPTIONS_H
#define RTLIB_EXCEPTIONS_H
//host only functuins
#ifdef __cplusplus
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <string>
#include <stdexcept>
#ifndef _DEBUG
#define RTLIB_CUDA_CHECK(CALL) CALL
#define RTLIB_CU_CHECK(CALL) CALL
#define RTLIB_OPTIX_CHECK(CALL) CALL
#define RTLIB_OPTIX_CHECK2(CALL,LOG) CALL
#define RTLIB_NVRTC_CHECK(CALL) CALL
#define RTLIB_NVRTC_CHECK2(CALL,LOG) CALL 
#else
//CUDA RUNTIME ERROR CHECK
#define RTLIB_CUDA_CHECK(CALL) \
do{ \
    cudaError_t result = CALL; \
    if (result != cudaSuccess){ \
        throw rtlib::CUDAException(result,#CALL,__FILE__,__LINE__); \
    } \
}while(0)
//CUDA DRIVER ERROR CHECK
#define RTLIB_CU_CHECK(CALL) \
do{ \
    CUresult result = CALL; \
    if (result != CUDA_SUCCESS){ \
        throw rtlib::CUException(result,#CALL,__FILE__,__LINE__); \
    } \
}while(0)
//OPTIX ERROR CHECK
#define RTLIB_OPTIX_CHECK(CALL) \
do{ \
    OptixResult result = CALL; \
    if (result != OPTIX_SUCCESS){ \
        throw rtlib::OptixException(result,#CALL,__FILE__,__LINE__); \
    } \
}while(0)
#define RTLIB_OPTIX_CHECK2(CALL,LOG) \
do{ \
    OptixResult result = CALL; \
    if (result != OPTIX_SUCCESS){ \
        throw rtlib::OptixException(result,#CALL,__FILE__,__LINE__,LOG); \
    } \
}while(0)
//NVRTC ERROR CHECK
#define RTLIB_NVRTC_CHECK( CALL) \
do{ \
    nvrtcResult result = CALL; \
    if (result != NVRTC_SUCCESS){ \
        throw rtlib::NVRTCException(result,#CALL,__FILE__,__LINE__); \
    } \
}while(0)
#define RTLIB_NVRTC_CHECK2(CALL,LOG) \
do{ \
    nvrtcResult result = CALL; \
    if (result != NVRTC_SUCCESS){ \
        throw rtlib::NVRTCException(result,#CALL,__FILE__,__LINE__,LOG); \
    } \
}while(0)
#endif
//Exception
namespace rtlib{
    class OptixException: public std::runtime_error{
    private:
        OptixResult m_Result;
        const char* m_Call;
        const char* m_File;
        size_t      m_Line;
        std::string m_Log;
    public:
        //Constructor
        OptixException(OptixResult        result, 
                       const char*        call, 
                       const char*        file, 
                       size_t             line,
                       const std::string& log = "")noexcept
                       :m_Result{result},
                        m_Call{call},
                        m_File{file},
                        m_Line{line},
                        m_Log{log},
                        std::runtime_error(getMessage(result,call,file,line,log)){}
        //getMessage
        static std::string getMessage(OptixResult        result, 
                                      const char*        call, 
                                      const char*        file, 
                                      size_t             line,
                                      const std::string& log)noexcept;
        //getter
        inline OptixResult getResult()const noexcept{
            return m_Result;
        }
        inline std::string getCall()const noexcept{
            return m_Call;
        }
        inline const char* getFile()const noexcept{
            return m_File;
        }
        inline size_t      getLine()const noexcept{
            return m_Line;
        }
        inline std::string getLog()const noexcept{
            return m_Log;
        }
        //Destructor
        virtual ~OptixException()noexcept{}
    };
    class    CUException: public std::runtime_error{
    private:
        CUresult    m_Result;
        std::string m_Call;
        const char* m_File;
        size_t      m_Line;
        std::string m_Log;
    public:
        //Constructor
        CUException(   CUresult           result, 
                       const char*        call, 
                       const char*        file, 
                       size_t             line,
                       const std::string& log = "")noexcept
                       :m_Result{result},
                        m_Call{call},
                        m_File{file},
                        m_Line{line},
                        m_Log{log},
                        std::runtime_error(getMessage(result,call,file,line,log)){}
        //getMessage
        static std::string getMessage(CUresult           result, 
                                      const char*        call, 
                                      const char*        file, 
                                      size_t             line,
                                      const std::string& log)noexcept;
        //getter
        inline CUresult getResult()const noexcept{
            return m_Result;
        }
        inline std::string getCall()const noexcept{
            return m_Call;
        }
        inline const char* getFile()const noexcept{
            return m_File;
        }
        inline size_t      getLine()const noexcept{
            return m_Line;
        }
        inline std::string getLog()const noexcept{
            return m_Log;
        }
        //Destructor
        virtual ~CUException()noexcept{}
    };
    class  CUDAException: public std::runtime_error{
    private:
        cudaError_t m_Result;
        std::string m_Call;
        const char* m_File;
        size_t      m_Line;
        std::string m_Log;
    public:
        //Constructor
        CUDAException( cudaError_t        result, 
                       const char*        call, 
                       const char*        file, 
                       size_t             line,
                       const std::string& log = "")noexcept
                       :m_Result{result},
                        m_Call{call},
                        m_File{file},
                        m_Line{line},
                        m_Log{log},
                        std::runtime_error(getMessage(result,call,file,line,log)){}
        //getMessage
        static std::string getMessage(cudaError_t        result, 
                                      const char*        call, 
                                      const char*        file, 
                                      size_t             line,
                                      const std::string& log)noexcept;
        //getter
        inline cudaError_t getResult()const noexcept{
            return m_Result;
        }
        inline std::string getCall()const noexcept{
            return m_Call;
        }
        inline const char* getFile()const noexcept{
            return m_File;
        }
        inline size_t      getLine()const noexcept{
            return m_Line;
        }
        inline std::string getLog()const noexcept{
            return m_Log;
        }
        //Destructor
        virtual ~CUDAException()noexcept{}
    };
    class NVRTCException: public std::runtime_error{
        private:
        nvrtcResult m_Result;
        std::string m_Call;
        const char* m_File;
        size_t      m_Line;
        std::string m_Log;
    public:
        //Constructor
        NVRTCException( nvrtcResult        result, 
                        const char*        call, 
                        const char*        file, 
                        size_t             line,
                        const std::string& log = "")noexcept
                       :m_Result{result},
                        m_Call{call},
                        m_File{file},
                        m_Line{line},
                        m_Log{log},
                        std::runtime_error(getMessage(result,call,file,line,log)){}
        //getMessage
        static std::string getMessage(nvrtcResult        result, 
                                      const char*        call, 
                                      const char*        file, 
                                      size_t             line,
                                      const std::string& log)noexcept;
        //getter
        inline nvrtcResult getResult()const noexcept{
            return m_Result;
        }
        inline std::string getCall()const noexcept{
            return m_Call;
        }
        inline const char* getFile()const noexcept{
            return m_File;
        }
        inline size_t      getLine()const noexcept{
            return m_Line;
        }
        inline std::string getLog()const noexcept{
            return m_Log;
        }
        //Destructor
        virtual ~NVRTCException()noexcept{}
    };
}
#endif
#endif