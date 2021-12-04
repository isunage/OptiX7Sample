#ifndef RT_TRACER_H
#define RT_TRACER_H
#include <RTLib/core/CUDA.h>
namespace test{
    struct RTTraceConfig
    {
        int      width;
        int      height;
        int      depth;
        bool     isSync;
        CUstream stream;
        void*    pUserData;
    };
    class RTTracer{
    public:
        virtual void Initialize()                        = 0;
        virtual void Launch(const RTTraceConfig& config) = 0;
        virtual void CleanUp()                           = 0;
        virtual void Update()                            = 0;
        virtual bool ShouldLock()const noexcept { return false; }
        virtual ~RTTracer(){}
    };
}
#endif