#ifndef RT_PIPELINE_H
#define RT_PIPELINE_H
#include "RTTracer.h"
namespace test
{
    class RTPipeline
    {
    public:
        virtual void Initialize() = 0;
        virtual void CleanUp()    = 0;
        virtual ~RTPipeline(){}
    };
}
#endif