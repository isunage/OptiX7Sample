#ifndef RTLIB_CORE_H
#define RTLIB_CORE_H
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <array>
#include <string>
#include <utility>
#include <memory>
#include <iterator>
#include "Exceptions.h"
#include "GL.h"
#include "CUDA.h"
#include "CUDA_GL.h"
//HOST CLASS AND FUNCTION(OpenGL,CUDA,Optix)
namespace rtlib{
    //SBT
    template<typename T>
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT)  SBTRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T      data;
    };
    //SceneGraph
    //IAS(TL)->*Instace->GAS->BI
    //       ->*Instace->GAS->BI
    //       -> Instace->IAS->*Instance->GAS->BI
    //                      ->*Instance->GAS->BI
    //class
    //Context->Pipeline->Module
    //       ->  RaygenProgramGroup(RG)
    //       ->    MissProgramGroup(MS)
    //       ->HitgroupProgramGroup(CH,AH,IS)
    class  OPXContext;
    class  OPXPipeline;
    class  OPXModule;
    class  OPXRaygenPG;
    class  OPXMissPG;
    class  OPXHitgroupPG;
    struct OPXProgramDesc;
    //Copy Move 禁止
    class OPXContext{
    public:
        friend class OPXPipeline;
        friend class OPXModule;
        friend class OPXRaygenPG;
        friend class OPXMissPG;
        friend class OPXHitgroupPG;
        struct Desc{
            using ValidationMode = OptixDeviceContextValidationMode;
            int            deviceIdx;
            unsigned int   cuCtxFlags;
            ValidationMode validationMode;
            int            logCallbackLevel;
        };
        struct AccelBuildOutput {
            CUDABuffer<void>       outputBuffer;
            OptixTraversableHandle traversableHandle;
            AccelBuildOutput() :outputBuffer{}, traversableHandle{ 0 } {}
            AccelBuildOutput(AccelBuildOutput&& accel) noexcept{
                this->outputBuffer = std::move(accel.outputBuffer);
                traversableHandle  = std::move(accel.traversableHandle);
            }
            AccelBuildOutput& operator=(AccelBuildOutput&& accel)noexcept {
                if (this != &accel) {
                    this->outputBuffer = std::move(accel.outputBuffer);
                    traversableHandle  = std::move(accel.traversableHandle);
                }
                return *this;
            }
        };
    public:
        OPXContext()noexcept;
        OPXContext(const OPXContext&)noexcept               = delete;
        OPXContext(OPXContext&&)noexcept                    = delete;
        OPXContext& operator=(const OPXContext&)noexcept    = delete;
        OPXContext& operator=(OPXContext&&)noexcept         = delete;
        explicit OPXContext(const Desc& desc)noexcept;
        explicit operator bool()const noexcept;
        OPXPipeline      createPipeline(const OptixPipelineCompileOptions&  compileOptions);
        auto buildAccel(const OptixAccelBuildOptions& accelBuildOptions,
                        const OptixBuildInput&        buildInput)const->AccelBuildOutput;
        auto buildAccel(const OptixAccelBuildOptions&       accelBuildOptions,
                        const std::vector<OptixBuildInput>& buildInputs)const->AccelBuildOutput;
        ~OPXContext()noexcept{}
    private:
        auto getHandle()const->OptixDeviceContext;
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
    //Copy禁止
    class OPXPipeline{
    public:
        friend class OPXContext;
        using CompileOptions = OptixPipelineCompileOptions;
        using LinkOptions    = OptixPipelineLinkOptions;
    public:
        OPXPipeline()noexcept;
        OPXPipeline(const OPXPipeline&)noexcept            = delete;
        OPXPipeline(OPXPipeline&&)noexcept;
        OPXPipeline& operator=(const OPXPipeline&)noexcept = delete;
        OPXPipeline& operator=(OPXPipeline&&)noexcept;
        explicit operator bool()const noexcept;
        OPXModule createModule(const std::string& ptx, const OptixModuleCompileOptions& compileOptions);
        //ProgramGroup
        OPXRaygenPG   createRaygenPG(  const OPXProgramDesc& raygenDesc);
        OPXMissPG     createMissPG(    const OPXProgramDesc& missDesc);
        OPXHitgroupPG createHitgroupPG(const OPXProgramDesc& chDesc,const OPXProgramDesc& ahDesc,const OPXProgramDesc& isDesc);
        //link
        void link(const LinkOptions& linkOptions);
        template<typename Params>
        void launch(CUstream stream,Params* d_Params,const OptixShaderBindingTable& sbt, unsigned int width, unsigned int height, unsigned int depth) {
            RTLIB_OPTIX_CHECK(optixLaunch(this->getHandle(),stream,reinterpret_cast<CUdeviceptr>(d_Params),sizeof(Params),&sbt,width,height,depth));
        }
        ~OPXPipeline()noexcept{}
    private:
        auto getHandle()const->OptixPipeline;
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
    //Copy Move 可能
    class  OPXModule{
    public:
        friend class OPXContext;
        friend class OPXPipeline;
        friend class OPXRaygenPG;
        friend class OPXMissPG;
        friend class OPXHitgroupPG;
        using CompileOptions = OptixModuleCompileOptions;
    public:
        OPXModule()noexcept;
        explicit operator bool()const noexcept;
        ~OPXModule()noexcept{}
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
    struct OPXProgramDesc{
        OPXModule   module;
        const char* entryName;
    };
    class  OPXRaygenPG{
    public:
        friend class OPXPipeline;
        friend class OPXContext;
    public:
        OPXRaygenPG()noexcept;
        explicit operator bool()const noexcept;
        template<typename T>
        auto getSBTRecord()const -> SBTRecord<T>{
            SBTRecord<T> record;
            RTLIB_OPTIX_CHECK(optixSbtRecordPackHeader(this->getHandle(),record.header));
            return record;
        }
        ~OPXRaygenPG()noexcept{}
    private:
        auto getHandle()const -> OptixProgramGroup;
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
    class  OPXMissPG{
    public:
        friend class OPXPipeline;
        friend class OPXContext;
    public:
        OPXMissPG()noexcept;
        explicit operator bool()const noexcept;
        template<typename T>
        auto getSBTRecord()const -> SBTRecord<T>{
            SBTRecord<T> record;
            RTLIB_OPTIX_CHECK(optixSbtRecordPackHeader(this->getHandle(),record.header));
            return record;
        }
        ~OPXMissPG()noexcept{}
    private:
        auto getHandle()const -> OptixProgramGroup;
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
    class  OPXHitgroupPG{
    public:
        friend class OPXPipeline;
        friend class OPXContext;
    public:
        OPXHitgroupPG()noexcept;
        explicit operator bool()const noexcept;
        template<typename T>
        auto getSBTRecord()const -> SBTRecord<T>{
            SBTRecord<T> record;
            RTLIB_OPTIX_CHECK(optixSbtRecordPackHeader(this->getHandle(),record.header));
            return record;
        }
        ~OPXHitgroupPG()noexcept{}
    private:
        auto getHandle()const -> OptixProgramGroup;
    private:
        class Impl;
        std::shared_ptr<Impl> m_Impl;
    };
}
//DEVICE CLASS AND FUNCTION
#endif