#include "../include/RTLib/Core.h"
#include <optix_function_table_definition.h>
namespace rtlib{
    namespace internal{
        template<typename OptixType>
        struct OptixDestroy;
        template<>
        struct OptixDestroy<OptixDeviceContext_t>{
        public:
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixDeviceContext_t* ptr)const noexcept{
                //std::cout << "destructing Device Context!\n";
                try{
                    RTLIB_OPTIX_CHECK(optixDeviceContextDestroy(ptr));
                }catch(rtlib::OptixException& error){
                    std::cerr << "internal::OptixDestroy<OptixDeviceContext_t> Catch Error!\n";
                    std::cerr << error.what() << std::endl;
                }
            }
            ~OptixDestroy()noexcept{}
        };
        template<>
        struct OptixDestroy<OptixModule_t>{
        public:
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixModule_t* ptr)const noexcept{
                //std::cout << "destructing Module!\n";
                try{
                    RTLIB_OPTIX_CHECK(optixModuleDestroy(ptr));
                }catch(rtlib::OptixException& error){
                    std::cerr << "internal::OptixDestroy<OptixModule_t> Catch Error!\n";
                    std::cerr << error.what() << std::endl;
                }
            }
            ~OptixDestroy()noexcept{}
        };
        template<>
        struct OptixDestroy<OptixProgramGroup_t>{
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixProgramGroup_t* ptr)const noexcept{
                //std::cout << "destructing ProgramGroup!\n";
                try{
                    RTLIB_OPTIX_CHECK(optixProgramGroupDestroy(ptr));
                }catch(rtlib::OptixException& error){
                    std::cerr << "internal::OptixDestroy<OptixProgramGroup_t> Catch Error!\n";
                    std::cerr << error.what() << std::endl;
                }
            }
            ~OptixDestroy()noexcept{}
        };
        template<>
        struct OptixDestroy<OptixPipeline_t>{
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixPipeline_t* ptr)const noexcept{
                //std::cout << "destructing Pipeline!\n";
                try {
                    RTLIB_OPTIX_CHECK(optixPipelineDestroy(ptr));
                }catch(rtlib::OptixException& error){
                    std::cerr << "internal::OptixDestroy<OptixPipeline_t> Catch Error!\n";
                    std::cerr << error.what() << std::endl;
                }
            }
            ~OptixDestroy()noexcept{}
        };
        template<>
        struct OptixDestroy<OptixDenoiser_t>{
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixDenoiser_t* ptr)const noexcept{
                //std::cout << "destructing Denosier!\n";
                try{
                    RTLIB_OPTIX_CHECK(optixDenoiserDestroy(ptr));
                }catch(rtlib::OptixException& error){
                    std::cerr << "internal::OptixDestroy<OptixDenoiser_t> Catch Error!\n";
                    std::cerr << error.what() << std::endl;
                }
            }
            ~OptixDestroy()noexcept{}
        };
    }
    static void debugLogCallback( unsigned int level, const char* tag, const char* message, void* cbdata ){
        std::cerr << "[" << level << "][" << tag << "]: " << message << "\n";
    }
    //Context
    class OPXContext::Impl{
    public:
        using HandleType     = std::unique_ptr<OptixDeviceContext_t,internal::OptixDestroy<OptixDeviceContext_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        explicit Impl(const Desc& desc){
            RTLIB_CU_CHECK(cuDeviceGet( &m_Device,desc.deviceIdx));
            RTLIB_CU_CHECK(cuCtxCreate(&m_Context,desc.cuCtxFlags,m_Device));
            m_Options.logCallbackData     = nullptr;
            m_Options.logCallbackFunction = debugLogCallback;
            m_Options.validationMode      = desc.validationMode;
            m_Options.logCallbackLevel    = desc.logCallbackLevel;
            OptixDeviceContext ptr = nullptr;
            RTLIB_OPTIX_CHECK(optixDeviceContextCreate(m_Context,&m_Options,&ptr));
            m_Handle.reset(ptr);
        }
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixDeviceContext,Handle,m_Handle.get());
        RTLIB_DECLARE_GET_BY_VALUE(Impl,CUdevice,Device,m_Device);
        RTLIB_DECLARE_GET_BY_VALUE(Impl,CUcontext,Context,m_Context);
        RTLIB_DECLARE_GET_BY_REFERENCE(Impl,OptixDeviceContextOptions,Options,m_Options);
        ~Impl()noexcept{
            try{
                m_Handle = nullptr;
                if(m_Context){
                    RTLIB_CU_CHECK(cuCtxDestroy(m_Context));
                    m_Context = nullptr;
                }
                m_Device  = 0;
                m_Options = {};
            }catch(std::runtime_error& error){
                std::cerr << "OPXContext::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        using Options = OptixDeviceContextOptions;
        HandleType m_Handle  = nullptr;
        CUdevice   m_Device  = 0;
        CUcontext  m_Context = nullptr;
        Options    m_Options = {};
    };
    OPXContext::OPXContext()noexcept:m_Impl{ std::make_shared<OPXContext::Impl>() } {}
    OPXContext::OPXContext(const OPXContext::Desc& desc)noexcept:m_Impl{std::make_shared<OPXContext::Impl>(desc)}{}
    OPXContext::operator bool()const noexcept{ return (bool)m_Impl; }
    auto OPXContext::getHandle()const->OptixDeviceContext {
        return m_Impl->getHandle();
    }
    //Pipeline
    class OPXPipeline::Impl{
    public:
        using ContextRef    = std::shared_ptr<   OPXContext::Impl>;
        using RaygenPGRef   = std::shared_ptr<  OPXRaygenPG::Impl>;
        using MissPGRef     = std::shared_ptr<    OPXMissPG::Impl>;
        using HitgroupPGRef = std::shared_ptr<OPXHitgroupPG::Impl>;
        using HandleType    = std::unique_ptr<OptixPipeline_t,internal::OptixDestroy<OptixPipeline_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixPipeline,Handle,m_Handle.get());
        RTLIB_DECLARE_GET_BY_REFERENCE(Impl,OPXPipeline::CompileOptions,CompileOptions,m_CompileOptions);
        auto getContext()const noexcept->std::shared_ptr<OPXContext::Impl>{
            return m_Context;
        }
        void setContext(const std::shared_ptr<OPXContext::Impl>& context)noexcept{
            m_Context = context;
        }
        void setCompileOptions(const OPXPipeline::CompileOptions& compileOptions)noexcept{
            m_CompileOptions = compileOptions;
        }
        void setProgramGroup(const RaygenPGRef& raygenPG)noexcept{
            m_RaygenPGs.insert(raygenPG);
        }
        void setProgramGroup(const MissPGRef& missPG)noexcept{
            m_MissPGs.insert(missPG);
        }
        void setProgramGroup(const HitgroupPGRef& hitgroupPG)noexcept{
            m_HitgroupPGs.insert(hitgroupPG);
        }
        void link(const OPXPipeline::LinkOptions& linkOptions);
        ~Impl()noexcept{
            try{
                m_RaygenPGs.clear();
                m_MissPGs.clear();
                m_HitgroupPGs.clear();
                m_Handle  = nullptr;
                m_Context.reset();
                m_CompileOptions = {};
            }catch(std::runtime_error& error){
                std::cerr << "OPXContext::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType                         m_Handle         = nullptr;
        ContextRef                         m_Context        = {};
        std::unordered_set<  RaygenPGRef>  m_RaygenPGs      = {};
        std::unordered_set<    MissPGRef>  m_MissPGs        = {};
        std::unordered_set<HitgroupPGRef>  m_HitgroupPGs    = {};
        OPXPipeline::CompileOptions        m_CompileOptions = {};
        OPXPipeline::LinkOptions           m_LinkOptions    = {};
    };
    OPXPipeline::OPXPipeline()noexcept :m_Impl{ std::make_shared<OPXPipeline::Impl>() } {}
    OPXPipeline OPXContext::createPipeline(const OptixPipelineCompileOptions& compileOptions){
        auto pipeline = OPXPipeline();
        pipeline.m_Impl->setContext(m_Impl);
        pipeline.m_Impl->setCompileOptions(compileOptions);
        return pipeline;
    }
    auto OPXContext::buildAccel(const OptixAccelBuildOptions& accelBuildOptions, const OptixBuildInput& buildInput) const -> AccelBuildOutput
    {
        OptixAccelBufferSizes bufferSizes = {};
        RTLIB_OPTIX_CHECK(optixAccelComputeMemoryUsage(this->getHandle(), &accelBuildOptions, &buildInput, 1, &bufferSizes));
        if ((accelBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
            AccelBuildOutput accel = {};
            accel.outputBuffer     = CUDABuffer<void>(bufferSizes.outputSizeInBytes);
            auto  tempBuffer       = CUDABuffer<void>(bufferSizes.tempSizeInBytes);
            RTLIB_OPTIX_CHECK(optixAccelBuild(this->getHandle(), 0, &accelBuildOptions,
                &buildInput, 1,
                reinterpret_cast<CUdeviceptr>(tempBuffer.getDevicePtr()), tempBuffer.getSizeInBytes(),
                reinterpret_cast<CUdeviceptr>(accel.outputBuffer.getDevicePtr()), accel.outputBuffer.getSizeInBytes(),
                &accel.traversableHandle, nullptr, 0));
            return accel;
        }
        else {
            AccelBuildOutput tempAccel     = {};
            auto tempBuffer                = CUDABuffer<void>(bufferSizes.tempSizeInBytes);
            tempAccel.outputBuffer         = CUDABuffer<void>(bufferSizes.outputSizeInBytes);
            auto compactSizeBuffer         = CUDABuffer<size_t>(0);
            OptixAccelEmitDesc compactDesc = {};
            compactDesc.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            compactDesc.result             = reinterpret_cast<CUdeviceptr>(compactSizeBuffer.getDevicePtr());
            RTLIB_OPTIX_CHECK(optixAccelBuild(this->getHandle(), 0, &accelBuildOptions,
                &buildInput, 1,
                reinterpret_cast<CUdeviceptr>(tempBuffer.getDevicePtr())  ,tempBuffer.getSizeInBytes(),
                reinterpret_cast<CUdeviceptr>(tempAccel.outputBuffer.getDevicePtr()), tempAccel.outputBuffer.getSizeInBytes(),
                &tempAccel.traversableHandle, &compactDesc, 1));
            size_t compactSize             = {};
            compactSizeBuffer.download(&compactSize, 1);
            std::cout << compactSize << "vs" << bufferSizes.outputSizeInBytes << std::endl;
            if (compactSize < bufferSizes.outputSizeInBytes) {
                AccelBuildOutput accel = {};
                accel.outputBuffer = CUDABuffer<void>(compactSize);
                RTLIB_OPTIX_CHECK(optixAccelCompact(this->getHandle(),0,tempAccel.traversableHandle, 
                    reinterpret_cast<CUdeviceptr>(accel.outputBuffer.getDevicePtr()),
                    accel.outputBuffer.getSizeInBytes(),&accel.traversableHandle));
                return accel;
            }
            else {
                return tempAccel;
            }
        }
        
    }
    auto OPXContext::buildAccel(const OptixAccelBuildOptions& accelBuildOptions, const std::vector<OptixBuildInput>& buildInputs) const -> AccelBuildOutput
    {
        OptixAccelBufferSizes bufferSizes = {};
        RTLIB_OPTIX_CHECK(optixAccelComputeMemoryUsage(this->getHandle(), &accelBuildOptions, buildInputs.data(), buildInputs.size(), &bufferSizes));
        if ((accelBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
            AccelBuildOutput accel = {};
            accel.outputBuffer = CUDABuffer<void>(bufferSizes.outputSizeInBytes);
            auto  tempBuffer = CUDABuffer<void>(bufferSizes.tempSizeInBytes);
            RTLIB_OPTIX_CHECK(optixAccelBuild(this->getHandle(), 0, &accelBuildOptions,
                buildInputs.data(), buildInputs.size(),
                reinterpret_cast<CUdeviceptr>(tempBuffer.getDevicePtr()), tempBuffer.getSizeInBytes(),
                reinterpret_cast<CUdeviceptr>(accel.outputBuffer.getDevicePtr()), accel.outputBuffer.getSizeInBytes(),
                &accel.traversableHandle, nullptr, 0));
            return accel;
        }
        else {
            AccelBuildOutput tempAccel = {};
            auto tempBuffer = CUDABuffer<void>(bufferSizes.tempSizeInBytes);
            tempAccel.outputBuffer = CUDABuffer<void>(bufferSizes.outputSizeInBytes);
            auto compactSizeBuffer = CUDABuffer<size_t>(0);
            OptixAccelEmitDesc compactDesc = {};
            compactDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            compactDesc.result = reinterpret_cast<CUdeviceptr>(compactSizeBuffer.getDevicePtr());
            RTLIB_OPTIX_CHECK(optixAccelBuild(this->getHandle(), 0, &accelBuildOptions,
                buildInputs.data(), buildInputs.size(),
                reinterpret_cast<CUdeviceptr>(tempBuffer.getDevicePtr()), tempBuffer.getSizeInBytes(),
                reinterpret_cast<CUdeviceptr>(tempAccel.outputBuffer.getDevicePtr()), tempAccel.outputBuffer.getSizeInBytes(),
                &tempAccel.traversableHandle, &compactDesc, 1));
            size_t compactSize = {};
            compactSizeBuffer.download(&compactSize, 1);
            std::cout << compactSize << "vs" << bufferSizes.outputSizeInBytes << std::endl;
            if (compactSize < bufferSizes.outputSizeInBytes) {
                AccelBuildOutput accel = {};
                accel.outputBuffer = CUDABuffer<void>(compactSize);
                RTLIB_OPTIX_CHECK(optixAccelCompact(this->getHandle(), 0, tempAccel.traversableHandle,
                    reinterpret_cast<CUdeviceptr>(accel.outputBuffer.getDevicePtr()),
                    accel.outputBuffer.getSizeInBytes(), &accel.traversableHandle));
                return accel;
            }
            else {
                return tempAccel;
            }
        }
    }
    OPXPipeline::OPXPipeline(OPXPipeline&& pipeline)noexcept{
        this->m_Impl = std::move(pipeline.m_Impl);
    }
    OPXPipeline& OPXPipeline::operator=(OPXPipeline&& pipeline)noexcept{
        if(this!=&pipeline){
            this->m_Impl = std::move(pipeline.m_Impl);
        }
        return *this;
    }
    OPXPipeline::operator bool()const noexcept{
        return (bool)m_Impl;
    }
    auto OPXPipeline::getHandle() const -> OptixPipeline
    {
        return m_Impl->getHandle();
    }
    //Module
    class OPXModule::Impl{
    public:
        using ContextRef     = std::shared_ptr<OPXContext::Impl>;
        using HandleType     = std::unique_ptr<OptixModule_t,internal::OptixDestroy<OptixModule_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        explicit operator bool()const noexcept{
            return (bool)m_Handle;
        }
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixModule,Handle,m_Handle.get());
        auto getContext()const noexcept->std::shared_ptr<OPXContext::Impl>{
            return m_Context;
        }
        void setContext(const std::shared_ptr<OPXContext::Impl>& context)noexcept{
            m_Context = context;
        }
        void setCompileOptions(const OPXModule::CompileOptions& compileOptions)noexcept{
            m_CompileOptions = compileOptions;
        }
        void buildFromPTX(const std::string& ptx, const OptixPipelineCompileOptions& pipelineCompileOptions){
            char log [1024];
            size_t sizeoflog = sizeof(log);
            OptixModule ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixModuleCreateFromPTX(m_Context->getHandle(),&m_CompileOptions,&pipelineCompileOptions,
            ptx.data(),ptx.size(),log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle  = nullptr;
                m_Context = nullptr;
                m_CompileOptions = {};
            }catch(std::runtime_error& error){
                std::cerr << "OPXModule::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType m_Handle  = nullptr;
        ContextRef m_Context = nullptr;
        OPXModule::CompileOptions m_CompileOptions = {};
    };
    OPXModule::OPXModule()noexcept:m_Impl{std::make_shared<OPXModule::Impl>()}{}
    OPXModule OPXPipeline::createModule(const std::string& ptx,const OptixModuleCompileOptions& compileOptions){
        OPXModule module = {};
        module.m_Impl->setContext(m_Impl->getContext());
        module.m_Impl->setCompileOptions(compileOptions);
        module.m_Impl->buildFromPTX(ptx,m_Impl->getCompileOptions());
        return module;
    }
    OPXModule::operator bool()const noexcept{
        return (bool)m_Impl;
    }
    class OPXRaygenPG::Impl{
    public:
        using ModuleRef      = std::shared_ptr< OPXModule::Impl>;
        using ContextRef     = std::shared_ptr<OPXContext::Impl>;
        using HandleType     = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixProgramGroup,Handle,m_Handle.get());
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ContextRef,Context,m_Context);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,Module,m_Module);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,std::string,EntryName,m_EntryName);
        void build(){
            OptixProgramGroupDesc    desc    = {};
            OptixProgramGroupOptions options = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            desc.raygen.module = this->getModule()->getHandle();
            desc.raygen.entryFunctionName = this->getEntryName().c_str();
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle      = nullptr;
                m_Module      = nullptr;
                m_EntryName   = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXRaygenPG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle    = nullptr;
        ContextRef  m_Context   = nullptr;
        ModuleRef   m_Module    = nullptr;
        std::string m_EntryName = {};
    };
    OPXRaygenPG::OPXRaygenPG()noexcept:m_Impl{std::make_shared<OPXRaygenPG::Impl>()}{}
    OPXRaygenPG::operator bool()const noexcept{
        return (bool)m_Impl;
    }
    auto OPXRaygenPG::getHandle()const -> OptixProgramGroup{
        return this->m_Impl->getHandle();
    }
    class OPXMissPG::Impl{
    public:
        using ModuleRef      = std::shared_ptr< OPXModule::Impl>;
        using ContextRef     = std::shared_ptr<OPXContext::Impl>;
        using HandleType     = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixProgramGroup,Handle,m_Handle.get());
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ContextRef,Context,m_Context);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,Module,m_Module);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,std::string,EntryName,m_EntryName);
        void build(){
            OptixProgramGroupDesc    desc    = {};
            OptixProgramGroupOptions options = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc.miss.module = this->getModule()->getHandle();
            desc.miss.entryFunctionName = this->getEntryName().c_str();
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle      = nullptr;
                m_Module      = nullptr;
                m_EntryName   = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXMissPG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle    = nullptr;
        ContextRef  m_Context   = nullptr;    
        ModuleRef   m_Module    = nullptr;
        std::string m_EntryName = {};
    };
    OPXMissPG::OPXMissPG()noexcept:m_Impl{std::make_shared<OPXMissPG::Impl>()}{}
    OPXMissPG::operator bool()const noexcept{
        return (bool)m_Impl;
    }
    auto OPXMissPG::getHandle()const -> OptixProgramGroup{
        return this->m_Impl->getHandle();
    }
    class OPXHitgroupPG::Impl{
    public:
        using ModuleRef       = std::shared_ptr< OPXModule::Impl>;
        using ContextRef      = std::shared_ptr<OPXContext::Impl>;
        using HandleType      = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
    public:
        Impl(){}
        Impl(const Impl& )            = delete;
        Impl(Impl&& )                 = delete;
        Impl& operator=(const Impl& ) = delete;
        Impl& operator=(Impl&& )      = delete;
        RTLIB_DECLARE_GET_BY_VALUE(Impl,OptixProgramGroup,Handle,m_Handle.get());
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ContextRef,Context,m_Context);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,CHModule,m_CHModule);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Impl,const char*,CHEntryName,m_CHEntryName);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,AHModule,m_AHModule);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Impl,const char*,AHEntryName,m_AHEntryName);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,ISModule,m_ISModule);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Impl,const char*,ISEntryName,m_ISEntryName);
        void build(){
            OptixProgramGroupDesc    desc    = {};
            OptixProgramGroupOptions options = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleCH            = this->getCHModule()->getHandle();
            desc.hitgroup.moduleAH            = this->getAHModule()->getHandle();
            desc.hitgroup.moduleIS            = this->getISModule()->getHandle();
            desc.hitgroup.entryFunctionNameCH = this->getCHEntryName();
            desc.hitgroup.entryFunctionNameAH = this->getAHEntryName();
            desc.hitgroup.entryFunctionNameIS = this->getISEntryName();
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle      = nullptr;
                m_CHModule    = nullptr;
                m_CHEntryName = {};
                m_AHModule    = nullptr;
                m_AHEntryName = {};
                m_ISModule    = nullptr;
                m_ISEntryName = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXHitgroupPG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle      = nullptr;
        ContextRef  m_Context     = nullptr;
        ModuleRef   m_CHModule    = nullptr;
        const char* m_CHEntryName = nullptr;
        ModuleRef   m_AHModule    = nullptr;
        const char* m_AHEntryName = nullptr;
        ModuleRef   m_ISModule    = nullptr;
        const char* m_ISEntryName = nullptr;
    };
    OPXHitgroupPG::OPXHitgroupPG()noexcept:m_Impl{std::make_shared<OPXHitgroupPG::Impl>()}{}
    OPXHitgroupPG::operator bool()const noexcept{
        return (bool)m_Impl;
    }
    auto OPXHitgroupPG::getHandle()const -> OptixProgramGroup{
        return this->m_Impl->getHandle();
    }
    OPXRaygenPG   OPXPipeline::createRaygenPG(  const OPXProgramDesc& raygenDesc){
        OPXRaygenPG raygenPG;
        //TODO���`�`�F�b�N������
        raygenPG.m_Impl->setContext(m_Impl->getContext());
        raygenPG.m_Impl->setModule(raygenDesc.module.m_Impl);
        raygenPG.m_Impl->setEntryName(raygenDesc.entryName);
        raygenPG.m_Impl->build();
        m_Impl->setProgramGroup(raygenPG.m_Impl);
        return raygenPG;
    }
    OPXMissPG     OPXPipeline::createMissPG(    const OPXProgramDesc& missDesc){
        OPXMissPG missPG;
        //TODO���`�`�F�b�N������
        missPG.m_Impl->setContext(m_Impl->getContext());
        missPG.m_Impl->setModule(missDesc.module.m_Impl);
        missPG.m_Impl->setEntryName(missDesc.entryName);
        missPG.m_Impl->build();
        m_Impl->setProgramGroup(missPG.m_Impl);
        return missPG;
    }
    OPXHitgroupPG OPXPipeline::createHitgroupPG(const OPXProgramDesc& chDesc,const OPXProgramDesc& ahDesc,const OPXProgramDesc& isDesc){
        OPXHitgroupPG hitgroupPG;
        //TODO���`�`�F�b�N������
        hitgroupPG.m_Impl->setContext(m_Impl->getContext());
        if (chDesc.module.m_Impl) {
            hitgroupPG.m_Impl->setCHModule(chDesc.module.m_Impl);
            hitgroupPG.m_Impl->setCHEntryName(chDesc.entryName);
        }
        if (ahDesc.module.m_Impl) {
            hitgroupPG.m_Impl->setAHModule(ahDesc.module.m_Impl);
            hitgroupPG.m_Impl->setAHEntryName(ahDesc.entryName);
        }
        if (isDesc.module.m_Impl) {
            hitgroupPG.m_Impl->setISModule(isDesc.module.m_Impl);
            hitgroupPG.m_Impl->setISEntryName(isDesc.entryName);
        }
        hitgroupPG.m_Impl->build();
        m_Impl->setProgramGroup(hitgroupPG.m_Impl);
        return hitgroupPG;
    }
    void OPXPipeline::Impl::link(const OPXPipeline::LinkOptions& linkOptions) {
        std::vector<OptixProgramGroup> programGroups = {};
        for (auto rg : m_RaygenPGs) {
            auto pg = rg->getHandle();
            if (pg) {
                programGroups.push_back(pg);
            }
        }
        for (auto ms : m_MissPGs) {
            auto pg = ms->getHandle();
            if (pg) {
                programGroups.push_back(pg);
            }
        }
        for (auto hg : m_HitgroupPGs) {
            auto pg = hg->getHandle();
            if (pg) {
                programGroups.push_back(pg);
            }
        }
        char log[1024]; size_t sizeoflog = sizeof(log);
        OptixPipeline ptr = nullptr;
        RTLIB_OPTIX_CHECK2(optixPipelineCreate(m_Context->getHandle(), &m_CompileOptions, &linkOptions, programGroups.data(), programGroups.size(), log, &sizeoflog, &ptr), log);
        m_LinkOptions = linkOptions;
        m_Handle.reset(ptr);
    }
    void OPXPipeline::link(const OPXPipeline::LinkOptions& linkOptions) {
        m_Impl->link(linkOptions);
    }
    //
    
}