#include <RTLib/core/Optix.h>
#include <optix_function_table_definition.h>
namespace rtlib{
    namespace internal       {
        template<typename OptixType>
        struct OptixDestroy;
        template<>
        struct OptixDestroy<OptixDeviceContext_t>{
        public:
            constexpr OptixDestroy()noexcept{}
            void operator()(OptixDeviceContext_t* ptr)const noexcept{
                if (!ptr){
                    return;
                }
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
                if (!ptr){
                    return;
                }
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
                if (!ptr){
                    return;
                }
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
                if (!ptr){
                    return;
                }
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
                if (!ptr){
                    return;
                }
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
    class OPXContext::Impl   {
    public:
        using HandleType = std::unique_ptr<OptixDeviceContext_t,internal::OptixDestroy<OptixDeviceContext_t>>;
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
                m_Handle.reset();
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
    //Pipeline
    class OPXPipeline::Impl  {
    public:
        using ContextRef     = std::shared_ptr<    OPXContext::Impl>;
        using RaygenPGRef    = std::shared_ptr<   OPXRaygenPG::Impl>;
        using MissPGRef      = std::shared_ptr<     OPXMissPG::Impl>;
        using HitgroupPGRef  = std::shared_ptr< OPXHitgroupPG::Impl>;
        using CCallablePGRef = std::shared_ptr<OPXCCallablePG::Impl>;
        using DCallablePGRef = std::shared_ptr<OPXDCallablePG::Impl>;
        using ExceptionPGRef = std::shared_ptr<OPXExceptionPG::Impl>;
        using HandleType     = std::unique_ptr<OptixPipeline_t,internal::OptixDestroy<OptixPipeline_t>>;
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
        void setProgramGroup(const    RaygenPGRef&    raygenPG)noexcept{
            m_RaygenPGs.insert(raygenPG);
        }
        void setProgramGroup(const      MissPGRef&      missPG)noexcept{
            m_MissPGs.insert(missPG);
        }
        void setProgramGroup(const  HitgroupPGRef&  hitgroupPG)noexcept{
            m_HitgroupPGs.insert(hitgroupPG);
        }
        void setProgramGroup(const CCallablePGRef& ccallablePG)noexcept{
            m_CCallablePGs.insert(ccallablePG);
        }
        void setProgramGroup(const DCallablePGRef& dcallablePG)noexcept{
            m_DCallablePGs.insert(dcallablePG);
        }
        void setProgramGroup(const ExceptionPGRef& exceptionPG)noexcept{
            m_ExceptionPGs.insert(exceptionPG);
        }
        void link(const OPXPipeline::LinkOptions& linkOptions);
        ~Impl()noexcept{
            try{
                m_RaygenPGs.clear();
                m_MissPGs.clear();
                m_HitgroupPGs.clear();
                m_Handle.reset();
                m_Context.reset();
                m_CompileOptions = {};
            }catch(std::runtime_error& error){
                std::cerr << "OPXContext::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType                          m_Handle         = nullptr;
        ContextRef                          m_Context        = {};
        std::unordered_set<   RaygenPGRef>  m_RaygenPGs      = {};
        std::unordered_set<     MissPGRef>  m_MissPGs        = {};
        std::unordered_set< HitgroupPGRef>  m_HitgroupPGs    = {};
        std::unordered_set<CCallablePGRef>  m_CCallablePGs   = {};
        std::unordered_set<DCallablePGRef>  m_DCallablePGs   = {};
        std::unordered_set<ExceptionPGRef>  m_ExceptionPGs   = {};
        OPXPipeline::CompileOptions         m_CompileOptions = {};
        OPXPipeline::LinkOptions            m_LinkOptions    = {};
    };
    //Module
    class OPXModule::Impl    {
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
                m_Handle .reset();
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
    //RayGen
    class OPXRaygenPG::Impl  {
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
                m_Handle.reset();
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
    //Miss
    class OPXMissPG::Impl    {
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
            if (this->getModule()) {
                desc.miss.module            = this->getModule()->getHandle();
                desc.miss.entryFunctionName = this->getEntryName().c_str();
            }
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle.reset();
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
    //HitGroup
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
        RTLIB_DECLARE_GET_BY_VALUE(Impl, const char*, CHEntryName, m_CHEntryName);
        RTLIB_DECLARE_SET_BY_VALUE(Impl,       char*, CHEntryName, m_CHEntryName);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,AHModule,m_AHModule);
        RTLIB_DECLARE_GET_BY_VALUE(Impl, const char*, AHEntryName, m_AHEntryName);
        RTLIB_DECLARE_SET_BY_VALUE(Impl,       char*, AHEntryName, m_AHEntryName);
        RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Impl,ModuleRef,ISModule,m_ISModule);
        RTLIB_DECLARE_GET_BY_VALUE(Impl, const char*, ISEntryName, m_ISEntryName);
        RTLIB_DECLARE_SET_BY_VALUE(Impl,       char*, ISEntryName, m_ISEntryName);
        void build(){
            OptixProgramGroupDesc    desc    = {};
            OptixProgramGroupOptions options = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            if (this->getCHModule()->getHandle()) {
                desc.hitgroup.moduleCH = this->getCHModule()->getHandle();
                desc.hitgroup.entryFunctionNameCH = this->getCHEntryName();
            }
            if (this->getAHModule()->getHandle()) {
                desc.hitgroup.moduleAH = this->getAHModule()->getHandle();
                desc.hitgroup.entryFunctionNameAH = this->getAHEntryName();
            }
            if (this->getISModule()->getHandle()) {
                desc.hitgroup.moduleIS = this->getISModule()->getHandle();
                desc.hitgroup.entryFunctionNameIS = this->getISEntryName();
            }
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle.reset();
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
    //CCallable
    class OPXCCallablePG::Impl{
        public:
        using ModuleRef   = std::shared_ptr< OPXModule::Impl>;
        using ContextRef  = std::shared_ptr<OPXContext::Impl>;
        using HandleType  = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
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
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            if (this->getModule()) {
                desc.callables.moduleCC            = this->getModule()->getHandle();
                desc.callables.entryFunctionNameCC = this->getEntryName().c_str();
                desc.callables.moduleDC            = nullptr;
                desc.callables.entryFunctionNameDC = nullptr;
            }
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle.reset();
                m_Module      = nullptr;
                m_EntryName   = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXCCallablePG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle    = nullptr;
        ContextRef  m_Context   = nullptr;    
        ModuleRef   m_Module    = nullptr;
        std::string m_EntryName = {};
    };
    //DCallable
    class OPXDCallablePG::Impl{
        public:
        using ModuleRef   = std::shared_ptr< OPXModule::Impl>;
        using ContextRef  = std::shared_ptr<OPXContext::Impl>;
        using HandleType  = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
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
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            if (this->getModule()) {
                desc.callables.moduleCC            = this->getModule()->getHandle();
                desc.callables.entryFunctionNameCC = this->getEntryName().c_str();
                desc.callables.moduleDC            = nullptr;
                desc.callables.entryFunctionNameDC = nullptr;
            }
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle.reset();
                m_Module      = nullptr;
                m_EntryName   = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXDCallablePG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle    = nullptr;
        ContextRef  m_Context   = nullptr;    
        ModuleRef   m_Module    = nullptr;
        std::string m_EntryName = {};
    };
    //Exception
    class OPXExceptionPG::Impl{
        public:
        using ModuleRef   = std::shared_ptr< OPXModule::Impl>;
        using ContextRef  = std::shared_ptr<OPXContext::Impl>;
        using HandleType  = std::unique_ptr<OptixProgramGroup_t,internal::OptixDestroy<OptixProgramGroup_t>>;
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
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
            if (this->getModule()) {
                desc.exception.module              = this->getModule()->getHandle();
                desc.exception.entryFunctionName   = this->getEntryName().c_str();
            }
            char log[1024];
            size_t sizeoflog = sizeof(log);
            OptixProgramGroup ptr = nullptr;
            RTLIB_OPTIX_CHECK2(optixProgramGroupCreate(m_Context->getHandle(),&desc,1,&options,log,&sizeoflog,&ptr),log);
            m_Handle.reset(ptr);
        }
        ~Impl()noexcept{
            try{
                m_Handle.reset();
                m_Module      = nullptr;
                m_EntryName   = {};
                m_Context     = nullptr;
            }catch(std::runtime_error& error){
                std::cerr << "OPXDCallablePG::~Impl Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    private:
        HandleType  m_Handle    = nullptr;
        ContextRef  m_Context   = nullptr;    
        ModuleRef   m_Module    = nullptr;
        std::string m_EntryName = {};
    };
    //保証したいこと
    //Pipeline, Module, ProgramGroupが生きている間、少なくともDeviceContextが生存
    //Pipelineが生きている間、少なくともProgramGroupが生存
    //ProgramGroupが生きている間、少なくともModuleは生存
    //
    //Pipeline    ---RefCnt--->DeviceContext
    //Module      ---RefCnt--->DeviceContext
    //ProgramGroup---RefCnt--->DeviceContext
    //    Pipeline---RefCnt--->ProgramGroup---RefCnt--->DeviceContext
    //ProgramGroup---RefCnt--->Module      ---RefCnt--->DeviceContext 

    //Pipeline    ---RefCnt--->DeviceContext
    //            ---RefCnt--->ProgramGroup---RefCnt--->DeviceContext
    //                                     ---RefCnt--->Module       ---RefCnt--->DeviceContext
    //---RefCnt--->DeviceContext
    //---RefCnt--->ProgramGroup ---RefCnt--->DeviceContext
    //                          ---RefCnt--->Module       ---RefCnt--->DeviceContext
    //---RefCnt--->DeviceContext
    //DeviceContext
    //Module       ---RefCnt--->DeviceContext
    //
}