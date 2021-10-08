#ifndef RTLIB_CUDA_H
#define RTLIB_CUDA_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include "Preprocessors.h"
#include "PixelFormat.h"
#include "Exceptions.h"
namespace rtlib{
    /*****Buffers*********/
    //CUDA Buffer
    template<typename T>
    class CUDABuffer{
        T*     m_DevicePtr = nullptr;
        size_t m_Count     = 0;
    private:
        void upload_unsafe(const T* hostPtr, size_t count){
            RTLIB_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(m_DevicePtr),
                reinterpret_cast<const void*>(hostPtr),
                sizeof(T)*count,cudaMemcpyHostToDevice));
        }
        void download_unsafe(T* hostPtr, size_t count){
            RTLIB_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hostPtr),
                reinterpret_cast<const void*>(m_DevicePtr),
                sizeof(T)*count,cudaMemcpyDeviceToHost));
        }
    public:
        //constructor,copy,move 
        CUDABuffer()noexcept{}
        CUDABuffer(const CUDABuffer&)noexcept = delete;
        CUDABuffer(CUDABuffer& buffer)noexcept{
            this->m_DevicePtr = buffer.m_DevicePtr;
            this->m_Count     = buffer.m_Count;
            buffer.m_DevicePtr = nullptr;
            buffer.m_Count     = 0;
        }
        CUDABuffer& operator=(const CUDABuffer&)noexcept = delete;
        CUDABuffer& operator=(CUDABuffer&& buffer){
            if(this!=&buffer){  
                this->reset();
                this->m_DevicePtr = buffer.m_DevicePtr;
                this->m_Count     = buffer.m_Count;
                buffer.m_DevicePtr = nullptr;
                buffer.m_Count     = 0;
            }
            return *this;
        }
        //user constructor
        explicit CUDABuffer(const T* hostPtr,size_t count){
            this->allocate(count);
            this->upload_unsafe(hostPtr,count);
        }
        explicit CUDABuffer(const T& hostData):CUDABuffer(&hostData,1){}
        template<size_t N>
        explicit CUDABuffer(const T (&hostData)[N]):CUDABuffer(std::data(hostData),std::size(hostData)){}
        template<size_t N>
        explicit CUDABuffer(const std::array<T,N>& hostData):CUDABuffer(std::data(hostData),std::size(hostData)){}
        explicit CUDABuffer(const std::vector<T>& hostData) :CUDABuffer(std::data(hostData),std::size(hostData)){}
        //bool
        explicit operator bool()const noexcept{
            return m_DevicePtr!=nullptr;
        }
        //member
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,T*,DevicePtr,m_DevicePtr);
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,size_t,Count,m_Count);
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,size_t,SizeInBytes,m_Count*sizeof(T));
        void allocate(size_t count){
            cudaMalloc(reinterpret_cast<void**>(&m_DevicePtr),sizeof(T)*count);
            m_Count = count;
        }
        bool resize(size_t count){
            if(m_Count!=count){
                this->reset();
                this->allocate(count);
                return true;
            }
            return false;
        }
        void upload(const T* hostPtr, size_t count){
            this->upload_unsafe(hostPtr, std::min(count,m_Count));
        }
        void upload(const std::vector<T>& hostData) {
            this->upload_unsafe(hostData.data(), std::min(hostData.size(), m_Count));
        }
        void download(T* hostPtr,size_t count){
            this->download_unsafe(hostPtr, std::min(count,m_Count));
        }
        void download(std::vector<T>& hostData){
            hostData.resize(m_Count);
            this->download_unsafe(hostData.data(),hostData.size());
        }
        void reset(){
            if(m_DevicePtr){
                RTLIB_CUDA_CHECK(cudaFree((void*)m_DevicePtr));
                m_DevicePtr = nullptr;
                m_Count     = 0;
            }
        }
        //destructor
        ~CUDABuffer()noexcept{
            try{
                this->reset();
            }catch(rtlib::CUDAException& err){
                std::cerr << "~CUDABuffer() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
        }
    };
    template<>
    class CUDABuffer<void>{
        void*  m_DevicePtr = nullptr;
        size_t m_Count     = 0;
    private:
        void upload_unsafe(const void* hostPtr, size_t count){
            RTLIB_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(m_DevicePtr),hostPtr,
                count,cudaMemcpyHostToDevice));
        }
        void download_unsafe(void* hostPtr, size_t count){
            RTLIB_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hostPtr),m_DevicePtr,
                count,cudaMemcpyDeviceToHost));
        }
    public:
        //constructor,copy,move 
        CUDABuffer()noexcept{}
        CUDABuffer(const CUDABuffer&)noexcept = delete;
        CUDABuffer(CUDABuffer& buffer)noexcept{
            this->m_DevicePtr = buffer.m_DevicePtr;
            this->m_Count     = buffer.m_Count;
            buffer.m_DevicePtr = nullptr;
            buffer.m_Count     = 0;
        }
        CUDABuffer& operator=(const CUDABuffer&)noexcept = delete;
        CUDABuffer& operator=(CUDABuffer&& buffer){
            if(this!=&buffer){  
                this->reset();
                this->m_DevicePtr = buffer.m_DevicePtr;
                this->m_Count     = buffer.m_Count;
                buffer.m_DevicePtr = nullptr;
                buffer.m_Count     = 0;
            }
            return *this;
        }
        //user constructor
        explicit CUDABuffer(size_t count) {
            this->allocate(count);
        }
        explicit CUDABuffer(const void* hostPtr,size_t count){
            this->allocate(count);
            this->upload_unsafe(hostPtr,count);
        }
        //bool
        explicit operator bool()const noexcept{
            return m_DevicePtr!=nullptr;
        }
        //member
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,void*,DevicePtr,m_DevicePtr);
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,size_t,Count,m_Count);
        RTLIB_DECLARE_GET_BY_VALUE(CUDABuffer,size_t,SizeInBytes,m_Count);
        void allocate(size_t count){
            cudaMalloc(&m_DevicePtr,count);
            m_Count = count;
        }
        bool resize(size_t count){
            if(m_Count!=count){
                this->reset();
                this->allocate(count);
                return true;
            }
            return false;
        }
        void upload(const void* hostPtr, size_t count){
            this->upload_unsafe(hostPtr, std::min(count,m_Count));
        }
        void download(void* hostPtr,size_t count){
            this->download_unsafe(hostPtr, std::min(count,m_Count));
        }
        void reset(){
            if(m_DevicePtr){
                RTLIB_CUDA_CHECK(cudaFree(m_DevicePtr));
                m_DevicePtr = nullptr;
                m_Count     = 0;
            }
        }
        //destructor
        ~CUDABuffer()noexcept{
            try{
                this->reset();
            }catch(rtlib::CUDAException& err){
                std::cerr << "~CUDABuffer() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
        }
    };
    template<typename T>
    struct CUDAUploadBuffer {
        std::vector<T>       cpuHandle   = {};
        rtlib::CUDABuffer<T> gpuHandle   = {};
    public:
        void Alloc() {
            if (!cpuHandle.empty()) {
                return;
            }
            if (gpuHandle.getDevicePtr()) {
                gpuHandle.reset();
            }
            gpuHandle.allocate(cpuHandle.size());
        }
        void Alloc(size_t count){
            if (gpuHandle.getDevicePtr()) {
                gpuHandle.reset();
            }
            cpuHandle.resize(count);
            gpuHandle.allocate(count);
        }
        void Upload() {
            gpuHandle.resize(cpuHandle.size());
            gpuHandle.upload(cpuHandle);
        }
        void Reset() {
            cpuHandle.clear();
            gpuHandle.reset();
        }
        //destructor
        ~CUDAUploadBuffer()noexcept {
            try {
                this->Reset();
            }
            catch (rtlib::CUDAException& err) {
                std::cerr << "~CUDAUploadBuffer() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
        }
    };
    template<typename T>
    struct CUDADownloadBuffer{
        std::vector<T>       cpuHandle   = {};
        rtlib::CUDABuffer<T> gpuHandle   = {};
    public:
        void Alloc(size_t count){
            gpuHandle.allocate(count);
        }
        void Download(size_t count) {
            cpuHandle.resize(count);
            gpuHandle.download(cpuHandle.data(),count);
        }
        void Reset() {
            cpuHandle.clear();
            gpuHandle.reset();
        }
        //destructor
        ~CUDADownloadBuffer()noexcept {
            try {
                this->Reset();
            }
            catch (rtlib::CUDAException& err) {
                std::cerr << "~CUDADownloadBuffer() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
        }
    };
    /*****Textures*********/
    //Array2D
    //CUDA Array
    template<typename PixelType>
    class CUDAArray2D {
        cudaArray_t           m_Handle  = nullptr;
        size_t                m_Width  = 0;
        size_t                m_Height = 0;
        cudaChannelFormatDesc m_ChannelFormatDesc = {};
    private:
        
    public:
        CUDAArray2D() {
            m_ChannelFormatDesc = cudaCreateChannelDesc<PixelType>();
        }
        CUDAArray2D(const CUDAArray2D&)noexcept = delete;
        CUDAArray2D(CUDAArray2D&& arr)noexcept{
            m_Handle = arr.m_Handle;
            m_Width  = arr.m_Width;
            m_Height = arr.m_Height;
            m_ChannelFormatDesc = arr.m_ChannelFormatDesc;
            arr.m_Handle = nullptr;
            arr.m_Width  = 0;
            arr.m_Height = 0;
        }
        CUDAArray2D& operator=(const CUDAArray2D&)noexcept = delete;
        CUDAArray2D& operator=(CUDAArray2D&&  arr){
            if(this!=&arr){
                this->reset();
                m_Handle = arr.m_Handle;
                m_Width  = arr.m_Width;
                m_Height = arr.m_Height;
                m_ChannelFormatDesc = arr.m_ChannelFormatDesc;
                arr.m_Handle = nullptr;
                arr.m_Width  = 0;
                arr.m_Height = 0;
            }
            return *this;
        }
        explicit operator bool()const noexcept{
            return m_Handle!=nullptr;
        }
        RTLIB_DECLARE_GET_BY_VALUE(CUDAArray2D,cudaArray_t,Handle,m_Handle);
        RTLIB_DECLARE_GET_BY_VALUE(CUDAArray2D,size_t, Width,m_Width);
        RTLIB_DECLARE_GET_BY_VALUE(CUDAArray2D,size_t,Height,m_Height);
        void allocate(size_t width, size_t height) {
            m_Width  = width;
            m_Height = height;
            RTLIB_CUDA_CHECK(cudaMallocArray(&m_Handle, &m_ChannelFormatDesc, m_Width, m_Height));
        }
        bool resize(  size_t width, size_t height){
            if(m_Width!=width||m_Height!=height){
                this->reset();
                this->allocate(width,height);
                return false;
            }
            return true;
        }
        void upload(const void* hostPtr, size_t width, size_t height, size_t spitch) {
            RTLIB_CUDA_CHECK(cudaMemcpy2DToArray(m_Handle, 0, 0, hostPtr, spitch, width* CUDAPixelTraits<PixelType>::numChannels* sizeof(CUDAPixelTraits<PixelType>::base_type), height, cudaMemcpyHostToDevice));
        }
        void upload(const void* hostPtr, size_t width, size_t height) {
            this->upload(hostPtr, width, height, width * CUDAPixelTraits<PixelType>::numChannels * sizeof(CUDAPixelTraits<PixelType>::base_type));
        }
        void download(void* hostPtr, size_t width, size_t height, size_t dpitch) {
            RTLIB_CUDA_CHECK(cudaMemcpy2DFromArray(hostPtr,dpitch,m_Handle,0,0,  width* CUDAPixelTraits<PixelType>::numChannels* sizeof(CUDAPixelTraits<PixelType>::base_type), height,cudaMemcpyDeviceToHost));
        }
        void download(void* hostPtr, size_t width, size_t height) {
            this->download(hostPtr, width, height, width * CUDAPixelTraits<PixelType>::numChannels * sizeof(CUDAPixelTraits<PixelType>::base_type));
        }
        void reset() {
            if (m_Handle) {
                RTLIB_CUDA_CHECK(cudaFreeArray(m_Handle));
                m_Handle  = nullptr;
                m_Width  = 0;
                m_Height = 0;
            }  
        }
        ~CUDAArray2D()noexcept {
            try {
                this->reset();
            }
            catch (rtlib::CUDAException& err) {
                std::cerr << "~CUDAArray2D() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
            
        }
    };
    //CUDA Texture
    template<typename PixelType>
    class CUDATexture2D{
        cudaTextureObject_t  m_Handle = 0;
        CUDAArray2D<PixelType> m_Array  = {};
    public:
        void allocateArray(size_t width, size_t height){
            m_Array.allocate(width,height);
        }
        void releaseArray(){
            m_Array.reset();
        }
        void createTextureObject(cudaTextureReadMode readMode,bool useSRGB){
            cudaTextureFilterMode filterMode = cudaFilterModePoint;
            if (readMode == cudaReadModeNormalizedFloat)
            {
                filterMode = cudaFilterModeLinear;
            }
            cudaResourceDesc resDesc            = {};
            resDesc.resType                     = cudaResourceTypeArray;
            resDesc.res.array.array             = m_Array.getHandle();
            cudaTextureDesc texDesc             = {};
            texDesc.addressMode[0]              = cudaAddressModeWrap;
            texDesc.addressMode[1]              = cudaAddressModeWrap;
            texDesc.filterMode                  = filterMode;
            texDesc.readMode                    = readMode;
            texDesc.normalizedCoords            = true;
            texDesc.maxAnisotropy               = 1;
            texDesc.maxMipmapLevelClamp         = 99;
            texDesc.minMipmapLevelClamp         = 0;
            texDesc.mipmapFilterMode            = filterMode;
            texDesc.borderColor[0]              = 1.0f;
            texDesc.sRGB                        = useSRGB;
            RTLIB_CUDA_CHECK(cudaCreateTextureObject(&m_Handle,&resDesc,&texDesc,nullptr));
        }
        void destroyTextureObject(){
            if(m_Handle){
                RTLIB_CUDA_CHECK(cudaDestroyTextureObject(m_Handle));
                m_Handle = 0;
            }
        }
    public:
        CUDATexture2D():m_Array{} {}
        CUDATexture2D(const CUDATexture2D&)noexcept = delete;
        CUDATexture2D(CUDATexture2D&& tex)noexcept{
            m_Handle = tex.m_Handle;
            m_Array  = std::move(tex.m_Array);
            tex.m_Handle = 0;
        }
        CUDATexture2D& operator=(const CUDATexture2D&)noexcept = delete;
        CUDATexture2D& operator=(CUDATexture2D&&  tex){
            if(this!=&tex){
                this->reset();
                m_Handle = tex.m_Handle;
                m_Array  = std::move(tex.m_Array);
                tex.m_Handle = 0;
            }
            return *this;
        }
        explicit operator bool()const noexcept{
            return m_Handle!=nullptr;
        }
        RTLIB_DECLARE_GET_BY_VALUE(CUDATexture2D, cudaTextureObject_t,Handle,m_Handle);
        RTLIB_DECLARE_GET_BY_REFERENCE(CUDATexture2D,CUDAArray2D<PixelType>,Array,m_Array);
        RTLIB_DECLARE_GET_BY_VALUE(CUDATexture2D,size_t, Width,m_Array.getWidth());
        RTLIB_DECLARE_GET_BY_VALUE(CUDATexture2D,size_t,Height,m_Array.getHeight());
        void allocate(size_t width, size_t height,cudaTextureReadMode readMode = cudaReadModeNormalizedFloat, bool useSRGB = false){
            this->allocateArray(width,height);
            this->createTextureObject(readMode,useSRGB);
        }
        bool resize(  size_t width, size_t height,cudaTextureReadMode readMode = cudaReadModeNormalizedFloat, bool useSRGB = false){
            if(m_Array.getWidth()!=width||m_Array.getHeight()!=height){
                this->reset();
                this->allocate(width,height,readMode,useSRGB);
                return true;
            }
            return false;
        }
        void upload(  const void* hostPtr, size_t width, size_t height, size_t spitch) {
            m_Array.upload(hostPtr,width,height,spitch);
        }
        void upload(  const void* hostPtr, size_t width, size_t height) {
            m_Array.upload(hostPtr,width,height);
        }
        void download(void* hostPtr, size_t width, size_t height, size_t dpitch) {
            m_Array.download(hostPtr,width,height,dpitch);
        }
        void download(void* hostPtr, size_t width, size_t height) {
            m_Array.download(hostPtr,width,height);
        }
        void reset(){
            this->destroyTextureObject();
            this->releaseArray();
        }
        ~CUDATexture2D()noexcept{
            try {
                this->reset();
            }
            catch (rtlib::CUDAException& err) {
                std::cerr << "~CUDATexture2D() Catch Error!\n";
                std::cerr << err.what() << std::endl;
            }
            
        }
    };  
    //NVRTC Options
    class NVRTCOptions{
    public:
        NVRTCOptions& setIncludeDir(const std::string& includeDir) {
            m_Options.push_back(std::string("-I") + includeDir);
            return *this;
        }
        NVRTCOptions& setIncludeDirs(const std::vector<std::string>& includeDirs) {
            for (auto& includeDir : includeDirs) {
                m_Options.push_back(std::string("-I") + includeDir);
            }
            return *this;
        }
        NVRTCOptions& setOtherOption(const std::string& option) {
            m_Options.push_back(option);
            return *this;
        }
        NVRTCOptions& setOtherOptions(const std::vector<std::string>& options) {
            for (auto& option : options) {
                m_Options.push_back(option);
            }
            return *this;
        }
        auto get()const->std::vector<std::string> {
            return m_Options;
        }
    private:
        std::vector<std::string> m_Options = {};
    };
    //NVRTC Program
    class NVRTCProgram{
        nvrtcProgram m_Handle = nullptr;
    public:
        NVRTCProgram()noexcept{}
        NVRTCProgram(const NVRTCProgram& program)noexcept = delete;
        NVRTCProgram(NVRTCProgram&& program)noexcept{
            m_Handle = program.m_Handle;
            program.m_Handle = nullptr; 
        }
        NVRTCProgram& operator=(const NVRTCProgram& program)noexcept = delete;
        NVRTCProgram& operator=(NVRTCProgram&& program){
            if(this!=&program){
                this->destroy();
                m_Handle = program.m_Handle;
                program.m_Handle = nullptr; 
            }
            return *this;
        }
        //
        NVRTCProgram(const char *src, const char *name, 
                    int numHeaders , const char *const *headers, 
                    const char *const *includeNames):NVRTCProgram(){
            this->create(src,name,numHeaders,headers,includeNames);
        }
        NVRTCProgram(const std::string& src, const std::string& name):NVRTCProgram(){
            this->create(src,name);
        }
        void create(const char *src, const char *name, 
                    int numHeaders , const char *const *headers, 
                    const char *const *includeNames){
            RTLIB_NVRTC_CHECK(nvrtcCreateProgram(&m_Handle,src,name,numHeaders,headers,includeNames));
        }
        explicit operator bool()const noexcept{
            return m_Handle!=nullptr;
        }
        void create(const std::string& src, const std::string& name){
            this->create(src.data(),name.data(),0,nullptr,nullptr);
        }
        void compile(int numOptions, const char* const * options)const{
            RTLIB_NVRTC_CHECK(nvrtcCompileProgram(m_Handle,numOptions,options));
        }
        void compile(const std::vector<std::string>& options)const{
            std::vector<const char*> t_Options;
            t_Options.reserve(options.size());
            for(const auto& option:options){
                t_Options.push_back(option.data());
            }
            this->compile(t_Options.size(),t_Options.data());
        }
        auto getPTX()const->std::string{
            std::string ptx;
            size_t sizeInBytes = 0;
            RTLIB_NVRTC_CHECK(nvrtcGetPTXSize(m_Handle,&sizeInBytes));
            ptx.resize(sizeInBytes+1,'\0');
            RTLIB_NVRTC_CHECK(nvrtcGetPTX(m_Handle,ptx.data()));
            ptx.resize(sizeInBytes);
            return ptx;
        }
        auto getLog()const->std::string{
            std::string log;
            size_t sizeInBytes = 0;
            RTLIB_NVRTC_CHECK(nvrtcGetProgramLogSize(m_Handle,&sizeInBytes));
            log.resize(sizeInBytes+1,'\0');
            RTLIB_NVRTC_CHECK(nvrtcGetProgramLog(m_Handle,log.data()));
            log.resize(sizeInBytes);
            return log;
        }
        void destroy(){
            if(m_Handle){
                RTLIB_NVRTC_CHECK(nvrtcDestroyProgram(&m_Handle));
                m_Handle = nullptr;
            }
        }
        ~NVRTCProgram()noexcept{
            try{
                this->destroy();
            }catch(rtlib::NVRTCException& error){
                std::cerr << "~NVRTCProgram() Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    };
    //CUDA Function
    class CUDAFunction {
        CUfunction m_Handle = nullptr;
    public:
        CUDAFunction()noexcept {}
        CUDAFunction(const CUDAFunction&)noexcept = delete;
        CUDAFunction(CUDAFunction&& func)noexcept {
            m_Handle = func.m_Handle;
            func.m_Handle = nullptr;
        }
        CUDAFunction& operator=(const CUDAFunction&)noexcept = delete;
        CUDAFunction& operator=(CUDAFunction&& func) {
            if (this != &func) {
                this->unload();
                m_Handle = func.m_Handle;
                func.m_Handle = nullptr;
            }
            return *this;
        }
        explicit operator bool()const noexcept {
            return m_Handle != nullptr;
        }
        RTLIB_DECLARE_GET_BY_VALUE(CUDAFunction, CUfunction, Handle, m_Handle);
        int  getAttribute(CUfunction_attribute attrib)const {
            int val;
            RTLIB_CU_CHECK(cuFuncGetAttribute(&val, attrib, m_Handle));
            return val;
        }
        void setAttribute(CUfunction_attribute attrib, int val)const {
            RTLIB_CU_CHECK(cuFuncSetAttribute(m_Handle, attrib, val));
        }
        void load(CUmodule module, const std::string& name) {
            RTLIB_CU_CHECK(cuModuleGetFunction(&m_Handle, module, name.c_str()));
        }
        void unload()noexcept{
            if (m_Handle) {
                m_Handle = nullptr;
            }
        }
        void launch(const uint3& gridDims, const uint3& blockDims, unsigned int sharedMemBytes,
                    CUstream hStream, void** kernelParams, void** extra) {
            RTLIB_CU_CHECK(cuLaunchKernel(m_Handle,
                gridDims.x, gridDims.y, gridDims.z,
                blockDims.x, blockDims.y, blockDims.z,
                sharedMemBytes, hStream, kernelParams, extra));
        }
        ~CUDAFunction()noexcept {
            this->unload();
        }
    };
    //CUDA Module
    class CUDAModule{
        CUmodule m_Handle = nullptr;
    public:
        CUDAModule()noexcept{}
        CUDAModule(const CUDAModule&)noexcept = delete;
        CUDAModule(CUDAModule&& module)noexcept {
            m_Handle = module.m_Handle;
            module.m_Handle = nullptr;
        }
        CUDAModule& operator=(const CUDAModule&)noexcept = delete;
        CUDAModule& operator=(CUDAModule&& module){
            if (this != &module) {
                this->unload();
                m_Handle = module.m_Handle;
                module.m_Handle = nullptr;
            }
            return *this;
        }
        //
        explicit CUDAModule(const std::string& filename) :CUDAModule(){
            this->load(filename);
        }
        explicit CUDAModule(const void* image) :CUDAModule() {
            this->load(image);
        }
        explicit CUDAModule(const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) :CUDAModule() {
            this->load(image,numOptions,options,optionValues);
        }
        explicit operator bool()const noexcept {
            return m_Handle != nullptr;
        }
        RTLIB_DECLARE_GET_BY_VALUE(CUDAModule, CUmodule, Handle, m_Handle);
        void load(const std::string& file_name) {
            RTLIB_CU_CHECK(cuModuleLoad(&m_Handle, file_name.c_str()));
        }
        void load(const void* image) {
            RTLIB_CU_CHECK(cuModuleLoadData(&m_Handle, image));
        }
        void load(const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
            RTLIB_CU_CHECK(cuModuleLoadDataEx(&m_Handle, image, numOptions, options, optionValues));
        }
        void unload() {
            if (m_Handle) {
                RTLIB_CU_CHECK(cuModuleUnload(m_Handle));
                m_Handle = nullptr;
            }
        }
        auto getFunction(const std::string& name)const->CUDAFunction {
            auto function = CUDAFunction();
            function.load(m_Handle, name);
            return function;
        }
        ~CUDAModule()noexcept{
            try {
                this->unload();
            }
            catch (CUException& error) {
                std::cerr << "~CUDAModule() Catch Error!\n";
                std::cerr << error.what() << std::endl;
            }
        }
    };
}
#endif