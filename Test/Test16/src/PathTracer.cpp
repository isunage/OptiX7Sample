#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/PathTracer.h"
#include <RTLib/ext/TraversalHandle.h>
void test::PathTracer::InitCUDA()
{
	RTLIB_CUDA_CHECK(cudaFree(0));
}

void test::PathTracer::InitOPX()
{
	RTLIB_OPTIX_CHECK(optixInit());
	m_OPXContext = std::make_shared<rtlib::OPXContext>(rtlib::OPXContext::Desc{ 0, 0, OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL, 4 });
}

auto test::PathTracer::GetOPXContext() const -> const OPXContextPtr&
{
    return m_OPXContext;
}

void test::PathTracer::LoadTexture(const std::string& keyName, const std::string& texPath)
{
    int texWidth, texHeight, texComp;
    auto img = stbi_load(texPath.c_str(), &texWidth, &texHeight, &texComp, 4);
    this->m_Textures[keyName] = rtlib::CUDATexture2D<uchar4>();
    this->m_Textures[keyName].allocate(texWidth, texHeight, cudaTextureReadMode::cudaReadModeElementType);
    this->m_Textures[keyName].upload(img, texWidth, texHeight);
    stbi_image_free(img);
}

auto test::PathTracer::GetTexture(const std::string& keyName) const -> const rtlib::CUDATexture2D<uchar4>&
{
    return m_Textures.at(keyName);
}

bool test::PathTracer::HasTexture(const std::string& keyName) const noexcept
{
    return this->m_Textures.count(keyName)!=0;
}

void test::PathTracer::SetPipeline(const std::string& keyName, const std::shared_ptr<Pipeline>& pipeline)
{
    m_Pipelines[keyName] = pipeline;
}

void test::PathTracer::SetGASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::GASHandle>& gasHandle)
{
    m_GASHandles[keyName] = gasHandle;
}

auto test::PathTracer::GetInstance(const std::string& gasKeyName) const -> rtlib::ext::Instance
{
    auto baseGASHandle = this->m_GASHandles.at(gasKeyName);
	rtlib::ext::Instance instance  = {};
    instance.instance.traversableHandle = baseGASHandle->GetHandle();
    instance.instance.instanceId        = 0;
    instance.instance.sbtOffset         = 0;
    instance.instance.visibilityMask    = 255;
    instance.instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    float transform[12] = {
                1.0f,0.0f,0.0f,0.0f,
                0.0f,1.0f,0.0f,0.0f,
                0.0f,0.0f,1.0f,0.0f
    };
    std::memcpy(instance.instance.transform, transform, sizeof(float) * 12);
    instance.baseGASHandle = baseGASHandle;
    return instance;
}

void test::PathTracer::SetIASHandle(const std::string& keyName, const std::shared_ptr<rtlib::ext::IASHandle>& iasHandle)
{
    m_IASHandles[keyName] = iasHandle;
}

void test::Pipeline::Launch(CUstream stream) noexcept
{
    this->pipeline.launch(stream, this->paramsBuffer.gpuHandle.getDevicePtr(), this->shaderbindingTable, this->width, this->height, this->depth);
}
