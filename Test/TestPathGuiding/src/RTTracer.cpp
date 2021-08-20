#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/RTTracer.h"
#include <stb_image.h>
#include <stb_image_write.h>
//MeshGroup

void test::RTTracer::SetContext(const ContextPtr& context) noexcept
{
    m_Context = context;
}

auto test::RTTracer::GetContext() const -> const ContextPtr&
{
    // TODO: return ステートメントをここに挿入します
    return m_Context;
}

void test::RTTracer::AddMeshGroup(const std::string& mgName, const rtlib::ext::MeshGroupPtr& meshGroup) noexcept {
	m_MeshGroups[mgName] = meshGroup;
}

auto test::RTTracer::GetMeshGroup(const std::string& mgName) const -> rtlib::ext::MeshGroupPtr {
	return m_MeshGroups.at(mgName);
}

//MaterialList

void test::RTTracer::AddMaterialList(const std::string& mlName, const rtlib::ext::MaterialListPtr& materialList) noexcept {
	m_MaterialLists[mlName] = materialList;
}

auto test::RTTracer::GetMaterialList(const std::string& mlName) const -> rtlib::ext::MaterialListPtr {
	return m_MaterialLists.at(mlName);
}

void test::RTTracer::LoadTexture(const std::string& keyName, const std::string& texPath)
{
    int texWidth, texHeight, texComp;
    std::unique_ptr<unsigned char> pixels;
    {
        auto img = stbi_load(texPath.c_str(), &texWidth, &texHeight, &texComp, 4);
        pixels = std::unique_ptr<unsigned char>(new unsigned char[texWidth * texHeight * 4]);
        {
            for (auto h = 0; h < texHeight; ++h) {
                auto srcData = img + 4 * texWidth * (texHeight - 1 - h);
                auto dstData = pixels.get() + 4 * texWidth * h;
                std::memcpy(dstData, srcData, 4 * texWidth);
            }
        }

        stbi_image_free(img);
    }

    this->m_Textures[keyName] = rtlib::CUDATexture2D<uchar4>();
    this->m_Textures[keyName].allocate(texWidth, texHeight, cudaTextureReadMode::cudaReadModeElementType);
    this->m_Textures[keyName].upload(pixels.get(), texWidth, texHeight);
}

auto test::RTTracer::GetTexture(const std::string& keyName) const -> const rtlib::CUDATexture2D<uchar4>&
{
	// TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_Textures.at(keyName);
}

bool test::RTTracer::HasTexture(const std::string& keyName) const noexcept{
    return this->m_Textures.count(keyName) != 0;
}

//GeometryAS

void test::RTTracer::NewGASHandle(const std::string& gasName) {
	m_GASHandles[gasName] = rtlib::ext::GASHandlePtr(new rtlib::ext::GASHandle());
}

auto test::RTTracer::GetGASHandle(const std::string& gasName) const -> rtlib::ext::GASHandlePtr {
	return m_GASHandles.at(gasName);
}

//InstanceAS

void test::RTTracer::NewIASHandle(const std::string& iasName) {
	m_IASHandles[iasName] = rtlib::ext::IASHandlePtr(new rtlib::ext::IASHandle());
}

auto test::RTTracer::GetIASHandle(const std::string& iasName) const -> rtlib::ext::IASHandlePtr {
	return m_IASHandles.at(iasName);
}

//Pipeline

void test::RTTracer::SetTracePipeline(const RTTracePipelinePtr& tracePipeline) noexcept {
	m_TracePipeline = tracePipeline;
}


void test::RTTracer::SetDebugPipeline(const RTDebugPipelinePtr& debugPipeline) noexcept {
	m_DebugPipeline = debugPipeline;
}

auto test::RTTracer::GetTracePipeline() const -> const RTTracePipelinePtr&
{
    // TODO: return ステートメントをここに挿入します
    return m_TracePipeline;
}


auto test::RTTracer::GetDebugPipeline() const -> const RTDebugPipelinePtr&
{
    // TODO: return ステートメントをここに挿入します
    return m_DebugPipeline;
}

//TLAS

void test::RTTracer::SetTLASName(const std::string& tlasName) {
	m_TLASName = tlasName;
}

auto test::RTTracer::GetTLAS() const -> rtlib::ext::IASHandlePtr {
	return m_IASHandles.at(m_TLASName);
}
