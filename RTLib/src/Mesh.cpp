#include "../include/RTLib/ext/Mesh.h"

void rtlib::ext::MeshGroup::SetSharedResource(const MeshSharedResourcePtr& res) noexcept
{
	this->m_SharedResource = res;
}

auto rtlib::ext::MeshGroup::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
	return this->m_SharedResource;
}

void rtlib::ext::MeshGroup::SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res) noexcept
{
	this->m_UniqueResources[name] = res;
}

auto rtlib::ext::MeshGroup::GetUniqueResource(const std::string& name) const -> MeshUniqueResourcePtr
{
	return this->m_UniqueResources.at(name);
}

auto rtlib::ext::MeshGroup::GetUniqueResources() const noexcept -> const std::unordered_map<std::string, MeshUniqueResourcePtr>&
{
	return m_UniqueResources;
}

auto rtlib::ext::MeshGroup::GetUniqueNames() const noexcept -> std::vector<std::string>
{
    auto uniqueNames = std::vector<std::string>();
    uniqueNames.reserve(m_UniqueResources.size());
    for (auto& [name, res] : m_UniqueResources) {
        uniqueNames.push_back(name);
    }
    return uniqueNames;
}

auto rtlib::ext::MeshGroup::LoadMesh(const std::string& name) const -> MeshPtr
{
	auto mesh              = std::make_shared<Mesh>();
	mesh->m_Name           = name;
	mesh->m_SharedResource = this->GetSharedResource();
	mesh->m_UniqueResource = this->GetUniqueResource(name);
	return mesh;
}

bool rtlib::ext::MeshGroup::RemoveMesh(const std::string& name)
{
	if (m_UniqueResources.count(name)>0) {
		m_UniqueResources.erase(name);
		return true;
	}
	return false;
}

void rtlib::ext::Mesh::SetSharedResource(const MeshSharedResourcePtr& res) noexcept
{
    m_SharedResource = res;
}

auto rtlib::ext::Mesh::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
	return m_SharedResource;
}

void rtlib::ext::Mesh::SetUniqueResource(const MeshUniqueResourcePtr& res) noexcept
{
	m_UniqueResource = res;
}

void rtlib::ext::Mesh::SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res) noexcept
{
    m_Name = name;
    m_UniqueResource = res;
}

auto rtlib::ext::Mesh::GetUniqueResource() const -> MeshUniqueResourcePtr
{
	return m_UniqueResource;
}
