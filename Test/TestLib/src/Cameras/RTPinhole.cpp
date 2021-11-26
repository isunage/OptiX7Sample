#include "RTPinhole.h"

test::RTPinhole::RTPinhole() noexcept
{
}

auto test::RTPinhole::GetTypeName() const noexcept -> RTString 
{
	return "Camera";
}

auto test::RTPinhole::GetPluginName() const noexcept -> RTString 
{
	return "Pinhole";
}

auto test::RTPinhole::GetID() const noexcept -> RTString 
{
    if (m_Properties.HasString("ID")){ 
        return m_Properties.GetString("ID");
    }
    return "";
}

auto test::RTPinhole::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTPinhole::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
	data["Type"]   = GetTypeName();
	data["Plugin"] = GetPluginName();
	if (!GetID().empty()) {
		data["ID"] = GetID();
	}
	data["Properties"] = GetProperties().GetJsonData();
	return data;
}

auto test::RTPinhole::GetEye() const noexcept -> RTPoint
{
	return m_Properties.GetPoint("Eye");
}

void test::RTPinhole::SetEye(const RTPoint& p) noexcept
{
	m_Properties.SetPoint("Eye", p);
}

auto test::RTPinhole::GetLookAt() const noexcept -> RTVector
{
	return m_Properties.GetVector("LookAt");
}

void test::RTPinhole::SetLookAt(const RTVector& d) noexcept
{
	m_Properties.SetVector("LookAt",d);
}

auto test::RTPinhole::GetVup() const noexcept -> RTVector
{
	return m_Properties.GetVector("Vup");
}

void test::RTPinhole::SetVup(const RTVector& d) noexcept
{
	m_Properties.SetPoint("Vup", d);
}

auto test::RTPinhole::GetFovY() const noexcept -> RTFloat
{
	return m_Properties.GetFloat("FovY");
}

void test::RTPinhole::SetFovY(const RTFloat fovY) noexcept
{
	m_Properties.SetFloat("FovY", fovY);
}

void test::RTPinhole::SetAspect(const RTFloat aspect) noexcept
{
	m_Properties.SetFloat("Aspect", aspect);
}

auto test::RTPinhole::GetAspect() const noexcept -> RTFloat
{

	return m_Properties.GetFloat("Aspect");
}
