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

auto test::RTPinhole::GetProperties() const noexcept -> const RTProperties & 
{
	return m_Properties;
}

auto test::RTPinhole::GetJsonAsData() const noexcept -> nlohmann::json 
{
	nlohmann::json data;
	data = GetProperties().GetJsonAsData();
	data["Type"]   = GetTypeName();
	data["Plugin"] = GetPluginName();
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
struct test::RTPinholeReader::Impl {

};
test::RTPinholeReader::RTPinholeReader() noexcept
{
	m_Impl = std::make_unique<test::RTPinholeReader::Impl>();
}

auto test::RTPinholeReader::GetPluginName() const noexcept -> RTString
{
	return "Pinhole";
}

auto test::RTPinholeReader::LoadJsonFromData(const nlohmann::json& json) noexcept -> RTCameraPtr 
{
	if (!json.contains("Type") || !json["Type"].is_string() || json["Type"].get<std::string>() != "Camera") {
		return nullptr;
	}

	if (!json.contains("Plugin") || !json["Plugin"].is_string() || json["Plugin"].get<std::string>() != "Pinhole") {
		return nullptr;
	}

	auto pinhole = std::make_shared<test::RTPinhole>();
	if (!pinhole->m_Properties.LoadPoint("Eye", json)) {
		return nullptr;
	}
	if (!pinhole->m_Properties.LoadVector("LookAt", json)) {
		return nullptr;
	}
	if (!pinhole->m_Properties.LoadVector("Vup", json)) {
		return nullptr;
	}
	if (!pinhole->m_Properties.LoadFloat("FovY", json)) {
		return nullptr;
	}
	if (!pinhole->m_Properties.LoadFloat("Aspect", json)) {
		return nullptr;
	}
	pinhole->m_Properties.LoadString("ID", json);
	return pinhole;
}

test::RTPinholeReader::~RTPinholeReader() noexcept
{
}
