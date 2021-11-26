#include "..\include\TestLib\RTProperties.h"

auto test::RTProperties::GetJsonData() const noexcept -> nlohmann::json
{
	nlohmann::json data;
	for (auto& [name,val] : m_MapFloat)
	{
		data[name] = val;
	}
	for (auto& [name, val] : m_MapInt32)
	{
		data[name] = val;
	}
	for (auto& [name, val] : m_MapPoint)
	{
		data[name] = std::array<float, 3>{val.x, val.y, val.z};
	}
	for (auto& [name, val] : m_MapVector)
	{
		data[name] = std::array<float, 3>{val.x, val.y, val.z};
	}
	for (auto& [name, val] : m_MapMat4x4)
	{
		data[name] = std::array<float,16>{
			val(0,0), val(1, 0), val(2, 0), val(3, 0),
			val(0,1), val(1, 1), val(2, 1), val(3, 1),
			val(0,2), val(1, 2), val(2, 2), val(3, 2),
			val(0,3), val(1, 3), val(2, 3), val(3, 3)
		};
	}
	for (auto& [name, val] : m_MapColor)
	{
		data[name] = std::array<float, 3>{val.x, val.y, val.z};
	}
	for (auto& [name, val] : m_MapString)
	{
		data[name] = val;
	}
	for (auto& [name, val] : m_MapTexture)
	{
		data[name] = val->GetJsonAsData();
	}
	return data;
}

bool test::RTProperties::LoadFloat(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_number_float()) {
		return false;
	}
	SetFloat(keyName, jsonData[keyName].get<RTFloat>());
	return true;
}

bool test::RTProperties::LoadInt32(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_number_integer()) {
		return false;
	}
	SetInt32(keyName, jsonData[keyName].get<RTInt32>());
	return true;
}

bool test::RTProperties::LoadPoint(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_array()) {
		return false;
	}
	std::vector<float> t_data;
	RTColor            t_color;
	for (auto& v : jsonData[keyName])
	{
		if (!v.is_number_float()) {
			return false;
		}
		t_data.push_back(v.get<float>());
	}
	if (t_data.size() > 3) {
		return false;
	}
	if (t_data.size() == 3) {
		t_color = make_float3(t_data[0], t_data[1], t_data[2]);
	}
	if (t_data.size() == 2) {
		t_color = make_float3(t_data[0], t_data[1], 0.0f);
	}
	if (t_data.size() == 1) {
		t_color = make_float3(t_data[0], 0.0f, 0.0f);
	}
	if (t_data.size() == 0) {
		t_color = make_float3(0.0f, t_data[1], 0.0f);
	}
	SetColor(keyName, t_color);
	return true;
}

bool test::RTProperties::LoadVector(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_array()) {
		return false;
	}
	std::vector<float> t_data;
	RTVector           t_color;
	for (auto& v : jsonData[keyName])
	{
		if (!v.is_number_float()) {
			return false;
		}
		t_data.push_back(v.get<float>());
	}
	if (t_data.size() > 3) {
		return false;
	}
	if (t_data.size() == 3) {
		t_color = make_float3(t_data[0], t_data[1], t_data[2]);
	}
	if (t_data.size() == 2) {
		t_color = make_float3(t_data[0], t_data[1], 0.0f);
	}
	if (t_data.size() == 1) {
		t_color = make_float3(t_data[0], 0.0f, 0.0f);
	}
	if (t_data.size() == 0) {
		t_color = make_float3(0.0f, t_data[1], 0.0f);
	}
	SetVector(keyName, t_color);
	return true;
}

bool test::RTProperties::LoadMat4x4(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_array()) {
		return false;
	}
	std::vector<float> t_data;
	for (auto& v : jsonData[keyName])
	{
		if (!v.is_number_float()) {
			return false;
		}
		t_data.push_back(v.get<float>());
	}
	if (t_data.size() == 16) {
		return false;
	}
	SetMat4x4(keyName, RTMat4x4(
		make_float4(t_data[4 * 0 + 0], t_data[4 * 1 + 0], t_data[4 * 2 + 0], t_data[4 * 3 + 0]),
		make_float4(t_data[4 * 0 + 1], t_data[4 * 1 + 1], t_data[4 * 2 + 1], t_data[4 * 3 + 1]),
		make_float4(t_data[4 * 0 + 2], t_data[4 * 1 + 2], t_data[4 * 2 + 2], t_data[4 * 3 + 2]),
		make_float4(t_data[4 * 0 + 3], t_data[4 * 1 + 3], t_data[4 * 2 + 3], t_data[4 * 3 + 3])
	));
	return true;
}

bool test::RTProperties::LoadColor(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_array()) {
		return false;
	}

	std::vector<float> t_data;
	RTColor           t_color;
	for (auto& v : jsonData[keyName])
	{
		if (!v.is_number_float()&&!v.is_number_integer()) {

			return false;
		}
		t_data.push_back(v.get<float>());
	}

	if (t_data.size() > 3) {
		return false;
	}
	if (t_data.size() == 3) {
		t_color = make_float3(t_data[0], t_data[1], t_data[2]);
	}
	if (t_data.size() == 2) {
		t_color = make_float3(t_data[0], t_data[1], 0.0f);
	}
	if (t_data.size() == 1) {
		t_color = make_float3(t_data[0], 0.0f, 0.0f);
	}
	if (t_data.size() == 0) {
		t_color = make_float3(0.0f, t_data[1], 0.0f);
	}
	SetColor(keyName, t_color);
	return true;
}

bool test::RTProperties::LoadString(const std::string& keyName, const nlohmann::json& jsonData) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_string()) {
		return false;
	}
	if (!jsonData[keyName].is_string()) {
		return false;
	}
	auto str = jsonData[keyName].get<std::string>();
	if (!str.empty()) {

		SetString(keyName, str);
		return true;
	}
	return false;
}

bool test::RTProperties::LoadTexture(const std::string& keyName, const nlohmann::json& jsonData, std::shared_ptr<RTTextureCache>& cache) noexcept
{
	if (!jsonData.contains(keyName) || !jsonData[keyName].is_string()) {
		return false;
	}
	if (!jsonData[keyName].is_string()) {
		return false;
	}
	if (!cache->HasTexture(jsonData[keyName].get<std::string>())) {
		return false;
	}
	SetTexture(keyName, cache->GetTexture(jsonData[keyName].get<std::string>()));
	return true;
}
