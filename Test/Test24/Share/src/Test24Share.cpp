#include "..\include\Test24Share.h"

auto test24::SpecifyMaterialType(const rtlib::ext::VariableMap& material) -> std::string
{
	auto emitCol = material.GetFloat3As<float3>("emitCol");
	auto specCol = material.GetFloat3As<float3>("specCol");
	auto tranCol = material.GetFloat3As<float3>("tranCol");
	auto refrIndx = material.GetFloat1("refrIndx");
	auto shinness = material.GetFloat1("shinness");
	auto illum = material.GetUInt32("illum");
	if (illum == 7)
	{
		return "Diffuse";
	}
	else if (emitCol.x + emitCol.y + emitCol.z > 0.0f)
	{
		return "Emission";
	}
	else
	{
		return "Diffuse";
	}
}
