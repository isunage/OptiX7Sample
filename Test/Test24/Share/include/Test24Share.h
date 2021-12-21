#ifndef TEST_TEST24_SHARE_H
#define TEST_TEST24_SHARE_H
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/VariableMap.h>
#include <RTLib/math/VectorFunction.h>
enum Test24EventFlagBits : unsigned int
{
	TEST24_EVENT_FLAG_BIT_NONE = 0,
	TEST24_EVENT_FLAG_BIT_FLUSH_FRAME = 1,
	TEST24_EVENT_FLAG_BIT_RESIZE_FRAME = 2,
	TEST24_EVENT_FLAG_BIT_CHANGE_TRACE = 4,
	TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA = 8,
	TEST24_EVENT_FLAG_BIT_UPDATE_LIGHT = 16,
	TEST24_EVENT_FLAG_BIT_LOCK = 32,
};
enum Test24EventFlag : unsigned int
{
	TEST24_EVENT_FLAG_NONE = TEST24_EVENT_FLAG_BIT_NONE,
	TEST24_EVENT_FLAG_FLUSH_FRAME = TEST24_EVENT_FLAG_BIT_FLUSH_FRAME,
	TEST24_EVENT_FLAG_RESIZE_FRAME = TEST24_EVENT_FLAG_BIT_FLUSH_FRAME | TEST24_EVENT_FLAG_BIT_RESIZE_FRAME | TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA,
	TEST24_EVENT_FLAG_CHANGE_TRACE = TEST24_EVENT_FLAG_BIT_FLUSH_FRAME | TEST24_EVENT_FLAG_BIT_CHANGE_TRACE,
	TEST24_EVENT_FLAG_UPDATE_CAMERA = TEST24_EVENT_FLAG_BIT_FLUSH_FRAME | TEST24_EVENT_FLAG_BIT_UPDATE_CAMERA,
	TEST24_EVENT_FLAG_UPDATE_LIGHT = TEST24_EVENT_FLAG_BIT_FLUSH_FRAME | TEST24_EVENT_FLAG_BIT_UPDATE_LIGHT,
	TEST24_EVENT_FLAG_LOCK = TEST24_EVENT_FLAG_BIT_LOCK,

};
namespace test24 {
	auto SpecifyMaterialType(const rtlib::ext::VariableMap& material)->std::string;
	auto ChooseNEE(const rtlib::ext::MeshPtr& mesh)->bool;
}
#endif