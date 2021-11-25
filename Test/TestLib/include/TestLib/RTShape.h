#ifndef TEST_RT_SHAPE_H
#define TEST_RT_SHAPE_H
#include <TestLib/RTSerializable.h>
#include <RTLib/ext/Mesh.h>
#include <RTLib/ext/Math/Matrix.h>
#include <RTLib/ext/VariableMap.h>
#include <TestLib/RTMaterial.h>
#include <string>
#include <memory>
namespace test
{
	class RTShape : public RTSerializable
	{
	public:
		RTShape()noexcept :RTSerializable(){}
		virtual ~RTShape()noexcept {}
	};
	using RTShapePtr = std::shared_ptr<RTShape>;
	class RTShapeReader: public RTSerializableReader
	{
	public:
		RTShapeReader()noexcept:RTSerializableReader() {}
		virtual ~RTShapeReader() {}
	};
	using RTShapeReaderPtr = std::shared_ptr<RTShapeReader>;
}
#endif