#ifndef TEST_RT_SHAPE_OBJ_MESH_H
#define TEST_RT_SHAPE_OBJ_MESH_H
#include "TestLib/RTShape.h"
#include "TestLib/RTShape.h"
#include <memory>
namespace test
{
	class RTObjMeshReader;
	class RTObjMesh: public RTShape
	{
	public:
		RTObjMesh()noexcept;
		virtual auto GetJsonData()const noexcept -> nlohmann::json override;
		virtual ~RTObjMesh()noexcept;
	private:
		friend class RTObjMeshReader;
		struct Impl;
		std::unique_ptr<RTObjMesh::Impl> m_Impl = nullptr;
	};
	class RTObjMeshReader :public RTShapeReader
	{
	public:
		RTObjMeshReader()noexcept;
		virtual auto LoadJsonData(const nlohmann::json&)const noexcept -> RTShapePtr override;
		virtual ~RTObjMeshReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<RTObjMeshReader::Impl> m_Impl = nullptr;
	};
}
#endif