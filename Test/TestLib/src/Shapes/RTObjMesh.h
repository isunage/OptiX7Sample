#ifndef TEST_RT_SHAPE_OBJ_MESH_H
#define TEST_RT_SHAPE_OBJ_MESH_H
#include <TestLib/RTMaterial.h>
#include <TestLib/RTShape.h>
#include <TestLib/RTProperties.h>
#include <variant>
#include <string>
namespace test
{
	class RTObjMeshReader;
	class RTObjMesh       : public RTShape
	{
	public:
		RTObjMesh()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
        virtual auto GetMaterial()   const noexcept -> RTMaterialPtr        override;
		virtual auto GetFlipNormals() const noexcept-> RTBool   override;
		virtual auto GetTransforms() const noexcept -> RTMat4x4 override;
		virtual void SetID(const std::string& id)noexcept override;
        auto GetFilename()    const noexcept -> RTString  ;
        auto GetMeshname()    const noexcept -> RTString  ;
        auto GetSubMeshname() const noexcept -> RTString  ;
        auto SetFilename(const RTString& filename)noexcept;
        auto SetMeshname(const RTString& meshname)noexcept;
        auto SetSubMeshname(const RTString& submeshname)noexcept;
		void SetFlipNormals(const RTBool  & val)noexcept ;
		auto SetTransforms( const RTMat4x4& mat)noexcept ;
		virtual ~RTObjMesh() noexcept;
	private:
		friend class RTObjMeshReader;
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
	class RTObjMtl: public RTMaterial
	{
	public:
		RTObjMtl()noexcept;
		virtual auto GetTypeName()   const noexcept -> RTString override;
		virtual auto GetPluginName() const noexcept -> RTString override;
		virtual auto GetID()         const noexcept -> RTString override;
		virtual auto GetProperties() const noexcept -> const RTProperties & override;
		virtual auto GetJsonAsData() const noexcept ->       nlohmann::json override;
		virtual void SetID(const RTString& id)noexcept;
		auto GetDiffuseReflectance() const noexcept -> RTTexturePtr;
		auto GetSpecularReflectance()const noexcept -> RTTexturePtr;
		auto GetSpecularExponent()   const noexcept -> RTTexturePtr;
		auto GetTransmittance()      const noexcept -> RTColor;
		auto GetIOR()                const noexcept -> RTFloat;
		auto GetIllumMode()          const noexcept -> RTInt32;
		void SetDiffuseReflectance (const RTTexturePtr& texture)noexcept;
		void SetSpecularReflectance(const RTTexturePtr& texture)noexcept;
		void SetSpecularExponent   (const RTTexturePtr& texture)noexcept;
		void SetTransmittance(const RTColor& color)noexcept;
		void SetIOR(const RTFloat& fv)noexcept;
		void SetIllumMode(const RTInt32& iv)noexcept;
		virtual ~RTObjMtl() noexcept;
	private:
		friend class RTPhongReader;
		RTProperties m_Properties;
	};
	class RTObjMeshReader : public RTShapeReader
	{
	public:
		RTObjMeshReader(const std::shared_ptr< RTMaterialCache>& matCache)noexcept;
		virtual auto GetPluginName()const noexcept -> RTString override;
		virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr override;
		virtual ~RTObjMeshReader()noexcept;
	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif