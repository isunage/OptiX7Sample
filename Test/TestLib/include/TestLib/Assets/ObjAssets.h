#ifndef RT_ASSETS_OBJ_ASSETS_H
#define RT_ASSETS_OBJ_ASSETS_H
#include "../RTAssets.h"
#include <RTLib/ext/Mesh.h>
#include <memory>
namespace test
{
    namespace assets
    {
        class ObjAsset : public test::RTAsset
        {
        public:
            static auto As(std::shared_ptr<test::RTAsset> ptr)->std::shared_ptr<ObjAsset>
            {
                return std::static_pointer_cast<ObjAsset, test::RTAsset>(ptr);
            }
            static auto New() -> std::shared_ptr<test::assets::ObjAsset>{
                return std::make_shared<test::assets::ObjAsset>();
            }
            //Load
            virtual bool Load(const rtlib::ext::VariableMap& params) override;
            //Free
            virtual void Free() override;
            //IsValid
            virtual bool IsValid() const override;
            //MeshGroup
            auto GetMeshGroup()const -> const rtlib::ext::MeshGroupPtr&
            {
                return m_MeshGroup;
            }
            auto GetMeshGroup()      ->       rtlib::ext::MeshGroupPtr&
            {
                return m_MeshGroup;
            }
            //Materials
            auto GetMaterials()const -> const std::vector<rtlib::ext::VariableMap>&
            {
                return m_Materials;
            }
            auto GetMaterials() ->  std::vector<rtlib::ext::VariableMap>&
            {
                return m_Materials;
            }
            //ObjAsset
            virtual ~ObjAsset() {
                try
                {
                    this->Free();
                }
                catch (...)
                {

                }
            }
        private:
            bool                                 m_Valid     = false;
            rtlib::ext::MeshGroupPtr             m_MeshGroup = {};
            std::vector<rtlib::ext::VariableMap> m_Materials = {};
        };
        using ObjAssetPtr = std::shared_ptr<ObjAsset>;
        class ObjAssetManager : public RTAssetManager
        {
        public:
            static auto New() -> std::shared_ptr<test::assets::ObjAssetManager> {
                return std::make_shared<test::assets::ObjAssetManager>();
            }
            // Load
            virtual bool LoadAsset(const std::string& key, const rtlib::ext::VariableMap& params) override;
            // Free
            virtual void FreeAsset(const std::string& key) override;
            virtual ~ObjAssetManager() {
                try
                {
                    for (auto& [name, asset] : GetAssets())
                    {
                        asset->Free();
                    }
                    GetAssets().clear();
                }
                catch (...)
                {

                }
            }
        };
        using ObjAssetManagerPtr = std::shared_ptr<ObjAssetManager>;
    }
}
#endif
