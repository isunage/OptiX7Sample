#ifndef RT_ASSETS_IMG_ASSETS_H
#define RT_ASSETS_IMG_ASSETS_H
#include "../RTAssets.h"
#include <RTLib/ext/Resources.h>
#include <memory>
namespace test
{
    namespace assets
    {
        class ImgAsset : public test::RTAsset
        {
        public:
            static auto As(std::shared_ptr<test::RTAsset> ptr)->std::shared_ptr<ImgAsset>
            {
                return std::static_pointer_cast<ImgAsset, test::RTAsset>(ptr);
            }
            static auto New() -> std::shared_ptr<test::assets::ImgAsset> {
                return std::make_shared<test::assets::ImgAsset>();
            }
            // RTAsset ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
            virtual bool Load(const rtlib::ext::VariableMap& params) override;
            virtual void Free() override;
            virtual bool IsValid() const override;
            // Image2D
            auto GetImage2D()const -> const rtlib::ext::CustomImage2D<uchar4>&
            {
                return m_Image;
            }
            auto GetImage2D()      ->       rtlib::ext::CustomImage2D<uchar4>&
            {
                return m_Image;
            }
            virtual ~ImgAsset() {
                try
                {
                    this->Free();
                }
                catch (...)
                {

                }
            }
        private:
            bool                              m_Valid = false;
            rtlib::ext::CustomImage2D<uchar4> m_Image = {};
        };
        using ImgAssetPtr        = std::shared_ptr<ImgAsset>;
        class ImgAssetManager : public RTAssetManager
        {
        public:
            static auto New() -> std::shared_ptr<test::assets::ImgAssetManager> {
                return std::make_shared<test::assets::ImgAssetManager>();
            }
            // Load
            virtual bool LoadAsset(const std::string& key, const rtlib::ext::VariableMap& params) override;
            // Free
            virtual void FreeAsset(const std::string& key) override;
            virtual ~ImgAssetManager() {
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
        using ImgAssetManagerPtr = std::shared_ptr<ImgAssetManager>;
    }
}
#endif