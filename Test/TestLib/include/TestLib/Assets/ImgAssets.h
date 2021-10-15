#ifndef RT_ASSETS_IMG_ASSETS_H
#define RT_ASSETS_IMG_ASSETS_H
#include "../RTAssets.h"
namespace test
{
    namespace assets
    {
        class ImgAsset : public test::RTAsset
        {
        public:
            // RTAsset ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
            virtual bool Load() override;
            virtual void Free() override;
            virtual bool IsValid() const override;
            virtual ~ImgAsset() {}
        };
    }
}
#endif