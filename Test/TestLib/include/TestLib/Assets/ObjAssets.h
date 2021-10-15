#ifndef RT_ASSETS_OBJ_ASSETS_H
#define RT_ASSETS_OBJ_ASSETS_H
#include "../RTAssets.h"
namespace test
{
    namespace assets
    {
        class ObjAsset : public test::RTAsset
        {
        public:
            // RTAsset ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
            virtual bool Load() override;
            virtual void Free() override;
            virtual bool IsValid() const override;
            virtual ~ObjAsset() {}
        };
    }
}
#endif
