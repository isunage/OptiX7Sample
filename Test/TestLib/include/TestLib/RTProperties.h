#ifndef TEST_RT_PROPERTIES_H
#define TEST_RT_PROPERTIES_H
#include <nlohmann/json.hpp>
#include <TestLib/RTTexture.h>
#include <RTLib/ext/Math/Matrix.h>
#include <unordered_map>
#include <string>
#define TEST_RT_PROPERTIES_DECLARE_METHOD(TYPE_NAME,TYPE_BASE) \
    bool Has##TYPE_BASE(const std::string& name)const noexcept { return m_Map##TYPE_BASE.count(name)>0;} \
    auto Get##TYPE_BASE(const std::string& name)const -> const TYPE_NAME& { return m_Map##TYPE_BASE.at(name); } \
    void Set##TYPE_BASE(const std::string& name, const TYPE_NAME& value)noexcept { m_Map##TYPE_BASE[name] = value; }
namespace test
{
    using RTFloat  = float;
    using RTInt32  = int32_t;
    using RTPoint  = float3;
    using RTVector = float3;
    using RTColor  = float3;
    using RTString = std::string;
    using RTMat4x4 = rtlib::Matrix4x4;
    class RTProperties
    {
    public:
        RTProperties()noexcept{}
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTFloat     ,Float  );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTInt32     ,Int32  );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTPoint     ,Point  );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTVector    ,Vector );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTMat4x4    ,Mat4x4 );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTColor     ,Color  );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTString    ,String );
        TEST_RT_PROPERTIES_DECLARE_METHOD(RTTexturePtr,Texture);
        bool LoadFloat  (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadInt32  (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadPoint  (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadVector (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadMat4x4 (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadColor  (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadString (const std::string& keyName, const nlohmann::json& jsonData)noexcept;
        bool LoadTexture(const std::string& keyName, const nlohmann::json& jsonData,std::shared_ptr<RTTextureCache>& cache)noexcept;
        auto GetJsonData()const noexcept -> nlohmann::json;
        ~RTProperties()noexcept{}
    private:
        std::unordered_map<RTString,RTFloat >     m_MapFloat ;
        std::unordered_map<RTString,RTInt32 >     m_MapInt32 ;
        std::unordered_map<RTString,RTPoint >     m_MapPoint ;
        std::unordered_map<RTString,RTVector>     m_MapVector;
        std::unordered_map<RTString,RTMat4x4>     m_MapMat4x4;
        std::unordered_map<RTString,RTColor >     m_MapColor ;
        std::unordered_map<RTString,RTString>     m_MapString;
        std::unordered_map<RTString,RTTexturePtr> m_MapTexture;
    };
}
#undef TEST_RT_PROPERTIES_DECLARE_METHOD
#endif
