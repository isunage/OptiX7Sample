#ifndef TEST_RT_TEXTURE_H
#define TEST_RT_TEXTURE_H
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test {
    class RTProperties;
    class RTTexture:public RTInterface
    {
    public:
        RTTexture()noexcept :RTInterface() {}
        virtual auto GetID()const noexcept -> std::string = 0;
        virtual void SetID(const std::string&)noexcept = 0;
        virtual ~RTTexture()noexcept       {}
    };
    using RTTexturePtr    = std::shared_ptr<RTTexture>;
    class RTTextureReader 
    {
    public:
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTTexturePtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto GetPluginName()const noexcept -> std::string = 0;
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr = 0;
        virtual ~RTTextureReader()noexcept {}
    };
    using RTTextureReaderPtr = std::shared_ptr<RTTextureReader>;
    class RTTextureCache
    {
    public:
        RTTextureCache()noexcept;
        bool AddTexture(const RTTexturePtr& texture)noexcept;
        bool HasTexture(const std::string & id)const noexcept;
        auto GetTexture(const std::string & id)const->RTTexturePtr;
        bool AddReader( const RTTextureReaderPtr & reader)noexcept;
        bool HasReader( const std::string & id)const noexcept;
        auto GetReader( const std::string & id)const->RTTextureReaderPtr;
        auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTTexturePtr;
        ~RTTextureCache()noexcept;
    private:
        std::unordered_map<std::string, RTTexturePtr>       m_Textures;
        std::unordered_map<std::string, RTTextureReaderPtr> m_Readers;
    };
    auto GetDefaultTextureCache() noexcept -> std::shared_ptr<RTTextureCache>;
}
#endif
