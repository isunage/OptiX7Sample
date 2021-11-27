#ifndef TEST_RT_INTERFACE_H
#define TEST_RT_INTERFACE_H
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test{
    class RTProperties;
    class RTInterface
    {
    public:
        RTInterface()noexcept {}
        auto GetJsonAsString()const noexcept -> std::string {
            return GetJsonAsData().dump();
        }
        virtual auto GetTypeName()  const noexcept -> std::string = 0;
        virtual auto GetPluginName()const noexcept -> std::string = 0;
        virtual auto GetProperties()const noexcept -> const RTProperties   & = 0;
        virtual auto GetJsonAsData()const noexcept ->       nlohmann::json   = 0;
        virtual ~RTInterface()noexcept{}
    };
}
#endif
