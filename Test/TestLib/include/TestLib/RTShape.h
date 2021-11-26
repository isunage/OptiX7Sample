#ifndef TEST_RT_SHAPE_H
#define TEST_RT_SHAPE_H
#include <TestLib/RTInterface.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
namespace test {
    class RTProperties;
    class RTShape :public RTInterface
    {
    public:
        RTShape()noexcept :RTInterface() {}
        virtual ~RTShape()noexcept {}
    };
    using RTShapePtr = std::shared_ptr<RTShape>;
    class RTShapeReader
    {
        auto LoadJsonFromString(const std::string& jsonStr)noexcept -> RTShapePtr {
            return LoadJsonFromData(nlohmann::json::parse(jsonStr));
        }
        virtual auto LoadJsonFromData(const nlohmann::json& json)noexcept -> RTShapePtr = 0;
        virtual ~RTShapeReader()noexcept {}
    };
    using RTShapeReaderPtr = std::shared_ptr<RTShapeReader>;
}
#endif