#ifndef TEST_RT_PROPERTIES_H
#define TEST_RT_PROPERTIES_H
#include <TestLib/RTSerializable.h>
#include <RTLib/ext/Math/Matrix.h>
#include <RTTexture.h>
#include <unordered_map>
#include <string>
namespace test
{
    using RTFloat = float;
    using RTColor = float3;
    using RTMatrix= rtlib::Matrix4x4;
    using RTString= std::string;
    class RTProperties : public RTSerializable
    {
    public:
        virtual ~RTProperties(){}
    private:
        std::unordered_map<std::string,RTFloat>      m_FloatValues   = {};
        std::unordered_map<std::string,RTColor>      m_ColorValues   = {};
        std::unordered_map<std::string,RTMatrix>     m_MatrixValues  = {};
        std::unordered_map<std::string,RTString>     m_StringValues  = {};
        std::unordered_map<std::string,RTTexturePtr> m_TextureValues = {};
    };
}
#endif
