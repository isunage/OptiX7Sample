#ifndef RTLIB_EXT_MATERIAL_H
#define RTLIB_EXT_MATERIAL_H
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
namespace rtlib{
    namespace ext 
    {
        class Material
        {
        public:
            void SetUInt32(const std::string& keyName, const uint32_t value)noexcept{ m_UInt32Data[keyName] = value;}
            void SetFloat1(const std::string& keyName, const float value)noexcept { m_Float1Data[keyName] = value; }
            void SetFloat2(const std::string& keyName, const std::array<float,2>& value)noexcept { m_Float2Data[keyName] = value; }
            void SetFloat3(const std::string& keyName, const std::array<float,3>& value)noexcept { m_Float3Data[keyName] = value; }
            void SetFloat4(const std::string& keyName, const std::array<float,4>& value)noexcept { m_Float4Data[keyName] = value; }
            void SetString(const std::string& keyName, const std::string& value)noexcept { m_StringData[keyName] = value; }
            auto GetUInt32(const std::string& keyName)const -> uint32_t { return m_UInt32Data.at(keyName); }
            auto GetFloat1(const std::string& keyName)const -> float { return m_Float1Data.at(keyName); }
            auto GetFloat2(const std::string& keyName)const -> std::array<float,2> { return m_Float2Data.at(keyName);}
            auto GetFloat3(const std::string& keyName)const -> std::array<float,3> { return m_Float3Data.at(keyName);}
            auto GetFloat4(const std::string& keyName)const -> std::array<float,4> { return m_Float4Data.at(keyName);}
            template<typename T>
            auto GetFloat1As(const std::string& keyName)const -> T { 
                static_assert(sizeof(T) == sizeof(float));
                T ans; 
                std::memcpy(&ans, m_Float1Data.at(keyName).data(), sizeof(float)); 
                return ans; 
            }
            template<typename T>
            auto GetFloat2As(const std::string& keyName)const -> T {
                static_assert(sizeof(T) == sizeof(float)*2);
                T ans;
                std::memcpy(&ans, m_Float2Data.at(keyName).data(), sizeof(float)*2);
                return ans;
            }
            template<typename T>
            auto GetFloat3As(const std::string& keyName)const -> T {
                static_assert(sizeof(T) == sizeof(float)*3);
                T ans;
                std::memcpy(&ans, m_Float3Data.at(keyName).data(), sizeof(float)*3);
                return ans;
            }
            template<typename T>
            auto GetFloat4As(const std::string& keyName)const -> T {
                static_assert(sizeof(T) == sizeof(float) * 4);
                T ans;
                std::memcpy(&ans, m_Float4Data.at(keyName).data(), sizeof(float) * 4);
                return ans;
            }
            auto GetString(const std::string& keyName)const -> std::string { return m_StringData.at(keyName);}
        private:
            std::unordered_map<std::string, uint32_t>             m_UInt32Data;
            std::unordered_map<std::string, float>                m_Float1Data;
            std::unordered_map<std::string, std::array<float,2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float,3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float,4>>  m_Float4Data;
            std::unordered_map<std::string, std::string>          m_StringData;
        };
        using MaterialList    = std::vector<Material>;
        using MaterialListPtr = std::shared_ptr<MaterialList>;
    }
}
#endif