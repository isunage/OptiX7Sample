#ifndef RTLIB_EXT_VARIABLE_MAP_H
#define RTLIB_EXT_VARIABLE_MAP_H
#include <memory>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Name) void Set##Name(const std::string& keyName, const Internal##Name& value)noexcept { m_##Name##Data[keyName] = value; }
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Name) auto Get##Name(const std::string& keyName)const -> Internal##Name { return m_##Name##Data.at(keyName); }
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Name) bool Has##Name(const std::string& keyName)const noexcept{ return m_##Name##Data.count(keyName) > 0; }

namespace rtlib{
    namespace ext 
    {
        class VariableMap
        {
        private:
            using InternalUInt32 = uint32_t;
            using InternalBool   = bool;
            using InternalFloat1 = float;
            using InternalFloat2 = std::array<float, 2>;
            using InternalFloat3 = std::array<float, 3>;
            using InternalFloat4 = std::array<float, 4>;
            using InternalMat4x4 = std::array<float,16>;
            using InternalMat4x3 = std::array<float,12>;
            //For String
            using InternalString = std::string;
        public:
            
            //Set
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(String);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Mat4x4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Mat4x3);
            //Get
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(String);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Mat4x4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Mat4x3);
            //Has
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(String);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Mat4x4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Mat4x3);
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
        private:
            std::unordered_map<std::string, uint32_t>             m_UInt32Data;
            std::unordered_map<std::string, bool>                 m_BoolData;
            std::unordered_map<std::string, float>                m_Float1Data;
            std::unordered_map<std::string, std::array<float,2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float,3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float,4>>  m_Float4Data;
            std::unordered_map<std::string, std::array<float,12>> m_Mat4x3Data;
            std::unordered_map<std::string, std::array<float,16>> m_Mat4x4Data;
            std::unordered_map<std::string, std::string>          m_StringData;
        };
        using VariableMapList    = std::vector<VariableMap>;
        using VariableMapListPtr = std::shared_ptr<VariableMapList>;
    }
}
#endif