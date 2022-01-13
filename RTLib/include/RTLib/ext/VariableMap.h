#ifndef RTLIB_EXT_VARIABLE_MAP_H
#define RTLIB_EXT_VARIABLE_MAP_H
#include <memory>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET(Name) void Set##Name(const std::string& keyName, const Internal##Name& value)noexcept { m_##Name##Data[keyName] = value; }
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(CNT) template<typename T> void SetFloat##CNT##From(const std::string& keyName, const T& value)noexcept { \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle; \
    std::memcpy(&middle, &value, sizeof(float)*CNT);\
    return SetFloat##CNT(keyName,middle); \
}
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Name) auto Get##Name(const std::string& keyName)const -> Internal##Name { return m_##Name##Data.at(keyName); }
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(CNT) template<typename T> auto GetFloat##CNT##As(const std::string& keyName)const noexcept -> T{ \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle = GetFloat##CNT(keyName); \
    T value {}; \
    std::memcpy(&value, &middle, sizeof(float)*CNT);\
    return value; \
}
#define RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Name) auto Pop##Name(const std::string& keyName)noexcept -> Internal##Name { \
    auto val = Get##Name(keyName); \
    m_##Name##Data.erase(keyName); \
    return val;\
}
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
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(4);
            //Get
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET(String);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(4);
            //Pop
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_POP(String);
            //Has
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(UInt32);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Bool);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float1);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float2);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float3);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(Float4);
            RTLIB_EXT_VARIABLE_MAP_METHOD_DECLARE_HAS(String);
        private:
            std::unordered_map<std::string, uint32_t>             m_UInt32Data;
            std::unordered_map<std::string, bool>                 m_BoolData;
            std::unordered_map<std::string, float>                m_Float1Data;
            std::unordered_map<std::string, std::array<float,2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float,3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float,4>>  m_Float4Data;
            std::unordered_map<std::string, std::string>          m_StringData;
        };
        using VariableMapList    = std::vector<VariableMap>;
        using VariableMapListPtr = std::shared_ptr<VariableMapList>;
    }
}
#endif