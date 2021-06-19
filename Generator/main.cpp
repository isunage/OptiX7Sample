#include <iostream>
#include <string_view>
#include <string>
enum UtilTypeFlags {
    UTIL_TYPE_FLAG_SIGNED_CHAR = 0,
    UTIL_TYPE_FLAG_SHORT,
    UTIL_TYPE_FLAG_INT,
    UTIL_TYPE_FLAG_LONG,
    UTIL_TYPE_FLAG_LONG_LONG,
    UTIL_TYPE_FLAG_UNSIGNED_CHAR,
    UTIL_TYPE_FLAG_UNSIGNED_SHORT,
    UTIL_TYPE_FLAG_UNSIGNED_INT,
    UTIL_TYPE_FLAG_UNSIGNED_LONG,
    UTIL_TYPE_FLAG_UNSIGNED_LONG_LONG,
    UTIL_TYPE_FLAG_FLOAT
};
struct    TypeSymbol {
    const char* m_name;
    const char* m_base_name;
    size_t      m_Dimension;
public:
    constexpr        TypeSymbol(const char* name): m_name{ name }, m_base_name{ name }, m_Dimension{ 0 }{}
    constexpr        TypeSymbol(const char* name, const char* base_name) : m_name{ name }, m_base_name{ base_name }, m_Dimension{ 0 }{}
    TypeSymbol       GetSymbolWith(size_t newDimension)const noexcept{
        TypeSymbol symbol = *this;
        symbol.m_Dimension = newDimension;
        return symbol;
    }
    constexpr size_t GetDimension()const noexcept { 
        return m_Dimension;  
    }

    std::string      GetTypeNameWith(size_t i)const noexcept{
        if (i==0) {
            return m_name;
        }
        return std::string(m_base_name) + std::to_string(i);
    }
    std::string      GetTypeName()const noexcept {
        return GetTypeNameWith(m_Dimension);
    }

    std::string      GetConstTypeNameWith(size_t i)const noexcept{
        if (i == 0) {
            return std::string("const ") + GetTypeNameWith(i);
        }
        else {
            return std::string("const ") + GetTypeNameWith(i) + "&";
        }
    }
    std::string      GetConstTypeName()const noexcept {
        return GetConstTypeNameWith(m_Dimension);
    }

    std::string      GetAccessorWith(const std::string& var_name,size_t dimension,size_t i)const noexcept {
        if (dimension == 0) {
            return var_name;
        }
        else {
            if (i == 0) {
                return var_name+".x";
            }
            if (i == 1){
                return var_name+".y";
            }
            if (i == 2) {
                return var_name+".z";
            }
            if (i == 3) {
                return var_name+".w";
            }
            return var_name;
        }
    }
    std::string      GetAccessor(const std::string& var_name, size_t i)const noexcept {
        return GetAccessorWith(var_name,m_Dimension, i);
    }

    std::string      GetAccessorsWith(const std::string& var_name, size_t dimension, size_t count,const std::string & linker = " ,")const noexcept {
        std::string ans;
        if (count == 0) {
            return "";
        }
        for (size_t j = 0; j < count -1; ++j) {
            ans += (GetAccessorWith(var_name, dimension, j) + linker);
        }
        ans += GetAccessorWith(var_name, dimension, count - 1);
        return ans;
    }
    std::string      GetAccessors(    const std::string& var_name, size_t count, const std::string& linker = " ,")const noexcept {
        return GetAccessorsWith(var_name, m_Dimension, count, linker);
    }

    std::string      GetAccessorOperator2With(const std::string& var_name0, const std::string& var_name1, const std::string& operator_2, size_t dimension0, size_t dimension1, size_t i, bool useBracket = true)const noexcept{
        std::string ans = GetAccessorWith(var_name0, dimension0, i) + operator_2 + GetAccessorWith(var_name1, dimension1, i);
        return useBracket ? (std::string("(") + ans + std::string(")")) : ans;
    }
    std::string      GetAccessorOperator2With(const std::string& var_name0, const std::string& var_name1, const std::string& operator_2, size_t dimension, size_t i, bool useBracket = true) const noexcept{
        return GetAccessorOperator2With(var_name0, var_name1, operator_2, dimension, dimension, i, useBracket);
    }
    std::string      GetAccessorOperator2(    const std::string& var_name0, const std::string& var_name1, const std::string& operator_2, size_t i, bool useBracket = true) const noexcept {
        return GetAccessorOperator2With(var_name0, var_name1, operator_2, m_Dimension, i, useBracket);
    }

    std::string      GetAccessorOperators2With(const std::string& var_name0, const std::string& var_name1,const std::string& operator_2, size_t dimension0,size_t dimension1,size_t count, const std::string& linker = " ,", bool useBracket = true) const noexcept {
        std::string ans;
        if (count == 0) {
            return "";
        }
        for (size_t j = 0; j < count - 1; ++j) {
            ans += (GetAccessorOperator2With(var_name0, var_name1, operator_2, dimension0,dimension1, j, useBracket) + linker);
        }
        ans += GetAccessorOperator2With(var_name0, var_name1, operator_2, dimension0, dimension1, count - 1, useBracket);
        return ans;
    }
    std::string      GetAccessorOperators2With(const std::string& var_name0, const std::string& var_name1,const std::string& operator_2, size_t dimension, size_t count, const std::string& linker = " ,", bool useBracket = true) const noexcept {
        return GetAccessorOperators2With(var_name0, var_name1, operator_2, dimension, dimension, count, linker, useBracket);
    }
    std::string      GetAccessorOperators2(    const std::string& var_name0, const std::string& var_name1,const std::string& operator_2, size_t count, const std::string& linker = " ,", bool useBracket = true) const noexcept {
        return GetAccessorOperators2With(var_name0, var_name1, operator_2, m_Dimension, count, linker, useBracket);
    }


    std::string      GetAccessorFunction1With(const std::string& var_name, const std::string& func_name, size_t dimension, size_t i)const noexcept {
        return func_name + "(" + GetAccessorWith(var_name, dimension, i) + ")";
    }
    std::string      GetAccessorFunction1(    const std::string& var_name, const std::string& func_name, size_t i)const noexcept {
        return GetAccessorFunction1With(var_name, func_name, m_Dimension, i);
    }

    std::string      GetAccessorFunctions1With(const std::string& var_name, const std::string& func_name, size_t dimension, size_t count, const std::string& linker = " ,")const noexcept {
        std::string ans;
        if (count == 0) {
            return "";
        }
        for (size_t j = 0; j < count - 1; ++j) {
            ans += (GetAccessorFunction1With(var_name, func_name, dimension, j) + linker);
        }
        ans += GetAccessorFunction1With(var_name, func_name, dimension, count - 1);
        return ans;
    }
    std::string      GetAccessorFunctions1(    const std::string& var_name, const std::string& func_name, size_t count, const std::string& linker = " ,")const noexcept {
        return GetAccessorFunctions1With(var_name, func_name, m_Dimension, count, linker);
    }

    std::string      GetAccessorFunction2With(const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t dimension0, size_t dimension1, size_t i)const noexcept {
        return func_name + "( " + GetAccessorWith(var_name0, dimension0, i) + " ," + GetAccessorWith(var_name1, dimension1, i) + " )";
    }
    std::string      GetAccessorFunction2With(const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t dimension, size_t i)const noexcept {
        return GetAccessorFunction2With(var_name0, var_name1, func_name, dimension, dimension, i);
    }
    std::string      GetAccessorFunction2(    const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t i)const noexcept {
        return GetAccessorFunction2With(var_name0, var_name1, func_name, m_Dimension, i);
    }

    std::string      GetAccessorFunctions2With(const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t dimension0, size_t dimension1, size_t count, const std::string& linker = " ,") const noexcept {
        std::string ans;
        if (count == 0) {
            return "";
        }
        for (size_t j = 0; j < count - 1; ++j) {
            ans += (GetAccessorFunction2With(var_name0, var_name1, func_name, dimension0, dimension1, j) + linker);
        }
        ans += GetAccessorFunction2With(var_name0, var_name1, func_name, dimension0, dimension1, count - 1);
        return ans;
    }
    std::string      GetAccessorFunctions2With(const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t dimension, size_t count, const std::string& linker = " ,") const noexcept {
        return GetAccessorFunctions2With(var_name0, var_name1, func_name, dimension, dimension, count, linker);
    }
    std::string      GetAccessorFunctions2(    const std::string& var_name0, const std::string& var_name1, const std::string& func_name, size_t count, const std::string& linker = " ,") const noexcept {
        return GetAccessorFunctions2With(var_name0, var_name1, func_name, m_Dimension, count, linker);
    }

    std::string      GetAccessorFunction3With(const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t dimension0, size_t dimension1, size_t dimension2, size_t i)const noexcept {
        return func_name + "( " + GetAccessorWith(var_name0, dimension0, i) + " ," + GetAccessorWith(var_name1, dimension1, i) + " ," + GetAccessorWith(var_name2, dimension2, i) + " )";
    }
    std::string      GetAccessorFunction3With(const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t dimension, size_t i)const noexcept {
        return GetAccessorFunction3With(var_name0, var_name1, var_name2, func_name, dimension, dimension, dimension, i);
    }
    std::string      GetAccessorFunction3(    const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t i)const noexcept {
        return GetAccessorFunction3With(var_name0, var_name1, var_name2, func_name, m_Dimension, i);
    }

    std::string      GetAccessorFunctions3With(const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t dimension0, size_t dimension1, size_t dimension2, size_t count, const std::string& linker = " ,") const noexcept {
        std::string ans;
        if (count == 0) {
            return "";
        }
        for (size_t j = 0; j < count - 1; ++j) {
            ans += (GetAccessorFunction3With(var_name0, var_name1, var_name2, func_name, dimension0, dimension1, dimension2, j) + linker);
        }
        ans += GetAccessorFunction3With(var_name0, var_name1, var_name2, func_name, dimension0, dimension1, dimension2, count - 1);
        return ans;
    }
    std::string      GetAccessorFunctions3With(const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t dimension, size_t count, const std::string& linker = " ,") const noexcept {
        return GetAccessorFunctions3With(var_name0, var_name1, var_name2, func_name, dimension, dimension, dimension, count, linker);
    }
    std::string      GetAccessorFunctions3(    const std::string& var_name0, const std::string& var_name1, const std::string& var_name2, const std::string& func_name, size_t count, const std::string& linker = " ,") const noexcept {
        return GetAccessorFunctions3With(var_name0, var_name1, var_name2, func_name, m_Dimension, count, linker);
    }

};
constexpr TypeSymbol typeSymbols[] = {
    TypeSymbol("signed char","char")   ,TypeSymbol{"short"},TypeSymbol{"int"},TypeSymbol{"long"},TypeSymbol{"long long","longlong"},
    TypeSymbol("unsigned char","uchar"),TypeSymbol{"unsigned short","ushort"},TypeSymbol{"unsigned int","uint"},
    TypeSymbol{"unsigned long","ulong"},TypeSymbol{"unsigned long long","ulonglong"},
    TypeSymbol{"float"},
};

int main_test(){
    ///Core
    //operator overload
    for (const auto& typeSymbol : typeSymbols) {
        for (size_t d = 2; d <= 4; ++d) {
            auto base_symbol = typeSymbol.GetSymbolWith(d);
            //operator==
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  return " << base_symbol.GetAccessorOperators2("v0", "v1", "==", d, "&&") << ";\n";
            std::cout << "}\n";
            //operator!=
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  return " << base_symbol.GetAccessorOperators2("v0", "v1", "!=", d, "||") << ";\n";
            std::cout << "}\n";
            //operator+=
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator+=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "+=", d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            //operator-=
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator-=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "-=", d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            //operator*=
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator*=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "*=", d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator*=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "*=", d, 1, d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            //operator/=
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator/=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "/=", d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << "& operator/=( ";
            std::cout << base_symbol.GetTypeName() << "& v0, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v1){\n";
            std::cout << "  ";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "/=", d, 1, d, ";\n  ", false) << ";\n";
            std::cout << "  return v0;\n";
            std::cout << "}\n";
            //operator+
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator+( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "+", d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            //operator-
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator-( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "-", d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            //operator*
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator*( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "*", d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator*( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "*", d, 1, d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator*( ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "*", 1, d, d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            //operator/
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator/( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2("v0", "v1", "/", d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator/( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "/", d, 1, d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE ";
            std::cout << base_symbol.GetTypeName() << " operator/( ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "    return make_" << base_symbol.GetTypeName() << "(";
            std::cout << base_symbol.GetAccessorOperators2With("v0", "v1", "/", 1, d, d, " ,", false);
            std::cout << ");\n";
            std::cout << "}\n";
        }
    }
    //make_vec
    for (const auto& typeSymbol : typeSymbols) {
        for (size_t d = 2; d <= 4; ++d) {
            //make_1
            auto base_symbol = typeSymbol.GetSymbolWith(d);
            for (size_t i = 1; i <= d; ++i) {
                auto temp_symbol = typeSymbol.GetSymbolWith(i);
                std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName();
                std::cout << " make_" << base_symbol.GetTypeName() << " (" << temp_symbol.GetConstTypeName() << " v){\n";
                std::cout << "  return make_" << base_symbol.GetTypeName() << "(";
                if (i == 1) {
                    std::cout << temp_symbol.GetAccessors("v", d) << ");\n";
                }
                else {
                    std::cout << temp_symbol.GetAccessors("v", i);
                    if (i != d) {
                        std::cout << " ," << typeSymbol.GetAccessorsWith("0.0f", 0, d - i) << ");\n";
                    }
                    else {
                        std::cout << ");\n";
                    }
                }
                std::cout << "}\n";
            }
            //make_2
            for (size_t i = 1; i <= d - 1; ++i) {
                if (i == 1 && i == d - 1) {
                    continue;
                }
                auto temp1_symbol = typeSymbol.GetSymbolWith(i);
                auto temp2_symbol = typeSymbol.GetSymbolWith(d - i);
                std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName();
                std::cout << " make_" << base_symbol.GetTypeName();
                std::cout << " (" << temp1_symbol.GetConstTypeName() << " v0, ";
                std::cout << temp2_symbol.GetConstTypeName() << " v1){\n";
                std::cout << "  return make_" << base_symbol.GetTypeName() << "(";
                std::cout << temp1_symbol.GetAccessors("v0", i);
                std::cout << " ," << temp2_symbol.GetAccessors("v1", d - i) << ");\n";
                std::cout << "}\n";
            }
            //make_3
            for (size_t i = 1; i <= d - 2; ++i) {
                auto temp1_symbol = typeSymbol.GetSymbolWith(i);
                for (size_t j = 1; j <= d - 1 - i; ++j) {
                    auto temp2_symbol = typeSymbol.GetSymbolWith(j);
                    if (i == 1 && j == 1 && (i + j) == d - 1) {
                        continue;
                    }
                    auto temp3_symbol = typeSymbol.GetSymbolWith(d - i - j);
                    std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName();
                    std::cout << " make_" << base_symbol.GetTypeName();
                    std::cout << " (" << temp1_symbol.GetConstTypeName() << " v0, ";
                    std::cout << temp2_symbol.GetConstTypeName() << " v1, ";
                    std::cout << temp3_symbol.GetConstTypeName() << " v2){\n";
                    std::cout << "  return make_" << base_symbol.GetTypeName() << "(";
                    std::cout << temp1_symbol.GetAccessors("v0", i);
                    std::cout << " ," << temp2_symbol.GetAccessors("v1", j);
                    std::cout << " ," << temp3_symbol.GetAccessors("v2", d - i - j) << ");\n";
                    std::cout << "}\n";
                }
            }

        }
    }
    {
        size_t zip_index = 0;
        for (const auto& typeSymbol : typeSymbols) {
            if (zip_index == UTIL_TYPE_FLAG_FLOAT) {
                break;
            }
            for (size_t d = 2; d <= 4; ++d) {
                //make_1
                auto float_symbol = typeSymbols[UTIL_TYPE_FLAG_FLOAT].GetSymbolWith(d);
                for (size_t i = 1; i <= d; ++i) {
                    auto temp_symbol = typeSymbol.GetSymbolWith(i);
                    std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << float_symbol.GetTypeName();
                    std::cout << " make_" << float_symbol.GetTypeName() << " (" << temp_symbol.GetConstTypeName() << " v){\n";
                    std::cout << "  return make_" << float_symbol.GetTypeName() << "(";
                    if (i == 1) {
                        std::cout << temp_symbol.GetAccessorFunctions1("v","static_cast<float>", d) << ");\n";
                    }
                    else {
                        std::cout << temp_symbol.GetAccessorFunctions1("v", "static_cast<float>", i);
                        if (i != d) {
                            std::cout << " ," << typeSymbol.GetAccessorsWith("0.0f", 0, d - i) << ");\n";
                        }
                        else {
                            std::cout << ");\n";
                        }
                    }
                    std::cout << "}\n";
                }
            }
            zip_index++;
        }
    }
    std::cout << "\nnamespace rtlib{\n";
    ///UTIL
    for (const auto& typeSymbol : typeSymbols) {
        for (size_t d = 2; d <= 4; ++d) {
            auto base_symbol = typeSymbol.GetSymbolWith(d);
            std::cout << "//max\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " max( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::max;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions2("v0", "v1", "max", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//min\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " min( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::min;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions2("v0", "v1", "min", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//mix\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " mix( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1, ";
            std::cout << base_symbol.GetConstTypeName() << " a){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions3("v0", "v1", "a", "mix", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " mix( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " a){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions3With("v0", "v1", "a", "mix", d, d, 1, d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//clamp\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " clamp( ";
            std::cout << base_symbol.GetConstTypeName() << " v, ";
            std::cout << base_symbol.GetConstTypeName() << " low, ";
            std::cout << base_symbol.GetConstTypeName() << " high){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions3("v", "low", "high", "clamp", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//powN\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow2( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "pow2", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow3( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "pow3", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow4( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "pow4", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow5( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "pow5", d, " ,") << ");\n";
            std::cout << "}\n";
        }
    }
    {
        //float only function
        const auto& typeSymbol = typeSymbols[UTIL_TYPE_FLAG_FLOAT];
        //dot,length(Sqr), distance(Sqr),
        std::cout << "//cross\n";
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float3 cross( const float3& v0, const float3& v1){\n";
        std::cout << "  return make_float3( v0.y*v1.z-v0.z*v1.y, v0.z*v1.x-v0.x*v1.z, v0.x*v1.y-v0.y*v1.x );\n";
        std::cout << "}\n";
        for (size_t d = 2; d <= 4; ++d) {
            auto base_symbol = typeSymbol.GetSymbolWith(d);
            std::cout << "//dot\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float dot( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  return " << base_symbol.GetAccessorOperators2("v0","v1","*",d,"+",true) << ";\n";
            std::cout << "}\n";
            std::cout << "//lengthSqr\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float lengthSqr( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return " << base_symbol.GetAccessorOperators2("v", "v", "*", d, "+", true) << ";\n";
            std::cout << "}\n";
            std::cout << "//length\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float length( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return sqrtf(lengthSqr(v));\n";
            std::cout << "}\n";
            std::cout << "//distanceSqr\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float distanceSqr(";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  return lengthSqr(v0-v1);\n";
            std::cout << "}\n";
            std::cout << "//distance\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE float distance(";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "  return length(v0-v1);\n";
            std::cout << "}\n";
            std::cout << "//normalize\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " normalize( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "  return v/length(v);\n";
            std::cout << "}\n";
            std::cout << "//expf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " expf( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::expf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "::expf", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//logf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " logf( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::logf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "::logf", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//sinf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " sinf( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::sinf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "::sinf", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//cosf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " cosf( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::cosf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "::cosf", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//tanf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " tanf( ";
            std::cout << base_symbol.GetConstTypeName() << " v){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::tanf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions1("v", "::tanf", d, " ,") << ");\n";
            std::cout << "}\n";
            std::cout << "//powf\n";
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " powf( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeNameWith(1) << " v1){\n";
            std::cout << "#if defined(__cplusplus) && !defined(__CUDA_ARCH__)\n";
            std::cout << "  using std::powf;\n";
            std::cout << "#endif\n";
            std::cout << "  return make_" << base_symbol.GetTypeName() << "(" << base_symbol.GetAccessorFunctions2With("v0","v1", "::powf", d,1,d, " ,") << ");\n";
            std::cout << "}\n";
        }
    }
    std::cout << "\n}\n";
    return 0;
}
int main_test2() {
    std::cout << "//max\n";
    {
        size_t i = 0;
        for (const auto& typeSymbol : typeSymbols) {
            auto base_symbol = typeSymbol.GetSymbolWith(1);
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " max( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "#if defined(__CUDA_ARCH__) || !defined(__cplusplus)\n";
            if (i == UTIL_TYPE_FLAG_SIGNED_CHAR || i == UTIL_TYPE_FLAG_UNSIGNED_CHAR) {
                std::cout << "  return v0>=v1?v0:v1;\n";
            }
            else {
                std::cout << "  return ::max(v0, v1);\n";
            }
            std::cout << "#elif defined(__cplusplus)\n";
            std::cout << "  return std::max(v0, v1);\n";
            std::cout << "#else\n";
            std::cout << "  return v0>=v1?v0:v1;\n";
            std::cout << "#endif\n";
            std::cout << "}\n";
            ++i;
        }
    }
    std::cout << "//min\n";
    {
        size_t i = 0;
        for (const auto& typeSymbol : typeSymbols) {
            auto base_symbol = typeSymbol.GetSymbolWith(1);
            std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " min( ";
            std::cout << base_symbol.GetConstTypeName() << " v0, ";
            std::cout << base_symbol.GetConstTypeName() << " v1){\n";
            std::cout << "#if defined(__CUDA_ARCH__) || !defined(__cplusplus)\n";
            if (i == UTIL_TYPE_FLAG_SIGNED_CHAR || i == UTIL_TYPE_FLAG_UNSIGNED_CHAR) {
                std::cout << "  return v0<=v1?v0:v1;\n";
            }
            else {
                std::cout << "  return ::min(v0, v1);\n";
            }
            std::cout << "#elif defined(__cplusplus)\n";
            std::cout << "  return std::min(v0, v1);\n";
            std::cout << "#else\n";
            std::cout << "  return v0<=v1?v0:v1;\n";
            std::cout << "#endif\n";
            std::cout << "}\n";
            ++i;
        }
    }
    std::cout << "//mix\n";
    for (const auto& typeSymbol : typeSymbols) {
        auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " mix( ";
        std::cout << base_symbol.GetConstTypeName() << " v0, ";
        std::cout << base_symbol.GetConstTypeName() << " v1, ";
        std::cout << base_symbol.GetConstTypeName() << " a){\n";
        std::cout << "  return v0*(1-a)+v1*a;\n";
        std::cout << "}\n";
    }
    std::cout << "//clamp\n";
    for (const auto& typeSymbol : typeSymbols) {
        auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " clamp( ";
        std::cout << base_symbol.GetConstTypeName() << " v, ";
        std::cout << base_symbol.GetConstTypeName() << " low, ";
        std::cout << base_symbol.GetConstTypeName() << " high){\n";
        std::cout << "#if defined(__CUDA_ARCH__)\n";
        std::cout << "  return min(max(v, low), high);\n";
        std::cout << "#elif defined(__cplusplus)\n";
        std::cout << "  return std::clamp(v,low,high);\n";
        std::cout << "#endif\n";
        std::cout << "}\n";
    }
    std::cout << "//step\n";
    for (const auto& typeSymbol : typeSymbols) {
        auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " step( ";
        std::cout << base_symbol.GetConstTypeName() << " edge, ";
        std::cout << base_symbol.GetConstTypeName() << " x){\n";
        std::cout << "  return x < edge ? 0 : 1;\n";
        std::cout << "}\n";
    }
    std::cout << "//powN\n";
    for (const auto& typeSymbol : typeSymbols) {
        auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow2( ";
        std::cout << base_symbol.GetConstTypeName() << " v){\n";
        std::cout << "  return v*v;\n";
        std::cout << "}\n";
        //auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow3( ";
        std::cout << base_symbol.GetConstTypeName() << " v){\n";
        std::cout << "  return v*v*v;\n";
        std::cout << "}\n";
        //auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow4( ";
        std::cout << base_symbol.GetConstTypeName() << " v){\n";
        std::cout << "  return v*v*v*v;\n";
        std::cout << "}\n";
        //auto base_symbol = typeSymbol.GetSymbolWith(1);
        std::cout << "RTLIB_INLINE RTLIB_HOST_DEVICE " << base_symbol.GetTypeName() << " pow5( ";
        std::cout << base_symbol.GetConstTypeName() << " v){\n";
        std::cout << "  return v*v*v*v*v;\n";
        std::cout << "}\n";
    }

    for (const auto& typeSymbol : typeSymbols) {
    }
    return 0;
}
int main_test3() {
    for (const auto& typeSymbol : typeSymbols) {
        {
            auto baseSymbol = typeSymbol.GetSymbolWith(0);
            std::cout << "template<>\n";
            std::cout << "struct CUDAPixelTraits<" << baseSymbol.GetTypeName() << ">{\n";
            std::cout << "  using base_type = " << typeSymbol.GetTypeName() << ";\n";
            std::cout << "  static inline constexpr size_t numChannels = " << 1 << ";\n";
            std::cout << "};\n";
        }
        for (size_t d = 1; d <= 4; ++d) {
            auto baseSymbol = typeSymbol.GetSymbolWith(d);
            std::cout << "template<>\n";
            std::cout << "struct CUDAPixelTraits<" << baseSymbol.GetTypeName() << ">{\n";
            std::cout << "  using base_type = " << typeSymbol.GetTypeName() << ";\n";
            std::cout << "  static inline constexpr size_t numChannels = " << d << ";\n";
            std::cout << "};\n";
        }
    }
    return 0;
}
int main() {
    return main_test3();
}