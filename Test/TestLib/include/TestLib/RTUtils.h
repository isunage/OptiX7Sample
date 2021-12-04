#ifndef TEST_RT_UTILS_H
#define TEST_RT_UTILS_H
#include <string>
#include <memory>
namespace test
{
    auto LoadImgFromPathAsUcharData(const std::string& filePath,       int& width,       int& height,       int& comp, int requestComp)->std::unique_ptr<unsigned char[]>;
    auto LoadImgFromPathAsFloatData(const std::string& filePath,       int& width,       int& height,       int& comp, int requestComp)->std::unique_ptr<float[]>;
}
#endif