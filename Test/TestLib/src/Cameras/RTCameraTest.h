#ifndef TEST_RT_CAMERA_TEST_H
#define TEST_RT_CAMERA_TEST_H
#include <filesystem>
#include <string>
namespace test {
	inline auto GetCameraTestBaseDir() noexcept -> std::filesystem::path
	{
		return std::filesystem::path(__FILE__).parent_path();
	}
}
#endif
