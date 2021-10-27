#ifndef RT_EXCEPTIONS_H
#define RT_EXCEPTIONS_H
#include <string>
#include <stdexcept>
namespace test
{
	namespace exceptions
	{
		class LibraryLoadError:public std::runtime_error
		{
		public:
			explicit LibraryLoadError(const std::string& libraryName) :
				std::runtime_error("LibraryLoadError: " + libraryName + " Failed To Load!") {}
			explicit LibraryLoadError(const char* libraryName) :
				LibraryLoadError(std::string(libraryName)) {}
			virtual ~LibraryLoadError() {}
		};
		class WindowCreateError: public std::runtime_error
		{
		public:
			explicit WindowCreateError(const std::string& windowName) :
				std::runtime_error("WindowCreateError: " + windowName + " Failed To Create!") {}
			explicit WindowCreateError(const char* windowName) :
				WindowCreateError(std::string(windowName)) {}
			virtual ~WindowCreateError() {}
		};
	}
}
#endif