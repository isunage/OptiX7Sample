#ifndef RTLIB_PIXEL_FORMAT_H
#define RTLIB_PIXEL_FORMAT_H
#include <glad/glad.h>
#include <cuda_runtime.h>
namespace rtlib{
    template<typename PixelType>
    struct CUDAPixelTraits;
    template<>
    struct CUDAPixelTraits<signed char>{
    using base_type = signed char;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<char1>{
    using base_type = signed char;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<char2>{
    using base_type = signed char;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<char3>{
    using base_type = signed char;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<char4>{
    using base_type = signed char;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<short>{
    using base_type = short;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<short1>{
    using base_type = short;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<short2>{
    using base_type = short;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<short3>{
    using base_type = short;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<short4>{
    using base_type = short;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<int>{
    using base_type = int;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<int1>{
    using base_type = int;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<int2>{
    using base_type = int;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<int3>{
    using base_type = int;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<int4>{
    using base_type = int;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<long>{
    using base_type = long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<long1>{
    using base_type = long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<long2>{
    using base_type = long;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<long3>{
    using base_type = long;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<long4>{
    using base_type = long;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<long long>{
    using base_type = long long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<longlong1>{
    using base_type = long long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<longlong2>{
    using base_type = long long;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<longlong3>{
    using base_type = long long;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<longlong4>{
    using base_type = long long;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<unsigned char>{
    using base_type = unsigned char;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<uchar1>{
    using base_type = unsigned char;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<uchar2>{
    using base_type = unsigned char;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<uchar3>{
    using base_type = unsigned char;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<uchar4>{
    using base_type = unsigned char;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<unsigned short>{
    using base_type = unsigned short;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ushort1>{
    using base_type = unsigned short;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ushort2>{
    using base_type = unsigned short;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<ushort3>{
    using base_type = unsigned short;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<ushort4>{
    using base_type = unsigned short;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<unsigned int>{
    using base_type = unsigned int;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<uint1>{
    using base_type = unsigned int;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<uint2>{
    using base_type = unsigned int;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<uint3>{
    using base_type = unsigned int;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<uint4>{
    using base_type = unsigned int;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<unsigned long>{
    using base_type = unsigned long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ulong1>{
    using base_type = unsigned long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ulong2>{
    using base_type = unsigned long;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<ulong3>{
    using base_type = unsigned long;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<ulong4>{
    using base_type = unsigned long;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<unsigned long long>{
    using base_type = unsigned long long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ulonglong1>{
    using base_type = unsigned long long;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<ulonglong2>{
    using base_type = unsigned long long;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<ulonglong3>{
    using base_type = unsigned long long;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<ulonglong4>{
    using base_type = unsigned long long;
    static inline constexpr size_t numChannels = 4;
    };
    template<>
    struct CUDAPixelTraits<float>{
    using base_type = float;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<float1>{
    using base_type = float;
    static inline constexpr size_t numChannels = 1;
    };
    template<>
    struct CUDAPixelTraits<float2>{
    using base_type = float;
    static inline constexpr size_t numChannels = 2;
    };
    template<>
    struct CUDAPixelTraits<float3>{
    using base_type = float;
    static inline constexpr size_t numChannels = 3;
    };
    template<>
    struct CUDAPixelTraits<float4>{
    using base_type = float;
    static inline constexpr size_t numChannels = 4;
    };
    namespace internal{
        template<typename BaseType, size_t Dim>
        struct GLPixelTraitsImpl;
        //8bit unsigned
        template<>
        struct GLPixelTraitsImpl<unsigned char,1>{
        using base_type = unsigned char;
        static inline constexpr size_t numChannels        = 1;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_R8;
        static inline constexpr GLenum internalFormatSRGB = GL_R8_SNORM;
        static inline constexpr GLenum format             = GL_RED;
        static inline constexpr GLenum type               = GL_UNSIGNED_BYTE;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned char,2>{
        using base_type = unsigned char;
        static inline constexpr size_t numChannels        = 2;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RG8;
        static inline constexpr GLenum internalFormatSRGB = GL_RG8_SNORM;
        static inline constexpr GLenum format             = GL_RG;
        static inline constexpr GLenum type               = GL_UNSIGNED_BYTE;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned char,3>{
        using base_type = unsigned char;
        static inline constexpr size_t numChannels        = 3;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RGB8;
        static inline constexpr GLenum internalFormatSRGB = GL_RGB8_SNORM;
        static inline constexpr GLenum format             = GL_RGB;
        static inline constexpr GLenum type               = GL_UNSIGNED_BYTE;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned char,4>{
        using base_type = unsigned char;
        static inline constexpr size_t numChannels        = 4;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RGBA8;
        static inline constexpr GLenum internalFormatSRGB = GL_RGBA8_SNORM;
        static inline constexpr GLenum format             = GL_RGBA;
        static inline constexpr GLenum type               = GL_UNSIGNED_BYTE;
        };
        //16bit unsigned
        template<>
        struct GLPixelTraitsImpl<unsigned short,1>{
        using base_type = unsigned short;
        static inline constexpr size_t numChannels        = 1;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_R16;
        static inline constexpr GLenum internalFormatSRGB = GL_R16_SNORM;
        static inline constexpr GLenum format             = GL_RED;
        static inline constexpr GLenum type               = GL_UNSIGNED_SHORT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned short,2>{
        using base_type = unsigned short;
        static inline constexpr size_t numChannels        = 2;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RG16;
        static inline constexpr GLenum internalFormatSRGB = GL_RG16_SNORM;
        static inline constexpr GLenum format             = GL_RG;
        static inline constexpr GLenum type               = GL_UNSIGNED_SHORT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned short,3>{
        using base_type = unsigned short;
        static inline constexpr size_t numChannels        = 3;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RGB16;
        static inline constexpr GLenum internalFormatSRGB = GL_RGB16_SNORM;
        static inline constexpr GLenum format             = GL_RGB;
        static inline constexpr GLenum type               = GL_UNSIGNED_SHORT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned short,4>{
        using base_type = unsigned short;
        static inline constexpr size_t numChannels        = 4;
        static inline constexpr bool   supportSRGB        = true;
        static inline constexpr GLenum internalFormat     = GL_RGBA16;
        static inline constexpr GLenum internalFormatSRGB = GL_RGBA16_SNORM;
        static inline constexpr GLenum format             = GL_RGBA;
        static inline constexpr GLenum type               = GL_UNSIGNED_SHORT;
        };
        //32bit unsigned
        template<>
        struct GLPixelTraitsImpl<unsigned int,1>{
        using base_type = unsigned int;
        static inline constexpr size_t numChannels        = 1;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_R32UI;
        static inline constexpr GLenum internalFormatSRGB = GL_R32UI;
        static inline constexpr GLenum format             = GL_RED;
        static inline constexpr GLenum type               = GL_UNSIGNED_INT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned int,2>{
        using base_type = unsigned int;
        static inline constexpr size_t numChannels        = 2;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RG32UI;
        static inline constexpr GLenum internalFormatSRGB = GL_RG32UI;
        static inline constexpr GLenum format             = GL_RG;
        static inline constexpr GLenum type               = GL_UNSIGNED_INT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned int,3>{
        using base_type = unsigned int;
        static inline constexpr size_t numChannels        = 3;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RGB32UI;
        static inline constexpr GLenum internalFormatSRGB = GL_RGB32UI;
        static inline constexpr GLenum format             = GL_RGB;
        static inline constexpr GLenum type               = GL_UNSIGNED_INT;
        };
        template<>
        struct GLPixelTraitsImpl<unsigned int,4>{
        using base_type = unsigned int;
        static inline constexpr size_t numChannels        = 4;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RGBA32UI;
        static inline constexpr GLenum internalFormatSRGB = GL_RGBA32UI;
        static inline constexpr GLenum format             = GL_RGBA;
        static inline constexpr GLenum type               = GL_UNSIGNED_INT;
        };
        //32bit float
        template<>
        struct GLPixelTraitsImpl<float,1>{
        using base_type = float;
        static inline constexpr size_t numChannels        = 1;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_R32F;
        static inline constexpr GLenum internalFormatSRGB = GL_R32F;
        static inline constexpr GLenum format             = GL_RED;
        static inline constexpr GLenum type               = GL_FLOAT;
        };
        template<>
        struct GLPixelTraitsImpl<float,2>{
        using base_type = float;
        static inline constexpr size_t numChannels        = 2;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RG32F;
        static inline constexpr GLenum internalFormatSRGB = GL_RG32F;
        static inline constexpr GLenum format             = GL_RG;
        static inline constexpr GLenum type               = GL_FLOAT;
        };
        template<>
        struct GLPixelTraitsImpl<float,3>{
        using base_type = float;
        static inline constexpr size_t numChannels        = 3;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RGB32F;
        static inline constexpr GLenum internalFormatSRGB = GL_RGB32F;
        static inline constexpr GLenum format             = GL_RGB;
        static inline constexpr GLenum type               = GL_FLOAT;
        };
        template<>
        struct GLPixelTraitsImpl<float,4>{
        using base_type = float;
        static inline constexpr size_t numChannels        = 4;
        static inline constexpr bool   supportSRGB        = false;
        static inline constexpr GLenum internalFormat     = GL_RGBA32F;
        static inline constexpr GLenum internalFormatSRGB = GL_RGBA32F;
        static inline constexpr GLenum format             = GL_RGBA;
        static inline constexpr GLenum type               = GL_FLOAT;
        };
    
    }
    template<typename PixelFormat>
    struct GLPixelTraits;
    //8bit unsigned
    template<>
    struct GLPixelTraits<unsigned char>:internal::GLPixelTraitsImpl<unsigned char,1>{};
    template<>
    struct GLPixelTraits<uchar1>:internal::GLPixelTraitsImpl<unsigned char,1>{};
    template<>
    struct GLPixelTraits<uchar2>:internal::GLPixelTraitsImpl<unsigned char,2>{};
    template<>
    struct GLPixelTraits<uchar3>:internal::GLPixelTraitsImpl<unsigned char,3>{};
    template<>
    struct GLPixelTraits<uchar4>:internal::GLPixelTraitsImpl<unsigned char,4>{};
    //16bit unsigned
    template<>
    struct GLPixelTraits<unsigned short>:internal::GLPixelTraitsImpl<unsigned short,1>{};
    template<>
    struct GLPixelTraits<ushort1>:internal::GLPixelTraitsImpl<unsigned short,1>{};
    template<>
    struct GLPixelTraits<ushort2>:internal::GLPixelTraitsImpl<unsigned short,2>{};
    template<>
    struct GLPixelTraits<ushort3>:internal::GLPixelTraitsImpl<unsigned short,3>{};
    template<>
    struct GLPixelTraits<ushort4>:internal::GLPixelTraitsImpl<unsigned short,4>{};
    //32bit unsigned
    template<>
    struct GLPixelTraits<unsigned int>:internal::GLPixelTraitsImpl<unsigned int,1>{};
    template<>
    struct GLPixelTraits<uint1>:internal::GLPixelTraitsImpl<unsigned int,1>{};
    template<>
    struct GLPixelTraits<uint2>:internal::GLPixelTraitsImpl<unsigned int,2>{};
    template<>
    struct GLPixelTraits<uint3>:internal::GLPixelTraitsImpl<unsigned int,3>{};
    template<>
    struct GLPixelTraits<uint4>:internal::GLPixelTraitsImpl<unsigned int,4>{};
    //32bit float
    template<>
    struct GLPixelTraits<float>:internal::GLPixelTraitsImpl<float,1>{};
    template<>
    struct GLPixelTraits<float1>:internal::GLPixelTraitsImpl<float,1>{};
    template<>
    struct GLPixelTraits<float2>:internal::GLPixelTraitsImpl<float,2>{};
    template<>
    struct GLPixelTraits<float3>:internal::GLPixelTraitsImpl<float,3>{};
    template<>
    struct GLPixelTraits<float4>:internal::GLPixelTraitsImpl<float,4>{};
}
#endif