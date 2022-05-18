#ifndef RTLIB_PAYLOAD_H
#define RTLIB_PAYLOAD_H
#include <RTLib/core/Preprocessors.h>
#include <RTLib/math/VectorFunction.h>
#if defined(__CUDACC__)
#include <optix.h>
#define RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(N)            \
    template <>                                                               \
    struct GlobalPayloadUIntImpl<N>                                           \
    {                                                                         \
        static __forceinline__ __device__ auto Get() noexcept -> unsigned int \
        {                                                                     \
            return optixGetPayload_##N##();                                   \
        }                                                                     \
        static __forceinline__ __device__ void Set(unsigned int v) noexcept   \
        {                                                                     \
            return optixSetPayload_##N##(v);                                  \
        }                                                                     \
    }

namespace rtlib
{
    template <size_t N>
    struct Payload;

    template <size_t N>
    static __forceinline__ __device__ auto GetGlobalPayload() noexcept -> Payload<N>;
    template <size_t N>
    static __forceinline__ __device__ void SetGlobalPayload(const Payload<N> &p) noexcept;
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadUInt() noexcept -> unsigned int;
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadUInt(unsigned int v) noexcept;
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadFloat()noexcept -> float
    {
        return __uint_as_float(GetGlobalPayloadUInt<index>());
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadFloat(float v)noexcept
    {
        return SetGlobalPayloadUInt<index>(__float_as_uint(v));
    }
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadFloat2()noexcept -> float2
    {
        return make_float2(GetGlobalPayloadFloat<index+0>(),GetGlobalPayloadFloat<index+1>());
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadFloat2(const float2& v)noexcept
    {
        SetGlobalPayloadFloat<index+0>(v.x);
        SetGlobalPayloadFloat<index+1>(v.y);
    }
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadFloat3()noexcept -> float3
    {
        return make_float3(GetGlobalPayloadFloat<index+0>(),GetGlobalPayloadFloat<index+1>(),GetGlobalPayloadFloat<index+2>());
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadFloat3(const float3& v)noexcept
    {
        SetGlobalPayloadFloat<index+0>(v.x);
        SetGlobalPayloadFloat<index+1>(v.y);
        SetGlobalPayloadFloat<index+2>(v.z);
    }
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadFloat4()noexcept -> float3
    {
        return make_float4(GetGlobalPayloadFloat<index+0>(),GetGlobalPayloadFloat<index+1>(),GetGlobalPayloadFloat<index+2>(),GetGlobalPayloadFloat<index+3>());
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadFloat4(const float4& v)noexcept
    {
        SetGlobalPayloadFloat<index+0>(v.x);
        SetGlobalPayloadFloat<index+1>(v.y);
        SetGlobalPayloadFloat<index+2>(v.z);
        SetGlobalPayloadFloat<index+3>(v.w);
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadUchar4(const uchar4& v) noexcept
    {
        unsigned short up = rtlib::to_combine(v.x, v.y);
        unsigned short lw = rtlib::to_combine(v.z, v.w);
        unsigned int ui = rtlib::to_combine(up, lw);
        SetGlobalPayloadUInt<index>(ui);
    }
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadUchar4() noexcept -> uchar4
    {
        auto ui = GetGlobalPayloadUInt<index>();
        unsigned short up = rtlib::to_upper(ui);
        unsigned char vx = rtlib::to_upper(up);
        unsigned char vy = rtlib::to_lower(up);
        unsigned short lw = rtlib::to_lower(ui);
        unsigned char vz = rtlib::to_upper(lw);
        unsigned char vw = rtlib::to_lower(lw);
        return make_uchar4(vx, vy, vz, vw);
    }
    template <size_t index,typename T>
    static __forceinline__ __device__ auto GetGlobalPayloadPointer()noexcept->T*
    {
        auto p0 = GetGlobalPayloadUInt<index + 0>();
        auto p1 = GetGlobalPayloadUInt<index + 1>();
        return reinterpret_cast<T *>(rtlib::to_combine(p0, p1));
    }
    template <size_t index, typename T>
    static __forceinline__ __device__ void SetGlobalPayloadPointer(T *ptr) noexcept
    {
        const unsigned long long llv = reinterpret_cast<const unsigned long long>(ptr);
        SetGlobalPayloadUInt<index + 0>(rtlib::to_upper(llv));
        SetGlobalPayloadUInt<index + 1>(rtlib::to_lower(llv));
    }


    template <size_t N>
    static __forceinline__ __device__ void Trace(OptixTraversableHandle handle,
                                                 float3 rayOrigin,
                                                 float3 rayDirection,
                                                 float tmin,
                                                 float tmax,
                                                 float rayTime,
                                                 OptixVisibilityMask visibilityMask,
                                                 unsigned int rayFlags,
                                                 unsigned int SBToffset,
                                                 unsigned int SBTstride,
                                                 unsigned int missSBTIndex, Payload<N> &p);

    template <size_t N>
    struct Payload
    {
        static __forceinline__ __device__ auto GetGlobal() noexcept -> Payload<N>
        {
            return GetGlobalPayload<N>();
        }
        static __forceinline__ __device__ void SetGlobal(const Payload<N> &v) noexcept
        {
            return SetGlobalPayload<N>(v);
        }
        template <size_t index>
        __forceinline__ __device__ void SetUInt(unsigned int val) noexcept
        {
            values[index] = val;
        }
        template <size_t index>
        __forceinline__ __device__ auto GetUInt() const noexcept -> unsigned int
        {
            return values[index];
        }
        template <size_t index>
        __forceinline__ __device__ void SetFloat(float value) noexcept
        {
            SetUInt<index>(__float_as_uint(value));
        }
        template <size_t index>
        __forceinline__ __device__ auto GetFloat() const noexcept -> float
        {
            return __uint_as_float(GetUInt<index>());
        }
        template <size_t index>
        __forceinline__ __device__ void SetFloat2(const float2 &value) noexcept
        {
            SetFloat<index + 0>(value.x);
            SetFloat<index + 1>(value.y);
        }
        template <size_t index>
        __forceinline__ __device__ auto GetFloat2() const noexcept -> float2
        {
            return make_float2(
                GetFloat<index + 0>(),
                GetFloat<index + 1>());
        }
        template <size_t index>
        __forceinline__ __device__ void SetFloat3(const float3 &value) noexcept
        {
            SetFloat<index + 0>(value.x);
            SetFloat<index + 1>(value.y);
            SetFloat<index + 2>(value.z);
        }
        template <size_t index>
        __forceinline__ __device__ auto GetFloat3() const noexcept -> float3
        {
            return make_float3(
                GetFloat<index + 0>(),
                GetFloat<index + 1>(),
                GetFloat<index + 2>());
        }
        template <size_t index>
        __forceinline__ __device__ void SetFloat4(const float4 &value) noexcept
        {
            SetFloat<index + 0>(value.x);
            SetFloat<index + 1>(value.y);
            SetFloat<index + 2>(value.z);
            SetFloat<index + 3>(value.w);
        }
        template <size_t index>
        __forceinline__ __device__ auto GetFloat4() const noexcept -> float4
        {
            return make_float3(
                GetFloat<index + 0>(),
                GetFloat<index + 1>(),
                GetFloat<index + 2>(),
                GetFloat<index + 3>());
        }
        template <size_t index>
        __forceinline__ __device__ void SetUchar4(const uchar4 &v) noexcept
        {
            unsigned short up = rtlib::to_combine(v.x, v.y);
            unsigned short lw = rtlib::to_combine(v.z, v.w);
            unsigned int ui = rtlib::to_combine(up, lw);
            SetUInt<index>(ui);
        }
        template <size_t index>
        __forceinline__ __device__ auto GetUchar4() const noexcept -> uchar4
        {
            auto ui = GetUInt<index>();
            unsigned short up = rtlib::to_upper(ui);
            unsigned char vx = rtlib::to_upper(up);
            unsigned char vy = rtlib::to_lower(up);
            unsigned short lw = rtlib::to_lower(ui);
            unsigned char vz = rtlib::to_upper(lw);
            unsigned char vw = rtlib::to_lower(lw);
            return make_uchar4(vx, vy, vz, vw);
        }
        template <size_t index, typename T>
        __forceinline__ __device__ void SetPointer(T *ptr) noexcept
        {
            const unsigned long long llv = reinterpret_cast<const unsigned long long>(ptr);
            SetUInt<index + 0>(rtlib::to_upper(llv));
            SetUInt<index + 1>(rtlib::to_lower(llv));
        }
        template <size_t index, typename T>
        __forceinline__ __device__ T *GetPointer() const noexcept
        {
            auto p0 = GetUInt<index + 0>();
            auto p1 = GetUInt<index + 1>();
            return reinterpret_cast<T *>(rtlib::to_combine(p0, p1));
        }

        unsigned int values[N];
    };

    template <size_t index>
    struct GlobalPayloadUIntImpl;
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(0);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(1);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(2);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(3);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(4);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(5);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(6);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(7);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(8);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(9);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(10);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(11);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(12);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(13);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(14);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(15);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(16);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(17);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(18);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(19);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(20);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(21);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(22);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(23);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(24);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(25);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(26);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(27);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(28);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(29);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(30);
    RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION(31);

    template <size_t N>
    struct GlobalPayloadImpl;
    template <>
    struct GlobalPayloadImpl<1>
    {
        static __forceinline__ __device__ Payload<1> Get() noexcept
        {
            Payload<1> p;
            p.values[0] = optixGetPayload_0();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<1> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
        }
    };
    template <>
    struct GlobalPayloadImpl<2>
    {
        static __forceinline__ __device__ Payload<2> Get() noexcept
        {
            Payload<2> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<2> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
        }
    };
    template <>
    struct GlobalPayloadImpl<3>
    {
        static __forceinline__ __device__ Payload<3> Get() noexcept
        {
            Payload<3> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<3> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
        }
    };
    template <>
    struct GlobalPayloadImpl<4>
    {
        static __forceinline__ __device__ Payload<4> Get() noexcept
        {
            Payload<4> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<4> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
        }
    };
    template <>
    struct GlobalPayloadImpl<5>
    {
        static __forceinline__ __device__ Payload<5> Get() noexcept
        {
            Payload<5> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<5> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
        }
    };
    template <>
    struct GlobalPayloadImpl<6>
    {
        static __forceinline__ __device__ Payload<6> Get() noexcept
        {
            Payload<6> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<6> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
        }
    };
    template <>
    struct GlobalPayloadImpl<7>
    {
        static __forceinline__ __device__ Payload<7> Get() noexcept
        {
            Payload<7> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<7> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
        }
    };
    template <>
    struct GlobalPayloadImpl<8>
    {
        static __forceinline__ __device__ Payload<8> Get() noexcept
        {
            Payload<8> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<8> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
        }
    };
    template <>
    struct GlobalPayloadImpl<9>
    {
        static __forceinline__ __device__ Payload<9> Get() noexcept
        {
            Payload<9> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<9> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
        }
    };
    template <>
    struct GlobalPayloadImpl<10>
    {
        static __forceinline__ __device__ Payload<10> Get() noexcept
        {
            Payload<10> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<10> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
        }
    };
    template <>
    struct GlobalPayloadImpl<11>
    {
        static __forceinline__ __device__ Payload<11> Get() noexcept
        {
            Payload<11> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<11> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
        }
    };
    template <>
    struct GlobalPayloadImpl<12>
    {
        static __forceinline__ __device__ Payload<12> Get() noexcept
        {
            Payload<12> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<12> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
        }
    };
    template <>
    struct GlobalPayloadImpl<13>
    {
        static __forceinline__ __device__ Payload<13> Get() noexcept
        {
            Payload<13> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<13> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
        }
    };
    template <>
    struct GlobalPayloadImpl<14>
    {
        static __forceinline__ __device__ Payload<14> Get() noexcept
        {
            Payload<14> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<14> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
        }
    };
    template <>
    struct GlobalPayloadImpl<15>
    {
        static __forceinline__ __device__ Payload<15> Get() noexcept
        {
            Payload<15> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<15> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
        }
    };
    template <>
    struct GlobalPayloadImpl<16>
    {
        static __forceinline__ __device__ Payload<16> Get() noexcept
        {
            Payload<16> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<16> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
        }
    };
    template <>
    struct GlobalPayloadImpl<17>
    {
        static __forceinline__ __device__ Payload<17> Get() noexcept
        {
            Payload<17> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<17> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
        }
    };
    template <>
    struct GlobalPayloadImpl<18>
    {
        static __forceinline__ __device__ Payload<18> Get() noexcept
        {
            Payload<18> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<18> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
        }
    };
    template <>
    struct GlobalPayloadImpl<19>
    {
        static __forceinline__ __device__ Payload<19> Get() noexcept
        {
            Payload<19> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<19> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
        }
    };
    template <>
    struct GlobalPayloadImpl<20>
    {
        static __forceinline__ __device__ Payload<20> Get() noexcept
        {
            Payload<20> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<20> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
        }
    };
    template <>
    struct GlobalPayloadImpl<21>
    {
        static __forceinline__ __device__ Payload<21> Get() noexcept
        {
            Payload<21> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<21> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
        }
    };
    template <>
    struct GlobalPayloadImpl<22>
    {
        static __forceinline__ __device__ Payload<22> Get() noexcept
        {
            Payload<22> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<22> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
        }
    };
    template <>
    struct GlobalPayloadImpl<23>
    {
        static __forceinline__ __device__ Payload<23> Get() noexcept
        {
            Payload<23> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<23> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
        }
    };
    template <>
    struct GlobalPayloadImpl<24>
    {
        static __forceinline__ __device__ Payload<24> Get() noexcept
        {
            Payload<24> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<24> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
        }
    };
    template <>
    struct GlobalPayloadImpl<25>
    {
        static __forceinline__ __device__ Payload<25> Get() noexcept
        {
            Payload<25> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<25> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
        }
    };
    template <>
    struct GlobalPayloadImpl<26>
    {
        static __forceinline__ __device__ Payload<26> Get() noexcept
        {
            Payload<26> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<26> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
        }
    };
    template <>
    struct GlobalPayloadImpl<27>
    {
        static __forceinline__ __device__ Payload<27> Get() noexcept
        {
            Payload<27> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<27> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
        }
    };
    template <>
    struct GlobalPayloadImpl<28>
    {
        static __forceinline__ __device__ Payload<28> Get() noexcept
        {
            Payload<28> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            p.values[27] = optixGetPayload_27();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<28> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
            optixSetPayload_27(p.values[27]);
        }
    };
    template <>
    struct GlobalPayloadImpl<29>
    {
        static __forceinline__ __device__ Payload<29> Get() noexcept
        {
            Payload<29> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            p.values[27] = optixGetPayload_27();
            p.values[28] = optixGetPayload_28();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<29> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
            optixSetPayload_27(p.values[27]);
            optixSetPayload_28(p.values[28]);
        }
    };
    template <>
    struct GlobalPayloadImpl<30>
    {
        static __forceinline__ __device__ Payload<30> Get() noexcept
        {
            Payload<30> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            p.values[27] = optixGetPayload_27();
            p.values[28] = optixGetPayload_28();
            p.values[29] = optixGetPayload_29();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<30> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
            optixSetPayload_27(p.values[27]);
            optixSetPayload_28(p.values[28]);
            optixSetPayload_29(p.values[29]);
        }
    };
    template <>
    struct GlobalPayloadImpl<31>
    {
        static __forceinline__ __device__ Payload<31> Get() noexcept
        {
            Payload<31> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            p.values[27] = optixGetPayload_27();
            p.values[28] = optixGetPayload_28();
            p.values[29] = optixGetPayload_29();
            p.values[30] = optixGetPayload_30();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<31> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
            optixSetPayload_27(p.values[27]);
            optixSetPayload_28(p.values[28]);
            optixSetPayload_29(p.values[29]);
            optixSetPayload_30(p.values[30]);
        }
    };
    template <>
    struct GlobalPayloadImpl<32>
    {
        static __forceinline__ __device__ Payload<32> Get() noexcept
        {
            Payload<32> p;
            p.values[0] = optixGetPayload_0();
            p.values[1] = optixGetPayload_1();
            p.values[2] = optixGetPayload_2();
            p.values[3] = optixGetPayload_3();
            p.values[4] = optixGetPayload_4();
            p.values[5] = optixGetPayload_5();
            p.values[6] = optixGetPayload_6();
            p.values[7] = optixGetPayload_7();
            p.values[8] = optixGetPayload_8();
            p.values[9] = optixGetPayload_9();
            p.values[10] = optixGetPayload_10();
            p.values[11] = optixGetPayload_11();
            p.values[12] = optixGetPayload_12();
            p.values[13] = optixGetPayload_13();
            p.values[14] = optixGetPayload_14();
            p.values[15] = optixGetPayload_15();
            p.values[16] = optixGetPayload_16();
            p.values[17] = optixGetPayload_17();
            p.values[18] = optixGetPayload_18();
            p.values[19] = optixGetPayload_19();
            p.values[20] = optixGetPayload_20();
            p.values[21] = optixGetPayload_21();
            p.values[22] = optixGetPayload_22();
            p.values[23] = optixGetPayload_23();
            p.values[24] = optixGetPayload_24();
            p.values[25] = optixGetPayload_25();
            p.values[26] = optixGetPayload_26();
            p.values[27] = optixGetPayload_27();
            p.values[28] = optixGetPayload_28();
            p.values[29] = optixGetPayload_29();
            p.values[30] = optixGetPayload_30();
            p.values[31] = optixGetPayload_31();
            return p;
        }
        static __forceinline__ __device__ void Set(const Payload<32> &p) noexcept
        {
            optixSetPayload_0(p.values[0]);
            optixSetPayload_1(p.values[1]);
            optixSetPayload_2(p.values[2]);
            optixSetPayload_3(p.values[3]);
            optixSetPayload_4(p.values[4]);
            optixSetPayload_5(p.values[5]);
            optixSetPayload_6(p.values[6]);
            optixSetPayload_7(p.values[7]);
            optixSetPayload_8(p.values[8]);
            optixSetPayload_9(p.values[9]);
            optixSetPayload_10(p.values[10]);
            optixSetPayload_11(p.values[11]);
            optixSetPayload_12(p.values[12]);
            optixSetPayload_13(p.values[13]);
            optixSetPayload_14(p.values[14]);
            optixSetPayload_15(p.values[15]);
            optixSetPayload_16(p.values[16]);
            optixSetPayload_17(p.values[17]);
            optixSetPayload_18(p.values[18]);
            optixSetPayload_19(p.values[19]);
            optixSetPayload_20(p.values[20]);
            optixSetPayload_21(p.values[21]);
            optixSetPayload_22(p.values[22]);
            optixSetPayload_23(p.values[23]);
            optixSetPayload_24(p.values[24]);
            optixSetPayload_25(p.values[25]);
            optixSetPayload_26(p.values[26]);
            optixSetPayload_27(p.values[27]);
            optixSetPayload_28(p.values[28]);
            optixSetPayload_29(p.values[29]);
            optixSetPayload_30(p.values[30]);
            optixSetPayload_31(p.values[31]);
        }
    };

    template <size_t N>
    struct TraceImpl;
    template <>
    struct TraceImpl<1>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<1> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0]);
        }
    };
    template <>
    struct TraceImpl<2>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<2> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1]);
        }
    };
    template <>
    struct TraceImpl<3>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<3> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2]);
        }
    };
    template <>
    struct TraceImpl<4>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<4> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3]);
        }
    };
    template <>
    struct TraceImpl<5>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<5> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4]);
        }
    };
    template <>
    struct TraceImpl<6>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<6> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5]);
        }
    };
    template <>
    struct TraceImpl<7>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<7> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6]);
        }
    };
    template <>
    struct TraceImpl<8>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<8> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7]);
        }
    };
    template <>
    struct TraceImpl<9>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<9> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8]);
        }
    };
    template <>
    struct TraceImpl<10>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<10> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9]);
        }
    };
    template <>
    struct TraceImpl<11>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<11> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10]);
        }
    };
    template <>
    struct TraceImpl<12>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<12> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11]);
        }
    };
    template <>
    struct TraceImpl<13>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<13> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12]);
        }
    };
    template <>
    struct TraceImpl<14>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<14> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13]);
        }
    };
    template <>
    struct TraceImpl<15>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<15> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14]);
        }
    };
    template <>
    struct TraceImpl<16>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<16> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15]);
        }
    };
    template <>
    struct TraceImpl<17>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<17> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16]);
        }
    };
    template <>
    struct TraceImpl<18>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<18> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17]);
        }
    };
    template <>
    struct TraceImpl<19>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<19> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18]);
        }
    };
    template <>
    struct TraceImpl<20>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<20> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19]);
        }
    };
    template <>
    struct TraceImpl<21>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<21> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20]);
        }
    };
    template <>
    struct TraceImpl<22>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<22> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21]);
        }
    };
    template <>
    struct TraceImpl<23>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<23> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22]);
        }
    };
    template <>
    struct TraceImpl<24>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<24> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23]);
        }
    };
    template <>
    struct TraceImpl<25>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<25> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24]);
        }
    };
    template <>
    struct TraceImpl<26>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<26> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25]);
        }
    };
    template <>
    struct TraceImpl<27>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<27> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26]);
        }
    };
    template <>
    struct TraceImpl<28>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<28> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26], payload.values[27]);
        }
    };
    template <>
    struct TraceImpl<29>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<29> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26], payload.values[27], payload.values[28]);
        }
    };
    template <>
    struct TraceImpl<30>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<30> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26], payload.values[27], payload.values[28], payload.values[29]);
        }
    };
    template <>
    struct TraceImpl<31>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<31> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26], payload.values[27], payload.values[28], payload.values[29], payload.values[30]);
        }
    };
    template <>
    struct TraceImpl<32>
    {
        static __forceinline__ __device__ void Eval(OptixTraversableHandle handle,
                                                    float3 rayOrigin,
                                                    float3 rayDirection,
                                                    float tmin,
                                                    float tmax,
                                                    float rayTime,
                                                    OptixVisibilityMask visibilityMask,
                                                    unsigned int rayFlags,
                                                    unsigned int SBToffset,
                                                    unsigned int SBTstride,
                                                    unsigned int missSBTIndex,
                                                    Payload<32> &payload)
        {
            optixTrace(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, payload.values[0], payload.values[1], payload.values[2], payload.values[3], payload.values[4], payload.values[5], payload.values[6], payload.values[7], payload.values[8], payload.values[9], payload.values[10], payload.values[11], payload.values[12], payload.values[13], payload.values[14], payload.values[15], payload.values[16], payload.values[17], payload.values[18], payload.values[19], payload.values[20], payload.values[21], payload.values[22], payload.values[23], payload.values[24], payload.values[25], payload.values[26], payload.values[27], payload.values[28], payload.values[29], payload.values[30], payload.values[31]);
        }
    };

    template <size_t N>
    static __forceinline__ __device__ auto GetGlobalPayload() noexcept -> Payload<N>
    {
        return GlobalPayloadImpl<N>::Get();
    }
    template <size_t N>
    static __forceinline__ __device__ void SetGlobalPayload(const Payload<N> &p) noexcept
    {
        return GlobalPayloadImpl<N>::Set(v);
    }
    template <size_t index>
    static __forceinline__ __device__ auto GetGlobalPayloadUInt() noexcept -> unsigned int
    {
        return GlobalPayloadUIntImpl<index>::Get();
    }
    template <size_t index>
    static __forceinline__ __device__ void SetGlobalPayloadUInt(unsigned int v) noexcept
    {
        return GlobalPayloadUIntImpl<index>::Set(v);
    }
    template <size_t N>
    static __forceinline__ __device__ void Trace(OptixTraversableHandle handle,
                                                 float3 rayOrigin,
                                                 float3 rayDirection,
                                                 float tmin,
                                                 float tmax,
                                                 float rayTime,
                                                 OptixVisibilityMask visibilityMask,
                                                 unsigned int rayFlags,
                                                 unsigned int SBToffset,
                                                 unsigned int SBTstride,
                                                 unsigned int missSBTIndex, Payload<N> &p)
    {
        return TraceImpl<N>::Eval(handle,rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, p);
    }
}
#undef RTLIB_PAYLOAD_MACRO_GLOBAL_PAYLOAD_UINT_IMPL_DEFINITION
#endif
#endif
