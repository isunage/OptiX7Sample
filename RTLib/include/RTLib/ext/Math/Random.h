#ifndef RTLIB_RANDOM_H
#define RTLIB_RANDOM_H
#include <cuda_runtime.h>
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
#include <type_traits>
#endif
#include "../../TypeTraits.h"
#include "../../Preprocessors.h"
#include "VectorFunction.h"
#include "../Math.h"
namespace rtlib{
    //xoshiro family random number generator
    //https://prng.di.unimi.it/を参考に実装
    //SplitMin64
    //Xoroshiroの初期化用RNGとして推奨されていた
    struct SplitMin64{
        unsigned long long m_seed;
    public:
        using result_type = unsigned long long;
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit SplitMin64(const unsigned long long s):m_seed{s}{}
        RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long  next(){
            unsigned long long  z = (m_seed+=0x9e3779b97f4a7c15);
            z = (z^(z>>30))*0xbf58476d1ce4e5b9;
            z = (z^(z>>27))*0x94d049bb133111eb;
            return m_seed = z^(z>>31);
        }
    };
    //Xorshift32
    struct Xorshift32 {
        unsigned int m_seed;
    public:
        using result_type = unsigned int;
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Xorshift32(const unsigned int s) :m_seed{ s } {}
        RTLIB_INLINE RTLIB_HOST_DEVICE result_type  next() {
            unsigned int y = m_seed;
            y = y ^ (y << 13); y = y ^ (y >> 17);
            return m_seed = y ^ (y << 5);
        }
    };
    //Xorshift128
    struct Xorshift128{
        unsigned int  m_seed[4];
    public:
        using result_type = unsigned int ;
        //constructor
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Xorshift128(const unsigned long long  seed){
            auto sm64 = SplitMin64(seed);
            auto rnd1 = sm64.next();
            auto rnd2 = sm64.next();
            m_seed[0] = static_cast<unsigned int >(rnd1>>32);
            m_seed[1] = static_cast<unsigned int >(rnd1&0x00000000FFFFFFFF);
            m_seed[2] = static_cast<unsigned int >(rnd2>>32);
            m_seed[3] = static_cast<unsigned int >(rnd2&0x00000000FFFFFFFF);
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE result_type next(){
            unsigned int  t = m_seed[0]^(m_seed[0]<<11);
            m_seed[0] = m_seed[1];
            m_seed[1] = m_seed[2];
            m_seed[2] = m_seed[3];
            return m_seed[3] = (m_seed[3]^(m_seed[3]>>19))^(t^(t>>8));
        }
    };
    //Xoroshiro128plus
    struct Xoroshiro128plus{
        unsigned long long  m_seed[2];
    public:
        using result_type = unsigned long long ;
        template<unsigned long long  k>
        static RTLIB_INLINE RTLIB_HOST_DEVICE   unsigned long long  rotl(const unsigned long long  x){
            return (x<<k)|(x>>(64-k));
        }
        //constructor
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Xoroshiro128plus(const unsigned long long  seed){
            auto sm64 = SplitMin64(seed);
            m_seed[0] = sm64.next();
            m_seed[1] = sm64.next();
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE result_type next(){
            const unsigned long long  s0 = m_seed[0];
            unsigned long long  s1 = m_seed[1];
            const unsigned long long  result = s0 + s1;
            s1 ^= s0;
            m_seed[0] = rotl<24>(s0)^s1^(s1<<16);
            m_seed[1] = rotl<37>(s1);
            return result;
        }
    };
    namespace internal{
        #if !defined(__CUDA_ARCH__) && defined(__cplusplus)
        using namespace std;
        #endif
        //
        template<typename RNG, bool is_result_type_64 = is_same<typename RNG::result_type,unsigned long long>::value>
        struct random_function_impl{
            //default algorithm
            RTLIB_INLINE RTLIB_HOST_DEVICE static double random_double(RNG& rng) {
                return to_uniform_double(rng.next());
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float  random_float1(RNG& rng) {
                return static_cast<float>(random_double(rng));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float2 random_float2(RNG& rng) {
                return make_float2(random_float1(rng), random_float1(rng));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float3 random_float3(RNG& rng) {
                return make_float3(random_float1(rng), random_float1(rng), random_float1(rng));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float4 random_float4(RNG& rng) {
                return make_float4(random_float1(rng), random_float1(rng), random_float1(rng), random_float1(rng));
            }
            //fast algorithm
            RTLIB_INLINE RTLIB_HOST_DEVICE static double random_double_fast(RNG& rng){
                return random_double(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float  random_float1_fast(RNG& rng) {
                return random_float1(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float2 random_float2_fast(RNG& rng){
                unsigned long long  rnd_v = rng.next();
                return make_float2(
                    to_uniform_float(to_upper(rnd_v)),
                    to_uniform_float(to_lower(rnd_v))
                );
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float3 random_float3_fast(RNG& rng){
                unsigned long long  rnd_v1 = rng.next();
                unsigned long long  rnd_v2 = rng.next();
                return make_float3(
                    to_uniform_float(to_upper(rnd_v1)),
                    to_uniform_float(to_lower(rnd_v1)),
                    to_uniform_float(to_upper(rnd_v2))
                );
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float4 random_float4_fast(RNG& rng){
                unsigned long long  rnd_v1 = rng.next();
                unsigned long long  rnd_v2 = rng.next();
                return make_float4(
                    to_uniform_float(to_upper(rnd_v1)),
                    to_uniform_float(to_lower(rnd_v1)),
                    to_uniform_float(to_upper(rnd_v2)),
                    to_uniform_float(to_lower(rnd_v2))
                );
            }

        };
        template<typename RNG>
        struct random_function_impl<RNG,false>{
            //default algorithm
            RTLIB_INLINE RTLIB_HOST_DEVICE static double random_double(RNG& rng) {
                return to_uniform_double(to_combine(rng.next(), rng.next()));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float  random_float1(RNG& rng) {
                return to_uniform_float(rng.next());
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float2 random_float2(RNG& rng) {
                return make_float2(random_float1(rng), random_float1(rng));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float3 random_float3(RNG& rng) {
                return make_float3(random_float1(rng), random_float1(rng), random_float1(rng));
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float4 random_float4(RNG& rng) {
                return make_float4(random_float1(rng), random_float1(rng), random_float1(rng), random_float1(rng));
            }
            //fast algorithm
            RTLIB_INLINE RTLIB_HOST_DEVICE static double random_double_fast(RNG& rng){
                return random_double(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float  random_float1_fast(RNG& rng) {
                return random_float1(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float2 random_float2_fast(RNG& rng){
                return random_float2(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float3 random_float3_fast(RNG& rng){
                return random_float3(rng);
            }
            RTLIB_INLINE RTLIB_HOST_DEVICE static float4 random_float4_fast(RNG& rng){
                return random_float4(rng);
            }
        };
    }
    //double
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE double random_double(RNG& rng) {
        return internal::random_function_impl<RNG>::random_double(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE double random_double(double low, double high, RNG& rng) {
        return low + (high - low) * random_double(rng);
    }
    //float
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float  random_float1(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float1(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float  random_float1(float  low, float  high, RNG& rng) {
        return low + (high - low) * random_float1(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float2(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2(float  low, float  high, RNG& rng) {
        return make_float2(low) + (high - low) * random_float2(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2(const float2& low, const float2& high, RNG& rng) {
        return low + (high - low) * random_float2(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float3(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3(float  low, float  high, RNG& rng) {
        return make_float3(low) + (high - low) * random_float3(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3(const float3& low, const float3& high, RNG& rng) {
        return low + (high - low) * random_float3(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float4(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4(float  low, float  high, RNG& rng) {
        return make_float4(low) + (high - low) * random_float4(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4(const float4& low, const float4& high, RNG& rng) {
        return low + (high - low) * random_float4(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_in_unit_triangle(const float3& v0, const float3& v1, const float3& v2, RNG& rng)
    {
        const float2 rnd2 = rtlib::random_float2(rng);
        const float  a    = rtlib::min(rnd2.x, rnd2.y);
        const float  b    = rtlib::max(rnd2.x, rnd2.y);
        return v0 * a + (1.0f - b) * v1 + (b - a) * v2;
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_in_unit_triangle(const float2& v0, const float2& v1, const float2& v2, RNG& rng)
    {
        const float2 rnd2 = rtlib::random_float2(rng);
        const float  a = rtlib::min(rnd2.x, rnd2.y);
        const float  b = rtlib::max(rnd2.x, rnd2.y);
        return v0 * a + (1.0f - b) * v1 + (b - a) * v2;
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_in_unit_sphere(RNG& rng) {
        //r=(1.0f,-1.0f)
        //t=(0.0f, 2.0fPI)
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
        using std::cosf;
        using std::sinf;
        using std::sqrtf;
#endif
        const float2 rnd = 2.0 * random_float2(rng);
        const float  r   = rnd.x - 1.0f;
        const float  z   = sqrtf(1.0f-r*r);
        const float  t   = rnd.y * RTLIB_M_PI;
        return make_float3(z * ::cosf(t), z * ::sinf(t), r);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_cosine_direction(RNG& rng) {
        //r=(1.0f,-1.0f)
        //t=(0.0f, 2.0fPI)
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
        using std::cosf;
        using std::sinf;
        using std::sqrtf;
#endif
        const float2 rnd = random_float2(rng);
        const float  r   = sqrtf(rnd.x);
        const float  z   = sqrtf(1.0f-rnd.x);
        const float  t   = 2.0f*rnd.y * RTLIB_M_PI;
        return make_float3(r * ::cosf(t), r * ::sinf(t), z);
    }
    //fastの場合、得られたbit長を最大限利用することで乱数の呼び出しを最小限にとどめる
    //そのため精度は保証できず、場合によっては単精度になる可能性あり
    //double
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE double random_double_fast(RNG& rng) {
        return internal::random_function_impl<RNG>::random_double_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE double random_double_fast(double low, double high, RNG& rng) {
        return low + (high - low) * random_double_fast(rng);
    }
    //float
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float  random_float1_fast(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float1_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float  random_float1_fast(float  low, float  high, RNG& rng) {
        return low + (high - low) * random_float1_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2_fast(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float2_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2_fast(float  low, float  high, RNG& rng) {
        return make_float2(low) + (high - low) * random_float2_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float2 random_float2_fast(const float2& low, const float2& high, RNG& rng) {
        return low + (high - low) * random_float2_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3_fast(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float3_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3_fast(float  low, float  high, RNG& rng) {
        return make_float3(low) + (high - low) * random_float3_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_float3_fast(const float3& low, const float3& high, RNG& rng) {
        return low + (high - low) * random_float3_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4_fast(RNG& rng) {
        return internal::random_function_impl<RNG>::random_float4_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4_fast(float  low, float  high, RNG& rng) {
        return make_float4(low) + (high - low) * random_float4_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float4 random_float4_fast(const float4& low, const float4& high, RNG& rng) {
        return low + (high - low) * random_float4_fast(rng);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_in_unit_sphere_fast(RNG& rng) {
        //r=(1.0f,-1.0f)
        //t=(0.0f, 2.0fPI)
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
        using std::cosf;
        using std::sinf;
        using std::sqrtf;
#endif
        const float2 rnd = 2.0 * random_float2_fast(rng);
        const float  r = rnd.x - 1.0f;
        const float  z = sqrtf(1.0f - r * r);
        const float  t = rnd.y * RTLIB_M_PI;
        return make_float3(z * ::cosf(t), z * ::sinf(t), r);
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE float3 random_cosine_direction_fast(RNG& rng) {
        //r=(1.0f,-1.0f)
        //t=(0.0f, 2.0fPI)
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
        using std::cosf;
        using std::sinf;
        using std::sqrtf;
#endif
        const float2 rnd = random_float2_fast(rng);
        const float  r   = sqrtf(rnd.x);
        const float  z   = sqrtf(1.0f-rnd.x);
        const float  t   = 2.0f*rnd.y * RTLIB_M_PI;
        return make_float3(r * ::cosf(t), r * ::sinf(t), z);
    }
}
#endif