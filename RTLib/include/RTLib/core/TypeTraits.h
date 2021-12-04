#ifndef RTLIB_TYPE_TRAITS_H
#define RTLIB_TYPE_TRAITS_H
#if !defined(__CUDA_ARCH__) && defined(__cplusplus)
#include <type_traits>
#else
#include <RTLib/core/Preprocessors.h>
namespace rtlib{
    template<class T, T v>
    struct integral_constant{
        static constexpr T value = v;
        using value_type = T;
        using type = integral_constant<T,v>;
        RTLIB_INLINE RTLIB_HOST_DEVICE constexpr operator value_type()const{ return value; }
    };
    template<bool B>
    using bool_constant = integral_constant<bool,B>;
    using true_type     = bool_constant<true>;
    using false_type    = bool_constant<false>;
    template<typename T,typename S>
    struct is_same:false_type{};
    template<typename T>
    struct is_same<T,T>:true_type{};
}
#endif
#endif