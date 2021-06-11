#ifndef RTLIB_PREPROCESSOR_H
#define RTLIB_PREPROCESSOR_H
#ifdef __CUDA_ARCH__
#define RTLIB_HOST_DEVICE __host__ __device__
#define RTLIB_INLINE __forceinline__
//Program
#define RTLIB_RAYGEN_PROGRAM_NAME(program_name) __raygen__##program_name
#define RTLIB_MISS_PROGRAM_NAME(program_name) __miss__##program_name
#define RTLIB_INTERSECT_PROGRAM_NAME(program_name) __intersect__##program_name
#define RTLIB_CLOSESTHIT_PROGRAM_NAME(program_name) __closesthit__##program_name
#define RTLIB_ANYHIT_PROGRAM_NAME(program_name) __anyhit__##program_name
#define RTLIB_DIRECT_CALLABLE_PROGRAM_NAME(program_name) __direct_callable__##program_name
#define RTLIB_CONTINUATION_CALLABLE_PROGRAM_NAME(program_name) __continuation_callable__##program_name
#define RTLIB_EXCEPTION_PROGRAM_NAME(program_name) __exception__##program_name
#else
#define RTLIB_HOST_DEVICE 
#define RTLIB_INLINE inline
//Program
#define RTLIB_RAYGEN_PROGRAM_STR(program_name) "__raygen__"#program_name
#define RTLIB_MISS_PROGRAM_STR(program_name) "__miss__"#program_name
#define RTLIB_INTERSECT_PROGRAM_STR(program_name) "__intersect__"#program_name
#define RTLIB_CLOSESTHIT_PROGRAM_STR(program_name) "__closesthit__"#program_name
#define RTLIB_ANYHIT_PROGRAM_STR(program_name) "__anyhit__"#program_name
#define RTLIB_DIRECT_CALLABLE_PROGRAM_STR(program_name) "__direct_callable__"#program_name
#define RTLIB_CONTINUATION_CALLABLE_PROGRAM_STR(program_name) "__continuation_callable__"#program_name
#define RTLIB_EXCEPTION_PROGRAM_STR(program_name) "__exception__"#program_name
#endif
#ifdef __cplusplus
#define RTLIB_DECLARE_GET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
const type_name& get##func_name_base()const noexcept{ return member_name; } 
#define RTLIB_DECLARE_GET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
type_name get##func_name_base()const noexcept{ return member_name; }
#define RTLIB_DECLARE_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
void  set##func_name_base(const type_name& v)noexcept{ member_name = v; } 
#define RTLIB_DECLARE_SET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
void  set##func_name_base(const type_name v)noexcept{ member_name = v; }
#define RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
RTLIB_DECLARE_GET_BY_REFERENCE(class_name,type_name,func_name_base,member_name); \
RTLIB_DECLARE_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name)
#define RTLIB_DECLARE_GET_AND_SET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
RTLIB_DECLARE_GET_BY_VALUE(class_name,type_name,func_name_base,member_name); \
RTLIB_DECLARE_SET_BY_VALUE(class_name,type_name,func_name_base,member_name)
#endif
#endif