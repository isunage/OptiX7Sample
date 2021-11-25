#ifndef RTLIB_MATRIX_H
#define RTLIB_MATRIX_H
#include "../../Preprocessors.h"
#include "VectorFunction.h"
#ifndef __CUDA_ARCH__
#include <cstdio>
#endif
#define RTLIB_IMPL_MATRIX_MUL_2(I1,J1,I2,J2) RTLIB_IMPL_MATRIX_AT(I1, J1)* RTLIB_IMPL_MATRIX_AT(I2, J2)
#define RTLIB_IMPL_MATRIX_MUL_2_TRANSPOSE(I1,J1,I2,J2) RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I1, J1)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I2, J2)
#define RTLIB_IMPL_MATRIX_MUL_3(I1,J1,I2,J2,I3,J3) RTLIB_IMPL_MATRIX_AT(I1, J1)* RTLIB_IMPL_MATRIX_AT(I2, J2)* RTLIB_IMPL_MATRIX_AT(I3, J3)
#define RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(I1,J1,I2,J2,I3,J3) RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I1, J1)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I2, J2)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I3, J3)
#define RTLIB_IMPL_MATRIX_MUL_4(I1,J1,I2,J2,I3,J3,I4,J4) RTLIB_IMPL_MATRIX_AT(I1, J1)* RTLIB_IMPL_MATRIX_AT(I2, J2)* RTLIB_IMPL_MATRIX_AT(I3, J3)* RTLIB_IMPL_MATRIX_AT(I4, J4)
#define RTLIB_IMPL_MATRIX_MUL_4_TRANSPOSE(I1,J1,I2,J2,I3,J3,I4,J4) RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I1, J1)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I2, J2)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I3, J3)* RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I4, J4)
#define RTLIB_IMPL_MATRIX_AT_TRANSPOSE(I, J) RTLIB_IMPL_MATRIX_AT(J, I)
#define RTLIB_IMPL_MATRIX_AT(I, J) RTLIB_IMPL_MATRIX_AT_##J(I)
#define RTLIB_IMPL_MATRIX_AT_0(I) m_Base[I].x
#define RTLIB_IMPL_MATRIX_AT_1(I) m_Base[I].y
#define RTLIB_IMPL_MATRIX_AT_2(I) m_Base[I].z
#define RTLIB_IMPL_MATRIX_AT_3(I) m_Base[I].w
namespace rtlib
{
    class Matrix4x4;
    class Matrix3x3
    {
    public:
        RTLIB_INLINE RTLIB_HOST_DEVICE          Matrix3x3()noexcept : Matrix3x3(
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f))
        {}
        RTLIB_INLINE RTLIB_HOST_DEVICE          Matrix3x3(const float3& c0, const float3& c1, const float3& c2)noexcept : m_Base
        {
            c0,c1,c2
        } {}
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Matrix3x3(const Matrix4x4& m)noexcept;
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Matrix3x3(float s)noexcept : Matrix3x3(
            make_float3(s, 0.0f, 0.0f),
            make_float3(0.0f, s, 0.0f),
            make_float3(0.0f, 0.0f, s))
        {}
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator+=(const Matrix3x3& m)noexcept -> Matrix3x3& {
            m_Base[0] += m.m_Base[0];
            m_Base[1] += m.m_Base[1];
            m_Base[2] += m.m_Base[2];
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator-=(const Matrix3x3& m)noexcept -> Matrix3x3& {
            m_Base[0] -= m.m_Base[0];
            m_Base[1] -= m.m_Base[1];
            m_Base[2] -= m.m_Base[2];
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator*=(const Matrix3x3& m)noexcept -> Matrix3x3& {
            *this = (*this) * m;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator*=(const float    & s)noexcept -> Matrix3x3& {
            m_Base[0] *= s;
            m_Base[1] *= s;
            m_Base[2] *= s;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator/=(const float    & s)noexcept -> Matrix3x3& {
            m_Base[0] /= s;
            m_Base[1] /= s;
            m_Base[2] /= s;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator()(int x, int y)const -> const float& {
            return At(x, y);
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator()(int x, int y)      ->       float& {
            return At(x, y);
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCol(int idx)const noexcept -> const float3& {
            return m_Base[idx];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCol(int idx)      noexcept ->       float3& {
            return m_Base[idx];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto At(int x, int y)const noexcept -> const float& {
            return reinterpret_cast<const float*>(m_Base + x)[y];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto At(int x, int y)      noexcept -> float& {
            return reinterpret_cast<float*>(m_Base + x)[y];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetRow(int idx)const noexcept ->       float3 {
            return make_float3(At(0, idx), At(1, idx), At(2, idx));
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Det()const noexcept -> float {
            return RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(1, 1) * RTLIB_IMPL_MATRIX_AT(2, 2) +
                   RTLIB_IMPL_MATRIX_AT(0, 1) * RTLIB_IMPL_MATRIX_AT(1, 2) * RTLIB_IMPL_MATRIX_AT(2, 0) +
                   RTLIB_IMPL_MATRIX_AT(0, 2) * RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(2, 1) -
                   RTLIB_IMPL_MATRIX_AT(0, 2) * RTLIB_IMPL_MATRIX_AT(1, 1) * RTLIB_IMPL_MATRIX_AT(2, 0) -
                   RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(1, 2) * RTLIB_IMPL_MATRIX_AT(2, 1) -
                   RTLIB_IMPL_MATRIX_AT(0, 1) * RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(2, 2)
                ;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Transpose()const noexcept -> Matrix3x3
        {
            return Matrix3x3(
                make_float3(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2)),
                make_float3(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2)),
                make_float3(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2))
            );
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Inverse()const noexcept -> Matrix3x3 {
            float det = Det();
            return Matrix3x3(
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT(1, 1) * RTLIB_IMPL_MATRIX_AT(2,2) - RTLIB_IMPL_MATRIX_AT(2, 1) * RTLIB_IMPL_MATRIX_AT(1, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT(2 ,1) * RTLIB_IMPL_MATRIX_AT(0,2) - RTLIB_IMPL_MATRIX_AT(0, 1) * RTLIB_IMPL_MATRIX_AT(2, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT(0, 1) * RTLIB_IMPL_MATRIX_AT(1,2) - RTLIB_IMPL_MATRIX_AT(1, 1) * RTLIB_IMPL_MATRIX_AT(0, 2)) / det
                ),
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT(2, 0) * RTLIB_IMPL_MATRIX_AT(1, 2) - RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(2, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(2, 2) - RTLIB_IMPL_MATRIX_AT(2, 0) * RTLIB_IMPL_MATRIX_AT(0, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(0, 2) - RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(1, 2)) / det
                ),
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(2, 1) - RTLIB_IMPL_MATRIX_AT(2, 0) * RTLIB_IMPL_MATRIX_AT(1, 1)) / det,
                    (RTLIB_IMPL_MATRIX_AT(2, 0) * RTLIB_IMPL_MATRIX_AT(0, 1) - RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(2, 1)) / det,
                    (RTLIB_IMPL_MATRIX_AT(0, 0) * RTLIB_IMPL_MATRIX_AT(1, 1) - RTLIB_IMPL_MATRIX_AT(1, 0) * RTLIB_IMPL_MATRIX_AT(0, 1)) / det
                )
            );
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto InverseTranspose()const noexcept -> Matrix3x3 {
            float det = Det();
            return Matrix3x3(
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2)) / det
                ),
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2)) / det
                ),
                make_float3(
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1)) / det,
                    (RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1) - RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0) * RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1)) / det
                )
            );
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE void Show()const {
            printf("%f %f %f\n", m_Base[0].x, m_Base[1].x, m_Base[2].x);
            printf("%f %f %f\n", m_Base[0].y, m_Base[1].y, m_Base[2].y);
            printf("%f %f %f\n", m_Base[0].z, m_Base[1].z, m_Base[2].z);
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Identity()noexcept
        {
            return Matrix3x3(1.0f);
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Scaling(const float3& sv) {
            return Matrix3x3(
                make_float3(sv.x, 0.0f, 0.0f),
                make_float3(0.0f, sv.y, 0.0f),
                make_float3(0.0f, 0.0f, sv.z)
            );
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Rotate(const float3& axis, float angleRadians)noexcept -> Matrix3x3
        {
            auto  n = rtlib::normalize(axis);
            float s = ::sinf(angleRadians);
            float o_s = 1.0f - s;
            float c = ::cosf(angleRadians);
            float o_c = 1.0f - c;
            float n_x_x = n.x * n.x;
            float n_y_y = n.y * n.y;
            float n_z_z = n.z * n.z;
            float n_x_y = n.x * n.y;
            float n_z_x = n.z * n.x;
            float n_y_z = n.y * n.z;
            return Matrix3x3(
                make_float3(c + n_x_x * o_c, n_x_y * o_c + n.z * s, n_z_x * o_c - n.y * s),
                make_float3(n_x_y * o_c - n.z * s, c + n_y_y * o_c, n_y_z * o_c + n.x * s),
                make_float3(n_z_x * o_c + n.y * s, n_y_z * o_c + n.x * s, c + n_z_z * o_c)
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator+(const Matrix3x3& m1, const Matrix3x3& m2)noexcept
        {
            return Matrix3x3(
                m1.m_Base[0] + m2.m_Base[0],
                m1.m_Base[1] + m2.m_Base[1],
                m1.m_Base[2] + m2.m_Base[2]
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator-(const Matrix3x3& m1, const Matrix3x3& m2)noexcept
        {
            return Matrix3x3(
                m1.m_Base[0] - m2.m_Base[0],
                m1.m_Base[1] - m2.m_Base[1],
                m1.m_Base[2] - m2.m_Base[2]
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator*(const Matrix3x3& m1, const float      s2)noexcept {
            return Matrix3x3(
                m1.m_Base[0] * s2,
                m1.m_Base[1] * s2,
                m1.m_Base[2] * s2
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator*(const float      s2, const Matrix3x3& m1)noexcept {
            return Matrix3x3(
                m1.m_Base[0] * s2,
                m1.m_Base[1] * s2,
                m1.m_Base[2] * s2
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE float3    operator*(const Matrix3x3& m1, const float3&    m2)noexcept
        {
            return make_float3(
                m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.z,
                m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.z,
                m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.z
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator*(const Matrix3x3& m1, const Matrix3x3& m2)noexcept
        {
            return Matrix3x3(
                make_float3(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 2),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 2),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 2) 
                ),
                make_float3(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 2),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 2),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 2)
                ),
                make_float3(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 2),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) ,
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) 
                )
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3 operator/(const Matrix3x3& m1, const float      s2)noexcept {
            return Matrix3x3(
                m1.m_Base[0] / s2,
                m1.m_Base[1] / s2,
                m1.m_Base[2] / s2
            );
        }
    private:
        float3 m_Base[3] = {};
    };
    class Matrix4x4
    {
    public:
        RTLIB_INLINE RTLIB_HOST_DEVICE          Matrix4x4()noexcept : Matrix4x4(
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, 0.0f))
        {}
        RTLIB_INLINE RTLIB_HOST_DEVICE          Matrix4x4(const float4& c0, const float4& c1, const float4& c2, const float4& c3)noexcept : m_Base
        {
            c0,c1,c2,c3
        } {}
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Matrix4x4(const Matrix3x3& m)noexcept : m_Base
        {
            make_float4(m.GetCol(0),0.0f),
            make_float4(m.GetCol(1),0.0f),
            make_float4(m.GetCol(2),0.0f),
            make_float4(0.0f)
        } {}
        RTLIB_INLINE RTLIB_HOST_DEVICE explicit Matrix4x4(float s)noexcept : Matrix4x4(
            make_float4(s, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, s, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, s, 0.0f),
            make_float4(0.0f, 0.0f, 0.0f, s))
        {}
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator+=(const Matrix4x4& m)noexcept -> Matrix4x4& {
            m_Base[0] += m.m_Base[0];
            m_Base[1] += m.m_Base[1];
            m_Base[2] += m.m_Base[2];
            m_Base[3] += m.m_Base[3];
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator-=(const Matrix4x4& m)noexcept -> Matrix4x4& {
            m_Base[0] -= m.m_Base[0];
            m_Base[1] -= m.m_Base[1];
            m_Base[2] -= m.m_Base[2];
            m_Base[3] -= m.m_Base[3];
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator*=(const Matrix4x4& m)noexcept -> Matrix4x4& {
            *this = (*this) * m;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator*=(const float& s)noexcept -> Matrix4x4& {
            m_Base[0] *= s;
            m_Base[1] *= s;
            m_Base[2] *= s;
            m_Base[3] *= s;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator/=(const float& s)noexcept -> Matrix4x4& {
            m_Base[0] /= s;
            m_Base[1] /= s;
            m_Base[2] /= s;
            m_Base[3] /= s;
            return *this;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator()(int x, int y)const -> const float& {
            return At(x, y);
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto operator()(int x, int y)      ->       float& {
            return At(x, y);
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCol(int idx)const noexcept -> const float4& {
            return m_Base[idx];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCol(int idx)      noexcept ->       float4& {
            return m_Base[idx];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto At(int x, int y)const noexcept -> const float& {
            return reinterpret_cast<const float*>(m_Base + x)[y];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto At(int x, int y)      noexcept -> float& {
            return reinterpret_cast<float*>(m_Base + x)[y];
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetRow(int idx)const noexcept ->       float4 {
            return make_float4(At(0, idx), At(1, idx), At(2, idx), At(3, idx));
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Det()const noexcept -> float {
            return 
                RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 1, 2, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 2, 2, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 3, 2, 1, 3, 2)+
                RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 0, 2, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 2, 2, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 3, 2, 2, 3, 0)+
                RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 0, 2, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 1, 2, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 3, 2, 0, 3, 1)+
                RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 0, 2, 2, 3, 1) + RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 1, 2, 0, 3, 2) + RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 2, 2, 1, 3, 0)-
                RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 1, 2, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 2, 2, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_4(0, 0, 1, 3, 2, 2, 3, 1)-
                RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 0, 2, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 2, 2, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_4(0, 1, 1, 3, 2, 0, 3, 2)-
                RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 0, 2, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 1, 2, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_4(0, 2, 1, 3, 2, 1, 3, 0)-
                RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 0, 2, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 1, 2, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_4(0, 3, 1, 2, 2, 0, 3, 1)
                ;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Transpose()const noexcept -> Matrix4x4
        {
            return Matrix4x4(
                make_float4(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 2), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(0, 3)),
                make_float4(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 2), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(1, 3)),
                make_float4(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 2), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(2, 3)), 
                make_float4(RTLIB_IMPL_MATRIX_AT_TRANSPOSE(3, 0), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(3, 1), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(3, 2), RTLIB_IMPL_MATRIX_AT_TRANSPOSE(3, 3))
            );
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Inverse()const noexcept -> Matrix4x4 {
            float det = Det();
            const auto b_00 = RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 2, 3, 1);
            const auto b_01 = RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 2, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 1, 3, 2);
            const auto b_02 = RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 2, 3, 1);
            const auto b_03 = RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 3, 2, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 1, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 2, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 2, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 3, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 1, 2, 2);
            const auto b_10 = RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 0, 3, 2);
            const auto b_11 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 0, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 2, 3, 0);
            const auto b_12 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 0, 3, 2);
            const auto b_13 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 2, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 3, 2, 0) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 0, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 3, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 0, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 2, 2, 0);
            const auto b_20 = RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(1, 3, 2, 1, 3, 0);
            const auto b_21 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 2, 0, 3, 1);
            const auto b_22 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 1, 3, 0);
            const auto b_23 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 3, 2, 1) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 0, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 1, 2, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 1, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 3, 2, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 3, 1, 0, 2, 1);
            const auto b_30 = RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 2, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 0, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(1, 0, 2, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(1, 1, 2, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(1, 2, 2, 0, 3, 1);
            const auto b_31 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 1, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 2, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 2, 2, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 2, 0, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 2, 1, 3, 0);
            const auto b_32 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 2, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 0, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 0, 3, 1);
            const auto b_33 = RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 1, 2, 2) + RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 2, 2, 0) + RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 0, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 0, 1, 2, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3(0, 1, 1, 0, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3(0, 2, 1, 1, 2, 0);
            return Matrix4x4
            (
                make_float4(b_00 / det, b_01 / det, b_02 / det, b_03 / det),
                make_float4(b_10 / det, b_11 / det, b_12 / det, b_13 / det),
                make_float4(b_20 / det, b_21 / det, b_22 / det, b_23 / det),
                make_float4(b_30 / det, b_31 / det, b_32 / det, b_33 / det)
           );

        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto InverseTranspose()const noexcept -> Matrix4x4 {
            float det = Det();
            const auto b_00 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 2, 3, 1);
            const auto b_01 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 2, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 1, 3, 2);
            const auto b_02 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 2, 3, 1);
            const auto b_03 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 3, 2, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 1, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 2, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 2, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 3, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 1, 2, 2);
            const auto b_10 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 0, 3, 2);
            const auto b_11 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 2, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 0, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 3, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 2, 3, 0);
            const auto b_12 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 3, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 2, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 0, 3, 2);
            const auto b_13 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 2, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 3, 2, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 0, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 3, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 0, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 2, 2, 0);
            const auto b_20 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 3, 2, 1, 3, 0);
            const auto b_21 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 3, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 0, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 1, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 3, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 2, 0, 3, 1);
            const auto b_22 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 1, 3, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 3, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 3, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 0, 3, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 1, 3, 0);
            const auto b_23 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 3, 2, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 0, 2, 3) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 1, 2, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 1, 2, 3) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 3, 2, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 3, 1, 0, 2, 1);
            const auto b_30 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 2, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 0, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 0, 2, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 1, 2, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(1, 2, 2, 0, 3, 1);
            const auto b_31 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 1, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 2, 3, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 0, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 2, 2, 3, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 2, 0, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 2, 1, 3, 0);
            const auto b_32 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 2, 3, 1) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 0, 3, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 1, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 1, 3, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 2, 3, 0) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 0, 3, 1);
            const auto b_33 = RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 1, 2, 2) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 2, 2, 0) + RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 0, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 0, 1, 2, 2, 1) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 1, 1, 0, 2, 2) - RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE(0, 2, 1, 1, 2, 0);
            return Matrix4x4
            (
                make_float4(b_00 / det, b_01 / det, b_02 / det, b_03 / det),
                make_float4(b_10 / det, b_11 / det, b_12 / det, b_13 / det),
                make_float4(b_20 / det, b_21 / det, b_22 / det, b_23 / det),
                make_float4(b_30 / det, b_31 / det, b_32 / det, b_33 / det)
            );
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE void Show()const {
            printf("%f %f %f %f\n", m_Base[0].x, m_Base[1].x, m_Base[2].x, m_Base[3].x);
            printf("%f %f %f %f\n", m_Base[0].y, m_Base[1].y, m_Base[2].y, m_Base[3].y);
            printf("%f %f %f %f\n", m_Base[0].z, m_Base[1].z, m_Base[2].z, m_Base[3].z);
            printf("%f %f %f %f\n", m_Base[0].w, m_Base[1].w, m_Base[2].w, m_Base[3].w);
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Identity()noexcept->Matrix4x4
        {
            return Matrix4x4(1.0f);
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Scaling(const float3& sv) ->Matrix4x4 {
            return Matrix4x4(
                make_float4(sv.x, 0.0f, 0.0f, 0.0f),
                make_float4(0.0f, sv.y, 0.0f, 0.0f),
                make_float4(0.0f, 0.0f, sv.z, 0.0f),
                make_float4(0.0f, 0.0f, 0.0f, 1.0f)
            );
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Translate(const float3& sv) ->Matrix4x4 {
            return Matrix4x4(
                make_float4( 1.0f, 0.0f, 0.0f, 0.0f),
                make_float4( 0.0f, 1.0f, 0.0f, 0.0f),
                make_float4( 0.0f, 0.0f, 1.0f, 0.0f),
                make_float4(-sv.x,-sv.y,-sv.z, 1.0f)
            );
        }
        static RTLIB_INLINE RTLIB_HOST_DEVICE auto Rotate(const float3& axis, float angleRadians)noexcept -> Matrix4x4
        {
            //BUG!
            auto  n     = rtlib::normalize(axis);
            float s     = ::sinf(angleRadians);
            float c     = ::cosf(angleRadians);
            auto  o_c   = (1.0f - c) * n;
            float n_x   = n.x;
            float n_y   = n.y;
            float n_z   = n.z;
            float n_x_x = n_x * n_x;
            float n_y_y = n_y * n_y;
            float n_z_z = n_z * n_z;
            float n_x_y = n_x * n_y;
            float n_z_x = n_y * n_x;
            float n_y_z = n_y * n_z;
            return Matrix4x4(
                make_float4(make_float3(       c, n_z * s, -n_y * s) + n_x * o_c, 0.0f),
                make_float4(make_float3(-n_z * s,       c,  n_x * s) + n_y * o_c, 0.0f),
                make_float4(make_float3( n_y * s,-n_x * s,        c) + n_z * o_c, 0.0f),
                make_float4(                0.0f,    0.0f,                  0.0f, 1.0f)
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator+(const Matrix4x4& m1, const Matrix4x4& m2)noexcept
        {
            return Matrix4x4(
                m1.m_Base[0] + m2.m_Base[0],
                m1.m_Base[1] + m2.m_Base[1],
                m1.m_Base[2] + m2.m_Base[2],
                m1.m_Base[3] + m2.m_Base[3]
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator-(const Matrix4x4& m1, const Matrix4x4& m2)noexcept
        {
            return Matrix4x4(
                m1.m_Base[0] - m2.m_Base[0],
                m1.m_Base[1] - m2.m_Base[1],
                m1.m_Base[2] - m2.m_Base[2],
                m1.m_Base[3] - m2.m_Base[3]
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator*(const Matrix4x4& m1, const float      s2)noexcept {
            return Matrix4x4(
                m1.m_Base[0] * s2,
                m1.m_Base[1] * s2,
                m1.m_Base[2] * s2,
                m1.m_Base[3] * s2
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator*(const float      s2, const Matrix4x4& m1)noexcept {
            return Matrix4x4(
                m1.m_Base[0] * s2,
                m1.m_Base[1] * s2,
                m1.m_Base[2] * s2,
                m1.m_Base[3] * s2
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE float4    operator*(const Matrix4x4& m1, const float4& m2)noexcept
        {
            return make_float4(
                m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.z +
                m1.RTLIB_IMPL_MATRIX_AT(0, 3) * m2.w,
                m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.z +
                m1.RTLIB_IMPL_MATRIX_AT(1, 3) * m2.w,
                m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.z +
                m1.RTLIB_IMPL_MATRIX_AT(2, 3) * m2.w,
                m1.RTLIB_IMPL_MATRIX_AT(3, 0) * m2.x +
                m1.RTLIB_IMPL_MATRIX_AT(3, 1) * m2.y +
                m1.RTLIB_IMPL_MATRIX_AT(3, 2) * m2.z +
                m1.RTLIB_IMPL_MATRIX_AT(3, 3) * m2.w
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator*(const Matrix4x4& m1, const Matrix4x4& m2)noexcept
        {
            return Matrix4x4(
                make_float4(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 0) * m2.RTLIB_IMPL_MATRIX_AT(0, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 1) * m2.RTLIB_IMPL_MATRIX_AT(0, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 2) * m2.RTLIB_IMPL_MATRIX_AT(0, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 3) * m2.RTLIB_IMPL_MATRIX_AT(0, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 3) * m2.RTLIB_IMPL_MATRIX_AT(0, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 3) * m2.RTLIB_IMPL_MATRIX_AT(0, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 3) * m2.RTLIB_IMPL_MATRIX_AT(0, 3)
                ),
                make_float4(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 0) * m2.RTLIB_IMPL_MATRIX_AT(1, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 1) * m2.RTLIB_IMPL_MATRIX_AT(1, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 2) * m2.RTLIB_IMPL_MATRIX_AT(1, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 3) * m2.RTLIB_IMPL_MATRIX_AT(1, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 3) * m2.RTLIB_IMPL_MATRIX_AT(1, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 3) * m2.RTLIB_IMPL_MATRIX_AT(1, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 3) * m2.RTLIB_IMPL_MATRIX_AT(1, 3)
                ),
                make_float4(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 0) * m2.RTLIB_IMPL_MATRIX_AT(2, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 1) * m2.RTLIB_IMPL_MATRIX_AT(2, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 2) * m2.RTLIB_IMPL_MATRIX_AT(2, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 3) * m2.RTLIB_IMPL_MATRIX_AT(2, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 3) * m2.RTLIB_IMPL_MATRIX_AT(2, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 3) * m2.RTLIB_IMPL_MATRIX_AT(2, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 3) * m2.RTLIB_IMPL_MATRIX_AT(2, 3)
                ),
                make_float4(
                    m1.RTLIB_IMPL_MATRIX_AT(0, 0) * m2.RTLIB_IMPL_MATRIX_AT(3, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 0) * m2.RTLIB_IMPL_MATRIX_AT(3, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 0) * m2.RTLIB_IMPL_MATRIX_AT(3, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 0) * m2.RTLIB_IMPL_MATRIX_AT(3, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 1) * m2.RTLIB_IMPL_MATRIX_AT(3, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 1) * m2.RTLIB_IMPL_MATRIX_AT(3, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 1) * m2.RTLIB_IMPL_MATRIX_AT(3, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 1) * m2.RTLIB_IMPL_MATRIX_AT(3, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 2) * m2.RTLIB_IMPL_MATRIX_AT(3, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 2) * m2.RTLIB_IMPL_MATRIX_AT(3, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 2) * m2.RTLIB_IMPL_MATRIX_AT(3, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 2) * m2.RTLIB_IMPL_MATRIX_AT(3, 3),
                    m1.RTLIB_IMPL_MATRIX_AT(0, 3) * m2.RTLIB_IMPL_MATRIX_AT(3, 0) +
                    m1.RTLIB_IMPL_MATRIX_AT(1, 3) * m2.RTLIB_IMPL_MATRIX_AT(3, 1) +
                    m1.RTLIB_IMPL_MATRIX_AT(2, 3) * m2.RTLIB_IMPL_MATRIX_AT(3, 2) +
                    m1.RTLIB_IMPL_MATRIX_AT(3, 3) * m2.RTLIB_IMPL_MATRIX_AT(3, 3)
                )
            );
        }
        friend RTLIB_INLINE RTLIB_HOST_DEVICE Matrix4x4 operator/(const Matrix4x4& m1, const float      s2)noexcept {
            return Matrix4x4(
                m1.m_Base[0] / s2,
                m1.m_Base[1] / s2,
                m1.m_Base[2] / s2,
                m1.m_Base[3] / s2
            );
        }
    private:
        float4 m_Base[4] = {};
    };
    RTLIB_INLINE RTLIB_HOST_DEVICE Matrix3x3::Matrix3x3(const Matrix4x4& m)noexcept :m_Base
    {
        make_float3(m.GetCol(0).x,m.GetCol(0).y ,m.GetCol(0).z),
        make_float3(m.GetCol(1).x,m.GetCol(1).y ,m.GetCol(1).z),
        make_float3(m.GetCol(2).x,m.GetCol(2).y ,m.GetCol(2).z) 
    }
    {}
}

#undef  RTLIB_IMPL_MATRIX_MUL_2
#undef  RTLIB_IMPL_MATRIX_MUL_2_TRANSPOSE
#undef  RTLIB_IMPL_MATRIX_MUL_3
#undef  RTLIB_IMPL_MATRIX_MUL_3_TRANSPOSE
#undef  RTLIB_IMPL_MATRIX_MUL_4
#undef  RTLIB_IMPL_MATRIX_MUL_4_TRANSPOSE
#undef  RTLIB_IMPL_MATRIX_AT_TRANSPOSE
#undef  RTLIB_IMPL_MATRIX_AT 
#undef  RTLIB_IMPL_MATRIX_AT_0 
#undef  RTLIB_IMPL_MATRIX_AT_1  
#undef  RTLIB_IMPL_MATRIX_AT_2 
#undef  RTLIB_IMPL_MATRIX_AT_3 
#endif