#ifndef RTLIB_UTILS_H
#define RTLIB_UTILS_H
#include <tiny_obj_loader.h>
namespace rtlib{
    namespace utils {
        enum AxisFlag {
            AXIS_FLAG_YZ = 0,
            AXIS_FLAG_ZX = 1,
            AXIS_FLAG_XY = 2,
        };
        struct   Rect {
            float    x0;
            float    x1;
            float    y0;
            float    y1;
            float    z;
            AxisFlag axis;
        public:
            auto getVertices()const noexcept-> std::array<float3, 4>{
                std::array<float3, 4> vertices = {};
                auto axisX = (axis + 1) % 3;
                auto axisY = (axis + 2) % 3;
                auto axisZ = (axis + 3) % 3;
                float vertex0[3] = {};
                vertex0[axisX] = x0;
                vertex0[axisY] = y0;
                vertex0[axisZ] = z;
                float vertexX[3] = {};
                vertexX[axisX] = x1;
                vertexX[axisY] = y0;
                vertexX[axisZ] = z;
                float vertexY[3] = {};
                vertexY[axisX] = x0;
                vertexY[axisY] = y1;
                vertexY[axisZ] = z;
                float vertex1[3] = {};
                vertex1[axisX] = x1;
                vertex1[axisY] = y1;
                vertex1[axisZ] = z;
                vertices[toIndex(0, 0)].x = vertex0[0];
                vertices[toIndex(0, 0)].y = vertex0[1];
                vertices[toIndex(0, 0)].z = vertex0[2];
                vertices[toIndex(1, 0)].x = vertexX[0];
                vertices[toIndex(1, 0)].y = vertexX[1];
                vertices[toIndex(1, 0)].z = vertexX[2];
                vertices[toIndex(1, 1)].x = vertex1[0];
                vertices[toIndex(1, 1)].y = vertex1[1];
                vertices[toIndex(1, 1)].z = vertex1[2];
                vertices[toIndex(0, 1)].x = vertexY[0];
                vertices[toIndex(0, 1)].y = vertexY[1];
                vertices[toIndex(0, 1)].z = vertexY[2];
                return vertices;
            }
            auto getIndices()const noexcept-> std::array<uint3, 2> {
                std::array<uint3, 2> indices = {};
                auto i00 = toIndex(0, 0);
                auto i10 = toIndex(1, 0);
                auto i11 = toIndex(1, 1);
                auto i01 = toIndex(0, 1);
                indices[0].x = i00;
                indices[0].y = i10;
                indices[0].z = i11;
                indices[1].x = i11;
                indices[1].y = i01;
                indices[1].z = i00;
                return indices;
            }
        private:
            static constexpr auto toIndex(uint32_t x, uint32_t y)noexcept->uint32_t {
                return 2 * y + x;
            }
        };
        struct    Box {
            float    x0;
            float    x1;
            float    y0;
            float    y1;
            float    z0;
            float    z1;
            auto getVertices()const noexcept-> std::array<float3, 8>{
                std::array<float3, 8> vertices = {};
                //z: 0->
                vertices[toIndex(0, 0, 0)].x = x0;
                vertices[toIndex(0, 0, 0)].y = y0;
                vertices[toIndex(0, 0, 0)].z = z0;
                vertices[toIndex(1, 0, 0)].x = x1;
                vertices[toIndex(1, 0, 0)].y = y0;
                vertices[toIndex(1, 0, 0)].z = z0;
                vertices[toIndex(0, 1, 0)].x = x0;
                vertices[toIndex(0, 1, 0)].y = y1;
                vertices[toIndex(0, 1, 0)].z = z0;
                vertices[toIndex(1, 1, 0)].x = x1;
                vertices[toIndex(1, 1, 0)].y = y1;
                vertices[toIndex(1, 1, 0)].z = z0;
                //z: 1->
                vertices[toIndex(0, 0, 1)].x = x0;
                vertices[toIndex(0, 0, 1)].y = y0;
                vertices[toIndex(0, 0, 1)].z = z1;
                vertices[toIndex(1, 0, 1)].x = x1;
                vertices[toIndex(1, 0, 1)].y = y0;
                vertices[toIndex(1, 0, 1)].z = z1;
                vertices[toIndex(0, 1, 1)].x = x0;
                vertices[toIndex(0, 1, 1)].y = y1;
                vertices[toIndex(0, 1, 1)].z = z1;
                vertices[toIndex(1, 1, 1)].x = x1;
                vertices[toIndex(1, 1, 1)].y = y1;
                vertices[toIndex(1, 1, 1)].z = z1;
                return vertices;
            }
            auto getIndices()const noexcept-> std::array<uint3, 12> {
                std::array<uint3, 12> indices = {};
                for (uint32_t i = 0; i < 3; ++i) {
                    for (uint32_t j = 0; j < 2; ++j) {
                        uint32_t index[3] = {};
                        //x...y...z
                        uint32_t iX = (i + 1) % 3;
                        uint32_t iY = (i + 2) % 3;
                        uint32_t iZ = (i + 3) % 3;
                        index[iX] = 0;
                        index[iY] = 0;
                        index[iZ] = j;
                        auto i00 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 1;
                        index[iY] = 0;
                        index[iZ] = j;
                        auto i10 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 1;
                        index[iY] = 1;
                        index[iZ] = j;
                        auto i11 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 0;
                        index[iY] = 1;
                        index[iZ] = j;
                        auto i01 = toIndex(index[0], index[1], index[2]);
                        indices[2 * (2 * i + j) + 0].x = i00;
                        indices[2 * (2 * i + j) + 0].y = i10;
                        indices[2 * (2 * i + j) + 0].z = i11;
                        indices[2 * (2 * i + j) + 1].x = i11;
                        indices[2 * (2 * i + j) + 1].y = i01;
                        indices[2 * (2 * i + j) + 1].z = i00;
                    }
                }
                return indices;
            }
        private:
            static constexpr auto toIndex(uint32_t x, uint32_t y, uint32_t z)noexcept->uint32_t {
                return 4 * z + 2 * y + x;
            }
        };
    }
}
#endif