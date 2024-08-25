#pragma once
#include <cstring>
#include <cmath>
#include "grid.h"
#include "arch.h"

namespace SwirlCraft
{
    namespace AdvectUtil
    {
        template <typename T>
        T lerp(const T a, const T b, const T bias)
        {
            return (1 - bias) * a + bias * b;
        }
        
        template <typename T, uint32_t Dims>
        T domainNearestInterpolate(T* f, const T (&x)[Dims], const T* collision, const Grid<T, Dims>& grid)
        {
            int32_t I[Dims];
            for (uint32_t i = 0; i < Dims; i++)
            {
                I[i] = static_cast<int32_t>(std::round(x[i]));
            }
            const auto index = linearIndex(I, grid);
            return collision[index] > 0 ? f[index] : 0;
        }
        
        template <typename T>
        T domainBilinearInterpolate(T* f, const T(&x)[2U], const T* collision, const Grid<T, 2U>& grid)
        {
            int32_t ci = static_cast<int32_t>(std::round(x[0]));
            int32_t cj = static_cast<int32_t>(std::round(x[1]));

            if (collision[ci * grid.stride[0] + cj * grid.stride[1]] <= 0)
            {
                return static_cast<T>(0);
            }

            int32_t i1 = static_cast<int32_t>(std::floor(x[0]));
            int32_t i2 = static_cast<int32_t>(std::ceil(x[0]));
            int32_t j1 = static_cast<int32_t>(std::floor(x[1]));
            int32_t j2 = static_cast<int32_t>(std::ceil(x[1]));
            auto index11 = i1 * grid.stride[0] + j1 * grid.stride[1];
            auto index12 = i1 * grid.stride[0] + j2 * grid.stride[1];
            auto index21 = i2 * grid.stride[0] + j1 * grid.stride[1];
            auto index22 = i2 * grid.stride[0] + j2 * grid.stride[1];

            auto a11 = f[index11];
            auto a12 = f[index12];
            auto a21 = f[index21];
            auto a22 = f[index22];

            T s = x[0] - i1;
            T t = x[1] - j1;

            T b1 = lerp(a11, a12, t);
            T b2 = lerp(a21, a22, t);
            
            return lerp(b1, b2, s);
        }

        template <typename T>
        T domainTrilinearInterpolate(T* f, const T(&x)[3U], const T* collision, const Grid<T, 3U>& grid)
        {
            int32_t I0[3];
            T u[3];
            T c[3];
            for (int i = 0; i < 3; i++)
            {
                c[i] = static_cast<int32_t>(std::round(x[i]));
                I0[i] = static_cast<int32_t>(std::floor(x[i]));
                u[i] = x[i] - I0[i];
            }

            if (collision[linearIndex(c, grid)] <= 0)
            {
                return static_cast<T>(0);
            }

            auto index0 = linearIndex(I0, grid);
            
            T A[2][2];

            for (auto i = 0; i < 2; i++)
            {
                for (auto j = 0; j < 2; j++)
                {
                    auto k = index0 + i * grid.stride[0] + j * grid.stride[1];
                    A[i][j] = lerp(f[k], f[k+grid.stride[2]], u[2]);
                }
            }

            T b1 = lerp(A[0][0], A[0][1], u[1]);
            T b2 = lerp(A[1][0], A[1][1], u[1]);
            
            return lerp(b1, b2, u[0]);
        }

        // Currently linear interpolation can only be done in 2 and 3 dimensions.
        // Nearest interpolation is used as fallback when Dims is not 2 or 3.
        template <typename T, uint32_t Dims>
        T domainInterpolate(T* f, const T (&x)[Dims], const T* collision, const Grid<T, Dims>& grid)
        {
            if constexpr (Dims == 2)
            {
                return domainBilinearInterpolate(f, x, collision, grid);
            }
            else if constexpr (Dims == 3)
            {
                return domainTrilinerInterpolate(f, x, collision, grid);
            }
            return domainNearestInterpolate(f, x, collision, grid);
        }
    }
    
    template <typename T, uint32_t Dims>
    void advectScalarField(T* f, T* f_old, const T* (&vel)[Dims], const T* collision, const Grid<T, Dims>& grid, const T dt, bool copy_f=true)
    {
        if (copy_f)
        {
            std::memcpy(f_old, f, grid.N * sizeof(T));
        }
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                bool inDomain = true;
                T u[Dims];
                size_t I[Dims];
                cartesianIndex(I, i, grid);
                for (size_t j = 0; j < Dims; j++)
                {
                    u[j] = I[j] - vel[j][i] * dt / grid.dx[j];
                    if (u[j] < 0 || u[j] > (grid.size[j]-1))
                    {
                        inDomain = false;
                        break;
                    }
                }

                if (inDomain)
                {
                    f[i] = AdvectUtil::domainInterpolate(f_old, u, collision, grid);
                }
            }

        }
    }

    template <typename T, uint32_t Dims>
    void advectScalarField(T* f, const T* (&vel)[Dims], const T* collision, const Grid<T, Dims>& grid, const T dt)
    {
        T* f_old = new T[grid.N];
        advectScalarField(f, f_old, vel, collision, grid, dt, true);
        delete[] f_old;
    }

    template <typename T, uint32_t Dims>
    void advectVectorField(T* (&F)[Dims], const T* (&vel)[Dims], const T* collision, const Grid<T, Dims>& grid, const T dt)
    {
        for (size_t i = 0; i < Dims; i++)
        {
            advectScalarField(F[i], vel, collision, grid, dt);
        }
    }
}