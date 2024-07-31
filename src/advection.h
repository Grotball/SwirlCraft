#pragma once
#include <cstring>
#include <cmath>
#include "domain.h"

namespace SwirlCraft
{
    namespace AdvectUtil
    {
        template <typename T>
        T lerp(const T a, const T b, const T bias)
        {
            return (1 - bias) * a + bias * b;
        }
        
        template <typename T, size_t Dims>
        T domainNearestInterpolate(T* f, const T (&x)[Dims], const T* collision, const Domain<T, Dims>& domain)
        {
            int32_t I[Dims];
            for (size_t i = 0; i < Dims; i++)
            {
                I[i] = static_cast<int32_t>(std::round(x[i]));
            }
            const auto index = linearIndex(I, domain);
            return collision[index] > 0 ? f[index] : 0;
        }
        
        template <typename T>
        T domainBilinearInterpolate(T* f, const T(&x)[2UL], const T* collision, const Domain<T, 2UL>& domain)
        {
            int32_t ci = static_cast<int32_t>(std::round(x[0]));
            int32_t cj = static_cast<int32_t>(std::round(x[1]));

            if (collision[ci * domain.dims[0].stride + cj * domain.dims[1].stride] <= 0)
            {
                return static_cast<T>(0);
            }

            int32_t i1 = static_cast<int32_t>(std::floor(x[0]));
            int32_t i2 = static_cast<int32_t>(std::ceil(x[0]));
            int32_t j1 = static_cast<int32_t>(std::floor(x[1]));
            int32_t j2 = static_cast<int32_t>(std::ceil(x[1]));
            auto index11 = i1 * domain.dims[0].stride + j1 * domain.dims[1].stride;
            auto index12 = i1 * domain.dims[0].stride + j2 * domain.dims[1].stride;
            auto index21 = i2 * domain.dims[0].stride + j1 * domain.dims[1].stride;
            auto index22 = i2 * domain.dims[0].stride + j2 * domain.dims[1].stride;

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
        T domainTrilinearInterpolate(T* f, const T(&x)[3UL], const T* collision, const Domain<T, 3UL>& domain)
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

            if (collision[linearIndex(c, domain)] <= 0)
            {
                return static_cast<T>(0);
            }

            auto index0 = linearIndex(I0, domain);
            
            T A[2][2];

            for (auto i = 0; i < 2; i++)
            {
                for (auto j = 0; j < 2; j++)
                {
                    auto k = index0 + i * domain.dims[0].stride + j * domain.dims[1].stride;
                    A[i][j] = lerp(f[k], f[k+domain.dims[2].stride], u[2]);
                }
            }

            T b1 = lerp(A[0][0], A[0][1], u[1]);
            T b2 = lerp(A[1][0], A[1][1], u[1]);
            
            return lerp(b1, b2, u[0]);
        }

        // Currently linear interpolation can only be done in 2 and 3 dimensions.
        // Nearest interpolation is used as fallback when Dims is not 2 or 3.
        template <typename T, size_t Dims>
        T domainInterpolate(T* f, const T (&x)[Dims], const T* collision, const Domain<T, Dims>& domain)
        {
            if constexpr (Dims == 2)
            {
                return domainBilinearInterpolate(f, x, collision, domain);
            }
            else if constexpr (Dims == 3)
            {
                return domainTrilinerInterpolate(f, x, collision, domain);
            }
            return domainNearestInterpolate(f, x, collision, domain);
        }
    }
    
    template <typename T, size_t Dims>
    void advectScalarField(T* f, const T* (&vel)[Dims], const T* collision, const Domain<T, Dims>& domain, const T dt)
    {
        T* f_old = new T[domain.N];
        std::memcpy(f_old, f, domain.N * sizeof(T));

        for (size_t i = 0; i < domain.N; i++)
        {
            if (collision[i] > 0)
            {
                bool inDomain = true;
                T u[Dims];
                size_t I[Dims];
                cartesianIndex(I, i, domain);
                for (size_t j = 0; j < Dims; j++)
                {
                    u[j] = I[j] - vel[j][i] * dt / domain.dims[j].dx;
                    if (u[j] < 0 || u[j] > domain.dims[j].n)
                    {
                        inDomain = false;
                        break;
                    }
                }

                if (inDomain)
                {
                    f[I] = AdvectUtil::domainInterpolate(f_old, u, collision, domain);
                }
            }

        }

        delete[] f_old;
    }

    template <typename T, size_t Dims>
    void advectVectorField(T* (&F)[Dims], const T* (&vel)[Dims], const T* collision, const Domain<T, Dims>& domain, const T dt)
    {
        for (size_t i = 0; i < Dims; i++)
        {
            advectScalarField(F[i], vel, collision, domain, dt);
        }
    }
}