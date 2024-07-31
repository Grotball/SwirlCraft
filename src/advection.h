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
        T domainNearestInterpolate(T* f, const T (&x)[Dims], const Domain<T, Dims>& domain)
        {
            int32_t I[Dims];
            for (size_t i = 0; i < Dims; i++)
            {
                I[i] = static_cast<int32_t>(std::round(x[i]));
            }
            const auto index = linearIndex(I, domain);
            return f[index];
        }
        
        template <typename T>
        T domainBilinearInterpolate(T* f, const T(&x)[2UL], const Domain<T, 2UL>& domain)
        {
            int32_t ci = static_cast<int32_t>(std::round(x[0]));
            int32_t cj = static_cast<int32_t>(std::round(x[1]));

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
    }
    
    template <typename T, size_t Dims>
    void advectScalarField(T* f, const T* (&vel)[Dims], const Domain<T, Dims>& domain, const T dt)
    {
        T* f_old = new T[domain.N];
        std::memcpy(f_old, f, domain.N * sizeof(T));

        for (size_t i = 0; i < domain.N; i++)
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
                f[I] = AdvectUtil::domainNearestInterpolate(f_old, u, domain);
            }

        }

        delete[] f_old;
    }

    template <typename T, size_t Dims>
    void advectVectorField(T* F[Dims], const T* vel[Dims], const Domain<T, Dims>& domain)
    {
        for (size_t i = 0; i < Dims; i++)
        {
            advectScalarField(F[i], vel, domain);
        }
    }
}