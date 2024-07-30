#pragma once
#include <cstring>
#include <cmath>
#include "domain.h"

namespace SwirlCraft
{
    namespace AdvecUtil
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
    }
    
    template <typename T, size_t Dims>
    void advectScalarField(T* f, const T* vel[Dims], const Domain<T, Dims>& domain, const T dt)
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
                f[I] = AdvecUtil::domainNearestInterpolate(f_old, u, domain);
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