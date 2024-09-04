#pragma once
#include "../grid.h"

namespace SwirlCraft
{
    
    template <typename T, uint32_t Dims>
    void conjugateGradientSolve(T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations, const T epsilon)
    {
        T dxn2[Dims];
        for (uint32_t i = 0; i < Dims; i++)
        {
            auto dx = grid.dx[i];
            dxn2[i] = 1 / (dx*dx);
        }

        T* r = new T[grid.N];
        T* p = new T[grid.N];
        T* v = new T[grid.N];

        T res2_sum = 0;
        T g2_sum = 0;
        
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                r[i] = g[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    auto a1 = collision[i-stride] > 0 ? f[i-stride] : f[i];
                    auto a2 = collision[i+stride] > 0 ? f[i+stride] : f[i];
                    r[i] -= (a1 + a2 - 2 * f[i]) * dxn2[j];
                }
                g2_sum += g[i]*g[i];
                res2_sum += r[i]*r[i];
            }
            else
            {
                r[i] = 0;
            }
            p[i] = r[i];
            v[i] = 0;
        }

        const T tol = epsilon * epsilon * g2_sum;
        int32_t iter = 0;
        while (iter < maxIterations && tol < res2_sum)
        {
            auto res2_sum_old = res2_sum;

            for (size_t i = 0; i < grid.N; i++)
            {
                v[i] = 0;
                if (collision[i] > 0)
                {
                    for (uint32_t j = 0; j < Dims; j++)
                    {
                        const auto stride = grid.stride[j];
                        auto a1 = collision[i-stride] > 0 ? p[i-stride] : p[i];
                        auto a2 = collision[i+stride] > 0 ? p[i+stride] : p[i];
                        v[i] += (a1 + a2 - 2 * p[i]) * dxn2[j];
                    }     
                }
            }

            T p_dot_v = 0;
            for (size_t i = 0; i < grid.N; i++)
            {
                p_dot_v += p[i] * v[i];
            }

            auto alpha = res2_sum_old / p_dot_v;
            
            for (size_t i = 0; i < grid.N; i++)
            {
                r[i] -= alpha * v[i];
                f[i] += alpha * p[i];
            }

            res2_sum = 0;
            for (size_t i = 0; i < grid.N; i++)
            {
                res2_sum += r[i] * r[i];
            }

            const T beta = res2_sum / res2_sum_old;

            for (size_t i = 0; i < grid.N; i++)
            {
                p[i] = r[i] + beta * p[i];
            }

            iter++;
        }

        delete[] r;
        delete[] p;
        delete[] v;
    }
}