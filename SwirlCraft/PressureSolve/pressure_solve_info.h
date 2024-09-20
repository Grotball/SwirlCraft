#pragma once
#include <cmath>
#include <cstdint>
#include <chrono>
#include "../grid.h"

namespace SwirlCraft
{
    struct PressureSolveInfo
    {
        int32_t iterations;
        std::chrono::duration<double> solve_time;
        double residual_norm;
    };

    template <typename T, uint32_t Dims>
    T pressureSolveResidualNorm(const T* f, const T* g, const T* collision, const Grid<T, Dims>& grid)
    {
        T dxn2[Dims];
        for (uint32_t j = 0; j < Dims; j++)
        {
            auto dx = grid.dx[j];
            dxn2[j] = 1 / (dx*dx);
        }
        
        T res_sum = 0;
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                auto res = g[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    const auto a1 = collision[i-stride] > 0 ? f[i-stride] : f[i];
                    const auto a2 = collision[i+stride] > 0 ? f[i+stride] : f[i];
                    res -= (a1 + a2 - 2 * f[i]) * dxn2[j];
                }
                res_sum += res*res;
            }
        }
        return std::sqrt(res_sum);
    }
}