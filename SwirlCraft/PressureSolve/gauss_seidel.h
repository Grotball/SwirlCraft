#pragma once
#include "../grid.h"
#include "pressure_solve_info.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    PressureSolveInfo gaussSeidelSolve(T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations)
    {
        T dxn2[Dims];
        T c[Dims];
        T sum = 0;
        const size_t N = grid.N;
        for (size_t i = 0; i < Dims; i++)
        {
            const T dx = grid.dx[i];
            dxn2[i] = 1 / (dx*dx);
            sum += dxn2[i];
        }

        const T c0 = static_cast<T>(0.5) / sum;
        for (size_t i = 0; i < Dims; i++)
        {
            c[i] = c0 * dxn2[i];
        }

        auto t1 = std::chrono::steady_clock::now();

        for (int32_t iter = 0; iter < maxIterations; iter++)
        {
            for (size_t i = 0; i < N; i++)
            {
                if (collision[i] > 0)
                {
                    T A = 0;
                    for (uint32_t k = 0; k < Dims; k++)
                    {
                        const auto stride = grid.stride[k];
                        const T b1 = collision[i-stride] > 0 ? f[i-stride] : f[i];
                        const T b2 = collision[i+stride] > 0 ? f[i+stride] : f[i]; 
                        A += c[k] * (b1 + b2);
                    }
                    f[i] = A - c0 * g[i];
                }
            }
        }

        auto t2 = std::chrono::steady_clock::now();
        return {maxIterations, t2 - t1, pressureSolveResidualNorm(f, g, collision, grid)};
    }
}