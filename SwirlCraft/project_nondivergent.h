#pragma once
#include "grid.h"
#include "pressure_solve.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    void projectNonDivergent(
        T* (&vel)[Dims], 
        T* p, 
        T* p_old, 
        T* div, 
        const T* collision, 
        T* (&collision_vel)[Dims], 
        const Grid<T, Dims>& grid, 
        PressureSolveMethod solveMethod=PressureSolveMethod::JacobiMethod, 
        const int32_t maxIterations=40
    )
    {
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                div[i] = 0;
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    auto b1 = collision[i - stride] > 0 ? vel[j][i - stride] : 2 * collision_vel[j][i - stride] - vel[j][i];
                    auto b2 = collision[i + stride] > 0 ? vel[j][i + stride] : 2 * collision_vel[j][i + stride] - vel[j][i];
                    div[i] += (b2 - b1) * static_cast<T>(0.5) / grid.dx[j];
                }
            }
        }

        switch (solveMethod)
        {
            case PressureSolveMethod::GaussSeidelMethod:
            {
                gaussSeidelSolve(p, div, collision, grid, maxIterations);
                break;
            }
            default:
            {
                jacobiSolve(p, p_old, div, collision, grid, maxIterations);
                break;
            }
        }

        for (uint32_t i = 0; i < Dims; i++)
        {
            auto stride = grid.stride[i];
            for (size_t j = 0; j < grid.N; j++)
            {
                if (collision[j] > 0)
                {
                    auto a1 = collision[j-stride] > 0 ? p[j-stride] : p[j];
                    auto a2 = collision[j+stride] > 0 ? p[j+stride] : p[j];

                    vel[i][j] -= (a2 - a1) / (2 * grid.dx[i]);
                }
            }
        }
    }
}