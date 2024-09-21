#pragma once
#include "grid.h"
#include "pressure_solve.h"

namespace SwirlCraft
{
    // Removes divergence from velocity field.
    template <typename T, uint32_t Dims, typename PressureSolver>
    PressureSolveInfo projectNonDivergent(
        T* (&vel)[Dims], 
        T* p, 
        T* div, 
        const T* collision, 
        const Grid<T, Dims>& grid, 
        PressureSolver& pressureSolver
    )
    {
        // Compute divergence of velocity while preserving
        // the free-slip boundary condition 
        // dot(vel, normal_dir) = dot(vel_solid, normal_dir).
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                div[i] = 0;
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    auto b1 = collision[i - stride] > 0 ? vel[j][i - stride] : 2 * vel[j][i - stride] - vel[j][i];
                    auto b2 = collision[i + stride] > 0 ? vel[j][i + stride] : 2 * vel[j][i + stride] - vel[j][i];
                    div[i] += (b2 - b1) * static_cast<T>(0.5) / grid.dx[j];
                }
            }
        }

        PressureSolveInfo psolveInfo = pressureSolver.solve(p, div, collision);

        // Subtract pressure gradient from velocity to remove divergence.
        for (uint32_t i = 0; i < Dims; i++)
        {
            auto stride = grid.stride[i];
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
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
        return psolveInfo;
    }
}