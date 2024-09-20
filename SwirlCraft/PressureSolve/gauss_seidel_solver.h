#pragma once
#include "base_pressure_solver.h"
#include "gauss_seidel.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    class GaussSeidelSolver : public BasePressureSolver<GaussSeidelSolver, T, Dims>
    {
        using BasePressureSolver<GaussSeidelSolver, T, Dims>::grid;
        public:
        GaussSeidelSolver(Grid<T, Dims> grid, int32_t maxIterations) : BasePressureSolver<GaussSeidelSolver, T, Dims>(grid, maxIterations) {}
        void _solve_impl(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            gaussSeidelSolve(f, g, collision, grid, maxIterations);
        }
    };
}