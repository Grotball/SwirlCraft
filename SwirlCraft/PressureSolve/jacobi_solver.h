#pragma once
#include "base_pressure_solver.h"
#include "jacobi.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    class JacobiSolver : public BasePressureSolver<JacobiSolver, T, Dims>
    {
        T* f_old;
        public:
        JacobiSolver(Grid<T, Dims> grid, int32_t maxIterations) : BasePressureSolver<JacobiSolver, T, Dims>(grid, maxIterations) 
        {
            f_old = new T[this->grid.N];
        }
        PressureSolveInfo _solve_impl(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            return jacobiSolve(f, f_old, g, collision, this->grid, maxIterations);
        }
        ~JacobiSolver()
        {
            delete[] f_old;
        }
    };
}