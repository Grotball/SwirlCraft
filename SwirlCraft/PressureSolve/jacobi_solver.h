#pragma once
#include "base_pressure_solver.h"
#include "jacobi.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    class JacobiSolver : public BasePressureSolver<JacobiSolver, T, Dims>
    {
        T* p_old;
        public:
        JacobiSolver(Grid<T, Dims> grid, int32_t maxIterations) : BasePressureSolver<JacobiSolver, T, Dims>(grid, maxIterations) 
        {
            p_old = new T[this->grid.N];
        }
        void _solve_impl(T* p, const T* div, const T* collision, const int32_t maxIterations)
        {
            jacobiSolve(p, p_old, div, collision, this->grid, maxIterations);
        }
        ~JacobiSolver()
        {
            delete[] p_old;
        }
    };
}