#pragma once
#include "base_pressure_solver.h"
#include "conjugate_gradient.h" 

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    class ConjugateGradientSolver : public BasePressureSolver<ConjugateGradientSolver, T, Dims>
    {
        private:
        using BasePressureSolver<ConjugateGradientSolver, T, Dims>::grid;
        T epsilon;
        T* p;
        T* r;
        T* v;
        public:
        ConjugateGradientSolver(Grid<T, Dims> grid, int32_t maxIterations, T epsilon=0) : BasePressureSolver<ConjugateGradientSolver, T, Dims>(grid, maxIterations), epsilon(epsilon)
        {
            p = new  T[grid.N];
            r = new  T[grid.N];
            v = new  T[grid.N];
        }
        void _solve_impl(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            conjugateGradientSolve(p, r, v, f, g, collision, grid, maxIterations, epsilon);
        }
        ~ConjugateGradientSolver()
        {
            delete[] p;
            delete[] r;
            delete[] v;
        }
    };
}