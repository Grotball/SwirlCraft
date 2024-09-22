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
        bool use_preconditioner;
        T* p;
        T* r;
        T* v;
        // L_diag, w, z used for preconditioned solve
        T* L_diag;
        T* w;
        T* z;
        public:
        ConjugateGradientSolver(Grid<T, Dims> grid, int32_t maxIterations, T epsilon=0, bool use_preconditioner=false) : 
            BasePressureSolver<ConjugateGradientSolver, T, Dims>(grid, maxIterations), 
            epsilon(epsilon), use_preconditioner(use_preconditioner)
        {
            p = new  T[grid.N];
            r = new  T[grid.N];
            v = new  T[grid.N];
            
            if (use_preconditioner)
            {
                L_diag = new T[grid.N];
                w = new T[grid.N];
                z = new T[grid.N];
            }
            else
            {
                L_diag = nullptr;
                w = nullptr;
                z = nullptr;
            }
        }
        PressureSolveInfo _solve_impl(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            for (size_t i = 0; i < grid.N; i++)
            {
                f[i] = 0;
            }
            if (use_preconditioner)
            {
                return preconditionedConjugateGradientSolve(L_diag, p, r, v, w, z, f, g, collision, grid, maxIterations, epsilon);
            }
            return conjugateGradientSolve(p, r, v, f, g, collision, grid, maxIterations, epsilon);
        }
        ~ConjugateGradientSolver()
        {
            delete[] p;
            delete[] r;
            delete[] v;

            if (use_preconditioner)
            {
                delete[] L_diag;
                delete[] w;
                delete[] z;
            }
        }
    };
}