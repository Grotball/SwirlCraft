#pragma once
#include <cstdint>
#include "pressure_solve_info.h"
#include "../grid.h"

namespace SwirlCraft
{
    template <template<class, uint32_t> class Derived, typename T, uint32_t Dims>
    class BasePressureSolver
    {
        int32_t maxIterations;
        protected:
        Grid<T, Dims> grid;
        BasePressureSolver(Grid<T, Dims> grid, int32_t maxIterations) : maxIterations(maxIterations), grid(grid) {}

        public:
        PressureSolveInfo solve(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            return static_cast<Derived<T, Dims>*>(this)->_solve_impl(f, g, collision, maxIterations);
        }
        PressureSolveInfo solve(T* f, const T* g, const T* collision)
        {
            return solve(f, g, collision, maxIterations);
        }
    };
}