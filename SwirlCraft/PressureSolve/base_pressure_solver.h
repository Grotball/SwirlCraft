#pragma once
#include <cstdint>
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
        void solve(T* f, const T* g, const T* collision, const int32_t maxIterations)
        {
            static_cast<Derived<T, Dims>*>(this)->_solve_impl(f, g, collision, maxIterations);
        }
        void solve(T* f, const T* g, const T* collision)
        {
            solve(f, g, collision, maxIterations);
        }
    };
}