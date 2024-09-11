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
        void solve(T* p, const T* div, const T* collision, const int32_t maxIterations)
        {
            static_cast<Derived<T, Dims>*>(this)->_solve_impl(p, div, collision, maxIterations);
        }
        void solve(T* p, const T* div, const T* collision)
        {
            solve(p, div, collision, maxIterations);
        }
    };
}