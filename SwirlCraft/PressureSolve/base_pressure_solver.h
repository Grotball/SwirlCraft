#pragma once
#include <cstdint>
#include <cstdlib>
#include <stddef.h>
#include "../grid.h"

namespace SwirlCraft
{
    template <template<class, uint32_t> class Derived, typename T, uint32_t Dims>
    class BasePressureSolver
    {
        protected:
        BasePressureSolver() = default;

        public:
        void solve(T* p, const T* div, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations)
        {
            static_cast<Derived<T, Dims>*>(this)->_solve_impl(p, div, collision, grid, maxIterations);
        }
    };
}