#pragma once
#include "../grid.h"
#include "../arch.h"
#include "pressure_solve_info.h"

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    void jacobiSolve_gpu(T* f_d, T* f_old_d, const T* g_d, const T* collision_d, const Grid<T, Dims>& grid, const int32_t maxIterations, const int32_t blockSize);
}
