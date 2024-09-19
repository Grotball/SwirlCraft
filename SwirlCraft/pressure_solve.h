#pragma once

#include "PressureSolve/jacobi.h"
#include "PressureSolve/gauss_seidel.h"
#include "PressureSolve/conjugate_gradient.h"
#include "PressureSolve/jacobi_solver.h"
#include "PressureSolve/gauss_seidel_solver.h"
#include "PressureSolve/conjugate_gradient_solver.h" 


namespace SwirlCraft
{
    enum class PressureSolveMethod
    {
        JacobiMethod,
        GaussSeidelMethod,
        ConjugateGradientMethod
    };

    template <typename T, uint32_t, PressureSolveMethod PSM>
    struct PressureSolver_traits {};

    template<typename T, uint32_t Dims>
    struct PressureSolver_traits<T, Dims, PressureSolveMethod::JacobiMethod>
    {
        typedef JacobiSolver<T, Dims> type;
    };
    template <typename T, uint32_t Dims>
    struct PressureSolver_traits<T, Dims, PressureSolveMethod::GaussSeidelMethod>
    {
        typedef GaussSeidelSolver<T, Dims> type;
    };
    template <typename T, uint32_t Dims>
    struct PressureSolver_traits<T, Dims, PressureSolveMethod::ConjugateGradientMethod>
    {
        typedef ConjugateGradientSolver<T, Dims> type;
    };

    template <typename T, uint32_t Dims, PressureSolveMethod PSM>
    using PressureSolver = typename PressureSolver_traits<T, Dims, PSM>::type;
}