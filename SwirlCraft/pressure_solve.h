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
        ConjugateGradientMethod,
        PreconditionedConjugateGradientMethod
    };
}