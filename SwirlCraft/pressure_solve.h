#pragma once

#include "PressureSolve/jacobi.h"
#include "PressureSolve/gauss_seidel.h"
#include "PressureSolve/conjugate_gradient.h"


namespace SwirlCraft
{
    enum class PressureSolveMethod
    {
        JacobiMethod,
        GaussSeidelMethod
    };
}