#pragma once

#include "PressureSolve/jacobi.h"
#include "PressureSolve/gauss_seidel.h"


namespace SwirlCraft
{
    enum class PressureSolveMethod
    {
        JacobiMethod,
        GaussSeidelMethod
    };
}