#pragma once

#include "PressureSolve/jacobi.h"
#include "PressureSolve/gauss_seidel.h"

enum class PressureSolveMethod
{
    JacobiMethod,
    GaussSeidelMethod
};