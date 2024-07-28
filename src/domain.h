#pragma once
#include <stddef.h>

namespace SwirlCraft
{
    template <typename T>
    struct DomainDim
    {
        size_t n;
        size_t stride;
        T dx;
    };

    template <typename T, size_t Dims>
    struct Domain
    {
        size_t N;
        DomainDim<T> dims[Dims];
    };
}