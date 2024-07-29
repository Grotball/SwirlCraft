#pragma once
#include <stddef.h>
#include <cstdlib>

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

    template <typename T, size_t Dims>
    void cartesianIndex(size_t (&I)[Dims], const size_t i, const Domain<T, Dims>& domain)
    {
        auto p = i;
        for (size_t j = 0; j < Dims; j++)
        {
            auto k = (Dims - 1) - j;
            auto d = std::div(p, static_cast<int>(domain.dims[k].stride));
            I[k] = d.quot;
            p = d.rem;
        }
    }
}