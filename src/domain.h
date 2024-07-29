#pragma once
#include <stddef.h>
#include <cstdlib>
#include <cstdint>

namespace SwirlCraft
{
    template <typename T>
    struct DomainDim
    {
        size_t n;
        int32_t stride;
        T dx;
    };

    template <typename T, size_t Dims>
    struct Domain
    {
        size_t N;
        DomainDim<T> dims[Dims];
    };

    template <typename T, size_t Dims>
    void cartesianIndex(int32_t (&I)[Dims], const size_t i, const Domain<T, Dims>& domain)
    {
        auto p = i;
        for (size_t j = 0; j < Dims; j++)
        {
            auto k = (Dims - 1) - j;
            auto d = std::div(p, domain.dims[k].stride);
            I[k] = d.quot;
            p = d.rem;
        }
    }
}