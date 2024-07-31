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

    template <typename Int_T, typename T, size_t Dims>
    void cartesianIndex(Int_T (&I)[Dims], const size_t i, const Domain<T, Dims>& domain)
    {
        auto p = i;
        for (size_t j = 0; j < Dims; j++)
        {
            auto k = (Dims - 1) - j;
            auto d = std::div(p, static_cast<int64_t>(domain.dims[k].stride));
            I[k] = static_cast<Int_T>(d.quot);
            p = d.rem;
        }
    }


    template <typename Int_T, typename T, size_t Dims>
    size_t linearIndex(Int_T (&I)[Dims], const Domain<T, Dims>& domain)
    {
        size_t index = 0;
        for (size_t j = 0; j < Dims; j++)
        {
            index += I[j] * domain.dims[j].stride;
        }
        return index;
    }
}