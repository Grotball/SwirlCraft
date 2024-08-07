#pragma once
#include <stddef.h>
#include <cstdlib>
#include <cstdint>

namespace SwirlCraft
{
    template <typename T, uint32_t Dims>
    struct Grid
    {
        size_t N;
        int64_t size[Dims];
        int64_t stride[Dims];
        T dx[Dims];
    };

    template <typename Int_T, typename T, uint32_t Dims>
    void cartesianIndex(Int_T (&I)[Dims], const size_t i, const Grid<T, Dims>& grid)
    {
        auto p = i;
        for (size_t j = 0; j < Dims; j++)
        {
            auto k = (Dims - 1) - j;
            auto d = std::div(p, static_cast<int64_t>(grid.stride[k]));
            I[k] = static_cast<Int_T>(d.quot);
            p = d.rem;
        }
    }


    template <typename Int_T, typename T, uint32_t Dims>
    size_t linearIndex(Int_T (&I)[Dims], const Grid<T, Dims>& grid)
    {
        size_t index = 0;
        for (size_t j = 0; j < Dims; j++)
        {
            index += I[j] * grid.stride[j];
        }
        return index;
    }
}