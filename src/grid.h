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
}