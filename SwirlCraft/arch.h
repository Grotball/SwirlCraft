#pragma once

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if __SSE4_1__
#define SWIRL_CRAFT_USE_SIMD
#endif

#ifdef SWIRL_CRAFT_GPU_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#endif



namespace SwirlCraft
{
    namespace Arch
    {
        #ifdef SWIRL_CRAFT_USE_SIMD
            #if __AVX__
            constexpr int PackedSize = 32;
            #elif __SSE4_1__
            constexpr int PackedSize = 16;
            #endif
        #else
            constexpr int PackedSize = 0;
        #endif

        constexpr int PackedFloats = PackedSize / sizeof(float);
        constexpr int PackedDoubles = PackedSize / sizeof(double);
    }
}
