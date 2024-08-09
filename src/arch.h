#pragma once

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64)
#include <immintrin.h>
#endif


#ifdef __SSE4_1__
#define SWIRL_CRAFT_USE_SIMD
#endif