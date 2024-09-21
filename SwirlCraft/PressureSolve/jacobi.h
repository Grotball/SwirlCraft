#pragma once
#include "../grid.h"
#include "../arch.h"
#include "pressure_solve_info.h"

namespace SwirlCraft
{
    template <typename T>
    inline void jacobiIteration(
        T* f, 
        T* f_old, 
        const T* g, 
        const T* collision, 
        const T c0, 
        const T* c, 
        const int64_t* strides, 
        const uint32_t dims, 
        const size_t N
    )
    {
        #ifdef _OPENMP
        #pragma omp parallel
        {
        #endif
            
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = 0; i < N; i++)
            {
                f_old[i] = f[i];
            }

            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = 0; i < N; i++)
            {
                if (collision[i] > 0)
                {
                    T A = 0;
                    for (size_t j = 0; j < dims; j++)
                    {
                        const auto stride = strides[j];
                        const T b1 = collision[i-stride] > 0 ? f_old[i-stride] : f_old[i];
                        const T b2 = collision[i+stride] > 0 ? f_old[i+stride] : f_old[i]; 
                        A += c[j] * (b1 + b2);
                    }
                    f[i] = A - c0 * g[i];
                }
            }
        #ifdef _OPENMP
        }
        #endif
    }

    #ifdef SWIRL_CRAFT_USE_SIMD
    template <>
    inline void jacobiIteration<float>(
        float* f, 
        float* f_old, 
        const float* g, 
        const float* collision, 
        const float c0, 
        const float* c, 
        const int64_t* strides, 
        const uint32_t dims, 
        const size_t N
    )
    {
        const size_t l = N % SwirlCraft::Arch::PackedFloats;
        #if __AVX__
        const auto c0_v = _mm256_set1_ps(c0);
        #elif __SSE4_1__
        const auto c0_v = _mm_set1_ps(c0);
        #endif

        #ifdef _OPENMP
        #pragma omp parallel
        {
        #endif
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = 0; i < N; i++)
            {
                f_old[i] = f[i];
            }

            #ifdef _OPENMP
            #pragma omp single nowait
            #endif
            for (size_t i = 0; i < l; i++)
            {
                if (collision[i] > 0)
                {
                    float A = 0;
                    for (uint32_t k = 0; k < dims; k++)
                    {
                        const auto stride = strides[k];
                        const auto b1 = collision[i-stride] > 0 ? f_old[i-stride] : f_old[i];
                        const auto b2 = collision[i+stride] > 0 ? f_old[i+stride] : f_old[i];
                        A += c[k] * (b1 + b2);
                    }
                    f[i] = A - c0 * g[i];
                }
            }

            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = l; i < N; i+=SwirlCraft::Arch::PackedFloats)
            {
                #if __AVX__
                __m256 A = _mm256_setzero_ps();
                const __m256 f0 = _mm256_loadu_ps(f_old + i);

                for (uint32_t j = 0; j < dims; j++)
                {
                    const auto stride = strides[j];
                    const auto cj = _mm256_set1_ps(c[j]);
                    const auto b1 = _mm256_blendv_ps(
                        _mm256_loadu_ps(f_old + i - stride),
                        f0,
                        _mm256_loadu_ps(collision + i - stride)
                    );
                    const auto b2 = _mm256_blendv_ps(
                        _mm256_loadu_ps(f_old + i + stride), 
                        f0, 
                        _mm256_loadu_ps(collision + i + stride)
                    );

                    A = _mm256_fmadd_ps(_mm256_add_ps(b1, b2), cj, A);
                }

                _mm256_storeu_ps(
                    f + i,
                    _mm256_fnmadd_ps(
                        c0_v, 
                        _mm256_loadu_ps(g + i),
                        A
                    )
                );
                #elif __SSE4_1__
                __m128 A = _mm_setzero_ps();
                const __m128 f0 = _mm_loadu_ps(f_old + i);

                for (uint32_t j = 0; j < dims; j++)
                {
                    const auto stride = strides[j];
                    const auto cj = _mm_set1_ps(c[j]);
                    const auto b1 = _mm_blendv_ps(
                        _mm_loadu_ps(f_old + i - stride),
                        f0,
                        _mm_loadu_ps(collision + i - stride)
                    );
                    const auto b2 = _mm_blendv_ps(
                        _mm_loadu_ps(f_old + i + stride), 
                        f0, 
                        _mm_loadu_ps(collision + i + stride)
                    );

                    A = _mm_fmadd_ps(_mm_add_ps(b1, b2), cj, A);
                }

                _mm_storeu_ps(
                    f + i,
                    _mm_fnmadd_ps(
                        c0_v, 
                        _mm_loadu_ps(g + i),
                        A
                    )
                );
                #endif
            }
        #ifdef _OPENMP
        }
        #endif
    }

    template <>
    inline void jacobiIteration<double>(
        double* f, 
        double* f_old, 
        const double* g, 
        const double* collision, 
        const double c0, 
        const double* c, 
        const int64_t* strides, 
        const uint32_t dims, 
        const size_t N
    )
    {
        const size_t l = N % SwirlCraft::Arch::PackedDoubles;
        #if __AVX__
        const auto c0_v = _mm256_set1_pd(c0);
        #elif __SSE4_1__
        const auto c0_v = _mm_set1_pd(c0);
        #endif
        
        #ifdef _OPENMP
        #pragma omp parallel
        {
        #endif
            
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = 0; i < N; i++)
            {
                f_old[i] = f[i];
            }

            #ifdef _OPENMP
            #pragma omp single nowait
            #endif
            for (size_t i = 0; i < l; i++)
            {
                if (collision[i] > 0)
                {
                    float A = 0;
                    for (uint32_t k = 0; k < dims; k++)
                    {
                        const auto stride = strides[k];
                        const auto b1 = collision[i-stride] > 0 ? f_old[i-stride] : f_old[i];
                        const auto b2 = collision[i+stride] > 0 ? f_old[i+stride] : f_old[i];
                        A += c[k] * (b1 + b2);
                    }
                    f[i] = A - c0 * g[i];
                }
            }

            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (size_t i = l; i < N; i+=SwirlCraft::Arch::PackedDoubles)
            {
                #if __AVX__
                __m256d A = _mm256_setzero_pd();
                const __m256d f0 = _mm256_loadu_pd(f_old + i);

                for (uint32_t j = 0; j < dims; j++)
                {
                    const auto stride = strides[j];
                    const auto cj = _mm256_set1_pd(c[j]);
                    const auto b1 = _mm256_blendv_pd(
                        _mm256_loadu_pd(f_old + i - stride),
                        f0,
                        _mm256_loadu_pd(collision + i - stride)
                    );
                    const auto b2 = _mm256_blendv_pd(
                        _mm256_loadu_pd(f_old + i + stride), 
                        f0, 
                        _mm256_loadu_pd(collision + i + stride)
                    );

                    A = _mm256_fmadd_pd(_mm256_add_pd(b1, b2), cj, A);
                }

                _mm256_storeu_pd(
                    f + i,
                    _mm256_fnmadd_pd(
                        c0_v, 
                        _mm256_loadu_pd(g + i),
                        A
                    )
                );

                #elif __SSE4_1__

                __m128d A = _mm_setzero_pd();
                const __m128d f0 = _mm_loadu_pd(f_old + i);

                for (uint32_t j = 0; j < dims; j++)
                {
                    const auto stride = strides[j];
                    const auto cj = _mm_set1_pd(c[j]);
                    const auto b1 = _mm_blendv_pd(
                        _mm_loadu_pd(f_old + i - stride),
                        f0,
                        _mm_loadu_pd(collision + i - stride)
                    );
                    const auto b2 = _mm_blendv_pd(
                        _mm_loadu_pd(f_old + i + stride), 
                        f0, 
                        _mm_loadu_pd(collision + i + stride)
                    );

                    A = _mm_fmadd_pd(_mm_add_pd(b1, b2), cj, A);
                }

                _mm_storeu_pd(
                    f + i,
                    _mm_fnmadd_pd(
                        c0_v, 
                        _mm_loadu_pd(g + i),
                        A
                    )
                );
                #endif
            }
        #ifdef _OPENMP
        }
        #endif
    }
    #endif

    template <typename T, uint32_t Dims>
    PressureSolveInfo jacobiSolve(T* f, T* f_old, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations)
    {
        T dxn2[Dims];
        T c[Dims];
        T sum = 0;
        const size_t N = grid.N;
        for (size_t i = 0; i < Dims; i++)
        {
            const T dx = grid.dx[i];
            dxn2[i] = 1 / (dx*dx);
            sum += dxn2[i];
        }

        const T c0 = static_cast<T>(0.5) / sum;
        for (size_t i = 0; i < Dims; i++)
        {
            c[i] = c0 * dxn2[i];
        }

        auto t1 = std::chrono::steady_clock::now();

        for (int32_t iter = 0; iter < maxIterations; iter++)
        {
            jacobiIteration(f, f_old, g, collision, c0, c, grid.stride, Dims, N);
        }

        auto t2 = std::chrono::steady_clock::now();

        return {maxIterations, t2 - t1, pressureSolveResidualNorm(f, g, collision, grid)};
    }    

    template <typename T, uint32_t Dims>
    PressureSolveInfo jacobiSolve(T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations)
    {
        T* f_old = new T[grid.N];

        PressureSolveInfo psolveInfo = jacobiSolve(f, f_old, g, collision, grid, maxIterations);

        delete[] f_old;
        return psolveInfo;
    }
}