#pragma once
#include "../grid.h"
#include "../arch.h"

namespace SwirlCraft
{
    template <typename T>
    inline void jacobiIteration(T* p, T* p_old, const T* div, const T* collision, const T c0, const T* c, const int64_t* strides, const uint32_t dims, const size_t N)
    {
        for (size_t i = 0; i < N; i++)
        {
            p_old[i] = p[i];
        }

        for (size_t i = 0; i < N; i++)
        {
            if (collision[i] > 0)
            {
                T A = 0;
                for (size_t j = 0; j < dims; j++)
                {
                    const auto stride = strides[j];
                    const T b1 = collision[i-stride] > 0 ? p_old[i-stride] : p_old[i];
                    const T b2 = collision[i+stride] > 0 ? p_old[i+stride] : p_old[i]; 
                    A += c[j] * (b1 + b2);
                }
                p[i] = A - c0 * div[i];
            }
        }
    }

    #ifdef SWIRL_CRAFT_USE_SIMD
    template <>
    inline void jacobiIteration<float>(float* p, float* p_old, const float* div, const float* collision, const float c0, const float* c, const int64_t* strides, const uint32_t dims, const size_t N)
    {
        const size_t l = N % SwirlCraft::Arch::PackedFloats;

        const auto c0_v = _mm_set1_ps(c0);

        for (size_t i = 0; i < N; i++)
        {
            p_old[i] = p[i];
        }

        for (size_t i = 0; i < l; i++)
        {
            if (collision[i] > 0)
            {
                float A = 0;
                for (uint32_t k = 0; k < dims; k++)
                {
                    const auto stride = strides[k];
                    const auto b1 = collision[i-stride] > 0 ? p_old[i-stride] : p_old[i];
                    const auto b2 = collision[i+stride] > 0 ? p_old[i+stride] : p_old[i];
                    A += c[k] * (b1 + b2);
                }
                p[i] = A - c0 * div[i];
            }
        }

        for (size_t i = l; i < N; i+=SwirlCraft::Arch::PackedFloats)
        {
            __m128 A = _mm_setzero_ps();
            const __m128 p0 = _mm_loadu_ps(p_old + i);

            for (uint32_t j = 0; j < dims; j++)
            {
                const auto stride = strides[j];
                const auto cj = _mm_set1_ps(c[j]);
                const auto b1 = _mm_blendv_ps(
                    _mm_loadu_ps(p_old + i - stride),
                    p0,
                    _mm_loadu_ps(collision + i - stride)
                );
                const auto b2 = _mm_blendv_ps(
                    _mm_loadu_ps(p_old + i + stride), 
                    p0, 
                    _mm_loadu_ps(collision + i + stride)
                );

                A = _mm_fmadd_ps(_mm_add_ps(b1, b2), cj, A);
            }

            _mm_storeu_ps(
                p + i,
                _mm_fnmadd_ps(
                    c0_v, 
                    _mm_loadu_ps(div + i),
                    A
                )
            );
        }
    }

    template <>
    inline void jacobiIteration<double>(double* p, double* p_old, const double* div, const double* collision, const double c0, const double* c, const int64_t* strides, const uint32_t dims, const size_t N)
    {
        const size_t l = N % SwirlCraft::Arch::PackedDoubles;
        const auto c0_v = _mm_set1_pd(c0);
        
        for (size_t i = 0; i < N; i++)
        {
            p_old[i] = p[i];
        }

        for (size_t i = 0; i < l; i++)
        {
            if (collision[i] > 0)
            {
                float A = 0;
                for (uint32_t k = 0; k < dims; k++)
                {
                    const auto stride = strides[k];
                    const auto b1 = collision[i-stride] > 0 ? p_old[i-stride] : p_old[i];
                    const auto b2 = collision[i+stride] > 0 ? p_old[i+stride] : p_old[i];
                    A += c[k] * (b1 + b2);
                }
                p[i] = A - c0 * div[i];
            }
        }

        for (size_t i = l; i < N; i+=SwirlCraft::Arch::PackedDoubles)
        {
            __m128d A = _mm_setzero_pd();
            const __m128d p0 = _mm_loadu_pd(p_old + i);

            for (uint32_t j = 0; j < dims; j++)
            {
                const auto stride = strides[j];
                const auto cj = _mm_set1_pd(c[j]);
                const auto b1 = _mm_blendv_pd(
                    _mm_loadu_pd(p_old + i - stride),
                    p0,
                    _mm_loadu_pd(collision + i - stride)
                );
                const auto b2 = _mm_blendv_pd(
                    _mm_loadu_pd(p_old + i + stride), 
                    p0, 
                    _mm_loadu_pd(collision + i + stride)
                );

                A = _mm_fmadd_pd(_mm_add_pd(b1, b2), cj, A);
            }

            _mm_storeu_pd(
                p + i,
                _mm_fnmadd_pd(
                    c0_v, 
                    _mm_loadu_pd(div + i),
                    A
                )
            );
        }
    }
    #endif

    template <typename T, uint32_t Dims>
    void jacobiSolve(T* p, const T* div, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations)
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


        T* p_old = new T[N];


        for (int32_t iter = 0; iter < maxIterations; iter++)
        {
            jacobiIteration(p, p_old, div, collision, c0, c, grid.stride, Dims, N);
        }
        
        delete[] p_old;
    }    
}