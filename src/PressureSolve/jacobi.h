#pragma once
#include "../grid.h"
#include <cstdlib>

namespace SwirlCraft
{
    template <typename T>
    void jacobiIteration(T* p, T* p_old, const T* div, const T* collision, const T c0, const T* c, const int64_t* strides, const uint32_t dims, const size_t N)
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