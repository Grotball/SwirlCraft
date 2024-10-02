#pragma once
#include <cmath>
#include "../grid.h"
#include "../arch.h"
#include "pressure_solve_info.h"

namespace SwirlCraft
{
    
    template <typename T, uint32_t Dims>
    PressureSolveInfo conjugateGradientSolve(T* p, T* r, T* v, T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations, const T epsilon)
    {
        T dxn2[Dims];
        for (uint32_t i = 0; i < Dims; i++)
        {
            auto dx = grid.dx[i];
            dxn2[i] = 1 / (dx*dx);
        }

        auto t1 = std::chrono::steady_clock::now();

        T res2_sum = 0;
        T g2_sum = 0;
        
        for (size_t i = 0; i < grid.N; i++)
        {
            if (collision[i] > 0)
            {
                r[i] = g[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    auto a1 = collision[i-stride] > 0 ? f[i-stride] : f[i];
                    auto a2 = collision[i+stride] > 0 ? f[i+stride] : f[i];
                    r[i] -= (a1 + a2 - 2 * f[i]) * dxn2[j];
                }
                g2_sum += g[i]*g[i];
                res2_sum += r[i]*r[i];
            }
            else
            {
                r[i] = 0;
            }
            p[i] = r[i];
            v[i] = 0;
        }

        const T tol = epsilon * epsilon * g2_sum;
        int32_t iter = 0;
        while (iter < maxIterations && tol < res2_sum)
        {
            auto res2_sum_old = res2_sum;
            T p_dot_v = 0;
            res2_sum = 0;
            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    v[i] = 0;
                    if (collision[i] > 0)
                    {
                        for (uint32_t j = 0; j < Dims; j++)
                        {
                            const auto stride = grid.stride[j];
                            auto a1 = collision[i-stride] > 0 ? p[i-stride] : p[i];
                            auto a2 = collision[i+stride] > 0 ? p[i+stride] : p[i];
                            v[i] += (a1 + a2 - 2 * p[i]) * dxn2[j];
                        }     
                    }
                }
                #ifdef _OPENMP
                #pragma omp for reduction(+:p_dot_v)
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    p_dot_v += p[i] * v[i];
                }

                auto alpha = res2_sum_old / p_dot_v;
                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    r[i] -= alpha * v[i];
                    f[i] += alpha * p[i];
                }
                #ifdef _OPENMP
                #pragma omp for reduction(+:res2_sum)
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    res2_sum += r[i] * r[i];
                }

                const T beta = res2_sum / res2_sum_old;
                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    p[i] = r[i] + beta * p[i];
                }
            }

            iter++;
        }

        auto t2 = std::chrono::steady_clock::now();

        return PressureSolveInfo{iter, t2-t1, std::sqrt(res2_sum)};
    }

    template <typename T, uint32_t Dims>
    PressureSolveInfo conjugateGradientSolve(T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations, const T epsilon)
    {
        T* r = new T[grid.N];
        T* p = new T[grid.N];
        T* v = new T[grid.N];
        
        auto psolveInfo = conjugateGradientSolve(p, r, v, f, g, collision, grid, maxIterations, epsilon);

        delete[] r;
        delete[] p;
        delete[] v;
        return psolveInfo;
    }


    // Apply incomplete Cholesky preconditioner.
    template <typename T, uint32_t Dims>
    void applyPreconditioner(T* z, T* w, const T* r, const T* L_diag_rcp, const T* collision, const T(&dxn2)[Dims], const Grid<T, Dims>& grid)
    {
        w[0] = collision[0] > 0 ? r[0] * L_diag_rcp[0] : 0;

        const int64_t N = static_cast<int64_t>(grid.N);
        for (int64_t i = 0; i < N; i++)
        {
            if (collision[i] > 0)
            {
                w[i] = r[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    if (i >= stride && collision[i - stride] > 0)
                    {
                        w[i] -= -dxn2[j] * w[i - stride] * L_diag_rcp[i - stride];
                    }
                }
                w[i] = w[i] * L_diag_rcp[i];
            }
            else
            {
                w[i] = 0;
            }
        }

        z[grid.N-1] = 0;

        for (size_t i_ = 0, i = grid.N-1; i_ < grid.N; i_++, i--)
        {
            if (collision[i] > 0)
            {
                z[i] = w[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    const auto stride = grid.stride[j];
                    if ((i + stride) < grid.N && collision[i + stride] > 0)
                    {
                        z[i] -= -dxn2[j] * z[i + stride] * L_diag_rcp[i];
                    }
                }
                z[i] = z[i] * L_diag_rcp[i];
            }
            else
            {
                z[i] = 0;
            }
        }
    }


    template <typename T, uint32_t Dims>
    PressureSolveInfo preconditionedConjugateGradientSolve(T* L_diag_rcp, T* p, T* r, T* v, T* w, T* z, T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations, const T epsilon)
    {
        
        T dxn2[Dims];
        T c0 = 0;
        for (uint32_t i = 0; i < Dims; i++)
        {
            auto dx = grid.dx[i];
            dxn2[i] = 1 / (dx*dx);
            c0 += 2 * dxn2[i];
        }

        auto t1 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < grid.N; i++) {r[i] = 0;}

        L_diag_rcp[0] = collision[0] <= 0 ? 0 : 1 / std::sqrt(c0);
        const int64_t N = static_cast<int64_t>(grid.N);
        for (int64_t i = 0; i < N; i++)
        {
            if (collision[i] > 0)
            {
                T q = 0;
                T A = c0;
                for (uint32_t j = 0; j < Dims; j++)
                {
                    auto stride = grid.stride[j];
                    if (0 <= (i - stride) && collision[i - stride] > 0)
                    {
                        auto a = -dxn2[j] * L_diag_rcp[i - stride];
                        q += a*a;
                    }
                    else
                    {
                        A += -dxn2[j];
                    }
                }
                L_diag_rcp[i] = 1 / std::sqrt(A - q);
            }
            else
            {
                L_diag_rcp[i] = 0;
            }
        }

        
        for (int64_t i = 0; i < N; i++)
        {
            if (collision[i] > 0)
            {
                r[i] = -g[i];
                for (uint32_t j = 0; j < Dims; j++)
                {
                    auto stride = grid.stride[j];
                    auto a1 = collision[i - stride] > 0 ? f[i - stride] : f[i];
                    auto a2 = collision[i + stride] > 0 ? f[i + stride] : f[i];
                    r[i] -= - (a1 + a2 - 2 * f[i]) * dxn2[j];
                }
            }
        }

        applyPreconditioner(z, w, r, L_diag_rcp, collision, dxn2, grid);

        for (size_t i = 0; i < grid.N; i++)
        {
            p[i] = z[i];
        }

        T r_dot_z = 0;
        T res_sum = 0;
        T g2_sum = 0;
        for (size_t i = 0; i < grid.N; i++)
        {
            res_sum += r[i]*r[i];
            r_dot_z += r[i] * z[i];
            g2_sum += g[i] * g[i];
        }

        T tol = epsilon*epsilon * g2_sum;
        int32_t iter = 0;
        while (iter < maxIterations && tol < res_sum)
        {
            T p_dot_v = 0;
            res_sum = 0;
            auto r_dot_z_old = r_dot_z;
            r_dot_z = 0;
            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    v[i] = 0;
                    if (collision[i] > 0)
                    {
                        for (uint32_t j = 0; j < Dims; j++)
                        {
                            auto stride = grid.stride[j];
                            auto a1 = collision[i - stride] > 0 ? p[i - stride] : p[i];
                            auto a2 = collision[i + stride] > 0 ? p[i + stride] : p[i];
                            v[i] += -(a1 + a2 - 2*p[i]) * dxn2[j];
                        }
                    }
                }

                #ifdef _OPENMP
                #pragma omp for reduction(+:p_dot_v)
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    p_dot_v += p[i] * v[i];
                }
                T alpha = r_dot_z_old / p_dot_v;

                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    f[i] += alpha * p[i];
                    r[i] -= alpha * v[i];
                }

                #ifdef _OPENMP
                #pragma omp single
                #endif
                {
                    applyPreconditioner(z, w, r, L_diag_rcp, collision, dxn2, grid);
                }

                #ifdef _OPENMP
                #pragma omp for reduction(+:res_sum, r_dot_z)
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    res_sum += r[i] * r[i];
                    r_dot_z += r[i] * z[i];
                }

                T beta = r_dot_z / r_dot_z_old;

                #ifdef _OPENMP
                #pragma omp for
                #endif
                for (size_t i = 0; i < grid.N; i++)
                {
                    p[i] = z[i] + beta * p[i];
                }
            }

            iter++;
        }

        auto t2 = std::chrono::steady_clock::now();

        return {iter, t2-t1, std::sqrt(res_sum)};
    }

    template <typename T, uint32_t Dims>
    PressureSolveInfo preconditionedConjugateGradientSolve(T* f, const T* g, const T* collision, const Grid<T, Dims>& grid, const int32_t maxIterations, const T epsilon)
    {
        T* L_diag_rcp = new T[grid.N];
        T* r = new T[grid.N];
        T* p = new T[grid.N];
        T* v = new T[grid.N];
        T* w = new T[grid.N];
        T* z = new T[grid.N];

        auto psolveInfo = preconditionedConjugateGradientSolve(L_diag_rcp, p, r, v, w, z, f, g, collision, grid, maxIterations, epsilon);

        delete[] L_diag_rcp;
        delete[] r;
        delete[] p;
        delete[] v;
        delete[] w;
        delete[] z;

        return psolveInfo;
    }
}