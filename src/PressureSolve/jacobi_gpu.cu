
#include <cstdint>
#include <SwirlCraft/grid.h>
#include <SwirlCraft/PressureSolve/jacobi_gpu.h>

namespace SwirlCraft
{
    template <typename T>
    __global__ void jacobiIterationKernel(T* f, T* f_old, const T* g, const T* collision, const T c0, const T* c, const int64_t* strides, const size_t n, uint32_t Dims)
    {
        const auto idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < n && collision[idx] > 0)
        {
            T A = 0;
            for (uint32_t j = 0; j < Dims; j++)
            {
                const auto stride = strides[j];
                const T a1 = collision[idx-stride] > 0 ? f_old[idx-stride] : f_old[idx];
                const T a2 = collision[idx+stride] > 0 ? f_old[idx+stride] : f_old[idx];
                A += c[j] * (a1 + a2);
            }
            f[idx] = A - c0 * g[idx];
        }
    }
    
    
    template <typename T, uint32_t Dims>
    void jacobiSolve_gpu(T* f_d, T* f_old_d, const T* g_d, const T* collision_d, const Grid<T, Dims>& grid, const int32_t maxIterations, const int32_t blockSize)
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

        int numBlocks = static_cast<int>(ceil(static_cast<double>(grid.N) / blockSize));

        const T c0 = static_cast<T>(0.5) / sum;
        for (size_t i = 0; i < Dims; i++)
        {
            c[i] = c0 * dxn2[i];
        }

        T* c_d;
        int64_t* strides_d;
        cudaMalloc(&c_d, Dims * sizeof(T));
        cudaMalloc(&strides_d, Dims * sizeof(int64_t));
        cudaMemcpy(c_d, c, Dims * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(strides_d, grid.stride, Dims * sizeof(int64_t), cudaMemcpyHostToDevice);

        for (int32_t iter = 0; iter < maxIterations; iter++)
        {
            cudaMemcpy(f_old_d, f_d, grid.N * sizeof(T), cudaMemcpyDeviceToDevice);
            jacobiIterationKernel<<<numBlocks, blockSize>>>(f_d, f_old_d, g_d, collision_d, c0, c_d, strides_d, N, Dims);
        }

        cudaFree(c_d);
        cudaFree(strides_d);
    }
    
}


template void SwirlCraft::jacobiSolve_gpu(float*, float*, const float*, const float*, const SwirlCraft::Grid<float, 2u>&, const int32_t, const int32_t);
template void SwirlCraft::jacobiSolve_gpu(float*, float*, const float*, const float*, const SwirlCraft::Grid<float, 3u>&, const int32_t, const int32_t);
template void SwirlCraft::jacobiSolve_gpu(double*, double*, const double*, const double*, const SwirlCraft::Grid<double, 2u>&, const int32_t, const int32_t);
template void SwirlCraft::jacobiSolve_gpu(double*, double*, const double*, const double*, const SwirlCraft::Grid<double, 3u>&, const int32_t, const int32_t);
