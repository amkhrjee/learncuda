// Computes the sum of two vectors of length N
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

__global__ void vectorAdd(int *__restrict a, int *__restrict b, int *__restrict c, int N)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c)
{
    for (int i = 0; i < a.size(); i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

int main(void)
{
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // These are on CPU (aka "the host")
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);

    for (int i = 0; i < N; i++)
    {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // Allocating memory on the GPU (aka "the device")
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from the host to the device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int NUM_THREADS = 1 << 10;
    constexpr int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a, b, c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}