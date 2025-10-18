#include <iostream>
#include "./cuda_error.h"

const int N = 33 * 1024;
const int threadsPerBlock = 1 << 10;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void dot(float *a, float *b, float *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    __shared__ float cache[threadsPerBlock];
    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x;
    }
    cache[cacheIndex] = temp;

    // Making sure all threads are done writing
    // to the cache before starting to read values from it
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        // Again making sure one batch of writing is done,
        // before we start to read again
        __syncthreads();
        i = i / 2;
    }

    // each block will have its sum at the first index of the cache
    if (cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc(&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_partial_c, blocksPerGrid * sizeof(float)));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // We still need sum over results of each thread block
    float dotprod = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        dotprod += partial_c[i];
    }

    std::cout << "The dot product from GPU is: " << dotprod << std::endl;

    float cpudotprod = 0;
    for (int i = 0; i < N; i++)
    {
        cpudotprod += a[i] * b[i];
    }

    std::cout << "The dot product from CPU is: " << cpudotprod << std::endl;

    free(a);
    free(b);
    free(partial_c);

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    return 0;
}
