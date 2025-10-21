#include "lock.cuh"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(Lock lock, float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    int i = threadsPerBlock / 2;
    while (i != 0)
    {
        if (cacheIdx < i)
        {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i = i / 2;
    }

    // Each threadblock now needs to add its final value
    // to get the dot product (the sum of each block is at cache[0])

    if (cacheIdx == 0)
    {
        lock.lock();
        *c += cache[0];
        lock.unlock();
    }
}

int main(void)
{
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    HANDLE_ERROR(cudaMalloc(&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dev_c, sizeof(float)));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, &c, sizeof(float), cudaMemcpyHostToDevice));

    Lock lock;
    dot<<<blocksPerGrid, threadsPerBlock>>>(lock, dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
}