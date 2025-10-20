#include "cuda_error.h"
#include <iostream>
#include <iomanip>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void)
{
    cudaDeviceProp props;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties_v2(&props, whichDevice));
    if (!props.deviceOverlap)
    {
        std::cout << "Device does not support streams :(" << std::endl;
        exit(EXIT_FAILURE);
    }

    int *host_a, *host_b, *host_c;

    // allocate page-locked memory used to stream
    HANDLE_ERROR(cudaHostAlloc(&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc(&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc(&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // initialiazing the stream
    cudaStream_t stream0, stream1;
    HANDLE_ERROR(cudaStreamCreate(&stream0));
    HANDLE_ERROR(cudaStreamCreate(&stream1));

    int *dev_a0, *dev_b0, *dev_c0; // For stream0
    int *dev_a1, *dev_b1, *dev_c1; // For stream1

    HANDLE_ERROR(cudaMalloc(&dev_a0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_b0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_c0, N * sizeof(int)));

    HANDLE_ERROR(cudaMalloc(&dev_a1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_b1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_c1, N * sizeof(int)));

    cudaEvent_t start, stop;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Launch kernels in chunks of N
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        // For stream0
        HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
        // For stream1

        kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

        // Copy the data from device to locked memory
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
    }

    HANDLE_ERROR(cudaStreamSynchronize(stream0));
    HANDLE_ERROR(cudaStreamSynchronize(stream1));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time taken:" << elapsedTime << "ms" << std::endl;
    // cleanup the streams and memory
    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a0));
    HANDLE_ERROR(cudaFree(dev_b0));
    HANDLE_ERROR(cudaFree(dev_c0));
    HANDLE_ERROR(cudaFree(dev_a1));
    HANDLE_ERROR(cudaFree(dev_b1));
    HANDLE_ERROR(cudaFree(dev_c1));

    HANDLE_ERROR(cudaStreamDestroy(stream0));
    HANDLE_ERROR(cudaStreamDestroy(stream1));

    return 0;
}