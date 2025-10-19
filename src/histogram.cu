#include "cuda_error.h"
#include <iostream>
#include <iomanip>

#define SIZE 100 * 1024 * 1024

void *big_random_block(int size)
{
    unsigned char *data = (unsigned char *)malloc(size);
    HANDLE_NULL(data);
    for (int i = 0; i < size; i++)
        data[i] = rand();

    return data;
}

__global__ void histo_kernel(unsigned char *buffer, int size, unsigned int *histo)
{
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0; // this initializes all indices as 0
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size)
    {
        atomicAdd(&temp[buffer[i]], 1);
        i += stride;
    }

    __syncthreads();
    atomicAdd(&histo[threadIdx.x], temp[threadIdx.x]);
}

int main(void)
{
    unsigned char *buffer = (unsigned char *)big_random_block(SIZE);
    unsigned int histo[256];

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    unsigned char *dev_buffer;
    unsigned int *dev_histo;

    HANDLE_ERROR(cudaMalloc(&dev_buffer, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&dev_histo, 256 * sizeof(long)));
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties_v2(&props, 0));
    int procs = props.multiProcessorCount; // Empirical result
    histo_kernel<<<2 * procs, 256>>>(dev_buffer, SIZE, dev_histo);

    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elasped_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elasped_time, start, stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Elapsed time: " << elasped_time << "ms" << std::endl;

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    free(buffer);
    cudaFree(dev_buffer);
    cudaFree(dev_histo);
    return 0;
}