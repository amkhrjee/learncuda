// Cuda graphs help curb kernel launch times
// This is primarily aimed at reducing CPU times,
// and thus the overall time it takes to run a program

// I am following this tutorial: https://developer.nvidia.com/blog/cuda-graphs/

#include <chrono>
#include <iostream>

#define N 500000 // tuned such that kernel takes a few microseconds

#define NSTEP 1000
#define NKERNEL 20

__global__ void shortKernel(float *out_d, float *in_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out_d[idx] = 1.23 * in_d[idx];
}

int main(void)
{
    float *host_in, *host_out;

    // Page-locked memory
    cudaHostAlloc(&host_in, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&host_out, N * sizeof(float), cudaHostAllocDefault);

    for (int i = 0; i < N; i++)
    {
        host_in[i] = rand();
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *dev_in, *dev_out;

    // Allocate GPU memory
    cudaMalloc(&dev_in, N * sizeof(float));
    cudaMalloc(&dev_out, N * sizeof(float));

    int threadsPerBlock = 1 << 10;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // Without using cuda graph

    // for (int istep = 0; istep < NSTEP; istep++)
    // {
    //     for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
    //     {
    //         shortKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dev_out, dev_in);
    //         // Subsequent kernel is not launched until current kernel is done
    //         // cudaStreamSynchronize(stream); <- if this is here
    //     }
    //     cudaStreamSynchronize(stream); // this is an improvement
    // }

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    for (int istep = 0; istep < NSTEP; istep++)
    {
        if (!graphCreated)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
            {
                shortKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dev_out, dev_in);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return 0;
}