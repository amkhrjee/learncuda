#ifndef CUDA_ERROR
#define CUDA_ERROR
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define HANDLE_ERROR(call)                                                                             \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0);

#endif