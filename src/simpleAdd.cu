#include <iostream>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main(void)
{
    int c;
    int *dev_c;

    cudaError_t err = cudaMalloc(&dev_c, sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    add<<<1, 1>>>(2, 7, dev_c);

    err = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c);

    return 0;
}