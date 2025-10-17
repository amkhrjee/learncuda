#include <iostream>

#define N 256

__global__ void MatAdd(int A[N][N], int B[N][N], int C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main(void)
{
    int A[N][N], B[N][N], C[N][N];
    int (*d_A)[N], (*d_B)[N], (*d_C)[N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = i * j;
            B[i][j] = i + j;
        }
    }

    cudaMalloc(&d_A, sizeof(int) * N * N);
    cudaMalloc(&d_B, sizeof(int) * N * N);
    cudaMalloc(&d_C, sizeof(int) * N * N);

    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    //  We're running one 2D thread block of dim N x N
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}