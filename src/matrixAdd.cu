#include <iostream>

#define N 10

__global__ void MatAdd(int A[N][N], int B[N][N], int C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
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

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    int numBlocks = 1;
    //  We're running one 2D thread block of dim N x N
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d\t", C[i][j]);
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}