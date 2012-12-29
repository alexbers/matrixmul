#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 2048
#define BLOCKSIZE 32

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute P = N * M
__global__ void matrixMultiply(double * N, double * M, double * P, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < size) && (col < size)) {
        double sum = 0;
        for(int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        P[row * size + col] = sum;
    }
}


int main() {
    struct timeval start, end;

    double *h_N = malloc(SIZE * SIZE * sizeof(double));
    double *h_M = malloc(SIZE * SIZE * sizeof(double));
    double *h_P = malloc(SIZE * SIZE * sizeof(double));

    long i, j, k;
    for(i = 0; i < SIZE * SIZE; i++) {
        N[i] = 2.0;
        M[i] = 2.0;
        P[i] = 0.0;
    }

    double *d_N;
    double *d_M;
    double *d_P;

    wbCheck(cudaMalloc((void **) &d_N, SIZE * SIZE * sizeof(double)));
    wbCheck(cudaMalloc((void **) &d_M, SIZE * SIZE * sizeof(double)));
    wbCheck(cudaMalloc((void **) &d_P, SIZe * SIZE * sizeof(double)));

    wbCheck(cudaMemcpy(d_N, h_N, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(d_M, h_M, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));

    dim3 dimGrid(SIZE / BLOCKSIZE, SIZE / BLOCKSIZE,1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);

    gettimeofday(&start, NULL);

    matrixMultiply<<<dimGrid,dimBlock>>>(d_N, d_M, d_P, SIZE);
    cudaThreadSynchronize();

    gettimeofday(&end, NULL);

    wbCheck(cudaMemcpy(h_P, d_P, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    // time calculation
    if(end.tv_sec < start.tv_sec) {
        printf("You are very unlucky, please, run me again\n");
        return 1;
    }

    double usec_diff = (end.tv_sec - start.tv_sec) +
                       (double)(end.tv_usec - start.tv_usec) / 1000 / 1000;
    double time_spent = (double)(usec_diff);

    printf("Multiplication finished, wallclock: %f sec\n", time_spent);

    return 0;
}
