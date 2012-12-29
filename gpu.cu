#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 2048
#define BLOCKSIZE 32

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            printf("Failed to run stmt %s", #stmt);    \
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
            sum += N[row * size + k] * M[k * size + col];
        }
        P[row * size + col] = sum;
    }
}


int main() {
    struct timeval start, end;

    double *h_N = (double *) malloc(SIZE * SIZE * sizeof(double));
    double *h_M = (double *) malloc(SIZE * SIZE * sizeof(double));
    double *h_P = (double *) malloc(SIZE * SIZE * sizeof(double));

    long i;
    for(i = 0; i < SIZE * SIZE; i++) {
        h_N[i] = 2.0;
        h_M[i] = 2.0;
        h_P[i] = 0.0;
    }

    double *d_N;
    double *d_M;
    double *d_P;

    wbCheck(cudaMalloc((void **) &d_N, SIZE * SIZE * sizeof(double)));
    wbCheck(cudaMalloc((void **) &d_M, SIZE * SIZE * sizeof(double)));
    wbCheck(cudaMalloc((void **) &d_P, SIZE * SIZE * sizeof(double)));

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
