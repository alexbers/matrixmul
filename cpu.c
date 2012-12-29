#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 1.5GB
#define SIZE 8192


int main() {
    struct timeval start, end;

    double *N = malloc(SIZE * SIZE * sizeof(double));
    double *M = malloc(SIZE * SIZE * sizeof(double));
    double *P = malloc(SIZE * SIZE * sizeof(double));

    // calculate P = N * M
    long i, j, k;
    for(i = 0; i < SIZE * SIZE; i++) {
        N[i] = 2.0;
        M[i] = 2.0;
        P[i] = 0.0;
    }

    // do the multiplication of two matrix
    gettimeofday(&start, NULL);

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            double sum = 0;
            for (k = 0; k < SIZE; k++)
                sum += N[i * SIZE + k] + M[k * SIZE + j];
            P[i * SIZE + j] = sum;
        }
    }

    gettimeofday(&end, NULL);


    // time calculation
    if(end.tv_usec < start.tv_usec) {
        printf("You are very unlucky, please, run me again\n");
        return 1;
    }

    double usec_diff = (end.tv_sec - start.tv_sec) +
                       (double)(end.tv_usec - start.tv_usec) / 1000 / 1000;
    double time_spent = (double)(usec_diff);

    printf("Multiplication finished, wallclock: %f sec\n", time_spent);

    return 0;
}