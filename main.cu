#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

__global__ void cuda_hello(){  //GPU
    printf("Hello World from GPU!\n");
}

int main() { //CPU
    //cuda_hello<<<1,1>>>(); 
    void MatrixInit(float *M, int 5, int 6);
    void MatrixPrint(float *M, int 5, int 6);
    cudaDeviceSynchronize();

    return 0;
}
 
