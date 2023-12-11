#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h> 



#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>


void MatrixInit(float *M, int n, int p){

    srand(40);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            M[i*p + j] = (float)(2*(rand()/(float)(RAND_MAX)-0.5));
        }
    }

}

void MatrixPrint(float *M, int n, int p){
    
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                printf("%f ", M[i*p + j]);
            }
            printf("\n");
        }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    
    *(Mout+i) = *(M1+i) + *(M2+i);
        
    
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            Mout[i*n + j] = 0;
            for(int k = 0; k < n; k++){
                Mout[i*n + j] += M1[i*n + k] * M2[k*n + j];
            }
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){

    int i = blockIdx.x;
    int j = threadIdx.x;  
       
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {

        sum += *(M1+i*blockDim.x+k)*( *(M2+k*blockDim.x+j));
        //sum += M1[i * n + k] * M2[k * n + j];
    }
    *(Mout +blockDim.x*i+j) = sum;
    //Mout[i * n + j] = sum;
}

int main(int argc, char *argv[]){

    int n = 3;
    int p = 4;
    float *M1, *M2, *Mout, *Mout_mult;
    M1   = (float*)malloc(sizeof(float) * n*p);
    M2   = (float*)malloc(sizeof(float) * n*p);
    Mout   = (float*)malloc(sizeof(float) * n*p);
    Mout_mult   = (float*)malloc(sizeof(float) * n*p);

    float *M1_c, *M2_c, *Mout_c, *Mout_multc;

    //cudaMalloc((void**)&M, sizeof(float)*n*p);
    cudaMallocManaged((void**)&M1_c, sizeof(float)*n*p);
    cudaMallocManaged((void**)&M2_c, sizeof(float)*n*p);
    cudaMallocManaged((void**)&Mout_c, sizeof(float)*n*p);
    cudaMallocManaged((void**)&Mout_multc, sizeof(float)*n*p);

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    MatrixInit(Mout, n, p);

    MatrixInit(M1_c, n, p);
    MatrixInit(M2_c, n, p);
    MatrixInit(Mout_c, n, p);

    cudaDeviceSynchronize();

    MatrixAdd(M1, M2, Mout, n, p);

    clock_t start, end;
    double cpu_time_used=0;

    start = clock();
    MatrixMult(M1, M2, Mout_mult, n);
    end = clock();
    cpu_time_used = ((double) (end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU matmult: %f ms\n", cpu_time_used);

    
    cudaMatrixAdd<<<n,p>>>(M1_c, M2_c, Mout_c, n, p);
    
    cudaDeviceSynchronize();
   
    // Measure GPU execution time
    cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);
    cudaMatrixMult<<<n, n>>>(M1_c, M2_c, Mout_multc, n);
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(stop_GPU);
    float elapsedTime_GPU;
    cudaEventElapsedTime(&elapsedTime_GPU, start_GPU, stop_GPU);
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);
    printf("GPU matmult: %f ms\n", elapsedTime_GPU);
    
    cudaDeviceSynchronize();
    
    //MatrixPrint(M1_c, n, p);
    //MatrixPrint(M2_c, n, p);
    //MatrixPrint(Mout_c, n, p);

    //MatrixPrint(Mout , n, p);

    //MatrixPrint(Mout_c , n, p);
    MatrixPrint(Mout_mult , n, p);
    MatrixPrint(Mout_multc , n, p);
    
    
    cudaFree(M1_c);
    cudaFree(M2_c);
    cudaFree(Mout_c);
    return 0;
}
