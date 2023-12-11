#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h> 


void MatrixInit(float *M, int n, int p, int l, bool init_zero){

    srand(40);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l;k++){
                if(init_zero){
                    M[i*p + j*l + k] = 0;    
                }
                else{
                    M[i*p + j*l + k] = (float)(rand()/(float)(RAND_MAX));    
                }
            }
            
                    }
    }

}

void MatrixPrint(float *M, int n, int p,int l){
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l; k++){
                printf("%f ", M[i*p + j*l + k]);
            }
            
        }
        printf("\n");
    }
}

void Matrix_kernel_init(float *M, int n, int p,int l){
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l; k++){
                printf("%f ", M[i*p + j*l + k]);
            }
            
        }
        printf("\n");
    }
}

__global__ void cudaMatrixMult_dot(float *M1, float *M2, float *Mout, int n){

    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    
    *(Mout+i) = *(M1+i)*(*(M2+i));
}

_global__ void cudaConvolution(float *M1, float *kernel, float *Mout, int input_size, int kernel_size, int n_kernels){

    //Stride = 1

    int n_points = input_size-(kernel_size-1) + 1;

    for(int i = 0; i < n_points;i++){
        for(int j = 0; i < n_points;j++){


            


            Mout[i*n_points+j] = 

        }

    }

    

    
}



int main(int argc, char *argv[]){

    float *raw_data,*C1_data,*S1_data,*C1_kernel;


    cudaMallocManaged((void**)&raw_data, sizeof(float)*32*32);
    cudaMallocManaged((void**)&C1_data, sizeof(float)*28*28*6);
    cudaMallocManaged((void**)&S1_data, sizeof(float)*14*14*6);
    cudaMallocManaged((void**)&C1_kernel, sizeof(float)*5*5*6);

    //initializations
    MatrixInit(raw_data,32,32,1,false);
    MatrixInit(C1_data,28,28,6,true);
    MatrixInit(S1_data,14,14,6,true);
    MatrixInit(C1_kernel,5,5,6,false);
    cudaDeviceSynchronize();

    MatrixPrint(C1_kernel,5,5,6);


}