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

__global__ void cudaConvolution2d(float *input, float *kernels, float *output, int input_size, int kernel_size, int n_kernels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //32
    int j = blockIdx.y * blockDim.y + threadIdx.y; //32
    int k = blockIdx.z; // 6

    int n_points = input_size - (kernel_size - 1); //28

    if (i < n_points && j < n_points && k < n_kernels) {
        float sum = 0.0;

        // Calcul des indices pour le noyau et l'output
        int kernelStartIndex = k * kernel_size * kernel_size;
        int outputIndex = k * n_points * n_points + i * n_points + j;

        // Boucle sur le noyau
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                int inputIndex = (i + ki) * input_size + (j + kj);
                int kernelIndex = kernelStartIndex + ki * kernel_size + kj;
                sum += (input[inputIndex]) * (kernels[kernelIndex]);
            }
        }

        output[outputIndex] = sum;
    }
}




int main(int argc, char *argv[]){

    float *raw_data,*C1_data,*S1_data,*C1_kernel;
    
    raw_data = (float*) malloc(sizeof(float)*32*32);
    C1_data = (float*) malloc(sizeof(float)*28*28*6);
    S1_data = (float*) malloc(sizeof(float)*14*14*6);
    C1_kernel = (float*) malloc(sizeof(float)*5*5*6);



    //initializations
    MatrixInit(raw_data,32,32,1,false);
    MatrixInit(C1_kernel,5,5,6,false);

    float *raw_data_d,*C1_data_d,*S1_data_d,*C1_kernel_d;
    cudaMalloc((void**)&raw_data_d, sizeof(float)*32*32);
    cudaMalloc((void**)&C1_data_d, sizeof(float)*28*28*6);
    cudaMalloc((void**)&S1_data_d, sizeof(float)*14*14*6);
    cudaMalloc((void**)&C1_kernel_d, sizeof(float)*5*5*6);


    dim3 dimBlock(6, 6); // Taille du bloc (ajustez en fonction des performances et des limitations du matériel)
    dim3 dimGrid((28 + dimBlock.x - 1) / dimBlock.x, (28 + dimBlock.y - 1) / dimBlock.y, 6);
    cudaMemcpy(raw_data_d,raw_data,sizeof(float)*32*32,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_d,C1_kernel,sizeof(float)*5*5*6,cudaMemcpyHostToDevice);

    cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);
    cudaConvolution2d<<<dimGrid, dimBlock>>>(raw_data_d, C1_kernel_d, C1_data_d, 32, 5, 6);
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(stop_GPU);
    float elapsedTime_GPU;
    cudaEventElapsedTime(&elapsedTime_GPU, start_GPU, stop_GPU);
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);
    printf("GPU conv: %f ms\n", elapsedTime_GPU);

    cudaMemcpy(C1_data,C1_data_d,sizeof(float)*28*28*6,cudaMemcpyDeviceToHost);

    //MatrixPrint(C1_kernel,5,5,6);
    //MatrixPrint(raw_data,32,32,1);
    MatrixPrint(C1_data,28,28,6);
    printf("GPU conv: %f ms\n", elapsedTime_GPU);

    cudaFree(raw_data_d);
    cudaFree(C1_data_d);
    cudaFree(S1_data_d);
    cudaFree(C1_kernel_d);
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}