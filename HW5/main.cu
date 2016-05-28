#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>

#define MIN_ARG_COUNT 3
#define MAX_ARG_COUNT 3
#define MAX_VALUE 10000

void usage(char* programName, FILE* outFileDisc, int exitStatus);
void errorExit(char* message, int exitStatus);
void printArray(uint* array, int length);

__global__ void mergSort(uint* in, uint* out, uint n){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId > n)
        return;

    out[threadId] = threadId;
}

int main(int argc, char** argv){
    uint *h_inArray, *h_outArray, *d_inArray, *d_outArray;
    uint3 blockDim, gridDim;
    uint n;

    // Deal with command line arguments
    if(argc < MIN_ARG_COUNT || argc > MAX_ARG_COUNT)
        usage(argv[0], stderr, 1);

    n = atoi(argv[1]);
    blockDim.x = atoi(argv[2]);
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim.x = ceil((float)n / blockDim.x);
    gridDim.y = 1;
    gridDim.z = 1;

    if(n % 2 != 0)
        errorExit((char*)"Error: the number of input elements must be even", 1);

    // Allocate memory
    h_inArray  = (uint*) calloc(sizeof(uint), n);
    h_outArray = (uint*) calloc(sizeof(uint), n);
    checkCudaErrors(cudaMalloc((void **)&d_inArray, n * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_outArray, n * sizeof(uint)));


    // Copy the host array to device array
    checkCudaErrors(cudaMemcpy(d_inArray, h_inArray, n * sizeof(uint), cudaMemcpyHostToDevice));


    if(h_inArray == NULL || h_outArray == NULL)
        errorExit((char*)"Error: host was unable to allocate array memory", 2);

    srand(time());

    uint i;
    for(i = 0; i < n; ++i){
         h_inArray[i] = rand() % MAX_VALUE;
    }

    mergSort<<<blockDim, gridDim>>>(d_inArray, d_outArray, n);

    checkCudaErrors(cudaMemcpy(h_outArray, d_outArray, n * sizeof(uint), cudaMemcpyDeviceToHost));
}

void errorExit(char* message, int exitStatus){
    fprintf(stderr, message);
    exit(exitStatus);
}

void usage(char* programName, FILE* outFileDisc, int exitStatus){
    fprintf(outFileDisc, "usage: %s arraySize blockSize");
    exit(exitStatus);
}

void printArray(uint* array, int length){
    int i = 0;
    for(; i < length; ++i){
        printf("$d\n", array[i]);
    }
}







