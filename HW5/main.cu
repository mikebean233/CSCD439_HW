#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MIN_ARG_COUNT 3
#define MAX_ARG_COUNT 3
#define MAX_VALUE 10000

void usage(char* programName, FILE* outFileDisc, int exitStatus);
void errorExit(char* message, int exitStatus);
void printArray(uint* array, int length);

__global__ void mergSort(uint* SrcKey, uint* SrcVal, uint* DstKey, uint* DstVal, uint n){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int blockId  = blockIdx.y * gridDim.x  + blockIdx.x;
    if(threadId > n)
        return;


    DstKey[threadId] = threadId;
}

int main(int argc, char** argv){
    uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
    uint *d_SrcKey, *d_SrcVal, *d_DstKey, *d_DstVal;
    uint3 blockDim, gridDim;
    uint n, noTiles;

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
    h_SrcKey  = (uint*) calloc(sizeof(uint), n);
    h_SrcVal  = (uint*) calloc(sizeof(uint), n);
    h_DstKey  = (uint*) calloc(sizeof(uint), n);
    h_DstVal  = (uint*) calloc(sizeof(uint), n);
    cudaMalloc((void **)&d_SrcKey, n * sizeof(uint));
    cudaMalloc((void **)&d_DstKey, n * sizeof(uint));
    cudaMalloc((void **)&d_SrcKey, n * sizeof(uint));
    cudaMalloc((void **)&d_DstKey, n * sizeof(uint));

    // Copy the host array to device array
    cudaMemcpy(d_SrcKey, h_SrcKey, n * sizeof(uint), cudaMemcpyHostToDevice);

    if(h_SrcKey == NULL || h_DstKey == NULL)
        errorExit((char*)"Error: host was unable to allocate array memory", 2);

    srand(time(NULL));

    uint i;
    for(i = 0; i < n; ++i){
        h_SrcKey[i] = rand() % MAX_VALUE;
        h_SrcVal[i] = i;
    }

    mergSort<<<blockDim, gridDim>>>(d_SrcKey, d_SrcVal, d_DstKey, d_DstVal, n);
    cudaMemcpy(h_DstKey, d_DstKey, n * sizeof(uint), cudaMemcpyDeviceToHost);

    printArray(h_DstKey, n);
}

void errorExit(char* message, int exitStatus){
    fprintf(stderr, "%s", message);
    exit(exitStatus);
}

void usage(char* programName, FILE* outFileDisc, int exitStatus){
    fprintf(outFileDisc, "usage: %s arraySize blockSize\n", programName);
    exit(exitStatus);
}

void printArray(uint* array, int length){
    int i = 0;
    for(; i < length; ++i){
        printf("%d\n", array[i]);
    }
}







