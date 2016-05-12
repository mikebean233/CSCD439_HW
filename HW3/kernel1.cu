#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];
    // TODO, implement this kernel below

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if(row >= width - 1 || col >= width)
        return;


    // Copy from global memory to shared memory
    int i = 0;
    for(i = 0; i < 3; ++i){
        s_data[(i * (blockDim.x + 2)) + threadIdx.x] = g_dataA[(row - (i - 1)) * floatpitch + col];

        if(threadIdx.x == 0 || threadIdx.x == blockDim.x + 2)
            s_data[(i * (blockDim.x + 2))] = g_dataA[(row - (i - 1)) * floatpitch];

        if(threadIdx.x == blockDim.x - 1)
            s_data[(i * (blockDim.x + 2)) + blockDim.x - 1] = g_dataA[(row - (i - 1)) * floatpitch + col + 1];
    }
    __syncthreads();


    /*
        s_data[(0 * (blockDim.x + 2)) + threadIdx.x] = g_dataA[(row - 1) * pitch + col];
        s_data[(1 * (blockDim.x + 2)) + threadIdx.x] = g_dataA[(row - 0) * pitch + col];
        s_data[(2 * (blockDim.x + 2)) + threadIdx.x] = g_dataA[(row + 1) * pitch + col];
    */

    g_dataB[row * floatpitch + col] = (
                              0.2f * s_data[1 * (blockDim.x + 2) + threadIdx.x]     + //itself
                              0.1f * s_data[0 * (blockDim.x + 2) + threadIdx.x]     + //N
                              0.1f * s_data[0 * (blockDim.x + 2) + threadIdx.x + 1] + //NE
                              0.1f * s_data[1 * (blockDim.x + 2) + threadIdx.x + 1] + //E
                              0.1f * s_data[2 * (blockDim.x + 2) + threadIdx.x + 1] + //SE
                              0.1f * s_data[2 * (blockDim.x + 2) + threadIdx.x]     + //S
                              0.1f * s_data[2 * (blockDim.x + 2) + threadIdx.x - 1] + //SW
                              0.1f * s_data[1 * (blockDim.x + 2) + threadIdx.x - 1] + //W
                              0.1f * s_data[0 * (blockDim.x + 2) + threadIdx.x - 1]   //NW
                             ) * 0.95f;

}


