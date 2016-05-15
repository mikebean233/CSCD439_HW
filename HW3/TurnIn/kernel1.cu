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

    if(row >= width - 1 || col >= width - 1 || row == 0 || col == 0)
        return;

    int sdataWidth = blockDim.x + 2;

    s_data[                 threadIdx.x + 1] = g_dataA[(row - 1) * floatpitch + col];
    s_data[    sdataWidth + threadIdx.x + 1] = g_dataA[(row    ) * floatpitch + col];
    s_data[2 * sdataWidth + threadIdx.x + 1] = g_dataA[(row + 1) * floatpitch + col];

    if(threadIdx.x == 0) {
        s_data[             0] = g_dataA[(row - 1) * floatpitch + col - 1];
        s_data[    sdataWidth] = g_dataA[(row    ) * floatpitch + col - 1];
        s_data[2 * sdataWidth] = g_dataA[(row + 1) * floatpitch + col - 1];
    }

    if(threadIdx.x == blockDim.x - 1 || col == width - 2){
        s_data[                 threadIdx.x + 2] = g_dataA[(row - 1) * floatpitch + col + 1];
        s_data[    sdataWidth + threadIdx.x + 2] = g_dataA[(row    ) * floatpitch + col + 1];
        s_data[2 * sdataWidth + threadIdx.x + 2] = g_dataA[(row + 1) * floatpitch + col + 1];
    }
    __syncthreads();

    g_dataB[row * floatpitch + col] = (
                              0.2f * s_data[    sdataWidth + threadIdx.x + 1] + //itself
                              0.1f * s_data[                 threadIdx.x + 1] + //N
                              0.1f * s_data[                 threadIdx.x + 2] + //NE
                              0.1f * s_data[    sdataWidth + threadIdx.x + 2] + //E
                              0.1f * s_data[2 * sdataWidth + threadIdx.x + 2] + //SE
                              0.1f * s_data[2 * sdataWidth + threadIdx.x + 1] + //S
                              0.1f * s_data[2 * sdataWidth + threadIdx.x    ] + //SW
                              0.1f * s_data[    sdataWidth + threadIdx.x    ] + //W
                              0.1f * s_data[                 threadIdx.x    ]   //NW
                             ) * 0.95f;

}


