#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "pgmProcess.h"
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] )
{
    int distanceX = p2[1] - p1[1];
    int distanceY = p2[0] - p1[0];
    return sqrt((float)(distanceX * distanceX) + (float)(distanceY * distanceY));
    return 0.0;
}

int  pgmDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, char **header) {
    int* dPixels;
    int blockSize = 64;
    int gridSize = ceil(((double)numRows * (double)numCols) / (double) blockSize);
    int arraySizeInBytes = sizeof(int) * numRows * numCols;

    // allocate device memory for the array
    cudaMalloc(&dPixels, arraySizeInBytes);

    // copy the cpu memory to the gpu
    cudaMemcpy(dPixels, pixels, arraySizeInBytes, cudaMemcpyHostToDevice);

    // run the kernel
    gpuDrawEdge<<<gridSize, blockSize>>>(dPixels, numRows, numCols, edgeWidth, numRows * numCols);

    // copy the results back to the host array
    cudaMemcpy(pixels, dPixels, arraySizeInBytes, cudaMemcpyDeviceToHost);

    // release the device array
    cudaFree(dPixels);
    return 0;
}

int pgmDrawCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header) {

    int* dPixels;
    int blockSize = 64;
    int gridSize = ceil(((double)numRows * (double)numCols) / (double) blockSize);
    int arraySizeInBytes = sizeof(int) * numRows * numCols;

    // allocate device memory for the array
    cudaMalloc(&dPixels, arraySizeInBytes);

    // copy the cpu memory to the gpu
    cudaMemcpy(dPixels, pixels, arraySizeInBytes, cudaMemcpyHostToDevice);

    // run the kernel
    gpuDrawCircle<<<gridSize, blockSize>>>(dPixels, numRows, numCols, centerRow, centerCol, radius, numRows * numCols);

    // copy the results back to the host array
    cudaMemcpy(pixels, dPixels, arraySizeInBytes, cudaMemcpyDeviceToHost);

    // release the device array
    cudaFree(dPixels);

    return 0;
}

int pgmDrawLine(int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col){
    int* dPixels;
    int blockSize = 64;
    int arraySizeInBytes = sizeof(int) * numRows * numCols;

    int pa[] = {0,0};
    int pb[] = {0,0};
    int vertical = 0;
    float slope = 0.0;
    if(p1col < p2col){
        pa[0] = p1row;
        pa[1] = p1col;
        pb[0] = p2row;
        pb[1] = p2col;
    }
    else{
        pa[0] = p2row;
        pa[1] = p2col;
        pb[0] = p1row;
        pb[1] = p1col;
    }

    if(pa[1] == pb[1])
        vertical = 1;
    else
        slope = (pb[0] - pa[0]) / (pb[1] - pa[1]);

    int startCol = pa[1];
    int noSamples = (vertical) ? ((pa[0] < pb[0]) ? pb[0] - pa[0] : pa[0] - pb[0]) : pb[1] - pa[0];

    int gridSize = ceil(((double)noSamples) / (double) blockSize);

    // allocate device memory for the array
    cudaMalloc(&dPixels, arraySizeInBytes);

    // copy the cpu memory to the gpu
    cudaMemcpy(dPixels, pixels, arraySizeInBytes, cudaMemcpyHostToDevice);

    // run the kernel
    gpuDrawLine<<<gridSize, blockSize>>>(dPixels, noRows, noCol, pa[0], pa[1], vertical, slope, noSamples);

    // copy the results back to the host array
    cudaMemcpy(pixels, dPixels, arraySizeInBytes, cudaMemcpyDeviceToHost);

    // release the device array
    cudaFree(dPixels);

    return 0;
}

__global__ void gpuDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int thisRow  = threadId / numCols;
    int thisCol  = threadId % numCols;

    if(thisRow <= edgeWidth ||
       thisRow >= numRows - edgeWidth ||
       thisCol <= edgeWidth ||
       thisCol >= numCols - edgeWidth){
        pixels[threadId] = 0;
    }
}

__global__ void gpuDrawCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, int n) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int thisRow  = threadId / numCols;
    int thisCol  = threadId % numCols;

    int p1[] = {thisRow, thisCol};
    int p2[] = {centerRow, centerCol};

    if(distance(p1, p2) <= radius)
        pixels[threadId] = 0;
}

__global__ void gpuDrawLine(int *pixels, int noRows, int noCols, int startRow, int startCol, int vertical, float slope, int noSamples);
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int position = startCol + threadId;
    int thisRow, thisCol;
    if(vertical){
        thisRow = startRow + position;
        thisCol = startCol;
    }
    else{
        thisCol = startCol + position;
        thisRow = startRow + (slope * startRow);
    }
    int index = noCols * thisRow + thisCol;
    pixels[index] = 0;
}


