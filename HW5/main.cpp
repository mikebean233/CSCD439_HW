/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"


void mergeSort(
        uint *d_DstKey,
        uint *d_DstVal,
        uint *d_BufKey,
        uint *d_BufVal,
        uint *d_SrcKey,
        uint *d_SrcVal,
        uint N,
        uint sortDir);



////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
    StopWatchInterface *hTimer = NULL;

    const uint   N = 48 * 1048576;
    const uint DIR = 1;
    const uint numValues = 65536;

    printf("%s Starting...\n\n", argv[0]);

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_SrcKey = (uint *)malloc(N * sizeof(uint));
    h_SrcVal = (uint *)malloc(N * sizeof(uint));
    h_DstKey = (uint *)malloc(N * sizeof(uint));
    h_DstVal = (uint *)malloc(N * sizeof(uint));

    srand(2009);

    for (uint i = 0; i < N; i++)
    {
        h_SrcKey[i] = rand() % numValues;
    }

    fillValues(h_SrcVal, N);

    printf("Allocating and initializing CUDA arrays...\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Initializing GPU merge sort...\n");
    initMergeSort();

    printf("Running GPU merge sort...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    mergeSort(
        d_DstKey,
        d_DstVal,
        d_BufKey,
        d_BufVal,
        d_SrcKey,
        d_SrcVal,
        N,
        DIR
    );
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

    printf("Reading back GPU merge sort results...\n");
    checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));
/*check last few elements
    for( int i = N - 10; i < N; i ++ )
    {
        printf("key: %d, value: %d\n", h_DstKey[i], h_DstVal[i]);
    }
*/

    printf("Inspecting the results...\n");
    uint keysFlag = validateSortedKeys(
                        h_DstKey,
                        h_SrcKey,
                        1,
                        N,
                        numValues,
                        DIR
                    );

    uint valuesFlag = validateSortedValues(
                          h_DstKey,
                          h_DstVal,
                          h_SrcKey,
                          1,
                          N
                      );

    printf("Shutting down...\n");
    closeMergeSort();
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_SrcVal));
    checkCudaErrors(cudaFree(d_SrcKey));
    checkCudaErrors(cudaFree(d_BufVal));
    checkCudaErrors(cudaFree(d_BufKey));
    checkCudaErrors(cudaFree(d_DstVal));
    checkCudaErrors(cudaFree(d_DstKey));
    free(h_DstVal);
    free(h_DstKey);
    free(h_SrcVal);
    free(h_SrcKey);
    cudaDeviceReset();

    exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}


void mergeSort(
        uint *d_DstKey,
        uint *d_DstVal,
        uint *d_BufKey,
        uint *d_BufVal,
        uint *d_SrcKey,
        uint *d_SrcVal,
        uint N
){
    uint stageCount = 0;

    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

    uint *ikey, *ival, *okey, *oval;

    if (stageCount & 1) {
        ikey = d_BufKey;
        ival = d_BufVal;
        okey = d_DstKey;
        oval = d_DstVal;
    }
    else {
        ikey = d_DstKey;
        ival = d_DstVal;
        okey = d_BufKey;
        oval = d_BufVal;
    }

        //    assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
    //    assert(N % SHARED_SIZE_LIMIT == 0);
    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT);

    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1) {
        uint lastSegmentElements = N % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N);

        if (lastSegmentElements <= stride) {
            //Last merge segment consists of a single array which just needs to be passed through
            checkCudaErrors(cudaMemcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements),lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements),lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        uint *t;
        t = ikey;
        ikey = okey;
        okey = t;
        t = ival;
        ival = oval;
        oval = t;
    }
}
